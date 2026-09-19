import os
import sys
import gc
import time
from pathlib import Path
from types import SimpleNamespace

# Ensure torch lib DLLs are loaded on Windows
torch_lib = r"D:\Vggt\drone_vggt_env\Lib\site-packages\torch\lib"
if os.path.exists(torch_lib) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(torch_lib)
    except Exception:
        pass

import torch
import torch.nn as nn
import numpy as np

# Add vggt_repo and training directories
sys.path.insert(0, str(Path("vggt_repo").resolve()))
sys.path.insert(0, str(Path("vggt_repo/training").resolve()))

from vggt.models.vggt import VGGT
from loss import MultitaskLoss
from train_utils.freeze import freeze_modules
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from data.datasets.uav_dataset import UAVDataset

def run_verification(target_res=392, device_name="cuda"):
    print(f"\n========================================================")
    print(f" DRY-RUN VERIFICATION: RES={target_res}x{target_res} on {device_name.upper()}")
    print(f"========================================================")

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Target Device: {device}")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {torch.cuda.get_device_name(0)} with {vram_total:.2f} GB total VRAM")

    # 1. Instantiate Model Architecture (img_size=518 matches DINOv2 pretrained weights)
    print("\n[STEP 1] Instantiating VGGT model (camera=True, depth=True, point=False, track=False, img_size=518)...")
    model = VGGT(enable_camera=True, enable_depth=True, enable_point=False, enable_track=False, img_size=518)
    
    # 2. Load Checkpoint
    ckpt_path = "models/VGGT-1B/model.pt"
    if os.path.exists(ckpt_path):
        print(f"[STEP 2] Loading weights from local checkpoint: {ckpt_path}...")
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Checkpoint loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        print(f"[WARNING] Checkpoint {ckpt_path} not found. Running with initialized weights.")

    # 3. PART 8: FREEZE AGGREGATOR
    print("\n[STEP 3 / PART 8] Freezing Aggregator module...")
    model = freeze_modules(model, patterns=["*aggregator*"])

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    agg_total = sum(p.numel() for p in model.aggregator.parameters())
    agg_trainable = sum(p.numel() for p in model.aggregator.parameters() if p.requires_grad)

    cam_total = sum(p.numel() for p in model.camera_head.parameters())
    cam_trainable = sum(p.numel() for p in model.camera_head.parameters() if p.requires_grad)

    depth_total = sum(p.numel() for p in model.depth_head.parameters())
    depth_trainable = sum(p.numel() for p in model.depth_head.parameters() if p.requires_grad)

    print(f"\n--- PARAMETER AUDIT ---")
    print(f"Total Parameters:      {total_params:,} (~{total_params/1e6:.2f}M)")
    print(f"Trainable Parameters:  {trainable_params:,} (~{trainable_params/1e6:.2f}M) ({trainable_params/total_params*100:.2f}%)")
    print(f"Frozen Parameters:     {frozen_params:,} (~{frozen_params/1e6:.2f}M)")
    print(f" - Aggregator:   Total={agg_total:,}, Trainable={agg_trainable:,} (requires_grad={agg_trainable > 0})")
    print(f" - Camera Head:  Total={cam_total:,}, Trainable={cam_trainable:,} (requires_grad={cam_trainable > 0})")
    print(f" - Depth Head:   Total={depth_total:,}, Trainable={depth_trainable:,} (requires_grad={depth_trainable > 0})")

    assert agg_trainable == 0, "ERROR: Aggregator has trainable parameters!"
    assert cam_trainable > 0, "ERROR: Camera head has no trainable parameters!"
    assert depth_trainable > 0, "ERROR: Depth head has no trainable parameters!"
    print("[SUCCESS] Aggregator freeze verified successfully!")

    # 4. Move Model to Device
    print(f"\n[STEP 4] Moving model to {device}...")
    if device.type == "cpu":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    try:
        model = model.to(device=device, dtype=dtype)
        if device.type == "cuda":
            model_mem_mb = torch.cuda.memory_allocated(0) / (1024**2)
            print(f"Model memory on GPU: {model_mem_mb:.2f} MB")
    except torch.cuda.OutOfMemoryError as e:
        print(f"[OOM] Cannot allocate model weights on GPU ({e}). Falling back to CPU for dry run.")
        device = torch.device("cpu")
        model = model.to(device=device, dtype=torch.float32)

    # 5. Load 1 Real Training Batch from UAVDataset
    print("\n[STEP 5 / PART 9] Loading 1 real multi-view batch from UAVDataset...")
    common_conf = SimpleNamespace(
        img_size=target_res,
        patch_size=14,
        augs=SimpleNamespace(scales=[0.8, 1.2]),
        rescale=True,
        rescale_aug=False,
        landscape_check=False,
        debug=False,
        training=True,
        get_nearby=True,
        load_depth=True,
        inside_random=False,
        allow_duplicate_img=False,
    )
    dataset = UAVDataset(
        common_conf=common_conf,
        split="train",
        DATA_DIR="data/uav_finetune_test",
        MANIFEST_PATH="data/uav_finetune_test/train.json",
        min_num_images=4,
        len_train=1,
    )

    sample = dataset[(0, 4, 1.0)]

    # Collate into tensors with batch dimension B=1
    images = torch.from_numpy(np.stack(sample["images"]).astype(np.float32)).permute(0, 3, 1, 2).div(255.0).unsqueeze(0) # [B=1, S=4, 3, H, W]
    depths = torch.from_numpy(np.stack(sample["depths"]).astype(np.float32)).unsqueeze(0) # [B=1, S=4, H, W]
    extrinsics = torch.from_numpy(np.stack(sample["extrinsics"]).astype(np.float32)).unsqueeze(0) # [B=1, S=4, 4, 4]
    extrinsics = extrinsics[..., :3, :] # [B=1, S=4, 3, 4] as required by normalization.py
    intrinsics = torch.from_numpy(np.stack(sample["intrinsics"]).astype(np.float32)).unsqueeze(0) # [B=1, S=4, 3, 3]
    cam_points = torch.from_numpy(np.stack(sample["cam_points"]).astype(np.float32)).unsqueeze(0) # [B=1, S=4, H, W, 3]
    world_points = torch.from_numpy(np.stack(sample["world_points"]).astype(np.float32)).unsqueeze(0) # [B=1, S=4, H, W, 3]
    point_masks = torch.from_numpy(np.stack(sample["point_masks"])).unsqueeze(0) # [B=1, S=4, H, W]

    # Batch Normalization (as in trainer._process_batch)
    norm_extri, norm_cam_pts, norm_world_pts, norm_depths = normalize_camera_extrinsics_and_points_batch(
        extrinsics=extrinsics,
        cam_points=cam_points,
        world_points=world_points,
        depths=depths,
        point_masks=point_masks,
    )

    batch = {
        "images": images.to(device=device, dtype=dtype if device.type == "cuda" else torch.float32),
        "depths": norm_depths.to(device=device),
        "extrinsics": norm_extri.to(device=device),
        "intrinsics": intrinsics.to(device=device),
        "cam_points": norm_cam_pts.to(device=device),
        "world_points": norm_world_pts.to(device=device),
        "point_masks": point_masks.to(device=device),
        "seq_name": "seq_uav_test_001",
    }

    print(f"Batch shapes on {device}:")
    print(f" - images:       {batch['images'].shape}")
    print(f" - depths:       {batch['depths'].shape}")
    print(f" - extrinsics:   {batch['extrinsics'].shape}")
    print(f" - intrinsics:   {batch['intrinsics'].shape}")
    print(f" - point_masks:  {batch['point_masks'].shape}")

    # 6. Instantiate MultitaskLoss
    print("\n[STEP 6] Initializing MultitaskLoss (camera=5.0, depth=1.0)...")
    loss_module = MultitaskLoss(
        camera={"weight": 5.0, "loss_type": "l1"},
        depth={"weight": 1.0, "gradient_loss_fn": "grad", "valid_range": 0.98},
        point=None,
        track=None,
    ).to(device)

    # 7. FORWARD PASS
    print(f"\n[STEP 7] Running FORWARD pass (Resolution: {target_res}x{target_res}, Sequence: {images.shape[1]})...")
    model.train()
    start_time = time.time()
    
    oom_occurred = False
    try:
        if device.type == "cuda":
            with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
                predictions = model(batch["images"])
        else:
            predictions = model(batch["images"])
        forward_time = time.time() - start_time
        print(f"[SUCCESS] Forward pass completed in {forward_time:.2f}s!")
        print(f"Predictions generated:")
        for k, v in predictions.items():
            if isinstance(v, list):
                print(f" - {k}: list of {len(v)} tensors (last shape: {v[-1].shape})")
            elif hasattr(v, "shape"):
                print(f" - {k}: shape={v.shape}")

    except torch.cuda.OutOfMemoryError as e:
        oom_occurred = True
        print(f"[CUDA OOM during Forward Pass]: {e}")
        return {"status": "OOM_FORWARD", "res": target_res}

    # 8. COMPUTE LOSS
    print("\n[STEP 8] Computing Multi-Task Loss...")
    loss_dict = loss_module(predictions, batch)
    objective = loss_dict["objective"]
    print(f"[SUCCESS] Loss computed successfully!")
    for k, v in loss_dict.items():
        val = v.item() if hasattr(v, "item") else v
        print(f" - {k}: {val:.6f}")

    # 9. BACKWARD PASS & GRADIENT VERIFICATION
    print("\n[STEP 9] Running BACKWARD pass...")
    try:
        objective.backward()
        print("[SUCCESS] Backward pass completed successfully!")
    except torch.cuda.OutOfMemoryError as e:
        print(f"[CUDA OOM during Backward Pass]: {e}")
        return {"status": "OOM_BACKWARD", "res": target_res}

    # Check gradients
    print("\n[STEP 10 / PART 8] Inspecting gradients across modules...")
    agg_has_grad = any(p.grad is not None for p in model.aggregator.parameters())
    cam_has_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.camera_head.parameters())
    depth_has_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.depth_head.parameters())

    print(f" - Aggregator gradients present: {agg_has_grad} (Expected: False)")
    print(f" - Camera Head gradients present: {cam_has_grad} (Expected: True)")
    print(f" - Depth Head gradients present:  {depth_has_grad} (Expected: True)")

    assert not agg_has_grad, "CRITICAL ERROR: Aggregator received gradients despite being frozen!"
    assert cam_has_grad, "CRITICAL ERROR: Camera Head received NO gradients!"
    assert depth_has_grad, "CRITICAL ERROR: Depth Head received NO gradients!"
    print("[SUCCESS] Gradient check PASSED: Aggregator has NONE, Camera and Depth have VALID gradients!")

    # 10. PART 10: MEMORY MEASUREMENTS
    if device.type == "cuda":
        alloc_mb = torch.cuda.memory_allocated(0) / (1024**2)
        reserved_mb = torch.cuda.memory_reserved(0) / (1024**2)
        peak_mb = torch.cuda.max_memory_allocated(0) / (1024**2)
        print(f"\n--- GPU MEMORY AUDIT ({target_res}x{target_res}) ---")
        print(f"Allocated Memory: {alloc_mb:.1f} MB ({alloc_mb/1024:.2f} GB)")
        print(f"Reserved Memory:  {reserved_mb:.1f} MB ({reserved_mb/1024:.2f} GB)")
        print(f"Peak Memory:      {peak_mb:.1f} MB ({peak_mb/1024:.2f} GB)")
        return {
            "status": "SUCCESS",
            "res": target_res,
            "allocated_mb": alloc_mb,
            "reserved_mb": reserved_mb,
            "peak_mb": peak_mb
        }
    else:
        return {"status": "SUCCESS_CPU", "res": target_res}

if __name__ == "__main__":
    print("\n--- RUNNING MATHEMATICAL PIPELINE VERIFICATION (FORWARD -> LOSS -> BACKWARD -> GRADIENTS) ---")
    res = run_verification(target_res=392, device_name="cpu")
    print(f"\nVerification Final Result: {res}")

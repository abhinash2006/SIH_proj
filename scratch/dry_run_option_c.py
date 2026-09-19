"""
Option C Mathematical Dry Run for VGGT-1B UAV Fine-Tuning:
Dataset -> DataLoader -> VGGT (Option C: Frozen Aggregator + Frozen Camera Trunk) -> Loss -> Backward -> Gradient Verification.
Resolution: 392x392, Sequence: 4, Batch: 1.
"""

import os
import sys
import time
import json
from types import SimpleNamespace

import torch
import numpy as np

sys.path.insert(0, os.path.abspath("vggt_repo"))
sys.path.insert(0, os.path.abspath("vggt_repo/training"))

from vggt.models.vggt import VGGT
from training.train_utils.freeze import freeze_modules
from loss import MultitaskLoss
from training.data.datasets.uav_dataset import UAVDataset

def run_option_c_dry_run():
    print("=" * 70)
    print("STARTING OPTION C ONE-BATCH DRY RUN")
    print("Architecture: Aggregator FROZEN | Camera Trunk FROZEN | PoseLN + PoseBranch + Depth TRAINABLE")
    print("Target: ~47.4M Trainable Parameters")
    print("=" * 70)

    device = torch.device("cpu") # CPU for universal mathematical proof without OOM
    print(f"Executing mathematical verification on: {device}")

    # 1. Instantiate Model
    print("\n[STEP 1] Instantiating VGGT-1B model (img_size=518)...")
    model = VGGT(
        enable_camera=True,
        enable_depth=True,
        enable_point=False,
        enable_track=False,
        img_size=518
    )

    # 2. Load Checkpoint
    ckpt_path = "models/VGGT-1B/model.pt"
    print(f"[STEP 2] Loading weights from: {ckpt_path}...")
    t0 = time.time()
    state_dict = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Checkpoint loaded in {time.time()-t0:.2f}s (missing: {len(missing)}, unexpected: {len(unexpected)}).")

    # 3. Apply Option C Freezing
    print("\n[STEP 3 / OPTION C] Freezing Aggregator and Camera Trunk...")
    patterns = ["*aggregator*", "*camera_head.trunk*"]
    model = freeze_modules(model, patterns=patterns, recursive=True)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print("\n--- PARAMETER AUDIT ---")
    print(f"Total Parameters:      {total_params:,}")
    print(f"Trainable Parameters:  {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"Frozen Parameters:     {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")

    # Submodule audit
    agg_train = sum(p.numel() for p in model.aggregator.parameters() if p.requires_grad)
    trunk_train = sum(p.numel() for p in model.camera_head.trunk.parameters() if p.requires_grad)
    poseln_train = sum(p.numel() for p in model.camera_head.poseLN_modulation.parameters() if p.requires_grad)
    branch_train = sum(p.numel() for p in model.camera_head.pose_branch.parameters() if p.requires_grad)
    depth_train = sum(p.numel() for p in model.depth_head.parameters() if p.requires_grad)

    print(f" - Aggregator:         Trainable={agg_train:,} (Expected: 0)")
    print(f" - Camera Trunk:       Trainable={trunk_train:,} (Expected: 0)")
    print(f" - PoseLN Modulation:  Trainable={poseln_train:,} (Expected: ~12.6M)")
    print(f" - Pose Branch:        Trainable={branch_train:,} (Expected: ~2.1M)")
    print(f" - Depth Head:         Trainable={depth_train:,} (Expected: ~32.7M)")

    assert agg_train == 0, "Aggregator must have 0 trainable parameters!"
    assert trunk_train == 0, "Camera Trunk must have 0 trainable parameters!"
    assert poseln_train > 0, "PoseLN Modulation must be trainable!"
    assert branch_train > 0, "Pose Branch must be trainable!"
    assert depth_train > 0, "Depth Head must be trainable!"
    assert 47_000_000 <= trainable_params <= 48_000_000, f"Unexpected trainable params: {trainable_params}"

    model.train()
    model = model.to(device)

    # 4. Load 1 Real Batch from data/uav_finetune/train.json
    print("\n[STEP 4] Loading 1 real training sample from data/uav_finetune...")
    common_conf = SimpleNamespace(
        img_size=392,
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
        DATA_DIR="data/uav_finetune",
        MANIFEST_PATH="data/uav_finetune/train.json",
        min_num_images=4,
        len_train=63,
    )

    sample = dataset[(0, 4, 1.0)]
    print(f"Sample sequence: {sample['seq_name']}")

    # Batch collation
    images = torch.from_numpy(np.stack(sample["images"]).astype(np.float32)).permute(0, 3, 1, 2).div(255.0).unsqueeze(0).to(device) # [1, 4, 3, 392, 392]
    depths = torch.from_numpy(np.stack(sample["depths"]).astype(np.float32)).unsqueeze(0).to(device) # [1, 4, 392, 392]
    extrinsics = torch.from_numpy(np.stack(sample["extrinsics"]).astype(np.float32)).unsqueeze(0)[..., :3, :].to(device) # [1, 4, 3, 4]
    intrinsics = torch.from_numpy(np.stack(sample["intrinsics"]).astype(np.float32)).unsqueeze(0).to(device) # [1, 4, 3, 3]
    point_masks = torch.from_numpy(np.stack(sample["point_masks"]).astype(np.bool_)).unsqueeze(0).to(device) # [1, 4, 392, 392]

    from train_utils.normalization import normalize_camera_extrinsics_and_points_batch

    cam_points = torch.from_numpy(np.stack(sample["cam_points"]).astype(np.float32)).unsqueeze(0)
    world_points = torch.from_numpy(np.stack(sample["world_points"]).astype(np.float32)).unsqueeze(0)

    norm_extri, norm_cam_pts, norm_world_pts, norm_depths = normalize_camera_extrinsics_and_points_batch(
        extrinsics=extrinsics,
        cam_points=cam_points,
        world_points=world_points,
        depths=depths,
        point_masks=point_masks,
    )

    batch = {
        "images": images,
        "depths": norm_depths.to(device),
        "extrinsics": norm_extri.to(device),
        "intrinsics": intrinsics.to(device),
        "cam_points": norm_cam_pts.to(device),
        "world_points": norm_world_pts.to(device),
        "point_masks": point_masks.to(device),
        "seq_name": sample["seq_name"],
    }

    # 5. Forward Pass
    print("\n[STEP 5] Running Forward pass (Res: 392x392, Seq: 4)...")
    t_fwd = time.time()
    preds = model(images)
    fwd_time = time.time() - t_fwd
    print(f"[SUCCESS] Forward pass completed in {fwd_time:.2f}s!")
    print(f" - pose_enc:   {preds['pose_enc'].shape}")
    print(f" - depth:      {preds['depth'].shape}")
    print(f" - depth_conf: {preds['depth_conf'].shape}")

    # Check for NaN / Inf in predictions
    assert not torch.isnan(preds['pose_enc']).any(), "NaN in pose_enc!"
    assert not torch.isnan(preds['depth']).any(), "NaN in depth!"
    assert not torch.isnan(preds['depth_conf']).any(), "NaN in depth_conf!"

    # 6. Loss Evaluation
    print("\n[STEP 6] Computing MultitaskLoss...")
    loss_fn = MultitaskLoss(
        camera={"weight": 5.0, "loss_type": "l1"},
        depth={"weight": 1.0, "gradient_loss_fn": "grad", "valid_range": 0.98},
        point=None,
        track=None,
    ).to(device)
    loss_dict = loss_fn(preds, batch)
    objective = loss_dict["objective"]

    print(f"[SUCCESS] Loss computed successfully:")
    for k, v in loss_dict.items():
        if isinstance(v, torch.Tensor):
            print(f" - {k:18s}: {v.item():.6f}")

    assert not torch.isnan(objective), "Objective loss is NaN!"
    assert not torch.isinf(objective), "Objective loss is Inf!"

    # 7. Backward Pass
    print("\n[STEP 7] Running Backward pass...")
    t_bwd = time.time()
    objective.backward()
    bwd_time = time.time() - t_bwd
    print(f"[SUCCESS] Backward pass completed in {bwd_time:.2f}s!")

    # 8. Gradient Verification
    print("\n[STEP 8] Verifying gradients across submodules...")

    # A. Aggregator (Expected: None)
    agg_has_grad = any(p.grad is not None for p in model.aggregator.parameters())
    print(f" - Aggregator gradients present:       {agg_has_grad} (Expected: False)")

    # B. Camera Trunk (Expected: None)
    trunk_has_grad = any(p.grad is not None for p in model.camera_head.trunk.parameters())
    print(f" - Camera Trunk gradients present:     {trunk_has_grad} (Expected: False)")

    # C. PoseLN Modulation (Expected: Present)
    poseln_has_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.camera_head.poseLN_modulation.parameters())
    print(f" - PoseLN Modulation grads present:   {poseln_has_grad} (Expected: True)")

    # D. Pose Branch (Expected: Present)
    branch_has_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.camera_head.pose_branch.parameters())
    print(f" - Pose Branch grads present:         {branch_has_grad} (Expected: True)")

    # E. Depth Head (Expected: Present)
    depth_has_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.depth_head.parameters())
    print(f" - Depth Head grads present:          {depth_has_grad} (Expected: True)")

    assert not agg_has_grad, "CRITICAL ERROR: Aggregator leaked gradients!"
    assert not trunk_has_grad, "CRITICAL ERROR: Camera Trunk leaked gradients!"
    assert poseln_has_grad, "PoseLN Modulation must have valid gradients!"
    assert branch_has_grad, "Pose Branch must have valid gradients!"
    assert depth_has_grad, "Depth Head must have valid gradients!"

    print("\n" + "=" * 70)
    print("ALL OPTION C DRY-RUN CHECKS PASSED PERFECTLY!")
    print(f"Final Objective Loss: {objective.item():.6f}")
    print("=" * 70)

    results = {
        "status": "PASS",
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "objective_loss": round(float(objective.item()), 6),
        "camera_loss": round(float(loss_dict["loss_camera"].item()), 6),
        "depth_loss": round(float(loss_dict["loss_reg_depth"].item()), 6),
        "fwd_time_s": round(fwd_time, 2),
        "bwd_time_s": round(bwd_time, 2),
        "agg_has_grad": agg_has_grad,
        "trunk_has_grad": trunk_has_grad,
        "poseln_has_grad": poseln_has_grad,
        "branch_has_grad": branch_has_grad,
        "depth_has_grad": depth_has_grad,
    }
    with open("scratch/option_c_dry_run_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_option_c_dry_run()

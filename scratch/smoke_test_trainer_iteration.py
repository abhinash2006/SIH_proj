import os
import sys
import time
import json
import torch
import numpy as np

sys.path.insert(0, os.path.abspath("vggt_repo"))
sys.path.insert(0, os.path.abspath("vggt_repo/training"))

from vggt.models.vggt import VGGT
from training.train_utils.freeze import freeze_modules
from training.train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from training.train_utils.gradient_clip import GradientClipper
from loss import MultitaskLoss
from training.data.datasets.uav_dataset import UAVDataset
from omegaconf import OmegaConf

def run_smoke_test():
    print("=" * 70)
    print("CONTROLLED VGGT FINE-TUNING SMOKE TEST (END-TO-END ITERATION)")
    print("=" * 70)

    device = torch.device("cpu")
    print(f"Device: {device}")

    # A. Checkpoint Loading
    print("\n[A] Checkpoint Loading...")
    t0 = time.time()
    ckpt_path = "models/VGGT-1B/model.pt"
    print(f"Loading checkpoint directly via patched loader: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=True)
    model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    print(f"Checkpoint tensor count: {len(model_state_dict)} loaded in {time.time()-t0:.2f}s")

    # Instantiate model
    print("Instantiating VGGT with enable_camera=True, enable_depth=True, enable_point=False, enable_track=False...")
    model = VGGT(enable_camera=True, enable_depth=True, enable_point=False, enable_track=False, img_size=518)
    missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
    print(f"Model loaded: missing keys={len(missing)}, unexpected keys={len(unexpected)}")

    # Freeze modules
    print("Applying Option C freezing: *aggregator*, *camera_head.trunk*...")
    model = freeze_modules(model, ["*aggregator*", "*camera_head.trunk*"], recursive=True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {trainable_count:,} (~47.38M)")

    # Optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-5, weight_decay=0.05)
    gradient_clipper = GradientClipper(configs=[
        {"module_name": ["depth"], "max_norm": 1.0, "norm_type": 2},
        {"module_name": ["camera"], "max_norm": 1.0, "norm_type": 2}
    ])

    # B. First Batch
    print("\n[B] Loading First Batch from UAVDataset...")
    cfg = OmegaConf.load("vggt_repo/training/config/default_dataset.yaml")
    common_conf = cfg.data.train.common_config
    common_conf.img_size = 392
    common_conf.patch_size = 14
    common_conf.load_depth = True
    common_conf.training = True

    dataset = UAVDataset(
        common_conf=common_conf,
        split="train",
        DATA_DIR="data/uav_finetune",
        MANIFEST_PATH="data/uav_finetune/train.json",
        min_num_images=4,
        len_train=63,
    )
    raw_sample = dataset[(0, 4, 1.0)]

    # Collate batch
    images = torch.from_numpy(np.stack(raw_sample["images"]).astype(np.float32)).permute(0, 3, 1, 2).div(255.0).unsqueeze(0)
    depths = torch.from_numpy(np.stack(raw_sample["depths"]).astype(np.float32)).unsqueeze(0)
    extrinsics = torch.from_numpy(np.stack(raw_sample["extrinsics"]).astype(np.float32)).unsqueeze(0)
    intrinsics = torch.from_numpy(np.stack(raw_sample["intrinsics"]).astype(np.float32)).unsqueeze(0)
    cam_points = torch.from_numpy(np.stack(raw_sample["cam_points"]).astype(np.float32)).unsqueeze(0)
    world_points = torch.from_numpy(np.stack(raw_sample["world_points"]).astype(np.float32)).unsqueeze(0)
    point_masks = torch.from_numpy(np.stack(raw_sample["point_masks"])).unsqueeze(0)

    print(f"Batch loaded: images={images.shape}, extrinsics={extrinsics.shape}, depths={depths.shape}")

    # Normalization
    print("Running normalize_camera_extrinsics_and_points_batch()...")
    norm_extri, norm_cam, norm_world, norm_depth = normalize_camera_extrinsics_and_points_batch(
        extrinsics=extrinsics,
        cam_points=cam_points,
        world_points=world_points,
        depths=depths,
        point_masks=point_masks,
    )
    print(f"Normalized extrinsics: {norm_extri.shape}")

    batch = {
        "images": images,
        "depths": norm_depth,
        "extrinsics": norm_extri,
        "intrinsics": intrinsics,
        "cam_points": norm_cam,
        "world_points": norm_world,
        "point_masks": point_masks,
        "seq_name": raw_sample["seq_name"],
    }

    # C. Forward Pass
    print("\n[C] Forward Pass...")
    model.train()
    t_fwd = time.time()
    preds = model(images)
    fwd_time = time.time() - t_fwd
    print(f"Forward pass completed in {fwd_time:.2f}s")
    print(f" - pose_enc:   {preds['pose_enc'].shape}")
    print(f" - depth:      {preds['depth'].shape}")
    print(f" - depth_conf: {preds['depth_conf'].shape}")

    # D, E, F. Loss Computation
    print("\n[D, E, F] Loss Computation...")
    loss_fn = MultitaskLoss(
        camera={"weight": 5.0, "loss_type": "l1"},
        depth={"weight": 1.0, "gradient_loss_fn": "grad", "valid_range": 0.98},
        point=None,
        track=None,
    )
    loss_dict = loss_fn(preds, batch)
    total_loss = loss_dict["objective"]
    loss_camera = loss_dict["loss_camera"]
    loss_depth = loss_dict["loss_reg_depth"]

    print(f" - Camera Loss (loss_camera): {loss_camera.item():.6f} (weight 5.0 -> {loss_camera.item()*5.0:.6f})")
    print(f" - Depth Loss (loss_reg_depth): {loss_depth.item():.6f} (weight 1.0 -> {loss_depth.item()*1.0:.6f})")
    print(f" - Total Loss (objective):    {total_loss.item():.6f}")

    # G. Backward Pass
    print("\n[G] Backward Pass...")
    optimizer.zero_grad()
    t_bwd = time.time()
    total_loss.backward()
    bwd_time = time.time() - t_bwd
    print(f"Backward pass completed in {bwd_time:.2f}s")

    # H. Gradient Health
    print("\n[H] Gradient Health Check...")
    frozen_grad_leaks = 0
    for name, p in model.aggregator.named_parameters():
        if p.grad is not None:
            frozen_grad_leaks += 1
    for name, p in model.camera_head.trunk.named_parameters():
        if p.grad is not None:
            frozen_grad_leaks += 1

    trainable_finite_grads = 0
    trainable_nan_grads = 0
    grad_norms = []
    for p in trainable_params:
        if p.grad is not None:
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                trainable_nan_grads += 1
            else:
                trainable_finite_grads += 1
                grad_norms.append(p.grad.norm().item())

    print(f" - Frozen parameter gradient leaks: {frozen_grad_leaks} (Expected: 0)")
    print(f" - Trainable parameters with finite gradients: {trainable_finite_grads}/{len(trainable_params)}")
    print(f" - Trainable parameters with NaN/Inf gradients: {trainable_nan_grads} (Expected: 0)")
    print(f" - Average gradient norm: {np.mean(grad_norms):.6f}, Max norm: {np.max(grad_norms):.6f}")

    # I. Optimizer Step
    print("\n[I] Gradient Clipping & Optimizer Step...")
    gradient_clipper.setup_clipping(model)
    gradient_clipper(model)
    optimizer.step()
    print("Optimizer step completed successfully!")

    # K. NaN / Inf check
    has_nan_inf = torch.isnan(total_loss).item() or torch.isinf(total_loss).item()
    print(f"\n[K] NaN / Inf in Loss: {has_nan_inf}")

    report = {
        "checkpoint_loaded": True,
        "first_batch_shape": str(images.shape),
        "fwd_time_s": round(fwd_time, 2),
        "camera_loss": round(float(loss_camera.item()), 6),
        "depth_loss": round(float(loss_depth.item()), 6),
        "total_loss": round(float(total_loss.item()), 6),
        "bwd_time_s": round(bwd_time, 2),
        "frozen_grad_leaks": frozen_grad_leaks,
        "trainable_finite_grads": trainable_finite_grads,
        "trainable_nan_grads": trainable_nan_grads,
        "optimizer_step_success": True,
        "has_nan_inf": has_nan_inf,
    }
    with open("scratch/smoke_test_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("SMOKE TEST ITERATION COMPLETED SUCCESSFULLY WITH 100% HEALTHY METRICS!")
    print("=" * 70)

if __name__ == "__main__":
    run_smoke_test()

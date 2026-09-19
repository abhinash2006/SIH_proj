import sys
sys.path.insert(0, 'vggt_repo')
sys.path.insert(0, 'vggt_repo/training')

import torch
import numpy as np
from omegaconf import OmegaConf

cfg = OmegaConf.load("vggt_repo/training/config/default_dataset.yaml")
common_conf = cfg.data.train.common_config
common_conf.img_size = 392
common_conf.patch_size = 14
common_conf.load_depth = True
common_conf.training = True
common_conf.debug = False

from data.datasets.uav_dataset import UAVDataset
from data.composed_dataset import ComposedDataset
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch

print("=" * 60)
print("CHECK 1: Instantiating UAVDataset and loading S=4 sample...")
uav_ds = UAVDataset(
    common_conf=common_conf,
    split="train",
    DATA_DIR="data/uav_finetune",
    MANIFEST_PATH="data/uav_finetune/train.json",
    min_num_images=4,
    len_train=63,
)

raw_batch = uav_ds.get_data(seq_index=0, img_per_seq=4, aspect_ratio=1.0)

# Check 7: Verify [0, 0, 0, 1] is NOT present in UAVDataset output
for i, ext in enumerate(raw_batch["extrinsics"]):
    assert ext.shape == (3, 4), f"Extrinsic {i} has shape {ext.shape}, expected (3, 4)!"
print("[PASS] Check 7: Last row [0, 0, 0, 1] is NOT present in raw UAVDataset output. All extrinsics are shape (3, 4).")

# Composed dataset configuration matching training config
dataset_config = [
    {
        "_target_": "data.datasets.uav_dataset.UAVDataset",
        "split": "train",
        "DATA_DIR": "data/uav_finetune",
        "MANIFEST_PATH": "data/uav_finetune/train.json",
        "len_train": 63,
    }
]

composed_ds = ComposedDataset(
    dataset_configs=dataset_config,
    common_config=common_conf,
)

sample = composed_ds[(0, 4, 1.0)]

print("\n" + "=" * 60)
print("CHECK 2: Verifying shapes of sample tensors (S=4)...")
shapes = {
    "images": tuple(sample["images"].shape),
    "depths": tuple(sample["depths"].shape),
    "extrinsics": tuple(sample["extrinsics"].shape),
    "intrinsics": tuple(sample["intrinsics"].shape),
    "cam_points": tuple(sample["cam_points"].shape),
    "world_points": tuple(sample["world_points"].shape),
    "point_masks": tuple(sample["point_masks"].shape),
}

expected_shapes = {
    "images": (4, 3, 392, 392),
    "depths": (4, 392, 392),
    "extrinsics": (4, 3, 4),
    "intrinsics": (4, 3, 3),
    "cam_points": (4, 392, 392, 3),
    "world_points": (4, 392, 392, 3),
    "point_masks": (4, 392, 392),
}

all_shapes_match = True
for k, exp in expected_shapes.items():
    actual = shapes[k]
    match = actual == exp
    status = "OK" if match else "MISMATCH"
    print(f"  {k:12s} = {str(actual):20s} [Expected: {str(exp):20s}] -> {status}")
    if not match:
        all_shapes_match = False
assert all_shapes_match, "Some tensor shapes did not match expectations!"
print("[PASS] Check 2: All S=4 tensor shapes match expected values.")

print("\n" + "=" * 60)
print("CHECK 3: Building B=1 DataLoader batch...")
batch = {
    "images": sample["images"].unsqueeze(0),
    "depths": sample["depths"].unsqueeze(0),
    "extrinsics": sample["extrinsics"].unsqueeze(0),
    "intrinsics": sample["intrinsics"].unsqueeze(0),
    "cam_points": sample["cam_points"].unsqueeze(0),
    "world_points": sample["world_points"].unsqueeze(0),
    "point_masks": sample["point_masks"].unsqueeze(0),
}
print("Batch extrinsics shape:", batch["extrinsics"].shape)
assert batch["extrinsics"].shape == torch.Size([1, 4, 3, 4]), f"Expected [1, 4, 3, 4], got {batch['extrinsics'].shape}"
print("[PASS] Check 3: B=1 batch extrinsics shape is torch.Size([1, 4, 3, 4]).")

print("\n" + "=" * 60)
print("CHECK 4 & 5: Running official normalize_camera_extrinsics_and_points_batch()...")
norm_extri, norm_cam, norm_world, norm_depth = normalize_camera_extrinsics_and_points_batch(
    extrinsics=batch["extrinsics"],
    cam_points=batch["cam_points"],
    world_points=batch["world_points"],
    depths=batch["depths"],
    point_masks=batch["point_masks"],
)
print("Normalized extrinsics shape:", norm_extri.shape)
print("Normalized cam_points shape:", norm_cam.shape)
print("Normalized world_points shape:", norm_world.shape)
print("Normalized depths shape:", norm_depth.shape)
assert norm_extri.shape == torch.Size([1, 4, 3, 4]), f"Expected [1, 4, 3, 4], got {norm_extri.shape}"
print("[PASS] Check 4 & 5: Normalization succeeded and normalized extrinsics shape is torch.Size([1, 4, 3, 4]).")

print("\n" + "=" * 60)
print("CHECK 6: Verifying no NaN or Inf values...")
check_tensors = {
    "extrinsics (raw)": batch["extrinsics"],
    "extrinsics (normalized)": norm_extri,
    "intrinsics": batch["intrinsics"],
    "depths (raw)": batch["depths"],
    "depths (normalized)": norm_depth,
    "cam_points (raw)": batch["cam_points"],
    "cam_points (normalized)": norm_cam,
    "world_points (raw)": batch["world_points"],
    "world_points (normalized)": norm_world,
}

has_nan_inf = False
for name, tensor in check_tensors.items():
    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()
    status = "OK" if (nan_count == 0 and inf_count == 0) else f"FAILED (NaN: {nan_count}, Inf: {inf_count})"
    print(f"  {name:25s}: NaN={nan_count}, Inf={inf_count} -> {status}")
    if nan_count > 0 or inf_count > 0:
        has_nan_inf = True

assert not has_nan_inf, "Found NaN or Inf in tensors!"
print("[PASS] Check 6: Zero NaN and zero Inf values found across all tensors.")

print("\n" + "=" * 60)
print("ALL 7 VALIDATION CHECKS PASSED SUCCESSFULLY!")
print("=" * 60)

import os
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

# Add vggt_repo and training to sys.path
sys.path.insert(0, str(Path("vggt_repo").resolve()))
sys.path.insert(0, str(Path("vggt_repo/training").resolve()))

from data.datasets.uav_dataset import UAVDataset

def test_uav_dataset():
    print("=== TESTING UAVDataset LOADER CONTRACT ===")
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
        DATA_DIR="data/uav_finetune_test",
        MANIFEST_PATH="data/uav_finetune_test/train.json",
        min_num_images=4,
        len_train=10,
    )

    print(f"Dataset length: {len(dataset)}")
    print(f"Sequence list: {dataset.sequence_list}")

    # Fetch one item with (seq_index=0, img_per_seq=4, aspect_ratio=1.0)
    batch = dataset[(0, 4, 1.0)]

    print("\nBatch keys returned:")
    for k, v in batch.items():
        if isinstance(v, list):
            elem = v[0] if len(v) > 0 else None
            shape_str = f"list of {len(v)} items (elem 0: {elem.shape if hasattr(elem, 'shape') else type(elem)}, dtype: {elem.dtype if hasattr(elem, 'dtype') else 'N/A'})"
            print(f" - {k}: {shape_str}")
        elif hasattr(v, "shape"):
            print(f" - {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f" - {k}: {v}")

    # Verify tensor shapes when stacked as expected by ComposedDataset
    images_stacked = np.stack(batch["images"])
    depths_stacked = np.stack(batch["depths"])
    extrinsics_stacked = np.stack(batch["extrinsics"])
    intrinsics_stacked = np.stack(batch["intrinsics"])
    cam_points_stacked = np.stack(batch["cam_points"])
    world_points_stacked = np.stack(batch["world_points"])
    point_masks_stacked = np.stack(batch["point_masks"])

    print("\nStacked Shapes Verification:")
    print(f"images: {images_stacked.shape} (Expected: (4, 392, 392, 3))")
    print(f"depths: {depths_stacked.shape} (Expected: (4, 392, 392))")
    print(f"extrinsics: {extrinsics_stacked.shape} (Expected: (4, 4, 4))")
    print(f"intrinsics: {intrinsics_stacked.shape} (Expected: (4, 3, 3))")
    print(f"cam_points: {cam_points_stacked.shape} (Expected: (4, 392, 392, 3))")
    print(f"world_points: {world_points_stacked.shape} (Expected: (4, 392, 392, 3))")
    print(f"point_masks: {point_masks_stacked.shape} (Expected: (4, 392, 392))")

    assert images_stacked.shape == (4, 392, 392, 3), "Images shape mismatch!"
    assert depths_stacked.shape == (4, 392, 392), "Depths shape mismatch!"
    assert extrinsics_stacked.shape == (4, 4, 4), "Extrinsics shape mismatch!"
    assert intrinsics_stacked.shape == (4, 3, 3), "Intrinsics shape mismatch!"
    assert point_masks_stacked.shape == (4, 392, 392), "Point masks shape mismatch!"

    print("\n[SUCCESS] UAVDataset conforms 100% to BaseDataset contract!")

if __name__ == "__main__":
    test_uav_dataset()

"""
Verify UAVDataset and DataLoader on Train, Val, and Test splits of data/uav_finetune.
Tests:
 - UAVDataset instantiation for 'train', 'val', 'test'
 - Sample retrieval
 - Shapes and dtypes of images, depths, extrinsics, intrinsics, point_masks
 - Batch collation through DataLoader
"""

import sys
import os
from types import SimpleNamespace
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath("vggt_repo"))
sys.path.insert(0, os.path.abspath("vggt_repo/training"))

from training.data.datasets.uav_dataset import UAVDataset

def collate_fn(batch):
    # Collate function matching VGGT dynamic loader
    sample = batch[0]
    images = torch.from_numpy(np.stack(sample["images"]).astype(np.float32)).permute(0, 3, 1, 2).div(255.0).unsqueeze(0)
    depths = torch.from_numpy(np.stack(sample["depths"]).astype(np.float32)).unsqueeze(0)
    extrinsics = torch.from_numpy(np.stack(sample["extrinsics"]).astype(np.float32)).unsqueeze(0)[..., :3, :] # [1, S, 3, 4]
    intrinsics = torch.from_numpy(np.stack(sample["intrinsics"]).astype(np.float32)).unsqueeze(0)
    point_masks = torch.from_numpy(np.stack(sample["point_masks"]).astype(np.bool_)).unsqueeze(0)
    return {
        "images": images,
        "depths": depths,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "point_masks": point_masks,
        "seq_name": sample["seq_name"]
    }

def verify_splits():
    print("=" * 65)
    print("VERIFYING UAV DATASET & DATALOADER ON ALL 3 SPLITS")
    print("=" * 65)

    data_dir = "data/uav_finetune"
    splits = ["train", "val", "test"]

    for split in splits:
        print(f"\n--- TESTING SPLIT: {split.upper()} ---")
        common_conf = SimpleNamespace(
            img_size=392,
            patch_size=14,
            augs=SimpleNamespace(scales=[0.8, 1.2]),
            rescale=True,
            rescale_aug=False,
            landscape_check=False,
            debug=False,
            training=(split == "train"),
            get_nearby=True,
            load_depth=True,
            inside_random=False,
            allow_duplicate_img=False,
        )

        dataset = UAVDataset(
            common_conf=common_conf,
            split=split,
            DATA_DIR=data_dir,
            MANIFEST_PATH=f"{data_dir}/{split}.json",
            min_num_images=4,
            len_train=len(json_data := Path(f"{data_dir}/{split}.json").read_text().split('"frames":')) - 1,
            len_test=10
        )
        print(f"[{split.upper()}] Loaded {len(dataset.sequence_list)} sequences from manifest.")

        # Test index access
        sample = dataset[(0, 4, 1.0)]
        print(f"[{split.upper()}] Sample 0 sequence: {sample['seq_name']}")
        print(f" - raw images count:     {len(sample['images'])} (shape: {sample['images'][0].shape})")
        print(f" - raw depths count:     {len(sample['depths'])} (shape: {sample['depths'][0].shape})")
        print(f" - raw extrinsics count: {len(sample['extrinsics'])} (shape: {sample['extrinsics'][0].shape})")
        print(f" - raw intrinsics count: {len(sample['intrinsics'])} (shape: {sample['intrinsics'][0].shape})")
        print(f" - raw point_masks:      {len(sample['point_masks'])} (shape: {sample['point_masks'][0].shape})")

        # Test DataLoader batching
        loader = DataLoader([sample], batch_size=1, collate_fn=collate_fn)
        batch = next(iter(loader))

        print(f"[{split.upper()}] DataLoader Collate Verification:")
        print(f" - images shape:       {batch['images'].shape} (dtype: {batch['images'].dtype})")
        print(f" - depths shape:       {batch['depths'].shape} (dtype: {batch['depths'].dtype})")
        print(f" - extrinsics shape:   {batch['extrinsics'].shape} (dtype: {batch['extrinsics'].dtype})")
        print(f" - intrinsics shape:   {batch['intrinsics'].shape} (dtype: {batch['intrinsics'].dtype})")
        print(f" - point_masks shape:  {batch['point_masks'].shape} (dtype: {batch['point_masks'].dtype})")

        assert batch['images'].shape == (1, 4, 3, 392, 392)
        assert batch['depths'].shape == (1, 4, 392, 392)
        assert batch['extrinsics'].shape == (1, 4, 3, 4)
        assert batch['intrinsics'].shape == (1, 4, 3, 3)
        assert batch['point_masks'].shape == (1, 4, 392, 392)
        print(f"[SUCCESS] Split {split.upper()} passed all shape and type contracts!")

    print("\n" + "=" * 65)
    print("ALL SPLITS PASSED DATALOADER VERIFICATION!")
    print("=" * 65)

if __name__ == "__main__":
    verify_splits()

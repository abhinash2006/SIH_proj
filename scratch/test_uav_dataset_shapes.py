import sys
sys.path.insert(0, 'vggt_repo')
sys.path.insert(0, 'vggt_repo/training')
import torch
import numpy as np
from types import SimpleNamespace

# Common config
common_conf = SimpleNamespace(
    img_size=392,
    patch_size=14,
    debug=False,
    training=True,
    get_nearby=False,
    load_depth=True,
    inside_random=False,
    allow_duplicate_img=False,
    augs=SimpleNamespace(scales=[0.8, 1.2], aspects=[1.0, 1.0]),
    rescale=True,
    rescale_aug=False,
    landscape_check=True,
)

from data.datasets.uav_dataset import UAVDataset

ds = UAVDataset(
    common_conf=common_conf,
    split="train",
    DATA_DIR="data/uav_finetune",
    MANIFEST_PATH="data/uav_finetune/train.json",
    min_num_images=4,
    len_train=63,
)

item = ds.get_data(seq_index=0, img_per_seq=4, aspect_ratio=1.0)
print("=== UAVDataset.get_data() raw outputs ===")
print("images:", len(item["images"]), [x.shape for x in item["images"][:2]])
print("depths:", len(item["depths"]), [x.shape for x in item["depths"][:2]])
print("extrinsics:", len(item["extrinsics"]), [x.shape for x in item["extrinsics"][:2]])
print("intrinsics:", len(item["intrinsics"]), [x.shape for x in item["intrinsics"][:2]])
print("cam_points:", len(item["cam_points"]), [x.shape for x in item["cam_points"][:2]])
print("world_points:", len(item["world_points"]), [x.shape for x in item["world_points"][:2]])
print("point_masks:", len(item["point_masks"]), [x.shape for x in item["point_masks"][:2]])

from train_utils.normalization import normalize_camera_extrinsics_and_points_batch

extri_fixed = torch.from_numpy(np.stack(item["extrinsics"])[:, :3, :].astype(np.float32)).unsqueeze(0)
cam_pts = torch.from_numpy(np.stack(item["cam_points"]).astype(np.float32)).unsqueeze(0)
world_pts = torch.from_numpy(np.stack(item["world_points"]).astype(np.float32)).unsqueeze(0)
depths = torch.from_numpy(np.stack(item["depths"]).astype(np.float32)).unsqueeze(0)
masks = torch.from_numpy(np.stack(item["point_masks"])).unsqueeze(0)

print("\nFixed batch extrinsics shape:", extri_fixed.shape)
try:
    res = normalize_camera_extrinsics_and_points_batch(
        extrinsics=extri_fixed,
        cam_points=cam_pts,
        world_points=world_pts,
        depths=depths,
        point_masks=masks,
    )
    norm_extri, norm_cam, norm_world, norm_depth = res
    print("Normalization SUCCEEDED with fixed extrinsics!")
    print("norm_extri shape:", norm_extri.shape)
    print("norm_cam shape:", norm_cam.shape)
    print("norm_world shape:", norm_world.shape)
    print("norm_depth shape:", norm_depth.shape)
except Exception as e:
    print(f"Normalization FAILED: {type(e).__name__}: {e}")

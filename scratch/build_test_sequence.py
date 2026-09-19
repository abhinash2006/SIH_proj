import os
import sys
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import open3d as o3d
from scipy.spatial.transform import Rotation
import cv2

def build_test_sequence():
    base_src = Path("data/raw/uav_sequence/1121222322212102-4")
    out_root = Path("data/uav_finetune_test")
    seq_name = "seq_uav_test_001"
    seq_dir = out_root / "sequences" / seq_name

    images_out = seq_dir / "images"
    depth_out = seq_dir / "depth"
    masks_out = seq_dir / "depth_masks"
    camera_out = seq_dir / "camera"

    for d in (images_out, depth_out, masks_out, camera_out):
        d.mkdir(parents=True, exist_ok=True)

    print("=== BUILDING ONE REAL UAV TEST SEQUENCE ===")
    print(f"Target directory: {seq_dir}")

    # 1. Load cameras.txt
    cameras = {}
    with open(base_src / "sparse/0/cameras.txt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split()
            cameras[int(parts[0])] = {
                "width": int(parts[2]), "height": int(parts[3]),
                "params": [float(x) for x in parts[4:]]
            }

    # 2. Load images.txt
    colmap_images = {}
    with open(base_src / "sparse/0/images.txt") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"): continue
        parts = line.split()
        if len(parts) >= 10:
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            cid = int(parts[8])
            name = parts[9]
            # skip 2D points line
            i += 1
            colmap_images[name] = {
                "q": np.array([qw, qx, qy, qz]),
                "t": np.array([tx, ty, tz]),
                "cam_id": cid,
                "name": name
            }

    # 3. Read train_list.txt and pick first 4 consecutive frames from same flight timestamp
    selected_names = []
    with open(base_src / "train_list.txt") as f:
        for line in f:
            name = line.strip()
            if name in colmap_images:
                selected_names.append(name)
            if len(selected_names) >= 4:
                break

    print(f"Selected {len(selected_names)} consecutive frames:")
    for n in selected_names:
        print(f" - {n}")

    # 4. Load dense point cloud
    print("\nLoading dense_points.ply...")
    pcd = o3d.io.read_point_cloud(str(base_src / "dense_points.ply"))
    pts_w = np.asarray(pcd.points, dtype=np.float32)
    print(f"Loaded {len(pts_w):,} dense 3D points.")

    # 5. Process each frame
    manifest_frames = []
    intrinsics_dict = {}
    extrinsics_dict = {}

    for idx, name in enumerate(selected_names):
        frame_num_str = f"{idx+1:06d}"
        src_img_path = base_src / "images" / name
        dst_img_path = images_out / f"{frame_num_str}.jpg"
        dst_depth_path = depth_out / f"{frame_num_str}.png"
        dst_mask_path = masks_out / f"{frame_num_str}.png"

        # Copy RGB image
        shutil.copy2(src_img_path, dst_img_path)

        # Camera calibration
        col_img = colmap_images[name]
        cam = cameras[col_img["cam_id"]]
        W, H = cam["width"], cam["height"]
        fx, fy, cx, cy = cam["params"]

        K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        q = col_img["q"]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        t = col_img["t"]

        extri_4x4 = np.eye(4, dtype=np.float64)
        extri_4x4[:3, :3] = R
        extri_4x4[:3, 3] = t

        intrinsics_dict[frame_num_str] = K.tolist()
        extrinsics_dict[frame_num_str] = extri_4x4.tolist()

        # Project dense points to camera frame
        pts_c = (R @ pts_w.T).T + t
        valid_z = pts_c[:, 2] > 0.5
        pts_c_front = pts_c[valid_z]

        u = (fx * pts_c_front[:, 0] / pts_c_front[:, 2]) + cx
        v = (fy * pts_c_front[:, 1] / pts_c_front[:, 2]) + cy
        z = pts_c_front[:, 2]

        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u_valid = np.clip(np.floor(u[in_bounds]).astype(int), 0, W - 1)
        v_valid = np.clip(np.floor(v[in_bounds]).astype(int), 0, H - 1)
        z_valid = z[in_bounds]

        depth_map = np.zeros((H, W), dtype=np.float32)
        sort_idx = np.argsort(-z_valid)
        depth_map[v_valid[sort_idx], u_valid[sort_idx]] = z_valid[sort_idx]

        # Dilate 3x3 to close single-pixel point cloud gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        depth_dilated = cv2.dilate(depth_map, kernel)
        valid_mask = (depth_dilated > 0) & (depth_dilated < 300.0)

        # Save 16-bit millimeter depth PNG (depth_mm = depth_m * 1000)
        depth_mm = (depth_dilated * 1000.0).clip(0, 65535).astype(np.uint16)
        cv2.imwrite(str(dst_depth_path), depth_mm)

        # Save binary mask PNG
        mask_uint8 = (valid_mask * 255).astype(np.uint8)
        cv2.imwrite(str(dst_mask_path), mask_uint8)

        print(f"Frame {frame_num_str} ({name}): valid depth pixels = {np.sum(valid_mask):,} ({np.mean(valid_mask)*100:.1f}%), Depth range: [{depth_dilated[valid_mask].min():.1f}m, {depth_dilated[valid_mask].max():.1f}m]")

        # Manifest frame item
        manifest_frames.append({
            "filepath": str(Path("sequences") / seq_name / "images" / f"{frame_num_str}.jpg").replace("\\", "/"),
            "depth_path": str(Path("sequences") / seq_name / "depth" / f"{frame_num_str}.png").replace("\\", "/"),
            "mask_path": str(Path("sequences") / seq_name / "depth_masks" / f"{frame_num_str}.png").replace("\\", "/"),
            "extri": extri_4x4.tolist(),
            "intri": K.tolist(),
            "depth_scale": 1000.0,
            "width": W,
            "height": H
        })

    # Save camera JSON files
    with open(camera_out / "intrinsics.json", "w") as f:
        json.dump(intrinsics_dict, f, indent=2)
    with open(camera_out / "extrinsics.json", "w") as f:
        json.dump(extrinsics_dict, f, indent=2)

    # Save metadata.json
    metadata = {
        "sequence_id": seq_name,
        "supervision_type": "PSEUDO_GROUND_TRUTH",
        "dataset_source": "UAVLight (CVPR 2024 / IEEE)",
        "num_frames": len(manifest_frames),
        "resolution": [W, H],
        "sensor": "DJI Pinhole Camera",
        "mean_flight_height_m": 70.0,
        "coordinate_convention": "OpenCV world-to-camera X_c = R @ X_w + t",
        "depth_unit": "millimeter (uint16 / 1000.0 = meters)"
    }
    with open(seq_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # 6. Save master manifests
    train_manifest = {
        seq_name: {
            "supervision_type": "PSEUDO_GROUND_TRUTH",
            "frames": manifest_frames
        }
    }
    val_manifest = {
        seq_name: {
            "supervision_type": "PSEUDO_GROUND_TRUTH",
            "frames": manifest_frames
        }
    }
    test_manifest = {
        seq_name: {
            "supervision_type": "PSEUDO_GROUND_TRUTH",
            "frames": manifest_frames
        }
    }

    with open(out_root / "train.json", "w") as f:
        json.dump(train_manifest, f, indent=2)
    with open(out_root / "val.json", "w") as f:
        json.dump(val_manifest, f, indent=2)
    with open(out_root / "test.json", "w") as f:
        json.dump(test_manifest, f, indent=2)

    print(f"\n[SUCCESS] One real UAV test sequence created at {seq_dir}")
    print(f"[SUCCESS] Manifests saved at {out_root}/(train.json, val.json, test.json)")

if __name__ == "__main__":
    build_test_sequence()

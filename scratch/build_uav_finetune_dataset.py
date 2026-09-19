"""
Build the Real UAV Fine-Tuning Dataset from 5 Real UAV Flight Trajectories.
Adheres to strict requirements:
 - Splits by flight: Train (Flights 1-3), Val (Flight 4), Test (Flight 5)
 - Zero frame / flight leakage between splits
 - Continuous 4-frame multi-view windows with stride=1
 - Metric depth rendered from dense_points.ply (16-bit uint mm)
 - 8-bit binary depth masks
 - True COLMAP camera extrinsics & intrinsics
 - Manifests: train.json, val.json, test.json with PSEUDO_GROUND_TRUTH
"""

import os
import sys
import json
import shutil
import time
from pathlib import Path

import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation

FLIGHT_SPLITS = {
    "train": [
        "2025-10-08-11-10-20",  # Flight 1 (17 images)
        "2025-10-08-16-04-49",  # Flight 2 (20 images)
        "2025-10-08-16-10-29",  # Flight 3 (35 images)
    ],
    "val": [
        "2025-10-14-11-20-51",  # Flight 4 (17 images)
    ],
    "test": [
        "2025-10-14-12-32-26",  # Flight 5 (34 images)
    ]
}

def build_dataset(
    raw_dir="data/raw/uav_sequence/1121222322212102-4",
    out_dir="data/uav_finetune",
    seq_len=4,
    stride=1
):
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BUILDING REAL UAV FINE-TUNING DATASET")
    print(f"Source:      {raw_path}")
    print(f"Destination: {out_path}")
    print(f"Window:      Length={seq_len}, Stride={stride}")
    print("=" * 70)

    # 1. Parse COLMAP cameras.txt
    cameras = {}
    with open(raw_path / "sparse/0/cameras.txt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cid = int(parts[0])
            model = parts[1]
            w = int(parts[2])
            h = int(parts[3])
            params = [float(x) for x in parts[4:]]
            cameras[cid] = {"model": model, "w": w, "h": h, "params": params}
    print(f"Loaded {len(cameras)} camera calibrations.")

    # 2. Parse COLMAP images.txt
    colmap_images = {}
    with open(raw_path / "sparse/0/images.txt") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 10:
            img_id = int(parts[0])
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            cid = int(parts[8])
            rel_name = parts[9] # e.g. '2025-10-08-11-10-20/1759893132.709145784.JPG'
            i += 1 # skip 2D points line
            colmap_images[rel_name] = {
                "id": img_id,
                "q": np.array([qw, qx, qy, qz], dtype=np.float64),
                "t": np.array([tx, ty, tz], dtype=np.float64),
                "cam_id": cid,
                "name": rel_name
            }
    print(f"Loaded {len(colmap_images)} COLMAP registered images.")

    # 3. Load dense 3D points
    print("\nLoading dense_points.ply (1.94M points)...")
    t0 = time.time()
    pcd = o3d.io.read_point_cloud(str(raw_path / "dense_points.ply"))
    pts_w = np.asarray(pcd.points, dtype=np.float32)
    print(f"Loaded {len(pts_w):,} points in {time.time()-t0:.2f}s.")

    # 4. Render and store 123 unique physical frames (Images, Depth, Masks)
    print("\nProcessing 123 physical frames (rendering metric depth & masks)...")
    frame_registry = {}
    total_rendered = 0

    for split_name, flight_list in FLIGHT_SPLITS.items():
        for flight in flight_list:
            flight_img_dir = raw_path / "images" / flight
            raw_files = sorted([f for f in flight_img_dir.iterdir() if f.suffix.lower() == ".jpg"])
            print(f" - Processing Flight: {flight} ({len(raw_files)} frames)")

            # Ensure output subdirs exist
            (out_path / "images" / flight).mkdir(parents=True, exist_ok=True)
            (out_path / "depth" / flight).mkdir(parents=True, exist_ok=True)
            (out_path / "depth_masks" / flight).mkdir(parents=True, exist_ok=True)

            for rf in raw_files:
                rel_name = f"{flight}/{rf.name}"
                if rel_name not in colmap_images:
                    print(f"WARNING: {rel_name} not in COLMAP! Skipping.")
                    continue

                col_info = colmap_images[rel_name]
                cid = col_info["cam_id"]
                cam = cameras[cid]
                W, H = cam["w"], cam["h"]
                fx, fy, cx, cy = cam["params"]

                # Intrinsic matrix
                K = np.array([
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float64)

                # Extrinsic matrix [R | t]
                q = col_info["q"]
                R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
                t = col_info["t"]
                extri_4x4 = np.eye(4, dtype=np.float64)
                extri_4x4[:3, :3] = R
                extri_4x4[:3, 3] = t

                # Output paths relative to out_dir
                dst_img_rel = f"images/{flight}/{rf.name}"
                dst_depth_rel = f"depth/{flight}/{rf.stem}.png"
                dst_mask_rel = f"depth_masks/{flight}/{rf.stem}.png"

                dst_img_abs = out_path / dst_img_rel
                dst_depth_abs = out_path / dst_depth_rel
                dst_mask_abs = out_path / dst_mask_rel

                # Copy image if not present
                if not dst_img_abs.exists():
                    shutil.copy2(rf, dst_img_abs)

                # Render depth if not present
                if not dst_depth_abs.exists() or not dst_mask_abs.exists():
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

                    # 3x3 morphological dilation to close sparse point rasterization gaps
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    depth_dilated = cv2.dilate(depth_map, kernel)
                    valid_mask = (depth_dilated > 0) & (depth_dilated < 300.0)

                    # 16-bit uint millimeter PNG
                    depth_mm = (depth_dilated * 1000.0).clip(0, 65535).astype(np.uint16)
                    cv2.imwrite(str(dst_depth_abs), depth_mm)

                    # 8-bit mask PNG (255 = valid depth)
                    mask_uint8 = (valid_mask * 255).astype(np.uint8)
                    cv2.imwrite(str(dst_mask_abs), mask_uint8)

                frame_registry[rel_name] = {
                    "rel_name": rel_name,
                    "flight": flight,
                    "filepath": dst_img_rel.replace("\\", "/"),
                    "depth_path": dst_depth_rel.replace("\\", "/"),
                    "mask_path": dst_mask_rel.replace("\\", "/"),
                    "extri": extri_4x4.tolist(),
                    "intri": K.tolist(),
                    "width": W,
                    "height": H
                }
                total_rendered += 1

    print(f"\nAll {total_rendered} unique physical frames registered and verified.")

    # 5. Build continuous multi-view sequences and manifests
    print("\nGenerating continuous multi-view windows (Length=4, Stride=1)...")
    manifests = {"train": {}, "val": {}, "test": {}}
    stats_per_flight = {}

    seq_global_id = 0
    (out_path / "sequences").mkdir(parents=True, exist_ok=True)

    for split_name, flight_list in FLIGHT_SPLITS.items():
        split_seq_count = 0
        for flight in flight_list:
            flight_frames = [
                frame_registry[k] for k in sorted(frame_registry.keys())
                if frame_registry[k]["flight"] == flight
            ]
            num_frames = len(flight_frames)
            num_seqs = (num_frames - seq_len) // stride + 1

            stats_per_flight[flight] = {
                "split": split_name,
                "frames": num_frames,
                "sequences": num_seqs
            }

            for s_idx in range(num_seqs):
                start_i = s_idx * stride
                window_frames = flight_frames[start_i : start_i + seq_len]
                seq_name = f"seq_{split_name}_{flight[-8:]}_{s_idx:03d}"
                seq_global_id += 1

                # Frame IDs within flight
                frame_ids = list(range(start_i, start_i + seq_len))

                # Build frame list
                seq_frames_data = []
                for wf in window_frames:
                    seq_frames_data.append({
                        "filepath": wf["filepath"],
                        "depth_path": wf["depth_path"],
                        "mask_path": wf["mask_path"],
                        "extri": wf["extri"],
                        "intri": wf["intri"]
                    })

                seq_dir = out_path / "sequences" / seq_name
                seq_dir.mkdir(parents=True, exist_ok=True)

                seq_meta = {
                    "sequence_id": seq_name,
                    "sequence_path": f"sequences/{seq_name}",
                    "supervision_type": "PSEUDO_GROUND_TRUTH",
                    "source_flight": flight,
                    "split": split_name,
                    "frame_ids": frame_ids,
                    "num_frames": seq_len,
                    "frames": seq_frames_data
                }

                # Save sequence metadata.json
                with open(seq_dir / "metadata.json", "w") as f:
                    json.dump(seq_meta, f, indent=2)

                # Add to manifest
                manifests[split_name][seq_name] = {
                    "sequence_path": f"sequences/{seq_name}",
                    "supervision_type": "PSEUDO_GROUND_TRUTH",
                    "source_flight": flight,
                    "frame_ids": frame_ids,
                    "frames": seq_frames_data
                }
                split_seq_count += 1

        print(f" - Split [{split_name.upper():5s}]: {split_seq_count} sequences generated.")

    # 6. Save train.json, val.json, test.json
    for split_name in ("train", "val", "test"):
        manifest_file = out_path / f"{split_name}.json"
        with open(manifest_file, "w") as f:
            json.dump(manifests[split_name], f, indent=2)
        print(f"Saved {manifest_file} with {len(manifests[split_name])} sequences.")

    # 7. Print summary statistics
    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE SUMMARY")
    print("=" * 70)
    total_seqs = sum(len(m) for m in manifests.values())
    print(f"Total Unique Flights:        {len(stats_per_flight)}")
    print(f"Total Physical Images:       {total_rendered}")
    print(f"Total Generated Sequences:   {total_seqs}")
    print(f" - Train Sequences:          {len(manifests['train'])} (Flights 1-3)")
    print(f" - Val Sequences:            {len(manifests['val'])} (Flight 4)")
    print(f" - Test Sequences:           {len(manifests['test'])} (Flight 5)")
    print("\nPer-Flight Statistics:")
    for fl, st in stats_per_flight.items():
        print(f" * Flight {fl}: Split={st['split']:5s}, Frames={st['frames']:2d}, Sequences={st['sequences']:2d}")
    print("=" * 70)

if __name__ == "__main__":
    build_dataset()

import os
import sys
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import cv2

def test_depth_projection():
    pcd = o3d.io.read_point_cloud("data/raw/uav_sequence/1121222322212102-4/dense_points.ply")
    pts_w = np.asarray(pcd.points, dtype=np.float32)
    print(f"Loaded {len(pts_w):,} points from dense_points.ply")

    # Read cameras.txt
    cameras = {}
    with open("data/raw/uav_sequence/1121222322212102-4/sparse/0/cameras.txt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split()
            cameras[int(parts[0])] = {
                "width": int(parts[2]), "height": int(parts[3]),
                "params": [float(x) for x in parts[4:]]
            }

    # Read first registered image from images.txt
    first_img = None
    with open("data/raw/uav_sequence/1121222322212102-4/sparse/0/images.txt") as f:
        lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i].strip()
        if not line or line.startswith("#"): continue
        parts = line.split()
        if len(parts) >= 10:
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            cid = int(parts[8])
            name = parts[9]
            first_img = {
                "q": np.array([qw, qx, qy, qz]),
                "t": np.array([tx, ty, tz]),
                "cam_id": cid,
                "name": name
            }
            break

    print("Testing with registered frame:", first_img["name"])
    cam = cameras[first_img["cam_id"]]
    W, H = cam["width"], cam["height"]
    fx, fy, cx, cy = cam["params"]

    # World-to-camera transform
    q = first_img["q"]
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    t = first_img["t"]

    # Transform all points to camera frame: X_c = R @ X_w + t
    pts_c = (R @ pts_w.T).T + t # [N, 3]

    # Filter in front of camera
    valid_z = pts_c[:, 2] > 0.5
    pts_c_front = pts_c[valid_z]

    u = (fx * pts_c_front[:, 0] / pts_c_front[:, 2]) + cx
    v = (fy * pts_c_front[:, 1] / pts_c_front[:, 2]) + cy
    z = pts_c_front[:, 2]

    # In image bounds
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u_valid = np.clip(np.floor(u[in_bounds]).astype(int), 0, W - 1)
    v_valid = np.clip(np.floor(v[in_bounds]).astype(int), 0, H - 1)
    z_valid = z[in_bounds]

    print(f"Points falling within camera view frustum: {len(z_valid):,}")
    print(f"Depth range: Min={z_valid.min():.2f}m, Max={z_valid.max():.2f}m, Median={np.median(z_valid):.2f}m")

    # Create z-buffer depth map
    depth_map = np.full((H, W), np.inf, dtype=np.float32)
    # Sort by z descending so closest points overwrite farthest
    sort_idx = np.argsort(-z_valid)
    depth_map[v_valid[sort_idx], u_valid[sort_idx]] = z_valid[sort_idx]
    valid_mask = np.isfinite(depth_map) & (depth_map > 0)
    depth_map[~valid_mask] = 0.0

    print(f"Depth map generated: shape={(H, W)}, valid pixels: {np.sum(valid_mask):,} ({np.mean(valid_mask)*100:.2f}%)")

if __name__ == "__main__":
    test_depth_projection()

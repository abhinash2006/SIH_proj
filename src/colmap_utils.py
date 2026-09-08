import os
import struct
import logging
from pathlib import Path
from typing import Dict, Any, List, Union
import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

class COLMAPExporter:
    """
    Exports VGGT camera parameters and reconstructed 3D points to standard
    COLMAP format (both text and binary .bin formats).
    Compatible with downstream Gaussian Splatting, NeRF, and Bundle Adjustment tools.
    """

    @staticmethod
    def rotmat_to_quaternion(R: np.ndarray) -> np.ndarray:
        """
        Converts 3x3 rotation matrix to COLMAP quaternion [qw, qx, qy, qz].
        """
        r = Rotation.from_matrix(R)
        # scipy returns [qx, qy, qz, qw]
        q = r.as_quat()
        return np.array([q[3], q[0], q[1], q[2]])

    @classmethod
    def export_colmap(
        cls,
        output_dir: Union[str, Path],
        frame_paths: List[str],
        extrinsics: np.ndarray, # [S, 3, 4]
        intrinsics: np.ndarray, # [S, 3, 3]
        points3d: np.ndarray,   # [N, 3]
        colors3d: np.ndarray,   # [N, 3] in [0, 1]
        export_binary: bool = True,
        export_text: bool = True
    ) -> str:
        """
        Exports scene to COLMAP sparse directory.
        Structure:
          output_dir/sparse/
          ├── cameras.txt / cameras.bin
          ├── images.txt / images.bin
          └── points3D.txt / points3D.bin
        """
        sparse_dir = Path(output_dir) / "sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        
        num_frames = len(frame_paths)
        num_points = len(points3d)
        
        # 1. Prepare Cameras (PINHOLE: camera_id, model="PINHOLE", width, height, fx, fy, cx, cy)
        # We assume 518x518 standard or take from intrinsics
        cameras = {}
        for i in range(num_frames):
            K = intrinsics[i]
            fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
            width, height = int(round(cx * 2)), int(round(cy * 2))
            if width <= 0 or height <= 0:
                width, height = 518, 518
            cameras[i + 1] = {
                "id": i + 1,
                "model": "PINHOLE",
                "width": width,
                "height": height,
                "params": [fx, fy, cx, cy]
            }

        # 2. Prepare Images (image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name)
        images = {}
        for i in range(num_frames):
            ext = extrinsics[i] # [3, 4]
            R = ext[:3, :3]
            t = ext[:3, 3]
            q = cls.rotmat_to_quaternion(R)
            img_name = Path(frame_paths[i]).name
            images[i + 1] = {
                "id": i + 1,
                "q": q,
                "t": t,
                "camera_id": i + 1,
                "name": img_name,
                "points2D": []
            }

        # Subsample 3D points if very large (> 200,000) for COLMAP text export
        pts_export = points3d
        col_export = (colors3d * 255.0).clip(0, 255).astype(np.uint8)
        if len(pts_export) > 200_000:
            sample_idx = np.random.choice(len(pts_export), 200_000, replace=False)
            pts_export = pts_export[sample_idx]
            col_export = col_export[sample_idx]

        # 3. Write Text Format
        if export_text:
            # cameras.txt
            with open(sparse_dir / "cameras.txt", "w") as f:
                f.write("# Camera list with one line of data per camera:\n")
                f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
                for cid, cam in cameras.items():
                    p = " ".join([str(x) for x in cam["params"]])
                    f.write(f"{cam['id']} {cam['model']} {cam['width']} {cam['height']} {p}\n")
                    
            # images.txt
            with open(sparse_dir / "images.txt", "w") as f:
                f.write("# Image list with two lines of data per image:\n")
                f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
                f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
                for iid, img in images.items():
                    qw, qx, qy, qz = img["q"]
                    tx, ty, tz = img["t"]
                    f.write(f"{img['id']} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {img['camera_id']} {img['name']}\n")
                    f.write("\n") # empty 2D-3D point tracks line
                    
            # points3D.txt
            with open(sparse_dir / "points3D.txt", "w") as f:
                f.write("# 3D point list with one line of data per point:\n")
                f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
                for idx in range(len(pts_export)):
                    x, y, z = pts_export[idx]
                    r, g, b = col_export[idx]
                    f.write(f"{idx + 1} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 0.5\n")

        # 4. Write Binary Format (.bin)
        if export_binary:
            # cameras.bin
            with open(sparse_dir / "cameras.bin", "wb") as f:
                f.write(struct.pack("<Q", len(cameras)))
                for cid, cam in cameras.items():
                    # PINHOLE model_id = 1
                    f.write(struct.pack("<iiQQ", cam["id"], 1, cam["width"], cam["height"]))
                    for p in cam["params"]:
                        f.write(struct.pack("<d", p))
                        
            # images.bin
            with open(sparse_dir / "images.bin", "wb") as f:
                f.write(struct.pack("<Q", len(images)))
                for iid, img in images.items():
                    qw, qx, qy, qz = img["q"]
                    tx, ty, tz = img["t"]
                    f.write(struct.pack("<i4d3di", img["id"], qw, qx, qy, qz, tx, ty, tz, img["camera_id"]))
                    name_bytes = img["name"].encode("utf-8") + b"\x00"
                    f.write(name_bytes)
                    f.write(struct.pack("<Q", 0)) # 0 2D points
                    
            # points3D.bin
            with open(sparse_dir / "points3D.bin", "wb") as f:
                f.write(struct.pack("<Q", len(pts_export)))
                for idx in range(len(pts_export)):
                    x, y, z = pts_export[idx]
                    r, g, b = col_export[idx]
                    f.write(struct.pack("<Q3d3Bdd", idx + 1, x, y, z, r, g, b, 0.5, 0.0))
                    f.write(struct.pack("<Q", 0)) # track length = 0

        logger.info(f"[SUCCESS] Exported COLMAP sparse reconstruction to: {sparse_dir}")
        return str(sparse_dir)

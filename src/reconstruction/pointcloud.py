"""
Point Cloud Builder using Open3D for UAV 3D Reconstruction.
Fuses RGB frames, depth/point maps, camera geometry, and dynamic exclusion masks
into a dense, colored, confidence-attributed 3D point cloud.
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import open3d as o3d

logger = logging.getLogger(__name__)


@dataclass
class PointCloudData:
    points: np.ndarray             # [N, 3] float32 (X, Y, Z)
    colors: np.ndarray             # [N, 3] float32 in [0, 1] (R, G, B)
    confidences: np.ndarray        # [N,] float32 in [0, 1]
    source_frames: np.ndarray      # [N,] int32 index of source keyframe
    pcd_o3d: o3d.geometry.PointCloud
    num_points: int


class PointCloudBuilder:
    """
    Builds Open3D point clouds from multi-view VGGT predictions or unprojected RGB-D frames.
    Filters dynamic pixels using YOLO + SAM 2 masks.
    """

    def __init__(
        self,
        min_confidence: float = 0.25,
        min_depth: float = 0.5,
        max_depth: float = 200.0,
    ):
        self.min_confidence = min_confidence
        self.min_depth = min_depth
        self.max_depth = max_depth

    def build_from_vggt_output(
        self,
        point_maps: np.ndarray,            # [S, H, W, 3]
        images: np.ndarray,                # [S, H, W, 3] uint8
        confidences: np.ndarray,           # [S, H, W] float32
        dynamic_masks: Optional[List[np.ndarray]] = None, # List of [H, W] uint8 (255 = dynamic)
    ) -> PointCloudData:
        """
        Extracts 3D point cloud directly from VGGT predicted world point maps and images.
        """
        S, H, W, _ = point_maps.shape

        all_points = []
        all_colors = []
        all_confs = []
        all_frame_ids = []

        for s in range(S):
            pts = point_maps[s]  # [H, W, 3]
            col = images[s].astype(np.float32) / 255.0  # [H, W, 3]
            cnf = confidences[s]  # [H, W]

            # Dynamic mask check
            if dynamic_masks is not None and s < len(dynamic_masks) and dynamic_masks[s] is not None:
                mask = dynamic_masks[s]
                if mask.shape != (H, W):
                    import cv2
                    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                static_valid = (mask == 0)
            else:
                static_valid = np.ones((H, W), dtype=bool)

            # Validity mask: high confidence + finite points + not masked dynamic
            valid = (
                static_valid &
                (cnf >= self.min_confidence) &
                np.isfinite(pts).all(axis=-1)
            )

            if not np.any(valid):
                continue

            all_points.append(pts[valid])
            all_colors.append(col[valid])
            all_confs.append(cnf[valid])
            all_frame_ids.append(np.full(np.count_nonzero(valid), s, dtype=np.int32))

        if not all_points:
            logger.warning("[POINTCLOUD] No valid 3D points passed quality filtering!")
            empty_pts = np.zeros((0, 3), dtype=np.float32)
            empty_col = np.zeros((0, 3), dtype=np.float32)
            pcd = o3d.geometry.PointCloud()
            return PointCloudData(
                points=empty_pts,
                colors=empty_col,
                confidences=np.zeros(0, dtype=np.float32),
                source_frames=np.zeros(0, dtype=np.int32),
                pcd_o3d=pcd,
                num_points=0,
            )

        points_np = np.concatenate(all_points, axis=0).astype(np.float32)
        colors_np = np.concatenate(all_colors, axis=0).astype(np.float32)
        confs_np = np.concatenate(all_confs, axis=0).astype(np.float32)
        frames_np = np.concatenate(all_frame_ids, axis=0).astype(np.int32)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors_np.astype(np.float64))

        logger.info(f"[POINTCLOUD] Generated point cloud with {len(points_np):,} points across {S} frames.")

        return PointCloudData(
            points=points_np,
            colors=colors_np,
            confidences=confs_np,
            source_frames=frames_np,
            pcd_o3d=pcd,
            num_points=len(points_np),
        )

    def save_ply(self, pcd_data: PointCloudData, output_path: str):
        """Saves Open3D point cloud to PLY file with ASCII/binary format."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(out_p), pcd_data.pcd_o3d, write_ascii=False)
        logger.info(f"[POINTCLOUD] Saved PLY point cloud ({pcd_data.num_points:,} points) to '{out_p}'.")

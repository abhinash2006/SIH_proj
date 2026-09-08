import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)

class PointCloudFilter:
    """
    Conservative geometric point cloud post-processor.
    Applies multi-stage filtering:
    1. Voxel grid downsampling
    2. Statistical Outlier Removal (SOR)
    3. Radius Outlier Removal (ROR)
    4. Bounding box spatial clipping
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        pcd_cfg = self.config.get("pointcloud_filtering", {})
        
        self.enabled = pcd_cfg.get("enabled", True)
        self.do_voxel = pcd_cfg.get("voxel_downsample", True)
        self.voxel_size = pcd_cfg.get("voxel_size", 0.02)
        
        self.do_sor = pcd_cfg.get("statistical_filter", True)
        self.sor_neighbors = pcd_cfg.get("nb_neighbors", 20)
        self.sor_std_ratio = pcd_cfg.get("std_ratio", 2.0)
        
        self.do_radius = pcd_cfg.get("radius_filter", True)
        self.radius_nb_points = pcd_cfg.get("radius_nb_points", 10)
        self.radius = pcd_cfg.get("radius", 0.08)

    def filter_pointcloud(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        confidence_scores: Optional[np.ndarray] = None,
        conf_threshold: float = 0.50
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Filters raw 3D point cloud using confidence gating + multi-stage geometric filtering.
        """
        if not self.enabled or len(points) == 0:
            return points, colors, {"initial_points": len(points), "final_points": len(points), "removed_points": 0}

        initial_count = len(points)
        
        # 0. Depth-Aware / Fused Confidence Gate
        if confidence_scores is not None and len(confidence_scores) == len(points):
            valid_idx = np.where(confidence_scores.flatten() >= conf_threshold)[0]
            if len(valid_idx) > 100:
                points = points[valid_idx]
                colors = colors[valid_idx]

        logger.info(f"[INFO] Filtering point cloud with {initial_count:,} points...")

        # Create Open3D PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        # 1. Voxel Grid Downsampling (Regularizes density across overlapping views)
        if self.do_voxel and self.voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            logger.info(f" - After voxel grid downsampling ({self.voxel_size}m): {len(pcd.points):,} points")

        # 2. Statistical Outlier Removal (SOR)
        if self.do_sor and len(pcd.points) > self.sor_neighbors:
            pcd, inlier_indices = pcd.remove_statistical_outlier(
                nb_neighbors=self.sor_neighbors,
                std_ratio=self.sor_std_ratio
            )
            logger.info(f" - After Statistical Outlier Removal (k={self.sor_neighbors}, std={self.sor_std_ratio}): {len(pcd.points):,} points")

        # 3. Radius Outlier Removal (ROR)
        if self.do_radius and len(pcd.points) > self.radius_nb_points:
            pcd, inlier_indices = pcd.remove_radius_outlier(
                nb_points=self.radius_nb_points,
                radius=self.radius
            )
            logger.info(f" - After Radius Outlier Removal (r={self.radius}m, min_pts={self.radius_nb_points}): {len(pcd.points):,} points")

        filtered_points = np.asarray(pcd.points, dtype=np.float32)
        filtered_colors = np.asarray(pcd.colors, dtype=np.float32)

        final_count = len(filtered_points)
        removed_count = initial_count - final_count
        reduction_pct = (removed_count / initial_count * 100.0) if initial_count > 0 else 0.0

        stats = {
            "initial_points": initial_count,
            "final_points": final_count,
            "removed_points": removed_count,
            "reduction_percentage": round(reduction_pct, 2)
        }

        logger.info(f"[SUCCESS] Point cloud filtering complete: {final_count:,} points retained ({reduction_pct:.1f}% noise removed)")
        return filtered_points, filtered_colors, stats

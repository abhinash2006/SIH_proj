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
        Dynamically adapts voxel sizing to point cloud spatial extent to prevent collapsing scene geometry.
        """
        if not self.enabled or len(points) == 0:
            return points, colors, {
                "initial_points": len(points),
                "final_points": len(points),
                "removed_points": 0,
                "reduction_percentage": 0.0,
                "stages": {"raw": len(points), "final": len(points)}
            }

        initial_count = len(points)
        stages_log = {"raw": initial_count}

        # 0. Depth-Aware / Fused Confidence Gate
        if confidence_scores is not None and len(confidence_scores) == len(points):
            valid_idx = np.where(confidence_scores.flatten() >= conf_threshold)[0]
            if len(valid_idx) > 100:
                points = points[valid_idx]
                colors = colors[valid_idx]
        stages_log["after_confidence"] = len(points)

        logger.info(f"[INFO] Filtering point cloud with {initial_count:,} raw points...")

        # Calculate spatial extent to determine scale-aware voxel size
        pts_min = np.min(points, axis=0)
        pts_max = np.max(points, axis=0)
        extent = pts_max - pts_min
        min_extent = float(np.min(extent)) if len(extent) > 0 else 1.0

        # In VGGT coordinate frame (extent ~ 1.0 to 3.0), voxel_size=0.02 is too coarse.
        # Scale voxel size adaptively: e.g. min_extent / 100, clamped between 0.003 and self.voxel_size
        eff_voxel_size = self.voxel_size
        if min_extent < 1.5:
            eff_voxel_size = float(np.clip(min_extent / 80.0, 0.0035, self.voxel_size))

        # Create Open3D PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        # 1. Voxel Grid Downsampling (Regularizes density across overlapping views)
        if self.do_voxel and eff_voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size=eff_voxel_size)
            logger.info(f" - After voxel grid downsampling ({eff_voxel_size:.4f}m): {len(pcd.points):,} points")
        stages_log["after_voxel"] = len(pcd.points)

        # 2. Statistical Outlier Removal (SOR)
        if self.do_sor and len(pcd.points) > self.sor_neighbors:
            pcd, inlier_indices = pcd.remove_statistical_outlier(
                nb_neighbors=self.sor_neighbors,
                std_ratio=self.sor_std_ratio
            )
            logger.info(f" - After Statistical Outlier Removal (k={self.sor_neighbors}, std={self.sor_std_ratio}): {len(pcd.points):,} points")
        stages_log["after_sor"] = len(pcd.points)

        # 3. Radius Outlier Removal (ROR)
        eff_radius = self.radius if min_extent >= 1.5 else float(np.clip(eff_voxel_size * 5.0, 0.02, self.radius))
        if self.do_radius and len(pcd.points) > self.radius_nb_points:
            pcd, inlier_indices = pcd.remove_radius_outlier(
                nb_points=self.radius_nb_points,
                radius=eff_radius
            )
            logger.info(f" - After Radius Outlier Removal (r={eff_radius:.4f}m, min_pts={self.radius_nb_points}): {len(pcd.points):,} points")
        stages_log["after_ror"] = len(pcd.points)

        # 4. Connected Component / Cluster Filtering (Removes isolated floating geometry clusters)
        if len(pcd.points) > 100:
            labels = np.array(pcd.cluster_dbscan(eps=eff_voxel_size * 5.0, min_points=10, print_progress=False))
            if len(labels) > 0 and labels.max() >= 0:
                counts = np.bincount(labels[labels >= 0])
                valid_clusters = np.where((counts >= 50) | (counts >= len(pcd.points) * 0.02))[0]
                keep_mask = np.isin(labels, valid_clusters)
                if np.sum(keep_mask) > 100:
                    pcd = pcd.select_by_index(np.where(keep_mask)[0])
                    logger.info(f" - After Connected Component Cluster Filtering: {len(pcd.points):,} points")
        stages_log["after_dbscan"] = len(pcd.points)

        filtered_points = np.asarray(pcd.points, dtype=np.float32)
        filtered_colors = np.asarray(pcd.colors, dtype=np.float32)

        final_count = len(filtered_points)
        removed_count = initial_count - final_count
        reduction_pct = (removed_count / initial_count * 100.0) if initial_count > 0 else 0.0

        stats = {
            "initial_points": initial_count,
            "final_points": final_count,
            "removed_points": removed_count,
            "reduction_percentage": round(reduction_pct, 2),
            "effective_voxel_size": round(eff_voxel_size, 5),
            "stages": stages_log
        }

        logger.info(f"[SUCCESS] Point cloud filtering complete: {final_count:,} points retained ({reduction_pct:.1f}% noise removed)")
        return filtered_points, filtered_colors, stats

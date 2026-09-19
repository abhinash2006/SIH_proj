"""
Point Cloud Filtering & Outlier Removal with Open3D.
Cleans reconstruction noise using Voxel Grid Downsampling, Statistical Outlier Removal (SOR),
Radius Outlier Removal (ROR), and Ground Plane Detection.
"""

import logging
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
import open3d as o3d

from .pointcloud import PointCloudData

logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    voxel_size: float = 0.05               # Voxel downsampling leaf size in meters (5cm)
    use_sor: bool = True                   # Statistical Outlier Removal
    sor_nb_neighbors: int = 20
    sor_std_ratio: float = 2.0
    use_ror: bool = True                   # Radius Outlier Removal
    ror_nb_points: int = 16
    ror_radius: float = 0.15
    estimate_normals: bool = True
    normal_radius: float = 0.2
    normal_max_nn: int = 30


class PointCloudFilter:
    """
    Cleans raw multi-view point clouds using progressive geometric filtering.
    """

    def __init__(self, config: Optional[FilterConfig] = None):
        self.config = config or FilterConfig()

    def filter(self, pcd_data: PointCloudData) -> PointCloudData:
        """
        Runs progressive filtering pipeline on input PointCloudData.
        """
        pcd = pcd_data.pcd_o3d
        init_pts = len(pcd.points)

        if init_pts == 0:
            return pcd_data

        logger.info(f"[PCD_FILTER] Starting filtering on {init_pts:,} points...")

        # 1. Voxel Downsampling (unifies density across overlapping frames, adapt to bounding box extent)
        bbox = pcd.get_axis_aligned_bounding_box()
        extent = float(bbox.get_max_extent())
        target_voxel = max(self.config.voxel_size, extent / 250.0) if extent > 1.0 else self.config.voxel_size
        if target_voxel > 0.001:
            pcd = pcd.voxel_down_sample(voxel_size=target_voxel)
            logger.info(f"[PCD_FILTER] After Voxel Downsampling ({target_voxel:.3f}m): {len(pcd.points):,} points.")

        # Estimate scale from point cloud spacing on a fast representative sample
        if len(pcd.points) > 3000:
            sample_pcd = pcd.random_down_sample(sampling_ratio=min(1.0, 3000.0 / len(pcd.points)))
        else:
            sample_pcd = pcd
        distances = sample_pcd.compute_nearest_neighbor_distance()
        avg_spacing = float(np.mean(distances)) if len(distances) > 0 else 0.1
        effective_ror_radius = max(self.config.ror_radius, avg_spacing * 2.5)
        effective_norm_radius = max(self.config.normal_radius, avg_spacing * 3.0)

        # 2. Statistical Outlier Removal (SOR)
        if self.config.use_sor and len(pcd.points) > self.config.sor_nb_neighbors:
            pcd_sor, ind = pcd.remove_statistical_outlier(
                nb_neighbors=min(self.config.sor_nb_neighbors, len(pcd.points) - 1),
                std_ratio=self.config.sor_std_ratio,
            )
            if len(pcd_sor.points) >= 10:
                pcd = pcd_sor
                logger.info(f"[PCD_FILTER] After Statistical Outlier Removal: {len(pcd.points):,} points.")

        # 3. Radius Outlier Removal (ROR)
        if self.config.use_ror and len(pcd.points) > self.config.ror_nb_points:
            pcd_ror, ind = pcd.remove_radius_outlier(
                nb_points=min(self.config.ror_nb_points, 4),
                radius=effective_ror_radius,
            )
            if len(pcd_ror.points) >= 10:
                pcd = pcd_ror
                logger.info(f"[PCD_FILTER] After Radius Outlier Removal: {len(pcd.points):,} points.")

        # 4. Normal Estimation
        if self.config.estimate_normals and len(pcd.points) >= 4:
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=effective_norm_radius,
                    max_nn=self.config.normal_max_nn,
                )
            )
            # Orient normals consistently towards camera / upward
            try:
                pcd.orient_normals_consistent_tangent_plane(k=min(15, len(pcd.points) - 1))
            except Exception:
                pass
            logger.info("[PCD_FILTER] Estimated point normals consistently.")

        filtered_pts = np.asarray(pcd.points, dtype=np.float32)
        filtered_cols = np.asarray(pcd.colors, dtype=np.float32)

        return PointCloudData(
            points=filtered_pts,
            colors=filtered_cols,
            confidences=np.ones(len(filtered_pts), dtype=np.float32),
            source_frames=np.zeros(len(filtered_pts), dtype=np.int32),
            pcd_o3d=pcd,
            num_points=len(filtered_pts),
        )

    @staticmethod
    def segment_ground_plane(
        pcd: o3d.geometry.PointCloud,
        distance_threshold: float = 0.1,
        ransac_n: int = 3,
        num_iterations: int = 1000,
    ) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, np.ndarray]:
        """
        Segments ground plane using RANSAC plane fitting.
        Returns (ground_pcd, non_ground_pcd, plane_model [a, b, c, d]).
        """
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        ground_pcd = pcd.select_by_index(inliers)
        non_ground_pcd = pcd.select_by_index(inliers, invert=True)
        return ground_pcd, non_ground_pcd, plane_model

import os
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)

class PointCloudReconstructor:
    """
    Constructs, formats, and exports 3D point clouds from VGGT predictions.
    Supports confidence masking, depth unprojection vs direct point maps,
    and Open3D PLY serialization.
    """

    @staticmethod
    def build_pointcloud_from_predictions(
        predictions: Dict[str, Any],
        use_depth_unprojection: bool = True,
        confidence_threshold: float = 1.1,
        max_points: Optional[int] = 1_000_000
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts 3D points, RGB colors, and confidence values from VGGT predictions.
        
        Args:
            predictions: Output dictionary from VGGTInferenceEngine.
            use_depth_unprojection: If True, uses camera + unprojected depth map (Reconstruction B).
                                   If False, uses direct point head predictions (Reconstruction A).
            confidence_threshold: Minimum confidence score to retain points.
            max_points: Maximum points to return (subsamples uniformly if exceeded).
            
        Returns:
            points: [N, 3] float32 array
            colors: [N, 3] float32 array in [0, 1]
            confidences: [N] float32 array
        """
        if use_depth_unprojection:
            pts_all = predictions["unprojected_points"] # [S, H, W, 3]
            conf_all = predictions["depth_conf"]        # [S, H, W]
        else:
            pts_all = predictions["point_maps"]         # [S, H, W, 3]
            conf_all = predictions["point_conf"]        # [S, H, W]

        images_rgb = predictions["images"]              # [S, H, W, 3] uint8

        S, H, W, _ = pts_all.shape
        
        # Flatten tensors
        flat_pts = pts_all.reshape(-1, 3)
        flat_conf = conf_all.reshape(-1)
        flat_colors = images_rgb.reshape(-1, 3).astype(np.float32) / 255.0

        # Confidence and validity mask
        valid_mask = (
            np.isfinite(flat_pts).all(axis=1) &
            (flat_conf >= confidence_threshold) &
            (np.linalg.norm(flat_pts, axis=1) > 1e-4) &
            (np.linalg.norm(flat_pts, axis=1) < 500.0) # remove extreme distant floating points
        )

        valid_pts = flat_pts[valid_mask]
        valid_colors = flat_colors[valid_mask]
        valid_conf = flat_conf[valid_mask]

        # Fallback if threshold was too high
        if len(valid_pts) < 1000:
            logger.warning(f"[WARNING] Confidence threshold {confidence_threshold} yielded only {len(valid_pts)} points. Relaxing threshold.")
            med_conf = np.median(flat_conf)
            valid_mask = np.isfinite(flat_pts).all(axis=1) & (flat_conf >= min(med_conf, 0.5))
            valid_pts = flat_pts[valid_mask]
            valid_colors = flat_colors[valid_mask]
            valid_conf = flat_conf[valid_mask]

        # Subsample if exceeding max_points
        if max_points and len(valid_pts) > max_points:
            sample_idx = np.random.choice(len(valid_pts), size=max_points, replace=False)
            valid_pts = valid_pts[sample_idx]
            valid_colors = valid_colors[sample_idx]
            valid_conf = valid_conf[sample_idx]

        logger.info(f"[SUCCESS] Reconstructed {len(valid_pts):,} 3D points (Confidence >= {confidence_threshold})")
        return valid_pts.astype(np.float32), valid_colors.astype(np.float32), valid_conf.astype(np.float32)

    @staticmethod
    def to_open3d_pcd(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
        """
        Converts points and colors to an Open3D PointCloud object.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        return pcd

    @staticmethod
    def save_ply(
        points: np.ndarray,
        colors: np.ndarray,
        output_path: Union[str, Path]
    ) -> str:
        """
        Saves point cloud to a binary PLY file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        pcd = PointCloudReconstructor.to_open3d_pcd(points, colors)
        
        # Estimate normals for realistic shading
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=False)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"[SUCCESS] Saved 3D Point Cloud PLY: {output_path} ({size_mb:.2f} MB, {len(points):,} points)")
        return str(output_path)

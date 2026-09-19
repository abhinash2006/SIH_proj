"""
Georeferencing Spatial Aligner for UAV 3D Reconstruction.
Aligns VGGT unscaled camera coordinate poses with real-world metric GPS/IMU trajectory
using the robust Umeyama algorithm (Procrustes analysis) with RANSAC outlier filtering.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GeoreferenceTransform:
    scale: float                 # Metric scale factor s
    rotation: np.ndarray         # 3x3 rotation matrix R
    translation: np.ndarray      # 3D translation vector t in meters [tx, ty, tz]
    alignment_rmse_m: float      # Root-Mean-Square Error of camera-GPS alignment in meters
    num_inliers: int             # Number of inlier anchor poses used
    total_poses: int             # Total camera poses processed


class GeoreferenceAligner:
    """
    Computes optimal 7-DoF Sim(3) similarity transformation:
        P_world = s * R * P_vggt + t
    matching VGGT camera centers to real-world GPS ENU coordinates.
    """

    def __init__(
        self,
        ransac_iterations: int = 1000,
        inlier_threshold_m: float = 2.5,
    ):
        self.ransac_iterations = int(ransac_iterations)
        self.inlier_threshold_m = float(inlier_threshold_m)

    @staticmethod
    def umeyama_alignment(
        src_points: np.ndarray,
        dst_points: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Computes the optimal similarity transformation (scale s, rotation R, translation t)
        minimizing || dst - (s * R * src + t) ||^2 via SVD (Umeyama, 1991).

        Args:
            src_points: [N, 3] points in source (VGGT) coordinate system.
            dst_points: [N, 3] points in destination (GPS ENU) coordinate system.

        Returns:
            (scale s, rotation R (3x3), translation t (3,))
        """
        assert src_points.shape == dst_points.shape
        n, m = src_points.shape
        assert m == 3 and n >= 3

        mu_src = np.mean(src_points, axis=0)
        mu_dst = np.mean(dst_points, axis=0)

        src_centered = src_points - mu_src
        dst_centered = dst_points - mu_dst

        var_src = np.sum(src_centered ** 2) / n
        if var_src < 1e-8:
            return 1.0, np.eye(3), mu_dst - mu_src

        # Cross-covariance matrix H = (dst^T * src) / n
        H = (dst_centered.T @ src_centered) / n

        U, S, Vt = np.linalg.svd(H)
        R = U @ Vt

        # Check reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt

        # Optimal scale
        scale = float(np.trace(np.diag(S)) / var_src) if var_src > 0 else 1.0
        if scale <= 0:
            scale = 1.0

        translation = mu_dst - scale * (R @ mu_src)

        return float(scale), R, translation

    def align_trajectory(
        self,
        camera_extrinsics: np.ndarray,  # [S, 3, 4] or [S, 4, 4]
        gps_positions_enu: np.ndarray,  # [N, 3]
        gps_timestamps: Optional[np.ndarray] = None,
    ) -> Tuple[GeoreferenceTransform, np.ndarray]:
        """
        Aligns VGGT camera positions to GPS trajectory.

        Args:
            camera_extrinsics: Camera extrinsic matrices [R | t] or [R | -R*C]
            gps_positions_enu: GPS positions in local ENU meters [N, 3]

        Returns:
            GeoreferenceTransform and world-aligned camera centers [S, 3]
        """
        S = camera_extrinsics.shape[0]
        N = gps_positions_enu.shape[0]

        # Extract camera optical centers C = -R^T * t
        cam_centers = np.zeros((S, 3), dtype=np.float64)
        for i in range(S):
            ext = camera_extrinsics[i]
            R_i = ext[:3, :3]
            t_i = ext[:3, 3]
            cam_centers[i] = -R_i.T @ t_i

        # Associate camera frames with GPS points (linear interpolation along sequence if timestamps not provided)
        if N == S:
            matched_gps = gps_positions_enu
        else:
            interp_indices = np.linspace(0, N - 1, S)
            matched_gps = np.zeros((S, 3), dtype=np.float64)
            for dim in range(3):
                matched_gps[:, dim] = np.interp(interp_indices, np.arange(N), gps_positions_enu[:, dim])

        # Run RANSAC Umeyama alignment
        best_scale = 1.0
        best_R = np.eye(3)
        best_t = np.zeros(3)
        best_inliers = []
        min_rmse = float("inf")

        if S < 4:
            # Direct solve on available points
            s, R, t = self.umeyama_alignment(cam_centers, matched_gps)
            transformed = (s * (R @ cam_centers.T)).T + t
            rmse = float(np.sqrt(np.mean(np.sum((transformed - matched_gps) ** 2, axis=1))))
            transform = GeoreferenceTransform(
                scale=round(s, 4),
                rotation=R,
                translation=t,
                alignment_rmse_m=round(rmse, 3),
                num_inliers=S,
                total_poses=S,
            )
            return transform, transformed

        # RANSAC loop
        rng = np.random.default_rng(42)
        for _ in range(self.ransac_iterations):
            sample_idx = rng.choice(S, size=3, replace=False)
            try:
                s_cand, R_cand, t_cand = self.umeyama_alignment(cam_centers[sample_idx], matched_gps[sample_idx])
                cand_transformed = (s_cand * (R_cand @ cam_centers.T)).T + t_cand
                resids = np.linalg.norm(cand_transformed - matched_gps, axis=1)
                inliers = np.where(resids < self.inlier_threshold_m)[0]

                if len(inliers) > len(best_inliers):
                    best_inliers = inliers
                    best_scale, best_R, best_t = s_cand, R_cand, t_cand
                    min_rmse = float(np.mean(resids[inliers]))
            except Exception:
                continue

        # Refine on all inliers
        if len(best_inliers) >= 3:
            s_ref, R_ref, t_ref = self.umeyama_alignment(cam_centers[best_inliers], matched_gps[best_inliers])
            final_transformed = (s_ref * (R_ref @ cam_centers.T)).T + t_ref
            final_resids = np.linalg.norm(final_transformed[best_inliers] - matched_gps[best_inliers], axis=1)
            final_rmse = float(np.sqrt(np.mean(final_resids ** 2)))
        else:
            s_ref, R_ref, t_ref = self.umeyama_alignment(cam_centers, matched_gps)
            final_transformed = (s_ref * (R_ref @ cam_centers.T)).T + t_ref
            final_rmse = float(np.sqrt(np.mean(np.sum((final_transformed - matched_gps) ** 2, axis=1))))
            best_inliers = np.arange(S)

        logger.info(
            f"[GEOREF] Sim(3) Alignment complete | Scale: {s_ref:.3f} | "
            f"RMSE: {final_rmse:.3f}m | Inliers: {len(best_inliers)}/{S}"
        )

        transform = GeoreferenceTransform(
            scale=round(s_ref, 4),
            rotation=R_ref,
            translation=t_ref,
            alignment_rmse_m=round(final_rmse, 3),
            num_inliers=len(best_inliers),
            total_poses=S,
        )

        return transform, final_transformed

    @staticmethod
    def transform_points(points: np.ndarray, transform: GeoreferenceTransform) -> np.ndarray:
        """Applies Sim(3) transform to 3D point cloud: P_world = s * R * P + t."""
        orig_shape = points.shape
        flat_pts = points.reshape(-1, 3)
        transformed = (transform.scale * (transform.rotation @ flat_pts.T)).T + transform.translation
        return transformed.reshape(orig_shape)

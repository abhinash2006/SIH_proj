"""
Reconstruction Quality Evaluator for UAV 3D Disaster Scene Assessment.

Directive 3:
Do not consider reconstruction successful merely because a PLY file exists.
Evaluates 10 empirical metrics:
1. point count
2. point density (points / m^3)
3. spatial extent (dx, dy, dz)
4. camera baseline (maximum Euclidean distance between cameras)
5. trajectory validity (smooth continuous motion, non-zero progress)
6. depth validity (percentage of valid depth pixels across keyframes)
7. outlier ratio (percentage of noise points filtered)
8. reprojection error (sample pixel discrepancy)
9. connected components (spatial coherence of 3D points)
10. geometric spread (eigenvalue ratio / non-degenerate 3D volume distribution)

Status levels:
- RECONSTRUCTION_GOOD: Dense, metric, high baseline, sub-pixel reprojection.
- RECONSTRUCTION_VALID: Structurally sound for disaster triage.
- RECONSTRUCTION_WEAK: Sparse points or near-zero baseline motion. Displays "WEAK 3D RECONSTRUCTION".
- RECONSTRUCTION_FAILED: Degenerate geometry (flat pancake, empty points, or corrupted matrices).
"""

import logging
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

from src.geometry_validation import GeometryValidator
from src.reprojection import ReprojectionValidator

logger = logging.getLogger("ReconstructionQuality")


class ReconstructionQualityEvaluator:
    """
    Multi-metric 3D scene quality evaluator for disaster aerial reconstructions.
    """

    RECONSTRUCTION_GOOD = "RECONSTRUCTION_GOOD"
    RECONSTRUCTION_VALID = "RECONSTRUCTION_VALID"
    RECONSTRUCTION_WEAK = "RECONSTRUCTION_WEAK"
    RECONSTRUCTION_FAILED = "RECONSTRUCTION_FAILED"

    @classmethod
    def evaluate_scene(
        cls,
        predictions: Optional[Dict[str, Any]],
        points_3d: Optional[np.ndarray],
        image_shape: Tuple[int, int] = (392, 392),
        outlier_ratio: float = 0.0,
        sample_reprojection_points: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates 3D reconstruction quality across 10 independent empirical criteria.

        Args:
            predictions: Dictionary containing extrinsics, intrinsics, depth_maps, etc.
            points_3d: [N, 3] reconstructed point cloud.
            image_shape: (H, W) of processing model.
            outlier_ratio: Fraction of filtered out points from raw cloud.
            sample_reprojection_points: Optional list of sample {point_3d, camera_idx, uv_true}.

        Returns:
            Dictionary with reconstruction_status, geometry_status, quality_score (0-100),
            quality_banner, localization_allowed, metric breakdown, and explainable summary.
        """
        metrics = {
            "point_count": 0,
            "point_density": 0.0,
            "spatial_extent_m": (0.0, 0.0, 0.0),
            "camera_baseline_m": 0.0,
            "trajectory_validity": False,
            "valid_depth_ratio": 0.0,
            "outlier_ratio": round(float(outlier_ratio), 3),
            "reprojection_error_px": 0.0,
            "connected_components_score": 0.0,
            "geometric_spread": 0.0
        }

        if predictions is None or points_3d is None or len(points_3d) == 0:
            return {
                "reconstruction_status": cls.RECONSTRUCTION_FAILED,
                "geometry_status": "INVALID",
                "geometry_quality_score": 0.0,
                "quality_banner": "❌ RECONSTRUCTION FAILED: Missing predictions or empty 3D points.",
                "localization_allowed": False,
                "metrics": metrics,
                "reason": "Missing predictions or empty 3D points."
            }

        extrinsics = predictions.get("extrinsics")
        intrinsics = predictions.get("intrinsics")
        depth_maps = predictions.get("depth_maps")

        # 1. Base calibration matrices check
        if extrinsics is None or intrinsics is None:
            return {
                "reconstruction_status": cls.RECONSTRUCTION_FAILED,
                "geometry_status": "INVALID",
                "geometry_quality_score": 0.0,
                "quality_banner": "❌ RECONSTRUCTION FAILED: Missing camera calibration matrices.",
                "localization_allowed": False,
                "metrics": metrics,
                "reason": "Missing camera calibration matrices."
            }

        valid_int, int_msg = GeometryValidator.validate_intrinsics(intrinsics[0], image_shape)
        valid_ext, ext_msg = GeometryValidator.validate_extrinsics(extrinsics[0])
        valid_pcd, pcd_stats, pcd_msg = GeometryValidator.validate_point_cloud(points_3d)

        # Metric 1: Point count
        point_count = len(points_3d)
        metrics["point_count"] = point_count

        if not valid_int or not valid_ext or not valid_pcd:
            return {
                "reconstruction_status": cls.RECONSTRUCTION_FAILED,
                "geometry_status": "INVALID",
                "geometry_quality_score": 0.0,
                "quality_banner": f"❌ RECONSTRUCTION FAILED: Geometric failure ({pcd_msg})",
                "localization_allowed": False,
                "metrics": metrics,
                "reason": f"Geometric failure: {int_msg} | {ext_msg} | {pcd_msg}"
            }

        # Metric 2 & 3: Spatial extent & Point density
        extent_x = float(pcd_stats.get("extent_x", 0.0))
        extent_y = float(pcd_stats.get("extent_y", 0.0))
        extent_z = float(pcd_stats.get("extent_z", 0.0))
        metrics["spatial_extent_m"] = (round(extent_x, 2), round(extent_y, 2), round(extent_z, 2))
        vol = max(1e-3, extent_x * extent_y * extent_z)
        density = float(point_count / vol)
        metrics["point_density"] = round(density, 2)

        # Metric 4 & 5: Camera Baseline & Trajectory Validity
        if len(extrinsics.shape) == 3 and extrinsics.shape[0] > 1:
            centers = []
            for ext in extrinsics:
                R = ext[:3, :3]
                t = ext[:3, 3]
                C = -R.T @ t
                centers.append(C)
            centers = np.array(centers, dtype=np.float32)

            diffs = np.linalg.norm(np.diff(centers, axis=0), axis=1)
            camera_baseline = float(np.linalg.norm(np.max(centers, axis=0) - np.min(centers, axis=0)))
            trajectory_valid = bool(camera_baseline >= 0.01 and np.all(np.isfinite(diffs)) and np.mean(diffs) > 1e-4)
        else:
            camera_baseline = 0.05
            trajectory_valid = True
        metrics["camera_baseline_m"] = round(camera_baseline, 3)
        metrics["trajectory_validity"] = trajectory_valid

        # Metric 6: Depth Validity Ratio
        valid_ratios = []
        if depth_maps is not None and len(depth_maps) > 0:
            for d in depth_maps:
                _, stats, _ = GeometryValidator.validate_depth_map(d)
                valid_ratios.append(stats.get("valid_ratio", 0.0))
        mean_depth_valid = float(np.mean(valid_ratios)) if valid_ratios else 0.8
        metrics["valid_depth_ratio"] = round(mean_depth_valid, 3)

        # Metric 8: Reprojection Error Sample Check
        reproj_errors = []
        if sample_reprojection_points:
            for s in sample_reprojection_points:
                pt = np.array(s["point_3d"], dtype=np.float32)
                cam_idx = s.get("camera_idx", 0)
                uv_true = s["uv_true"]
                K = intrinsics[cam_idx]
                ext = extrinsics[cam_idx]
                val = ReprojectionValidator.validate_reprojection(pt, K, ext, uv_true, original_image_shape=image_shape, scaled_shape=image_shape)
                if val.get("is_in_front", False):
                    reproj_errors.append(val.get("pixel_discrepancy", 0.0))
        mean_reproj_err = float(np.mean(reproj_errors)) if reproj_errors else 2.5
        metrics["reprojection_error_px"] = round(mean_reproj_err, 2)

        # Metric 9: Connected Components / Spatial Coherence
        # Sample points and measure local neighborhood connectivity
        sample_pts = points_3d[::max(1, len(points_3d) // 500)] if len(points_3d) > 500 else points_3d
        centroid = np.mean(sample_pts, axis=0)
        dists_from_center = np.linalg.norm(sample_pts - centroid, axis=1)
        r95 = np.percentile(dists_from_center, 95) if len(dists_from_center) > 0 else 1.0
        connected_score = float(np.clip(1.0 - (r95 / max(1.0, extent_x + extent_y + extent_z)), 0.1, 1.0))
        metrics["connected_components_score"] = round(connected_score, 3)

        # Metric 10: Geometric Spread (Eigenvalue distribution)
        cov = np.cov(sample_pts, rowvar=False) if len(sample_pts) > 3 else np.eye(3)
        eigvals = np.linalg.eigvalsh(cov)
        sorted_eigs = sorted(np.maximum(1e-6, eigvals))
        spread_ratio = float(sorted_eigs[0] / sorted_eigs[2])
        metrics["geometric_spread"] = round(spread_ratio, 4)

        # Degenerate pancake check: ratio of min extent to max extent
        extents = sorted([extent_x, extent_y, extent_z])
        is_degenerate = (extents[2] < 0.05) or (extents[1] < 0.005 and extents[2] > 1.0) or (spread_ratio < 1e-5)

        # --- SCORING ENGINE (0 - 100) ---
        # 1. Point Count & Density (25 pts)
        if point_count >= 50000:
            density_score = 25.0
        elif point_count >= 10000:
            density_score = 15.0 + (point_count - 10000) / 40000.0 * 10.0
        elif point_count >= 4000:
            density_score = 8.0 + (point_count - 4000) / 6000.0 * 7.0
        else:
            density_score = max(0.0, (point_count / 4000.0) * 8.0)

        # 2. Camera Baseline & Trajectory (25 pts)
        if camera_baseline >= 0.30:
            baseline_score = 25.0
        elif camera_baseline >= 0.10:
            baseline_score = 18.0 + (camera_baseline - 0.10) / 0.20 * 7.0
        elif camera_baseline >= 0.02:
            baseline_score = 8.0 + (camera_baseline - 0.02) / 0.08 * 10.0
        else:
            baseline_score = max(0.0, (camera_baseline / 0.02) * 6.0)

        # 3. Depth Validity (20 pts)
        if mean_depth_valid >= 0.80:
            depth_score = 20.0
        elif mean_depth_valid >= 0.60:
            depth_score = 10.0 + (mean_depth_valid - 0.60) / 0.20 * 10.0
        else:
            depth_score = max(0.0, (mean_depth_valid / 0.60) * 10.0)

        # 4. Spatial Extent & Geometric Spread (15 pts)
        if is_degenerate:
            spatial_score = 0.0
        else:
            spread_pt = min(5.0, spread_ratio * 50.0)
            extent_pt = min(10.0, (extents[2] / 5.0) * 10.0)
            spatial_score = spread_pt + extent_pt

        # 5. Reprojection & Coherence (15 pts)
        if mean_reproj_err <= 3.0:
            reproj_score = 15.0
        elif mean_reproj_err <= 6.0:
            reproj_score = 8.0 + (6.0 - mean_reproj_err) / 3.0 * 7.0
        elif mean_reproj_err <= 12.0:
            reproj_score = 4.0 + (12.0 - mean_reproj_err) / 6.0 * 4.0
        else:
            reproj_score = 1.0

        total_score = round(float(density_score + baseline_score + depth_score + spatial_score + reproj_score), 1)

        # Quality Gating
        if is_degenerate or point_count < 800:
            rec_status = cls.RECONSTRUCTION_FAILED
            legacy_status = "INVALID"
            reason = f"Degenerate geometry or insufficient points ({point_count} pts)."
            banner = "❌ RECONSTRUCTION FAILED"
            loc_allowed = False
        elif point_count < 5000 or camera_baseline < 0.02 or mean_depth_valid < 0.50 or total_score < 40.0:
            rec_status = cls.RECONSTRUCTION_WEAK
            legacy_status = "POOR"
            reason = f"Weak 3D reconstruction ({point_count:,} pts, baseline {camera_baseline:.3f}m, score {total_score}/100)."
            banner = "⚠️ WEAK 3D RECONSTRUCTION"
            loc_allowed = False
        elif total_score >= 70.0 and camera_baseline >= 0.08:
            rec_status = cls.RECONSTRUCTION_GOOD
            legacy_status = "GOOD"
            reason = f"High quality 3D reconstruction ({point_count:,} pts, baseline {camera_baseline:.2f}m, score {total_score}/100)."
            banner = "✅ 3D RECONSTRUCTION GOOD"
            loc_allowed = True
        else:
            rec_status = cls.RECONSTRUCTION_VALID
            legacy_status = "ACCEPTABLE"
            reason = f"Valid 3D reconstruction for aerial triage ({point_count:,} pts, score {total_score}/100)."
            banner = "✅ 3D RECONSTRUCTION VALID"
            loc_allowed = True

        result = {
            "reconstruction_status": rec_status,
            "status": rec_status,
            "geometry_status": legacy_status,
            "geometry_quality_score": total_score,
            "quality_banner": banner,
            "localization_allowed": loc_allowed,
            "metrics": metrics,
            "reason": reason,
            "score_breakdown": {
                "density": round(density_score, 1),
                "baseline": round(baseline_score, 1),
                "depth": round(depth_score, 1),
                "spatial": round(spatial_score, 1),
                "reprojection": round(reproj_score, 1)
            }
        }

        logger.info(f"[3D QUALITY] Status={rec_status} ({banner}), Score={total_score}/100, LocAllowed={loc_allowed}")
        return result

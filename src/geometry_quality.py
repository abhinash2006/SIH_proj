import numpy as np
import logging
from typing import Dict, Any, Tuple
from src.geometry_validation import GeometryValidator

logger = logging.getLogger("GeometryQuality")

class GeometryQualityEvaluator:
    """
    Evaluates 3D reconstruction quality based on empirical measurements of
    camera baseline, point density, depth validity, spatial variance, and intrinsics.
    """

    @classmethod
    def evaluate_reconstruction_quality(
        cls,
        predictions: Dict[str, Any],
        points_3d: np.ndarray,
        image_shape: Tuple[int, int] = (392, 392)
    ) -> Dict[str, Any]:
        """
        Computes geometry quality score and status (GOOD, ACCEPTABLE, POOR, INVALID).
        """
        if predictions is None or points_3d is None or len(points_3d) == 0:
            return {
                "geometry_status": "INVALID",
                "geometry_quality_score": 0.0,
                "reason": "Missing predictions or empty 3D points.",
                "localization_allowed": False
            }

        extrinsics = predictions.get("extrinsics")
        intrinsics = predictions.get("intrinsics")
        depth_maps = predictions.get("depth_maps")

        # 1. Intrinsics & Extrinsics Validation
        valid_int, int_msg = GeometryValidator.validate_intrinsics(intrinsics[0], image_shape)
        valid_ext, ext_msg = GeometryValidator.validate_extrinsics(extrinsics[0])
        valid_pcd, pcd_stats, pcd_msg = GeometryValidator.validate_point_cloud(points_3d)

        if not valid_int or not valid_ext or not valid_pcd:
            reason = f"Geometric failure: {int_msg} | {ext_msg} | {pcd_msg}"
            logger.warning(f"[RECONSTRUCTION INVALID] {reason}")
            return {
                "geometry_status": "INVALID",
                "geometry_quality_score": 0.0,
                "reason": reason,
                "localization_allowed": False,
                "point_count": pcd_stats.get("point_count", 0)
            }

        # 2. Camera Motion / Baseline Check
        translations = extrinsics[:, :3, 3] # [S, 3]
        camera_baseline = float(np.linalg.norm(np.max(translations, axis=0) - np.min(translations, axis=0)))
        
        # 3. Depth Valid Ratio
        valid_ratios = []
        for d in depth_maps:
            _, stats, _ = GeometryValidator.validate_depth_map(d)
            valid_ratios.append(stats.get("valid_ratio", 0.0))
        mean_depth_valid_ratio = float(np.mean(valid_ratios)) if valid_ratios else 0.0

        # 4. Point Count & Spatial Extent
        point_count = len(points_3d)
        extent_norm = float(np.sqrt(pcd_stats["extent_x"]**2 + pcd_stats["extent_y"]**2 + pcd_stats["extent_z"]**2))

        # Score computation (0-100 scale)
        baseline_score = min(30.0, (camera_baseline / 0.5) * 30.0)
        density_score = min(30.0, (point_count / 10000.0) * 30.0)
        depth_score = min(20.0, (mean_depth_valid_ratio / 0.8) * 20.0)
        extent_score = min(20.0, (extent_norm / 5.0) * 20.0)

        quality_score = round(float(baseline_score + density_score + depth_score + extent_score), 1)

        if quality_score >= 70.0 and camera_baseline >= 0.1:
            status = "GOOD"
            loc_allowed = True
        elif quality_score >= 40.0 and camera_baseline >= 0.02:
            status = "ACCEPTABLE"
            loc_allowed = True
        elif quality_score >= 20.0:
            status = "POOR"
            loc_allowed = False
        else:
            status = "INVALID"
            loc_allowed = False

        result = {
            "geometry_status": status,
            "geometry_quality_score": quality_score,
            "camera_baseline_m": round(camera_baseline, 3),
            "mean_depth_valid_ratio": round(mean_depth_valid_ratio, 3),
            "point_count": point_count,
            "extent_xyz_m": (round(pcd_stats["extent_x"], 2), round(pcd_stats["extent_y"], 2), round(pcd_stats["extent_z"], 2)),
            "localization_allowed": loc_allowed,
            "reason": f"Quality score {quality_score}/100 based on baseline {camera_baseline:.3f}m, point count {point_count:,}, depth valid {mean_depth_valid_ratio*100:.1f}%"
        }
        
        logger.info(f"[GEOMETRY EVAL] Status: {status} (Score: {quality_score}/100, Baseline: {camera_baseline:.3f}m)")
        return result

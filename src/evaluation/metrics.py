"""
Quantitative Reconstruction Evaluation & Accuracy Benchmarking Module.
Computes Chamfer distance, Point-to-Point RMSE/MAE, Absolute Trajectory Error (ATE),
Relative Pose Error (RPE), GPS alignment residuals, point density, and computational profiling.
"""

import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)


@dataclass
class PointDistanceMetrics:
    chamfer_distance: float
    rmse: float
    mae: float
    max_error: float
    completeness_ratio: float  # Percentage of ground truth points with a nearest neighbor < threshold
    estimated_density_pts_per_m3: float


@dataclass
class TrajectoryErrorMetrics:
    ate_rmse: float            # Absolute Trajectory Error RMSE in meters
    ate_mae: float             # ATE Mean Absolute Error
    rpe_trans_rmse: float      # Relative Pose Translation Error RMSE
    scale_error: float         # |estimated_scale - 1.0|
    gps_alignment_rmse: float  # GPS residual RMSE in meters


@dataclass
class ReconstructionMetrics:
    mission_id: str
    reconstruction_mode: str
    num_input_frames: int
    num_keyframes: int
    num_points: int
    num_triangles: int
    num_vertices: int
    point_metrics: Optional[PointDistanceMetrics]
    trajectory_metrics: Optional[TrajectoryErrorMetrics]
    total_processing_time_s: float
    peak_vram_mb: float
    stage_timings_s: Dict[str, float]


class ReconstructionEvaluator:
    """
    Computes rigorous spatial geometry and photogrammetry accuracy metrics.
    Ensures that reconstruction claims are supported by verified numerical benchmarks.
    """

    @staticmethod
    def evaluate_point_cloud_accuracy(
        predicted_points: np.ndarray,      # [N, 3]
        ground_truth_points: np.ndarray,   # [M, 3]
        max_dist_threshold_m: float = 0.5,
    ) -> PointDistanceMetrics:
        """
        Computes bidirectional Chamfer distance, RMSE, MAE, and completeness against ground truth.
        """
        if len(predicted_points) == 0 or len(ground_truth_points) == 0:
            return PointDistanceMetrics(
                chamfer_distance=999.0,
                rmse=999.0,
                mae=999.0,
                max_error=999.0,
                completeness_ratio=0.0,
                estimated_density_pts_per_m3=0.0,
            )

        # Downsample for computational speed if point clouds are massive (> 50,000 points)
        pred_sample = predicted_points
        if len(pred_sample) > 50000:
            idx = np.random.choice(len(pred_sample), 50000, replace=False)
            pred_sample = pred_sample[idx]

        gt_sample = ground_truth_points
        if len(gt_sample) > 50000:
            idx = np.random.choice(len(gt_sample), 50000, replace=False)
            gt_sample = gt_sample[idx]

        # KDTree search: Pred -> GT
        tree_gt = KDTree(gt_sample)
        dist_pred_to_gt, _ = tree_gt.query(pred_sample, k=1)

        # KDTree search: GT -> Pred
        tree_pred = KDTree(pred_sample)
        dist_gt_to_pred, _ = tree_pred.query(gt_sample, k=1)

        # Chamfer Distance (average bidirectional nearest neighbor distance)
        chamfer = float((np.mean(dist_pred_to_gt) + np.mean(dist_gt_to_pred)) / 2.0)
        rmse = float(np.sqrt(np.mean(dist_pred_to_gt ** 2)))
        mae = float(np.mean(dist_pred_to_gt))
        max_err = float(np.max(dist_pred_to_gt))

        # Completeness: fraction of GT points within max_dist_threshold_m of any predicted point
        completeness = float(np.count_nonzero(dist_gt_to_pred < max_dist_threshold_m) / len(dist_gt_to_pred))

        # Estimate volumetric bounding box density
        bbox_min = np.min(predicted_points, axis=0)
        bbox_max = np.max(predicted_points, axis=0)
        vol = float(np.prod(np.maximum(bbox_max - bbox_min, 0.1)))
        density = float(len(predicted_points) / max(vol, 1.0))

        logger.info(
            f"[EVALUATION] Point Cloud Benchmark | Chamfer: {chamfer:.3f}m | "
            f"RMSE: {rmse:.3f}m | MAE: {mae:.3f}m | Completeness: {completeness:.1%}"
        )

        return PointDistanceMetrics(
            chamfer_distance=round(chamfer, 4),
            rmse=round(rmse, 4),
            mae=round(mae, 4),
            max_error=round(max_err, 4),
            completeness_ratio=round(completeness, 4),
            estimated_density_pts_per_m3=round(density, 2),
        )

    @staticmethod
    def evaluate_trajectory_error(
        estimated_camera_centers: np.ndarray, # [S, 3] in world space
        ground_truth_positions: np.ndarray,   # [S, 3] (GPS ENU or COLMAP)
        estimated_scale: float = 1.0,
    ) -> TrajectoryErrorMetrics:
        """
        Computes Absolute Trajectory Error (ATE) and Relative Pose Error (RPE).
        """
        assert estimated_camera_centers.shape == ground_truth_positions.shape
        diffs = estimated_camera_centers - ground_truth_positions
        dists = np.linalg.norm(diffs, axis=1)

        ate_rmse = float(np.sqrt(np.mean(dists ** 2)))
        ate_mae = float(np.mean(dists))

        # Relative Pose Error (frame-to-frame translation step error)
        if len(estimated_camera_centers) > 1:
            est_deltas = np.diff(estimated_camera_centers, axis=0)
            gt_deltas = np.diff(ground_truth_positions, axis=0)
            rpe_errs = np.linalg.norm(est_deltas - gt_deltas, axis=1)
            rpe_rmse = float(np.sqrt(np.mean(rpe_errs ** 2)))
        else:
            rpe_rmse = 0.0

        scale_err = float(abs(estimated_scale - 1.0))

        logger.info(
            f"[EVALUATION] Trajectory Benchmark | ATE RMSE: {ate_rmse:.3f}m | "
            f"ATE MAE: {ate_mae:.3f}m | RPE RMSE: {rpe_rmse:.3f}m"
        )

        return TrajectoryErrorMetrics(
            ate_rmse=round(ate_rmse, 3),
            ate_mae=round(ate_mae, 3),
            rpe_trans_rmse=round(rpe_rmse, 3),
            scale_error=round(scale_err, 4),
            gps_alignment_rmse=round(ate_rmse, 3),
        )

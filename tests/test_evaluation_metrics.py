import numpy as np
import pytest
from src.evaluation.metrics import (
    ReconstructionEvaluator,
    PointDistanceMetrics,
    TrajectoryErrorMetrics,
)


def test_point_cloud_accuracy():
    # Ground truth box points
    gt_pts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])

    # Predicted points shifted by 0.1m in X
    pred_pts = gt_pts + np.array([0.1, 0.0, 0.0])

    evaluator = ReconstructionEvaluator()
    metrics: PointDistanceMetrics = evaluator.evaluate_point_cloud_accuracy(pred_pts, gt_pts)

    assert np.isclose(metrics.rmse, 0.1, atol=1e-3)
    assert np.isclose(metrics.mae, 0.1, atol=1e-3)
    assert metrics.completeness_ratio == 1.0


def test_trajectory_error():
    gt_traj = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0],
    ])
    est_traj = gt_traj + np.array([0.05, 0.0, 0.0])

    evaluator = ReconstructionEvaluator()
    metrics: TrajectoryErrorMetrics = evaluator.evaluate_trajectory_error(est_traj, gt_traj, estimated_scale=1.05)

    assert np.isclose(metrics.ate_rmse, 0.05, atol=1e-3)
    assert np.isclose(metrics.scale_error, 0.05, atol=1e-3)

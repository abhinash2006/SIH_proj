import pytest
import numpy as np

def test_depth_alignment_ransac():
    from src.depth_alignment import RobustDepthAligner
    from src.depth_consistency import DepthConsistencyAnalyzer
    from src.depth_confidence import DepthConfidenceFused

    # Create synthetic relative depth and metric scale
    true_scale, true_shift = 3.5, 0.8
    raw_da2 = np.random.uniform(0.5, 5.0, (100, 100)).astype(np.float32)
    vggt_depth = (true_scale * raw_da2 + true_shift).astype(np.float32)

    aligner = RobustDepthAligner(method="robust_ransac")
    aligned_da2, metrics = aligner.align_depth_maps(raw_da2, vggt_depth)

    assert aligned_da2.shape == (100, 100)
    assert abs(metrics["scale"] - true_scale) < 0.2
    assert abs(metrics["shift"] - true_shift) < 0.2
    assert metrics["residual_rmse"] < 0.1

    analyzer = DepthConsistencyAnalyzer(sigma=0.5)
    abs_diff, consistency, stats = analyzer.compute_consistency(vggt_depth, aligned_da2)
    assert consistency.shape == (100, 100)
    assert stats["high_consistency_percentage"] > 80.0

    fuser = DepthConfidenceFused()
    vggt_conf = np.ones((100, 100), dtype=np.float32)
    fused_conf = fuser.fuse_confidence(vggt_conf, consistency)
    assert fused_conf.shape == (100, 100)

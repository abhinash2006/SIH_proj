import numpy as np
import pytest
from src.geometry.depth_fusion import DepthFusionEngine, DepthComparisonReport


def test_depth_fusion_vggt_mode():
    engine = DepthFusionEngine(mode="vggt")
    v_depth = np.ones((50, 50), dtype=np.float32) * 10.0
    v_conf = np.ones((50, 50), dtype=np.float32) * 0.9

    report: DepthComparisonReport = engine.process(v_depth, v_conf)
    assert report.final_depth_source == "vggt"
    assert report.scale_factor == 1.0
    assert report.alignment_rmse == 0.0
    np.testing.assert_allclose(report.final_depth, 10.0)


def test_depth_fusion_comparison_alignment():
    engine = DepthFusionEngine(mode="comparison")
    # Ground truth metric depth varying from 5m to 15m
    v_depth = np.tile(np.linspace(5.0, 15.0, 60, dtype=np.float32)[:, None], (1, 60))
    v_conf = np.full((60, 60), 0.9, dtype=np.float32)

    # Relative monocular depth with scale 0.5 (so estimated scale factor should be 2.0)
    m_depth = v_depth * 0.5
    m_conf = np.full((60, 60), 0.8, dtype=np.float32)

    report: DepthComparisonReport = engine.process(v_depth, v_conf, m_depth, m_conf)
    assert report.mode == "comparison"
    assert report.final_depth_source == "vggt"
    assert np.isclose(report.scale_factor, 2.0, atol=0.1)
    assert np.isclose(report.alignment_rmse, 0.0, atol=0.1)

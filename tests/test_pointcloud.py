import os
import numpy as np
import pytest
from src.reconstruction.pointcloud import PointCloudBuilder, PointCloudData
from src.reconstruction.filtering import PointCloudFilter, FilterConfig


def test_pointcloud_builder_and_filter(tmp_path):
    builder = PointCloudBuilder(min_confidence=0.2)

    # 2 synthetic frames with 20x20 points
    S, H, W = 2, 20, 20
    point_maps = np.random.uniform(-5.0, 5.0, size=(S, H, W, 3)).astype(np.float32)
    images = np.full((S, H, W, 3), 180, dtype=np.uint8)
    conf = np.ones((S, H, W), dtype=np.float32)

    pcd_data: PointCloudData = builder.build_from_vggt_output(point_maps, images, conf)
    assert pcd_data.num_points == S * H * W
    assert pcd_data.points.shape == (S * H * W, 3)

    # Test filtering
    pcd_filter = PointCloudFilter(FilterConfig(voxel_size=0.1, use_sor=False, use_ror=False))
    filtered_data = pcd_filter.filter(pcd_data)

    assert filtered_data.num_points > 0
    assert filtered_data.num_points <= pcd_data.num_points

    # Test PLY saving
    ply_out = tmp_path / "test.ply"
    builder.save_ply(filtered_data, str(ply_out))
    assert ply_out.exists()

import os
import pytest
import numpy as np
from pathlib import Path
from src.reconstruction import PointCloudReconstructor
from src.pointcloud_filter import PointCloudFilter
from src.colmap_utils import COLMAPExporter

def test_pointcloud_reconstruction_and_ply(tmp_path):
    # Create synthetic predictions
    S, H, W = 2, 32, 32
    preds = {
        "unprojected_points": np.random.randn(S, H, W, 3).astype(np.float32),
        "point_maps": np.random.randn(S, H, W, 3).astype(np.float32),
        "depth_conf": np.ones((S, H, W), dtype=np.float32) * 1.5,
        "point_conf": np.ones((S, H, W), dtype=np.float32) * 1.5,
        "images": np.random.randint(0, 255, (S, H, W, 3), dtype=np.uint8)
    }
    
    pts, cols, confs = PointCloudReconstructor.build_pointcloud_from_predictions(
        preds, use_depth_unprojection=True, confidence_threshold=1.0
    )
    assert len(pts) > 0
    assert len(pts) == len(cols) == len(confs)
    
    # Save PLY
    ply_out = tmp_path / "test.ply"
    saved = PointCloudReconstructor.save_ply(pts, cols, ply_out)
    assert os.path.exists(saved)

def test_pointcloud_filter():
    # Regular points + isolated outliers
    pts = np.random.randn(1000, 3).astype(np.float32) * 0.5
    outliers = np.random.uniform(50.0, 100.0, (50, 3)).astype(np.float32)
    all_pts = np.vstack([pts, outliers])
    all_cols = np.ones((len(all_pts), 3), dtype=np.float32)
    
    filter_obj = PointCloudFilter({"pointcloud_filtering": {"enabled": True, "voxel_downsample": False, "statistical_filter": True}})
    filt_pts, filt_cols, stats = filter_obj.filter_pointcloud(all_pts, all_cols)
    
    assert len(filt_pts) < len(all_pts)
    assert stats["removed_points"] > 0

def test_colmap_export(tmp_path):
    sparse_out = tmp_path / "colmap_test"
    S = 3
    frame_paths = [f"frame_{i}.jpg" for i in range(S)]
    extrinsics = np.tile(np.eye(4)[:3][None, ...], (S, 1, 1))
    intrinsics = np.tile(np.eye(3)[None, ...], (S, 1, 1))
    pts3d = np.random.randn(100, 3).astype(np.float32)
    cols3d = np.ones((100, 3), dtype=np.float32)
    
    out_dir = COLMAPExporter.export_colmap(
        output_dir=sparse_out,
        frame_paths=frame_paths,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        points3d=pts3d,
        colors3d=cols3d,
        export_binary=True,
        export_text=True
    )
    
    assert os.path.exists(os.path.join(out_dir, "cameras.txt"))
    assert os.path.exists(os.path.join(out_dir, "images.txt"))
    assert os.path.exists(os.path.join(out_dir, "points3D.txt"))
    assert os.path.exists(os.path.join(out_dir, "cameras.bin"))
    assert os.path.exists(os.path.join(out_dir, "images.bin"))
    assert os.path.exists(os.path.join(out_dir, "points3D.bin"))

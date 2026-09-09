import pytest
import numpy as np
from src.disaster_inspection.geo_projection import GeoProjection3D

def test_yolo_2d_to_3d_localization_valid():
    # Synthetic detection
    detection = {
        "class_name": "person",
        "confidence": 0.85,
        "raw_detector_confidence": 0.85,
        "validated_detection_confidence": 0.85,
        "bbox": [100, 100, 200, 300],
        "pixel_center": (150, 200),
        "frame_idx": 0
    }
    
    # Synthetic depth map
    depth_map = np.ones((518, 518), dtype=np.float32) * 5.0 # 5 meters depth
    
    # Standard identity camera geometry
    extrinsic = np.eye(4)[:3, :]
    intrinsic = np.array([
        [500.0, 0.0, 259.0],
        [0.0, 500.0, 259.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    proj3d = GeoProjection3D.project_detection_to_3d(
        detection=detection,
        depth_map=depth_map,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        image_shape=(518, 518),
        max_reprojection_error_px=30.0
    )
    
    assert proj3d["location_status"] == "LOCALIZED_3D"
    assert proj3d["x_m"] is not None
    assert proj3d["y_m"] is not None
    assert proj3d["z_m"] is not None
    assert proj3d["z_m"] == 5.0
    assert proj3d["reprojection_error_px"] <= 30.0
    assert "object_confidence" in proj3d

def test_yolo_reprojection_failure_returns_unlocalized():
    detection = {
        "class_name": "person",
        "confidence": 0.80,
        "raw_detector_confidence": 0.80,
        "validated_detection_confidence": 0.80,
        "bbox": [100, 100, 200, 300],
        "pixel_center": (150, 200),
        "frame_idx": 0
    }
    
    # Invalid zero depth map
    depth_map = np.zeros((518, 518), dtype=np.float32)
    extrinsic = np.eye(4)[:3, :]
    intrinsic = np.eye(3)
    
    proj3d = GeoProjection3D.project_detection_to_3d(
        detection=detection,
        depth_map=depth_map,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        image_shape=(518, 518)
    )
    
    assert proj3d["location_status"] == "UNLOCALIZED"
    assert proj3d["x_m"] is None
    assert proj3d["y_m"] is None
    assert proj3d["z_m"] is None
    # Crucial requirement: No fake (0,0,0) coordinates!
    assert (proj3d["x_m"], proj3d["y_m"], proj3d["z_m"]) != (0.0, 0.0, 0.0)

def test_multiframe_3d_fusion():
    obs = [
        {"track_id": 1, "class_name": "person", "location_status": "LOCALIZED_3D", "x_m": 1.0, "y_m": 2.0, "z_m": 3.0, "reprojection_error_px": 2.0, "confidence": 0.8, "frame_idx": 1},
        {"track_id": 1, "class_name": "person", "location_status": "LOCALIZED_3D", "x_m": 1.1, "y_m": 2.1, "z_m": 3.0, "reprojection_error_px": 2.2, "confidence": 0.82, "frame_idx": 2},
        {"track_id": 1, "class_name": "person", "location_status": "LOCALIZED_3D", "x_m": 0.9, "y_m": 1.9, "z_m": 3.0, "reprojection_error_px": 1.8, "confidence": 0.85, "frame_idx": 3}
    ]
    
    fused = GeoProjection3D.fuse_multiframe_object_3d(obs)
    assert fused["location_status"] == "LOCALIZED_3D"
    assert fused["observation_count"] == 3
    assert fused["source_frames"] == [1, 2, 3]
    assert abs(fused["x_m"] - 1.0) < 0.2

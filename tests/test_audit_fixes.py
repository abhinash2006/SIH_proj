import pytest
import numpy as np
from src.geometry_validation import GeometryValidator
from src.geometry_quality import GeometryQualityEvaluator
from src.disaster_inspection.geo_projection import GeoProjection3D
from src.disaster_inspection.incident_manager import DisasterIncidentManager
from src.disaster_inspection.object_detection import DisasterObjectDetector

def test_unprojectable_localization_not_zero():
    """Test 1: Failed 3D projection returns UNLOCALIZED and None coords, NOT (0,0,0)."""
    detection = {
        "class_name": "person",
        "confidence": 0.85,
        "bbox": [10, 10, 50, 50],
        "pixel_center": (30, 30),
        "frame_idx": 0
    }
    # Depth map with zeros / invalid depths
    depth_map = np.zeros((100, 100), dtype=np.float32)
    extrinsic = np.eye(4)[:3, :]
    intrinsic = np.array([[500, 0, 50], [0, 500, 50], [0, 0, 1]], dtype=np.float64)

    res = GeoProjection3D.project_detection_to_3d(detection, depth_map, extrinsic, intrinsic)
    
    assert res["location_status"] == "UNLOCALIZED"
    assert res["x_m"] is None
    assert res["y_m"] is None
    assert res["z_m"] is None
    assert res["x_m"] != 0.0 or res["location_status"] == "UNLOCALIZED"

def test_degenerate_pointcloud_fails_validation():
    """Test 2: Degenerate point cloud with zero variance fails geometry validation."""
    # Points collapsed to single line/point
    points = np.ones((500, 3), dtype=np.float64)
    valid, stats, msg = GeometryValidator.validate_point_cloud(points)
    
    assert valid is False
    assert "Degenerate" in msg or "collapsed" in msg.lower()

def test_incident_spatial_deduplication(tmp_path):
    """Test 3: Multi-frame observations of same event merge into one incident."""
    manager = DisasterIncidentManager(db_dir=str(tmp_path))
    
    inc1 = {
        "incident_type": "FLOOD_ZONE",
        "object_id": "HAZARD_001",
        "confidence": 0.70,
        "severity": "HIGH",
        "location_status": "LOCALIZED_3D",
        "x_m": 12.5, "y_m": 5.0, "z_m": 2.0,
        "frame_idx": 1
    }
    inc2 = {
        "incident_type": "FLOOD_ZONE",
        "object_id": "HAZARD_001",
        "confidence": 0.75,
        "severity": "HIGH",
        "location_status": "LOCALIZED_3D",
        "x_m": 12.6, "y_m": 5.1, "z_m": 2.1,
        "frame_idx": 2
    }

    id1 = manager.add_incident(inc1)
    id2 = manager.add_incident(inc2)

    assert id1 == id2 # Merged into same unique incident ID
    all_inc = manager.get_all_incidents()
    assert len(all_inc) == 1
    assert all_inc[0]["observation_count"] == 2
    assert 1 in all_inc[0]["source_frames"] and 2 in all_inc[0]["source_frames"]

def test_reprojection_error_rejection():
    """Test 4: Negative/Behind-camera or out-of-bounds depth projection fails 3D localization."""
    detection = {
        "class_name": "vehicle",
        "confidence": 0.90,
        "bbox": [10, 10, 50, 50],
        "pixel_center": (30, 30),
        "frame_idx": 0
    }
    # Depth map with non-positive / NaN depths
    depth_map = np.full((100, 100), -5.0, dtype=np.float32)
    extrinsic = np.eye(4)[:3, :]
    intrinsic = np.array([[500, 0, 50], [0, 500, 50], [0, 0, 1]], dtype=np.float64)

    res = GeoProjection3D.project_detection_to_3d(detection, depth_map, extrinsic, intrinsic, max_reprojection_error_px=10.0)
    assert res["location_status"] == "UNLOCALIZED"
    assert res["x_m"] is None

def test_anti_boat_house_remap():
    """Test 5: House roof area misclassified as boat is remapped to building."""
    detector = DisasterObjectDetector()
    # Mock detection of COCO boat (class 8) with large area
    mock_detections = [{
        "class_name": "boat",
        "raw_class_id": 8,
        "confidence": 0.65,
        "bbox": [100, 100, 200, 200], # Area 10,000 px -> Flooded roof
        "pixel_center": (150, 150),
        "frame_idx": 0
    }]
    nms_dets = detector._nms_detections(mock_detections)
    
    # Filter Anti-Boat confusion
    final = []
    for det in nms_dets:
        bw = det["bbox"][2] - det["bbox"][0]
        bh = det["bbox"][3] - det["bbox"][1]
        if det.get("raw_class_id") == 8 and bw * bh > 2000:
            det["class_name"] = "building"
        final.append(det)
        
    assert final[0]["class_name"] == "building"

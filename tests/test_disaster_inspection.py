import pytest
import numpy as np

def test_disaster_inspection_pipeline():
    from src.disaster_inspection.object_detection import DisasterObjectDetector
    from src.disaster_inspection.tracking import MultiFrameObjectTracker
    from src.disaster_inspection.geo_projection import GeoProjection3D
    from src.disaster_inspection.damage_detection import StructuralDamageDetector
    from src.disaster_inspection.hazard_detection import HazardDetector
    from src.disaster_inspection.incident_manager import DisasterIncidentManager
    from src.disaster_inspection.risk_scoring import DisasterRiskScorer

    detector = DisasterObjectDetector()
    dummy_img = (np.random.rand(200, 300, 3) * 255).astype(np.uint8)
    dets = detector.detect_objects_in_frame(dummy_img, frame_idx=0)
    
    tracker = MultiFrameObjectTracker()
    tracked_dets = tracker.track_frame_detections(dets, frame_idx=0)

    # 3D projection test
    depth_map = np.ones((200, 300), dtype=np.float32) * 5.0
    intrinsic = np.array([[300., 0., 150.], [0., 300., 100.], [0., 0., 1.]])
    extrinsic = np.eye(4)[:3]

    sample_det = {
        "class_name": "person",
        "confidence": 0.90,
        "pixel_center": (150, 100),
        "bbox": [140, 90, 160, 110]
    }
    proj3d = GeoProjection3D.project_detection_to_3d(sample_det, depth_map, extrinsic, intrinsic)
    assert "x_m" in proj3d
    assert "y_m" in proj3d
    assert "z_m" in proj3d

    # Hazard & damage test
    damage_det = StructuralDamageDetector()
    dmg_incidents = damage_det.analyze_structural_damage(dummy_img, depth_map, [])

    hazard_det = HazardDetector()
    hazards = hazard_det.analyze_hazards(dummy_img, depth_map, [])

    risk_level, score, reasons, priorities = DisasterRiskScorer.calculate_scene_risk(
        [proj3d], hazards, dmg_incidents
    )
    assert risk_level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    mgr = DisasterIncidentManager(db_dir="outputs/test_incidents")
    inc_id = mgr.add_incident({
        "incident_type": "PERSON_DETECTED",
        "confidence": 0.90,
        "x_m": proj3d["x_m"],
        "y_m": proj3d["y_m"],
        "z_m": proj3d["z_m"]
    })
    assert inc_id.startswith("INCIDENT_")

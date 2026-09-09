import pytest
from src.disaster_inspection.incident_manager import DisasterIncidentManager

def test_person_detection_not_automatically_victim():
    inc_mgr = DisasterIncidentManager(db_dir="outputs/test_db_yolo")
    
    # Person detection input
    person_inc = {
        "incident_type": "PERSON_DETECTED",
        "target_object_class": "person",
        "object_id": "PERSON_001",
        "confidence": 0.84,
        "detection_confidence": 0.84,
        "localization_confidence": 0.90,
        "severity": "MEDIUM",
        "location_status": "LOCALIZED_3D",
        "x_m": 0.42, "y_m": 0.31, "z_m": 2.18,
        "frame_idx": 9,
        "evidence": "Person detected in drone image.",
        "status": "NEEDS_VERIFICATION"
    }
    
    inc_id = inc_mgr.add_incident(person_inc)
    all_inc = inc_mgr.get_all_incidents()
    
    assert len(all_inc) == 1
    inc = all_inc[0]
    
    # Verify exact label enforcement: PERSON_DETECTED != VICTIM
    assert inc["incident_type"] == "PERSON_DETECTED"
    assert inc["incident_type"] != "VICTIM"
    assert inc["incident_type"] != "VICTIM_PERSON_DETECTED"
    assert inc["status"] == "NEEDS_VERIFICATION"

def test_vehicle_detection_not_automatically_blocked_road():
    inc_mgr = DisasterIncidentManager(db_dir="outputs/test_db_yolo")
    
    vehicle_det = {
        "incident_type": "VEHICLE_DETECTED",
        "target_object_class": "vehicle",
        "object_id": "VEHICLE_001",
        "confidence": 0.78,
        "location_status": "LOCALIZED_3D",
        "x_m": 5.2, "y_m": 3.1, "z_m": 1.5,
        "frame_idx": 4,
        "status": "NEEDS_VERIFICATION"
    }
    
    inc_id = inc_mgr.add_incident(vehicle_det)
    all_inc = inc_mgr.get_all_incidents()
    
    inc = all_inc[0]
    assert inc["incident_type"] == "VEHICLE_DETECTED"
    assert inc["incident_type"] != "BLOCKED_ROAD_DEBRIS"
    assert inc["incident_type"] != "BLOCKED_ROAD"

def test_house_roof_not_automatically_boat():
    from src.yolo_detector import YOLODetector
    detector = YOLODetector()
    
    # Synthetic boat detection on a large house roof (area = 3000px, aspect_ratio = 1.2)
    synthetic_raw = [{
        "frame_id": "frame_0001.jpg",
        "frame_idx": 1,
        "source_frame": "frame_0001.jpg",
        "class_id": 8,
        "class_name": "boat",
        "raw_class_name": "boat",
        "bbox_xyxy": [100, 100, 200, 180],
        "bbox_width": 100,
        "bbox_height": 80,
        "image_width": 1000,
        "image_height": 1000,
        "raw_detector_confidence": 0.75,
        "validated_detection_confidence": 0.75,
        "location_status": "UNLOCALIZED"
    }]
    
    filtered = detector._apply_anti_boat_filter(synthetic_raw)
    
    # Boat on large structure roof must be filtered/flagged out
    assert len(filtered) == 0 or filtered[0]["validated_detection_confidence"] == 0.0
    detector.release()

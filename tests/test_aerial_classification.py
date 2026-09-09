"""
Regression Test Suite for Aerial Class Validation & False Classification Auditing.

Covers all 6 mandatory user audit regression tests (A-F) plus:
- Temporal consistency multi-frame voting
- Raw YOLO output immutability preservation
- Class-specific confidence thresholds
- UNCERTAIN_OBJECT incident gating
"""

import numpy as np
import pytest
from src.aerial_classifier import (
    AerialContextValidator,
    TrackClassHistory,
    ClassThresholdConfig
)
from src.incident_engine import IncidentEngine


@pytest.fixture
def validator():
    cfg = ClassThresholdConfig(
        person_thresh=0.08,
        vehicle_thresh=0.15,
        building_thresh=0.15,
        boat_thresh=0.15,
        vehicle_min_area=250.0,
        vehicle_max_area=12000.0,
        building_min_area=2500.0,
        boat_min_water_overlap=0.15
    )
    return AerialContextValidator(cfg)


# -------------------------------------------------------------
# Test A: House roof must not become vehicle
# -------------------------------------------------------------
def test_regression_a_house_roof_not_vehicle(validator):
    """Large rectangular house roof predicted as truck/vehicle must be remapped to BUILDING or UNCERTAIN_OBJECT."""
    # Roof of 14,000 px area (140x100) predicted as 'truck' with conf 0.22
    det = {
        "raw_class_name": "truck",
        "raw_detector_confidence": 0.22,
        "bbox_xyxy": [100, 100, 240, 200],  # 140x100 -> area=14000
        "frame_id": "frame_0001.jpg",
        "track_id": 1
    }
    res = validator.validate_detection(det)
    assert res["final_class"] in ["BUILDING", "UNCERTAIN_OBJECT"]
    assert res["final_class"] != "VEHICLE"
    assert "oversized" in res["validation_evidence"].lower() or "building-like" in res["validation_evidence"].lower()


# -------------------------------------------------------------
# Test B: Real vehicle must not become building
# -------------------------------------------------------------
def test_regression_b_real_vehicle_not_building(validator):
    """Compact vehicle on ground with standard vehicle dimensions must remain VEHICLE."""
    # Car of 1,200 px area (60x20) on dry ground
    det = {
        "raw_class_name": "car",
        "raw_detector_confidence": 0.35,
        "bbox_xyxy": [500, 400, 560, 420],  # 60x20 -> area=1200, aspect=3.0
        "frame_id": "frame_0002.jpg",
        "track_id": 2
    }
    water_mask = np.zeros((1080, 1920), dtype=np.uint8)
    res = validator.validate_detection(det, water_mask=water_mask)
    assert res["final_class"] == "VEHICLE"
    assert res["validation_confidence"] >= 0.35


# -------------------------------------------------------------
# Test C: Boat must not become vehicle unless supported
# -------------------------------------------------------------
def test_regression_c_boat_not_vehicle_unless_supported(validator):
    """Vessel floating in water must remain BOAT and not become VEHICLE."""
    # Boat of 1,800 px area in water
    det = {
        "raw_class_name": "boat",
        "raw_detector_confidence": 0.40,
        "bbox_xyxy": [200, 300, 290, 320],  # 90x20 -> area=1800, aspect=4.5
        "frame_id": "frame_0003.jpg",
        "track_id": 3
    }
    water_mask = np.ones((1080, 1920), dtype=np.uint8)  # 100% water overlap
    res = validator.validate_detection(det, water_mask=water_mask)
    assert res["final_class"] == "BOAT"
    assert "water context" in res["validation_evidence"].lower()


# -------------------------------------------------------------
# Test D: Water reflection must not become vehicle
# -------------------------------------------------------------
def test_regression_d_water_reflection_not_vehicle(validator):
    """Low-confidence shimmering artifact in deep water must be flagged UNCERTAIN_OBJECT."""
    # Transient reflection detection in deep water (95% water overlap, conf 0.18)
    det = {
        "raw_class_name": "car",
        "raw_detector_confidence": 0.18,
        "bbox_xyxy": [700, 600, 750, 630],
        "frame_id": "frame_0004.jpg",
        "track_id": 4
    }
    water_mask = np.zeros((1080, 1920), dtype=np.uint8)
    water_mask[590:640, 690:760] = 1  # 100% water coverage under bbox
    res = validator.validate_detection(det, water_mask=water_mask)
    assert res["final_class"] == "UNCERTAIN_OBJECT"
    assert "water reflection" in res["validation_evidence"].lower()


# -------------------------------------------------------------
# Test E: Tree / shadow must not become person
# -------------------------------------------------------------
def test_regression_e_tree_shadow_not_person(validator):
    """Oversized or extreme aspect ratio shadow blob must not become PERSON."""
    # Elongated shadow blob (width 200, height 15 -> aspect ratio 13.3)
    det = {
        "raw_class_name": "person",
        "raw_detector_confidence": 0.14,
        "bbox_xyxy": [100, 200, 300, 215],
        "frame_id": "frame_0005.jpg",
        "track_id": 5
    }
    res = validator.validate_detection(det)
    assert res["final_class"] == "UNCERTAIN_OBJECT"
    assert "extreme aspect ratio" in res["validation_evidence"].lower()


# -------------------------------------------------------------
# Test F: Flooded house remains building even when partially surrounded by water
# -------------------------------------------------------------
def test_regression_f_flooded_house_remains_building(validator):
    """A building partially surrounded by floodwaters must remain BUILDING and not become BOAT."""
    det = {
        "raw_class_name": "building",
        "raw_detector_confidence": 0.65,
        "bbox_xyxy": [300, 300, 420, 400],  # 120x100 -> area=12000
        "frame_id": "frame_0006.jpg",
        "track_id": 6
    }
    # 40% water overlap surrounding the foundation
    water_mask = np.zeros((1080, 1920), dtype=np.uint8)
    water_mask[300:340, 300:420] = 1
    res = validator.validate_detection(det, water_mask=water_mask)
    assert res["final_class"] == "BUILDING"
    assert "persistent structural footprint" in res["validation_evidence"].lower()


# -------------------------------------------------------------
# Test G: Temporal consistency prevents single-frame class flipping
# -------------------------------------------------------------
def test_regression_g_temporal_consistency_voting(validator):
    """A persistent vehicle with 4 vehicle predictions and 1 transient building prediction must remain VEHICLE."""
    history = TrackClassHistory(track_id=24)
    history.add_observation(0, "f0", "car", 0.21, [10, 10, 50, 30])
    history.add_observation(1, "f1", "car", 0.32, [12, 10, 52, 30])
    history.add_observation(2, "f2", "building", 0.18, [14, 10, 54, 30])  # Transient flip
    history.add_observation(3, "f3", "truck", 0.27, [16, 10, 56, 30])
    history.add_observation(4, "f4", "car", 0.30, [18, 10, 58, 30])

    metrics = history.compute_temporal_metrics()
    assert metrics["dominant_class"] in ["vehicle", "car"]
    assert metrics["class_vote_ratio"] >= 0.60
    assert metrics["class_switch_count"] == 2

    # Frame 2 transient building detection evaluated with history
    frame_2_det = {
        "raw_class_name": "building",
        "raw_detector_confidence": 0.18,
        "bbox_xyxy": [14, 10, 66, 60],  # 52x50 -> area=2600px (matches building footprint)
        "frame_id": "f2",
        "track_id": 24
    }
    res = validator.validate_detection(frame_2_det, track_history=history)
    # The dominant class is CAR / VEHICLE
    assert res["final_class"] == "VEHICLE"
    assert "temporal track consistency" in res["validation_evidence"].lower()


# -------------------------------------------------------------
# Test H: Preserve raw YOLO output immutability
# -------------------------------------------------------------
def test_regression_h_raw_yolo_immutability(validator):
    """Raw YOLO output must remain unmodified and fully auditable."""
    det = {
        "raw_class_name": "car",
        "raw_detector_confidence": 0.25,
        "bbox_xyxy": [100, 100, 150, 130],
        "frame_id": "frame_audit.jpg",
        "track_id": 99,
        "source_tile": [0, 0, 640, 640]
    }
    res = validator.validate_detection(det)
    raw = res["raw_yolo_output"]
    assert raw["class"] == "car"
    assert raw["confidence"] == 0.25
    assert raw["bbox"] == [100, 100, 150, 130]
    assert raw["source_tile"] == [0, 0, 640, 640]
    assert raw["original_image_coordinates"] == [100, 100, 150, 130]


# -------------------------------------------------------------
# Test I: Class-specific thresholds preserve person recall
# -------------------------------------------------------------
def test_regression_i_class_thresholds_person_recall(validator):
    """Person threshold (0.08) preserves tiny humans, while vehicle threshold (0.15) rejects tiny noise."""
    # Small aerial human at 0.10 confidence passes
    person_det = {
        "raw_class_name": "person",
        "raw_detector_confidence": 0.10,
        "bbox_xyxy": [10, 10, 25, 40],  # 15x30 px
        "frame_id": "frame_p.jpg",
        "track_id": 101
    }
    res_p = validator.validate_detection(person_det)
    assert res_p["final_class"] == "PERSON"

    # Weak vehicle candidate at 0.10 confidence (< 0.15 threshold) is filtered
    veh_det = {
        "raw_class_name": "car",
        "raw_detector_confidence": 0.10,
        "bbox_xyxy": [10, 10, 70, 40],
        "frame_id": "frame_v.jpg",
        "track_id": 102
    }
    res_v = validator.validate_detection(veh_det)
    assert res_v["final_class"] == "UNCERTAIN_OBJECT"


# -------------------------------------------------------------
# Test J: UNCERTAIN_OBJECT gated from canonical incidents
# -------------------------------------------------------------
def test_regression_j_uncertain_object_gated_from_incidents():
    """Objects marked UNCERTAIN_OBJECT must never produce canonical vehicle incidents."""
    unc_data = {
        "final_class": "UNCERTAIN_OBJECT",
        "class_name": "UNCERTAIN_OBJECT",
        "validation_confidence": 0.20,
        "point_3d": [1.0, 2.0, 3.0]
    }
    inc = IncidentEngine.create_vehicle_incident(unc_data, source_frame_id="f0.jpg", flood_status="VALIDATED_INUNDATION")
    assert inc is None

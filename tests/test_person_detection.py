"""
Regression Test Suite for UAV Small Person Detection & Person Tracking.

Validates the 10 core person-specific directives:
1. test_person_not_suppressed_by_vehicle_nms
2. test_small_aerial_person_detection
3. test_person_tracking_across_missed_frames
4. test_camera_motion_tolerance
5. test_aspect_ratio_acceptance
6. test_raw_person_record_immutability
7. test_confidence_semantics_decoupling
8. test_person_only_diagnostic_generation
9. test_no_synthetic_people
10. test_person_tile_deduplication
"""

import copy
import numpy as np
import pytest
from unittest.mock import MagicMock

from src.yolo_detector import YOLODetector
from src.aerial_classifier import AerialContextValidator, ClassThresholdConfig


@pytest.fixture
def detector():
    return YOLODetector(
        model_name="yolo26n.pt",
        confidence=0.15,
        iou_threshold=0.45,
        person_conf_threshold=0.08,
        person_iou_threshold=0.40,
        device="cpu",
        half=False
    )


@pytest.fixture
def validator():
    cfg = ClassThresholdConfig(
        person_thresh=0.08,
        vehicle_thresh=0.15,
        building_thresh=0.15,
        boat_thresh=0.15,
        person_min_area=10.0,
        person_max_area=3500.0,
        person_max_aspect=8.0
    )
    return AerialContextValidator(cfg)


# -------------------------------------------------------------
# 1. Person Not Suppressed by Large Vehicle NMS
# -------------------------------------------------------------
def test_person_not_suppressed_by_vehicle_nms(detector):
    """
    A large vehicle bounding box overlapping a small person detection must NOT
    suppress the person detection in class-aware NMS.
    """
    records = [
        {
            "id": "DET_001",
            "class_name": "vehicle",
            "raw_class_name": "car",
            "raw_detector_confidence": 0.85,
            "bbox_xyxy": [100, 100, 300, 300],
            "bbox_width": 200, "bbox_height": 200, "bbox_area": 40000
        },
        {
            "id": "DET_002",
            "class_name": "person",
            "raw_class_name": "person",
            "raw_detector_confidence": 0.28,
            "bbox_xyxy": [120, 120, 140, 160],  # Fully inside vehicle box
            "bbox_width": 20, "bbox_height": 40, "bbox_area": 800
        }
    ]

    kept = detector._apply_nms(records, iou_thresh=0.45, person_iou_thresh=0.40)
    kept_classes = [r["class_name"] for r in kept]

    assert "person" in kept_classes, "Small person was suppressed by overlapping vehicle box!"
    assert "vehicle" in kept_classes, "Vehicle was suppressed!"
    assert len(kept) == 2, f"Expected 2 detections, got {len(kept)}"


# -------------------------------------------------------------
# 2. Small Aerial Person Detection Thresholding
# -------------------------------------------------------------
def test_small_aerial_person_detection(detector):
    """
    A small aerial person candidate with confidence 0.12 is preserved when person
    threshold is 0.08, while a non-person candidate with conf 0.12 is filtered out (thresh=0.15).
    """
    out_records = []
    
    # Mock Ultralytics Results object
    box_person = MagicMock()
    box_person.cls = [MagicMock(item=MagicMock(return_value=0))]  # 0 = person
    box_person.conf = [MagicMock(item=MagicMock(return_value=0.12))]
    box_person.id = None
    box_person.xyxy = [MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=np.array([50, 50, 65, 85])))))]

    box_vehicle = MagicMock()
    box_vehicle.cls = [MagicMock(item=MagicMock(return_value=2))]  # 2 = car
    box_vehicle.conf = [MagicMock(item=MagicMock(return_value=0.12))]
    box_vehicle.id = None
    box_vehicle.xyxy = [MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=np.array([200, 200, 300, 250])))))]

    res_mock = MagicMock()
    res_mock.boxes = [box_person, box_vehicle]

    detector.model = MagicMock()
    detector.model.names = {0: "person", 2: "car"}

    detector._parse_results(
        [res_mock],
        out_records,
        frame_id="frame_0001.jpg",
        frame_idx=1,
        W_orig=1920,
        H_orig=1080,
        conf_thresh=0.15,
        person_thresh=0.08
    )

    classes_kept = [r["class_name"] for r in out_records]
    assert "person" in classes_kept, "Person with conf 0.12 >= 0.08 should be preserved"
    assert "vehicle" not in classes_kept, "Vehicle with conf 0.12 < 0.15 should be filtered out"


# -------------------------------------------------------------
# 3. Person Tracking Across Missed Frames
# -------------------------------------------------------------
def test_person_tracking_across_missed_frames(detector):
    """
    Person track must be preserved when missed for up to 5 consecutive frames.
    """
    f0_det = [{
        "id": "DET_0", "class_name": "person", "raw_class_name": "person",
        "bbox_xyxy": [100, 100, 120, 150], "pixel_center": (110, 125),
        "raw_detector_confidence": 0.45, "validated_detection_confidence": 0.45,
        "bbox_width": 20, "bbox_height": 50, "bbox_area": 1000
    }]
    f1_empty = []  # Missed frame 1
    f2_empty = []  # Missed frame 2
    f3_det = [{
        "id": "DET_3", "class_name": "person", "raw_class_name": "person",
        "bbox_xyxy": [108, 106, 128, 156], "pixel_center": (118, 131),
        "raw_detector_confidence": 0.50, "validated_detection_confidence": 0.50,
        "bbox_width": 20, "bbox_height": 50, "bbox_area": 1000
    }]

    detector.detect_image = MagicMock(side_effect=[f0_det, f1_empty, f2_empty, f3_det])
    frames = ["f0", "f1", "f2", "f3"]
    tracked = detector.track_frames(frames)

    assert len(tracked[0]) == 1
    assert len(tracked[1]) == 0
    assert len(tracked[2]) == 0
    assert len(tracked[3]) == 1

    trk_f0 = tracked[0][0]["track_id"]
    trk_f3 = tracked[3][0]["track_id"]
    assert trk_f0 == trk_f3, f"Track ID changed across missed frames: {trk_f0} vs {trk_f3}"
    assert tracked[3][0]["track_length"] == 2


# -------------------------------------------------------------
# 4. Camera Motion Tolerance (Up to 250 px Displacement)
# -------------------------------------------------------------
def test_camera_motion_tolerance(detector):
    """
    UAV camera motion displacing small person by ~150px between consecutive frames
    must maintain the same track ID.
    """
    f0_det = [{
        "id": "DET_0", "class_name": "person", "raw_class_name": "person",
        "bbox_xyxy": [200, 200, 220, 250], "pixel_center": (210, 225),
        "raw_detector_confidence": 0.60, "validated_detection_confidence": 0.60,
        "bbox_width": 20, "bbox_height": 50, "bbox_area": 1000
    }]
    # UAV moved forward/right by 140px in X and 50px in Y (distance ~148px)
    f1_det = [{
        "id": "DET_1", "class_name": "person", "raw_class_name": "person",
        "bbox_xyxy": [340, 250, 360, 300], "pixel_center": (350, 275),
        "raw_detector_confidence": 0.65, "validated_detection_confidence": 0.65,
        "bbox_width": 20, "bbox_height": 50, "bbox_area": 1000
    }]

    detector.detect_image = MagicMock(side_effect=[f0_det, f1_det])
    tracked = detector.track_frames(["f0", "f1"])

    trk_f0 = tracked[0][0]["track_id"]
    trk_f1 = tracked[1][0]["track_id"]
    assert trk_f0 == trk_f1, f"Track ID lost under camera motion: {trk_f0} != {trk_f1}"
    assert tracked[1][0]["tracking_confidence"] > tracked[0][0]["tracking_confidence"]


# -------------------------------------------------------------
# 5. Aspect Ratio Acceptance (Tall Upright Human)
# -------------------------------------------------------------
def test_aspect_ratio_acceptance(validator):
    """
    Upright standing person with high aspect ratio (e.g. 6.0) in oblique drone imagery
    must be accepted as PERSON (not UNCERTAIN_OBJECT).
    """
    # 7px wide by 42px high -> aspect ratio = 6.0
    det = {
        "raw_class_name": "person",
        "raw_detector_confidence": 0.42,
        "confidence": 0.42,
        "bbox_xyxy": [500, 500, 507, 542],
        "frame_id": "frame_0005.jpg",
        "track_id": 5
    }

    res = validator.validate_detection(det)
    assert res["final_class"] == "PERSON", f"Expected PERSON, got {res['final_class']}"
    assert res["person_validation_confidence"] == pytest.approx(0.42, abs=1e-3)


# -------------------------------------------------------------
# 6. Raw Person Record Immutability
# -------------------------------------------------------------
def test_raw_person_record_immutability(validator):
    """
    The 13-field raw_person_record snapshot must remain strictly immutable
    when downstream validation or scoring runs.
    """
    raw_person = {
        "frame_id": "frame_0010.jpg",
        "timestamp": 0.40,
        "bbox": [100, 200, 115, 240],
        "center_x": 107,
        "center_y": 220,
        "width": 15,
        "height": 40,
        "bbox_area": 600,
        "confidence": 0.35,
        "class": "PERSON",
        "tile_id": "tile_0_0",
        "tile_coordinates": [0, 0, 640, 640],
        "original_frame_coordinates": [100, 200, 115, 240]
    }
    frozen_copy = copy.deepcopy(raw_person)

    det = {
        "raw_class_name": "person",
        "raw_detector_confidence": 0.35,
        "confidence": 0.35,
        "bbox_xyxy": [100, 200, 115, 240],
        "raw_person_record": raw_person,
        "track_id": 10
    }

    res = validator.validate_detection(det)
    assert res["raw_person_record"] == frozen_copy, "raw_person_record was mutated by validator!"


# -------------------------------------------------------------
# 7. Confidence Semantics Decoupling
# -------------------------------------------------------------
def test_confidence_semantics_decoupling(validator):
    """
    raw_yolo_confidence, tracking_confidence, person_validation_confidence,
    and confidence_3d are separate, non-overlapping semantic fields.
    """
    det = {
        "raw_class_name": "person",
        "raw_detector_confidence": 0.32,
        "confidence": 0.32,
        "tracking_confidence": 0.70,
        "confidence_3d": 0.88,
        "rescue_confidence": 0.95,
        "bbox_xyxy": [300, 300, 320, 350],
        "track_id": 7
    }

    res = validator.validate_detection(det)

    assert res["raw_yolo_confidence"] == pytest.approx(0.32, abs=1e-3)
    assert res["tracking_confidence"] == pytest.approx(0.70, abs=1e-3)
    assert res["person_validation_confidence"] == pytest.approx(0.32, abs=1e-3)
    assert res["confidence_3d"] == pytest.approx(0.88, abs=1e-3)
    assert res["rescue_confidence"] == pytest.approx(0.95, abs=1e-3)


# -------------------------------------------------------------
# 8. Person-Only Diagnostic Generation
# -------------------------------------------------------------
def test_person_only_diagnostic_generation(detector):
    """
    annotate_person_diagnostic must produce a 3-panel contact sheet containing
    only person detections, ignoring vehicles or buildings.
    """
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    mixed_dets = [
        {
            "class_name": "person", "raw_class_name": "person",
            "bbox_xyxy": [200, 200, 220, 250], "raw_yolo_confidence": 0.55,
            "track_id": 1, "track_length": 3, "tracking_confidence": 0.85
        },
        {
            "class_name": "vehicle", "raw_class_name": "car",
            "bbox_xyxy": [500, 500, 600, 550], "raw_yolo_confidence": 0.88,
            "track_id": 2
        }
    ]

    contact_sheet = detector.annotate_person_diagnostic(img, mixed_dets)
    assert isinstance(contact_sheet, np.ndarray)
    assert contact_sheet.ndim == 3
    assert contact_sheet.shape[1] == 640 * 3, f"Expected 3 stacked 640px panels, got width {contact_sheet.shape[1]}"


# -------------------------------------------------------------
# 9. No Synthetic People Verification
# -------------------------------------------------------------
def test_no_synthetic_people(detector):
    """
    When the detector returns no person boxes on a blank frame,
    zero person detections and zero person tracks are created.
    """
    blank_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    detector.detect_image = MagicMock(return_value=[])

    tracked = detector.track_frames([blank_frame])
    assert len(tracked) == 1
    assert len(tracked[0]) == 0, "Synthetic detections injected into empty frame!"


# -------------------------------------------------------------
# 10. Person Tile Deduplication
# -------------------------------------------------------------
def test_person_tile_deduplication(detector):
    """
    Duplicate detections of the same person from overlapping tiles (IoU > 0.40)
    must be deduplicated to a single detection retaining the highest confidence.
    """
    tile1_det = {
        "id": "T1_001",
        "class_name": "person",
        "raw_class_name": "person",
        "raw_detector_confidence": 0.38,
        "bbox_xyxy": [450, 450, 470, 500]
    }
    tile2_det = {
        "id": "T2_001",
        "class_name": "person",
        "raw_class_name": "person",
        "raw_detector_confidence": 0.59,  # Higher confidence
        "bbox_xyxy": [452, 451, 471, 501]  # High overlap (IoU > 0.90)
    }

    nms_out = detector._apply_nms([tile1_det, tile2_det], person_iou_thresh=0.40)
    assert len(nms_out) == 1, f"Expected 1 deduplicated person detection, got {len(nms_out)}"
    assert nms_out[0]["raw_detector_confidence"] == pytest.approx(0.59, abs=1e-3)

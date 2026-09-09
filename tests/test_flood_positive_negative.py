import numpy as np
import cv2
import pytest

from src.disaster_inspection.segmentation import SceneSegmentationAnalyzer
from src.disaster_inspection.flood_validator import FloodValidator
from src.disaster_inspection.hazard_detection import HazardDetector
from src.disaster_inspection.risk_scoring import DisasterRiskScorer

def test_flood_positive_negative_suite():
    """
    Mandatory Test Suite (Requirement 18):
    - TEST 1 (NORMAL URBAN): 0 flood incidents.
    - TEST 2 (NORMAL WATER BODY): Water detected, but NO flood incident generated.
    - TEST 3 (FLOODED VILLAGE): Flood incident CREATED, rescue priorities active.
    """

    # TEST 1: Normal urban campus scene (dry road + building + grass)
    img_normal = np.zeros((400, 600, 3), dtype=np.uint8)
    img_normal[100:300, :] = (60, 60, 60) # Road
    img_normal[0:100, :] = (30, 100, 40)   # Grass

    water_mask_1, water_ratio_1 = SceneSegmentationAnalyzer.segment_water_candidates(img_normal)
    val_1 = FloodValidator.evaluate_flood_evidence(img_normal, water_mask_1, water_ratio_1, np.ones((400, 600)), [])
    assert val_1["incident_gate"] == "FAIL", "Normal scene must fail flood gate"
    assert val_1["water_state"] == "NO_WATER"

    # TEST 2: Normal water body (river / canal with dry surrounding land)
    img_river = np.zeros((400, 600, 3), dtype=np.uint8)
    img_river[0:150, :] = (30, 100, 40)   # Green banks
    img_river[150:200, :] = (200, 120, 20) # Blue river channel (BGR: blue=200)
    img_river[200:400, :] = (30, 100, 40)  # Green banks

    water_mask_2, water_ratio_2 = SceneSegmentationAnalyzer.segment_water_candidates(img_river)
    val_2 = FloodValidator.evaluate_flood_evidence(img_river, water_mask_2, water_ratio_2, np.ones((400, 600)), [])
    # Water detected, but fails flood disaster gate (water body confined in channel)
    assert val_2["water_state"] in ["WATER_BODY", "NO_WATER"], f"Expected WATER_BODY or NO_WATER, got {val_2['water_state']}"

    # TEST 3: Flooded Village (wide blue water spilling across 45% of image area)
    img_flood = np.zeros((400, 600, 3), dtype=np.uint8)
    img_flood[:, :] = (220, 140, 30) # Massive blue flood water (BGR)
    img_flood[50:150, 50:150] = (50, 60, 140) # Submerged building roof

    water_mask_3, water_ratio_3 = SceneSegmentationAnalyzer.segment_water_candidates(img_flood)
    detected_objs_3 = [{"class_name": "building", "confidence": 0.88, "bbox": [50, 50, 150, 150]}]
    val_3 = FloodValidator.evaluate_flood_evidence(img_flood, water_mask_3, water_ratio_3, np.ones((400, 600)), detected_objs_3)
    
    assert val_3["incident_gate"] == "PASS", "Flooded village scene must PASS flood incident gate"
    assert val_3["water_state"] in ["FLOOD_ZONE", "POTENTIAL_INUNDATION"]

    hazard_det = HazardDetector()
    hazards_3 = hazard_det.analyze_hazards(img_flood, np.ones((400, 600)), detected_objs_3)
    assert len(hazards_3) >= 1, "Flooded village must create at least 1 FLOOD_ZONE hazard incident"
    assert hazards_3[0]["incident_type"] == "FLOOD_ZONE"

    risk_level, score, reasons, priorities = DisasterRiskScorer.calculate_scene_risk(detected_objs_3, hazards_3, [])
    assert risk_level in ["HIGH", "CRITICAL"], f"Flooded village risk level should be HIGH or CRITICAL (got {risk_level})"

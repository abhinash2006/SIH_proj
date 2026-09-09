import numpy as np
import cv2
import pytest

from src.disaster_inspection.scene_classifier import SceneDisasterClassifier
from src.disaster_inspection.segmentation import SceneSegmentationAnalyzer
from src.disaster_inspection.hazard_detection import HazardDetector
from src.disaster_inspection.risk_scoring import DisasterRiskScorer

def test_normal_urban_campus_negative_test():
    """
    Negative Test Case: A normal urban/campus drone image must NOT produce false flood or debris disaster alerts.
    """
    # Create synthetic normal urban campus image (asphalt road + green grass + building roof + person)
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # Asphalt road (dark gray)
    img[200:400, :] = (60, 60, 60)
    
    # Green grass lawn (dark green)
    img[0:200, :] = (30, 100, 40)
    
    # Building roof (reddish brown)
    img[400:600, 0:400] = (50, 60, 140)

    # Person detection in normal scene
    detected_objects = [
        {"class_name": "person", "confidence": 0.85, "bbox": [100, 250, 120, 290]}
    ]

    # 1. Candidate segmentations
    water_mask, water_ratio = SceneSegmentationAnalyzer.segment_water_candidates(img)
    debris_mask, debris_ratio = SceneSegmentationAnalyzer.segment_debris_and_anomalies(img)

    # 2. Scene Disaster Classification
    scene_res = SceneDisasterClassifier.classify_scene(
        rgb_image=img,
        water_ratio=water_ratio,
        debris_ratio=debris_ratio,
        fire_ratio=0.0,
        detected_objects=detected_objects
    )

    assert scene_res["scene_class"] == "NORMAL", f"Expected NORMAL scene class, got {scene_res['scene_class']}"
    assert not scene_res["is_disaster"], "Normal scene should not be flagged as disaster"

    # 3. Hazard Detector
    hazard_detector = HazardDetector()
    depth_dummy = np.ones((600, 800), dtype=np.float32) * 5.0
    hazards = hazard_detector.analyze_hazards(img, depth_dummy, detected_objects, frame_idx=0)

    assert len(hazards) == 0, f"Expected 0 false hazard incidents on normal scene, got {len(hazards)}"

    # 4. Disaster Risk Scorer
    risk_level, score, reasons, priorities = DisasterRiskScorer.calculate_scene_risk(detected_objects, hazards, [])
    assert risk_level == "LOW / NO DISASTER EVIDENCE", f"Expected LOW / NO DISASTER EVIDENCE risk, got {risk_level}"
    assert score == 0.0, f"Expected score 0.0 on normal scene, got {score}"

def test_shadow_and_dark_asphalt_non_water():
    """
    Verifies that dark shadows and asphalt roads are NOT misclassified as flood water.
    """
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    # Dark shadow / asphalt
    img[:, :] = (20, 20, 20)

    water_mask, water_ratio = SceneSegmentationAnalyzer.segment_water_candidates(img)
    assert water_ratio < 0.05, f"Asphalt/shadow should not trigger flood water segmentation (got {water_ratio*100:.1f}%)"

import numpy as np
import cv2
import logging
from typing import List, Dict, Any
from pathlib import Path

from src.disaster_inspection.segmentation import SceneSegmentationAnalyzer
from src.disaster_inspection.scene_classifier import SceneDisasterClassifier
from src.disaster_inspection.flood_validator import FloodValidator

logger = logging.getLogger("HazardDetector")

class HazardDetector:
    """
    Detects environmental hazards: flood water coverage, road blockages/debris anomalies,
    and visible smoke/fire indicators from drone aerial observations.
    Integrates FloodValidator for multi-evidence flood inundation analysis.
    """

    def analyze_hazards(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray,
        detected_objects: List[Dict[str, Any]],
        frame_idx: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Runs multi-hazard inspection on a single frame.
        """
        H, W = rgb_image.shape[:2]
        hazards = []

        # 1. Candidate Segmentations
        water_candidate_mask, water_ratio = SceneSegmentationAnalyzer.segment_water_candidates(rgb_image)
        debris_mask, debris_ratio = SceneSegmentationAnalyzer.segment_debris_and_anomalies(rgb_image)

        # Fire / Smoke visual indicator
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        lower_fire = np.array([0, 120, 200])
        upper_fire = np.array([25, 255, 255])
        fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)
        fire_ratio = float(np.mean(fire_mask > 0))

        # 2. Multi-Evidence Flood Validation (No hard-blocking binary gate)
        val_res = FloodValidator.evaluate_flood_evidence(
            rgb_image=rgb_image,
            water_candidate_mask=water_candidate_mask,
            water_candidate_ratio=water_ratio,
            depth_map=depth_map,
            detected_objects=detected_objects,
            frame_idx=frame_idx
        )

        if val_res["incident_gate"] == "PASS":
            M = cv2.moments(water_candidate_mask)
            cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else W // 2
            cy = int(M["m01"] / M["m00"]) if M["m01"] > 0 else H // 2

            flood_conf = val_res["flood_incident_score"]
            hazards.append({
                "incident_type": "FLOOD_ZONE",
                "hazard_class": "flood",
                "confidence": round(flood_conf, 3),
                "segmentation_confidence": round(val_res["semantic_water_confidence"], 3),
                "inundation_confidence": round(val_res["inundation_confidence"], 3),
                "severity": "CRITICAL" if water_ratio > 0.35 else "HIGH",
                "coverage_pct": round(water_ratio * 100.0, 1),
                "pixel_center": (cx, cy),
                "bbox": [max(0, cx - 120), max(0, cy - 120), min(W, cx + 120), min(H, cy + 120)],
                "frame_idx": frame_idx,
                "evidence": f"Flooded water inundation detected ({water_ratio*100.0:.1f}% area, Score: {flood_conf:.2f}). State: {val_res['water_state']}.",
                "status": "NEEDS_VERIFICATION"
            })
        else:
            logger.info(f"[HAZARD DETECTOR] Frame {frame_idx}: Flood evidence status: {val_res['water_state']} ({val_res['rejection_reason']})")

        # 4. Debris & Road Obstruction (Requires Road Region evidence + confirmed disaster)
        if debris_ratio > 0.35:
            M = cv2.moments(debris_mask)
            cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else W // 3
            cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else H // 3

            has_road = any(d.get("class_name") in ["vehicle", "road"] for d in detected_objects)
            inc_type = "BLOCKED_ROAD_DEBRIS" if has_road else "POTENTIAL_GROUND_OBSTRUCTION"

            conf = min(0.90, round(0.40 + debris_ratio * 1.1, 3))
            hazards.append({
                "incident_type": inc_type,
                "hazard_class": "debris",
                "confidence": conf,
                "segmentation_confidence": round(float(debris_ratio), 3),
                "severity": "HIGH" if debris_ratio > 0.45 else "MEDIUM",
                "coverage_pct": round(debris_ratio * 100.0, 1),
                "pixel_center": (cx, cy),
                "bbox": [max(0, cx - 80), max(0, cy - 80), min(W, cx + 80), min(H, cy + 80)],
                "frame_idx": frame_idx,
                "evidence": f"Severe rubble concentration ({debris_ratio*100.0:.1f}% area) {'obstructing road' if has_road else 'on ground region'}.",
                "status": "NEEDS_VERIFICATION"
            })

        # 3. Fire / Smoke visual indicator (uses fire_mask/fire_ratio computed above)

        if fire_ratio > 0.015:
            M = cv2.moments(fire_mask)
            cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else W // 2
            cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else H // 2

            conf = min(0.95, round(0.60 + fire_ratio * 10.0, 3))
            hazards.append({
                "incident_type": "FIRE_SMOKE_HAZARD",
                "hazard_class": "fire",
                "confidence": conf,
                "severity": "CRITICAL",
                "coverage_pct": round(fire_ratio * 100.0, 1),
                "pixel_center": (cx, cy),
                "bbox": [max(0, cx - 50), max(0, cy - 50), min(W, cx + 50), min(H, cy + 50)],
                "frame_idx": frame_idx,
                "evidence": f"Visible high-temperature flame / active fire hue detected over {fire_ratio*100.0:.2f}% frame area.",
                "status": "NEEDS_VERIFICATION"
            })

        return hazards

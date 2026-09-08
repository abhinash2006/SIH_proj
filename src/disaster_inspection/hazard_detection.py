import numpy as np
import cv2
from typing import List, Dict, Any

from src.disaster_inspection.segmentation import SceneSegmentationAnalyzer

class HazardDetector:
    """
    Detects environmental and situational hazards: flooding, road blockages, debris anomalies,
    and visible smoke/fire indicators from drone aerial observations.
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

        # 1. Flood Water Detection
        water_mask, water_ratio = SceneSegmentationAnalyzer.segment_water_and_flood(rgb_image)
        if water_ratio > 0.08:
            # Locate flood centroid
            M = cv2.moments(water_mask)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = W // 2, H // 2

            hazards.append({
                "incident_type": "FLOOD_ZONE",
                "hazard_class": "flood",
                "confidence": min(0.95, round(0.70 + water_ratio, 2)),
                "severity": "CRITICAL" if water_ratio > 0.25 else "HIGH",
                "coverage_pct": round(water_ratio * 100.0, 1),
                "pixel_center": (cx, cy),
                "frame_idx": frame_idx,
                "evidence": f"Flooded water coverage detected over {water_ratio*100.1:.1f}% of aerial frame.",
                "status": "NEEDS_VERIFICATION"
            })

        # 2. Debris & Road Obstruction
        debris_mask, debris_ratio = SceneSegmentationAnalyzer.segment_debris_and_anomalies(rgb_image)
        if debris_ratio > 0.12:
            M = cv2.moments(debris_mask)
            cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else W // 3
            cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else H // 3

            hazards.append({
                "incident_type": "BLOCKED_ROAD_DEBRIS",
                "hazard_class": "debris",
                "confidence": min(0.92, round(0.65 + debris_ratio, 2)),
                "severity": "HIGH" if debris_ratio > 0.25 else "MEDIUM",
                "coverage_pct": round(debris_ratio * 100.0, 1),
                "pixel_center": (cx, cy),
                "frame_idx": frame_idx,
                "evidence": f"High density debris & rubble concentration obstructing ground route ({debris_ratio*100.0:.1f}% area).",
                "status": "NEEDS_VERIFICATION"
            })

        # 3. Fire / Smoke visual indicator
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        lower_fire = np.array([0, 120, 200])
        upper_fire = np.array([25, 255, 255])
        fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)
        fire_ratio = float(np.mean(fire_mask > 0))

        if fire_ratio > 0.015:
            M = cv2.moments(fire_mask)
            cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else W // 2
            cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else H // 2

            hazards.append({
                "incident_type": "FIRE_SMOKE_HAZARD",
                "hazard_class": "fire",
                "confidence": min(0.95, round(0.75 + fire_ratio * 5, 2)),
                "severity": "CRITICAL",
                "coverage_pct": round(fire_ratio * 100.0, 1),
                "pixel_center": (cx, cy),
                "frame_idx": frame_idx,
                "evidence": "Visible high-temperature flame / active fire hue detected in drone view.",
                "status": "NEEDS_VERIFICATION"
            })

        return hazards

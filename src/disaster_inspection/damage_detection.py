import numpy as np
import cv2
from typing import List, Dict, Any

class StructuralDamageDetector:
    """
    Detects visual indicators of building structural damage, roof discontinuities, debris accumulation,
    and structural anomalies from drone imagery and depth maps.
    """

    def analyze_structural_damage(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray,
        detected_buildings: List[Dict[str, Any]],
        frame_idx: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Analyzes building bounding regions for structural damage indicators.
        """
        H, W = rgb_image.shape[:2]
        if depth_map.ndim == 3:
            depth_map = depth_map.squeeze(-1)
        if depth_map.shape[:2] != (H, W):
            depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)

        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        gray_lap = cv2.Laplacian(gray, cv2.CV_64F)
        damage_incidents = []

        # If no specific building bounding box detected, perform grid analysis over central regions
        regions_to_check = detected_buildings if len(detected_buildings) > 0 else [
            {"object_id": f"BUILDING_GRID_{i+1}", "bbox": [int(W*0.15*(i+1)), int(H*0.2), int(W*0.15*(i+2)), int(H*0.6)]}
            for i in range(min(4, max(1, W // 300)))
        ]

        for idx, bldg in enumerate(regions_to_check):
            x1, y1, x2, y2 = bldg["bbox"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)

            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue

            crop_lap = gray_lap[y1:y2, x1:x2]
            crop_depth = depth_map[y1:y2, x1:x2]

            if crop_depth.shape[0] < 3 or crop_depth.shape[1] < 3 or crop_lap.size == 0:
                continue

            # 1. Texture irregularity / roof disruption score
            lap_variance = float(np.var(crop_lap))
            
            # 2. Depth gradient / surface discontinuity
            try:
                gy, gx = np.gradient(crop_depth)
                depth_grad_mag = np.sqrt(gx**2 + gy**2)
                high_grad_ratio = float(np.mean(depth_grad_mag > 0.15))
            except Exception:
                high_grad_ratio = 0.0

            # Damage classification
            damage_level = "intact"
            confidence = 0.65
            evidence = "Uniform surface texture & smooth depth profile."

            if high_grad_ratio > 0.25 and lap_variance > 1200:
                damage_level = "potential_major_damage"
                confidence = 0.84
                evidence = "High depth variance, roof discontinuity & high edge texture chaos."
            elif high_grad_ratio > 0.15 or lap_variance > 800:
                damage_level = "minor_damage"
                confidence = 0.72
                evidence = "Moderate surface irregularity and edge disruption detected."

            if damage_level != "intact":
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                damage_incidents.append({
                    "incident_type": "STRUCTURAL_DAMAGE",
                    "damage_level": damage_level,
                    "confidence": round(confidence, 2),
                    "target_object_id": bldg.get("object_id", f"BUILDING_{idx+1:03d}"),
                    "frame_idx": frame_idx,
                    "pixel_center": (cx, cy),
                    "evidence": evidence,
                    "status": "NEEDS_VERIFICATION",
                    "severity": "HIGH" if damage_level == "potential_major_damage" else "MEDIUM"
                })

        return damage_incidents

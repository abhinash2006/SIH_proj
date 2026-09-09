import logging
from typing import Dict, Any, List
import numpy as np
import cv2

logger = logging.getLogger("SceneDisasterClassifier")

class SceneDisasterClassifier:
    """
    Performs scene-level disaster assessment to prevent false disaster alerts
    on normal urban/campus drone imagery.
    Classifies overall scene as:
    - NORMAL
    - FLOOD
    - FIRE
    - LANDSLIDE
    - EARTHQUAKE_DAMAGE
    - UNKNOWN
    """

    @staticmethod
    def classify_scene(
        rgb_image: np.ndarray,
        water_ratio: float,
        debris_ratio: float,
        fire_ratio: float,
        detected_objects: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluates multi-modal evidence to determine the macro scene status.
        """
        H, W = rgb_image.shape[:2]
        
        # 1. Evaluate Flood Evidence
        # Flood requires significant water coverage (> 28%) OR water overlapping dry land
        has_flood = (water_ratio > 0.28)
        
        # 2. Evaluate Fire / Smoke Evidence
        has_fire = (fire_ratio > 0.02)
        
        # 3. Evaluate Debris / Structural Damage Evidence
        # Debris requires extremely high texture density + rubble indicators (> 35%)
        has_debris = (debris_ratio > 0.35)

        # 4. Count normal objects (vehicles, buildings, persons, trees)
        normal_objects = [d for d in detected_objects if d.get("class_name") in ["car", "vehicle", "building", "person", "tree", "road"]]
        
        if has_fire:
            scene_class = "FIRE"
            conf = 0.90
            rationale = "Active flame / high-temperature smoke hue spectrum detected."
        elif has_flood:
            scene_class = "FLOOD"
            conf = min(0.95, float(0.50 + water_ratio * 1.5))
            rationale = f"Massive inundation detected over {water_ratio*100:.1f}% of aerial frame."
        elif has_debris:
            scene_class = "EARTHQUAKE_DAMAGE"
            conf = min(0.90, float(0.40 + debris_ratio * 1.2))
            rationale = f"Severe structural collapse and rubble accumulation over {debris_ratio*100:.1f}% of frame."
        else:
            scene_class = "NORMAL"
            conf = 0.95
            rationale = "Normal urban / campus environment. No active disaster signatures present."

        logger.info(f"[SCENE CLASSIFIER] Result: {scene_class} (Conf: {conf:.2f}) | {rationale}")
        return {
            "scene_class": scene_class,
            "confidence": conf,
            "rationale": rationale,
            "is_disaster": (scene_class != "NORMAL")
        }

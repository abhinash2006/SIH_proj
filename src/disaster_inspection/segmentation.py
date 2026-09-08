import numpy as np
import cv2
from typing import Tuple, Dict, Any

class SceneSegmentationAnalyzer:
    """
    Performs color-space HSV and texture-based semantic segmentation for flood water regions,
    road surfaces, vegetation, and debris.
    """

    @staticmethod
    def segment_water_and_flood(rgb_bgr_img: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Segments water / flooded regions based on HSV water spectrum & low variance reflection characteristics.
        Returns:
            water_mask: [H, W] uint8 binary mask (255 for water, 0 for land)
            water_coverage_ratio: float ratio in [0.0, 1.0]
        """
        hsv = cv2.cvtColor(rgb_bgr_img, cv2.COLOR_BGR2HSV)
        
        # Water HSV range: dark murky water, muddy flood water, and reflective water
        lower_blue = np.array([80, 30, 20])
        upper_blue = np.array([135, 255, 240])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # Muddy flood water HSV range (brownish/grayish low saturation)
        lower_mud = np.array([10, 20, 30])
        upper_mud = np.array([35, 120, 180])
        mask_mud = cv2.inRange(hsv, lower_mud, upper_mud)

        # Combine masks
        water_mask = cv2.bitwise_or(mask_blue, mask_mud)

        # Morphological opening to remove isolated noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)

        coverage_ratio = float(np.mean(water_mask > 0))
        return water_mask, coverage_ratio

    @staticmethod
    def segment_debris_and_anomalies(rgb_bgr_img: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Segments high-texture debris / rubble regions using Canny edge density & local entropy.
        """
        gray = cv2.cvtColor(rgb_bgr_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Dense edge accumulator
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        edge_density = cv2.boxFilter(edges.astype(np.float32), -1, (15, 15))
        
        debris_mask = (edge_density > 45.0).astype(np.uint8) * 255
        debris_mask = cv2.morphologyEx(debris_mask, cv2.MORPH_CLOSE, kernel)
        
        debris_ratio = float(np.mean(debris_mask > 0))
        return debris_mask, debris_ratio

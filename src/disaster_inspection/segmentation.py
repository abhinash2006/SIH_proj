import numpy as np
import cv2
from typing import Tuple, Dict, Any

class SceneSegmentationAnalyzer:
    """
    Performs color-space HSV and texture-based semantic segmentation for flood water regions,
    road surfaces, vegetation, and debris.
    """

    @staticmethod
    def segment_water_candidates(rgb_bgr_img: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Segments water candidates across multi-spectral regimes: blue/cyan and turbid/muddy floodwater.
        Returns:
            water_candidate_mask: [H, W] uint8 binary mask (255 for water, 0 for land)
            water_coverage_ratio: float ratio in [0.0, 1.0]
        """
        try:
            from src.flood_analyzer import FloodAnalyzer
            water_candidate_mask = FloodAnalyzer.extract_water_candidates(rgb_bgr_img)
        except Exception:
            hsv = cv2.cvtColor(rgb_bgr_img, cv2.COLOR_BGR2HSV)
            lower_blue = np.array([85, 40, 30])
            upper_blue = np.array([135, 255, 240])
            water_candidate_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
            water_candidate_mask = cv2.morphologyEx(water_candidate_mask, cv2.MORPH_OPEN, kernel)
            water_candidate_mask = cv2.morphologyEx(water_candidate_mask, cv2.MORPH_CLOSE, kernel)

        coverage_ratio = float(np.mean(water_candidate_mask > 0))
        return water_candidate_mask, coverage_ratio

    @staticmethod
    def segment_water_and_flood(rgb_bgr_img: np.ndarray) -> Tuple[np.ndarray, float]:
        return SceneSegmentationAnalyzer.segment_water_candidates(rgb_bgr_img)

    @staticmethod
    def segment_debris_and_anomalies(rgb_bgr_img: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Segments severe high-texture debris / rubble regions using edge density & local variance.
        """
        gray = cv2.cvtColor(rgb_bgr_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 220)
        
        # Dense edge accumulator filter
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        edge_density = cv2.boxFilter(edges.astype(np.float32), -1, (25, 25))
        
        # High threshold to prevent normal roofs / trees / windows from being marked as debris
        debris_mask = (edge_density > 130.0).astype(np.uint8) * 255
        debris_mask = cv2.morphologyEx(debris_mask, cv2.MORPH_OPEN, kernel)
        debris_mask = cv2.morphologyEx(debris_mask, cv2.MORPH_CLOSE, kernel)
        
        debris_ratio = float(np.mean(debris_mask > 0))
        return debris_mask, debris_ratio

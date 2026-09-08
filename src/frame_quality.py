import cv2
import numpy as np
import logging
from typing import Dict, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class FrameQualityAnalyzer:
    """
    Computes visual and geometric quality metrics for video frames:
    - Sharpness (Laplacian variance)
    - Exposure & Brightness (underexposure, overexposure, mean luminance)
    - Contrast (Standard deviation of luminance)
    - Sky dominance (Outdoor aerial sky detection)
    - Combined quality score
    """

    @staticmethod
    def calculate_sharpness(gray_img: np.ndarray) -> float:
        """
        Calculates sharpness using variance of Laplacian.
        Higher is sharper.
        """
        laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
        variance = float(laplacian.var())
        return variance

    @staticmethod
    def calculate_exposure(gray_img: np.ndarray) -> Dict[str, float]:
        """
        Calculates exposure statistics:
        - Mean brightness (0-255)
        - Underexposure ratio (fraction of pixels < 10)
        - Overexposure ratio (fraction of pixels > 245)
        - Contrast (std deviation of luminance)
        """
        total_pixels = gray_img.size
        mean_brightness = float(np.mean(gray_img))
        std_contrast = float(np.std(gray_img))
        
        underexposed_count = np.count_nonzero(gray_img < 12)
        overexposed_count = np.count_nonzero(gray_img > 245)
        
        underexposed_ratio = float(underexposed_count / total_pixels)
        overexposed_ratio = float(overexposed_count / total_pixels)
        
        return {
            "mean_brightness": round(mean_brightness, 2),
            "std_contrast": round(std_contrast, 2),
            "underexposed_ratio": round(underexposed_ratio, 4),
            "overexposed_ratio": round(overexposed_ratio, 4),
        }

    @staticmethod
    def estimate_sky_ratio(bgr_img: np.ndarray) -> float:
        """
        Estimates the ratio of the image that corresponds to clear/cloudy sky.
        In drone aerial imagery, sky is characterized by high value (V),
        low saturation (S), and typically localized in the upper portion of the frame.
        """
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Sky heuristic: High brightness (V > 180), Low to moderate saturation (S < 90)
        # or bright cyan/blue sky (Hue in [85, 135], S > 30, V > 140)
        sky_mask_cloud = (v > 190) & (s < 60)
        sky_mask_blue = (h >= 85) & (h <= 135) & (v > 140) & (s >= 30)
        sky_mask = sky_mask_cloud | sky_mask_blue
        
        # Give higher weight to upper half of image
        height, width = sky_mask.shape
        upper_mask = np.zeros_like(sky_mask, dtype=bool)
        upper_mask[:int(height * 0.7), :] = True
        
        sky_pixels = np.count_nonzero(sky_mask & upper_mask)
        sky_ratio = float(sky_pixels / sky_mask.size)
        return round(sky_ratio, 4)

    @classmethod
    def analyze_frame(cls, frame_path: Union[str, Path]) -> Dict[str, Union[float, bool]]:
        """
        Comprehensive single frame quality analysis.
        """
        frame_path = str(frame_path)
        img = cv2.imread(frame_path)
        if img is None:
            raise ValueError(f"Could not read image for quality analysis: {frame_path}")
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        sharpness = cls.calculate_sharpness(gray)
        exposure = cls.calculate_exposure(gray)
        sky_ratio = cls.estimate_sky_ratio(img)
        
        # Composite quality score (0 - 100 normalized scale)
        # Sharpness contribution via smooth log scaling
        sharpness_score = float(min(60.0, np.log1p(sharpness) * 8.0))
        exposure_penalty = float((exposure["underexposed_ratio"] + exposure["overexposed_ratio"]) * 50.0)
        brightness_penalty = float(abs(exposure["mean_brightness"] - 128.0) / 128.0 * 15.0)
        sky_penalty = float(max(0.0, (sky_ratio - 0.4)) * 25.0)
        
        quality_score = max(0.0, min(100.0, sharpness_score + 40.0 - exposure_penalty - brightness_penalty - sky_penalty))
        
        return {
            "file_path": frame_path,
            "sharpness": round(sharpness, 2),
            "mean_brightness": exposure["mean_brightness"],
            "std_contrast": exposure["std_contrast"],
            "underexposed_ratio": exposure["underexposed_ratio"],
            "overexposed_ratio": exposure["overexposed_ratio"],
            "sky_ratio": sky_ratio,
            "quality_score": round(quality_score, 2),
            "is_valid": sharpness > 20.0 and exposure["underexposed_ratio"] < 0.25 and exposure["overexposed_ratio"] < 0.35
        }

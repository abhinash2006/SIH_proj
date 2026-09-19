"""
Depth Anything V2 Monocular Depth Estimator for UAV 3D Reconstruction.
Provides dense relative depth maps for scale/shift validation and geometric comparison with VGGT.
"""

import os
import cv2
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DepthResult:
    depth_map: np.ndarray        # 2D relative depth map [H, W] float32
    confidence: np.ndarray       # 2D confidence map [H, W] float32 in [0, 1]
    scale_type: str              # "relative" or "metric"
    source_model: str            # "depth_anything_v2" or "mock"
    processing_time_ms: float    # Execution latency in ms


class DepthAnythingV2Estimator:
    """
    Standardized Depth Anything V2 Estimator.
    Outputs normalized relative inverse depth maps and confidence scores.
    """

    def __init__(
        self,
        checkpoint_path: str = "models/depth_anything_v2/depth_anything_v2_vits.pth",
        encoder: str = "vits",
        device: str = "auto",
        use_fp16: bool = True,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.encoder = encoder
        self.device_pref = device
        self.use_fp16 = use_fp16
        self.model = None

    def load_model(self):
        """Loads model weights onto device if available."""
        if self.model is not None:
            return self.model

        if not self.checkpoint_path.exists():
            logger.warning(
                f"[DEPTH] Checkpoint not found at '{self.checkpoint_path}'. "
                "Depth Anything V2 will use heuristic depth estimation."
            )
            return None

        try:
            import torch
            dev = "cuda" if (self.device_pref in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
            logger.info(f"[DEPTH] Initializing Depth Anything V2 from '{self.checkpoint_path}' on {dev}...")
            # We can import from workspace if available
            try:
                from src.depth_anything_v2 import DepthAnythingV2Estimator as DAEngine
                self.engine = DAEngine(checkpoint_path=str(self.checkpoint_path), device=dev)
                self.model = self.engine.load_model()
            except Exception as e:
                logger.warning(f"[DEPTH] Direct load failed ({e}). Will use fallback estimation.")
            return self.model
        except Exception as e:
            logger.warning(f"[DEPTH] Failed to initialize Depth Anything V2 ({e}).")
            return None

    def estimate(self, image_input: Union[str, np.ndarray]) -> DepthResult:
        """
        Estimates monocular relative depth map from an input image.

        Args:
            image_input: Filepath string or numpy image array (BGR or RGB).

        Returns:
            DepthResult containing standardized relative depth and confidence.
        """
        import time

        start_t = time.perf_counter()

        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise FileNotFoundError(f"Cannot load image: {image_input}")
        else:
            img = image_input

        h, w = img.shape[:2]

        self.load_model()

        if self.model is not None and hasattr(self, "engine"):
            try:
                res = self.engine.estimate_depth(img)
                depth_map = res["depth_map"].astype(np.float32)
                confidence = res.get("confidence", np.ones((h, w), dtype=np.float32) * 0.9)
                source_name = "depth_anything_v2"
            except Exception as e:
                logger.warning(f"[DEPTH] Inference failed ({e}), using gradient heuristic fallback.")
                depth_map, confidence = self._estimate_fallback(img)
                source_name = "fallback_heuristic"
        else:
            depth_map, confidence = self._estimate_fallback(img)
            source_name = "fallback_heuristic"

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        return DepthResult(
            depth_map=depth_map,
            confidence=confidence,
            scale_type="relative",
            source_model=source_name,
            processing_time_ms=round(elapsed_ms, 2),
        )

    @staticmethod
    def _estimate_fallback(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Realistic aerial gradient depth heuristic (ground plane perspective gradient)
        for testing and CI environments.
        """
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        # Vertical aerial gradient (nadir / oblique view: top = far, bottom = near)
        y_grid = np.linspace(0.2, 1.0, h, dtype=np.float32)[:, None]
        base_depth = np.tile(y_grid, (1, w))

        # Add subtle local edge modulation
        blur = cv2.GaussianBlur(gray.astype(np.float32) / 255.0, (15, 15), 0)
        depth_map = (base_depth * 0.8 + blur * 0.2).astype(np.float32)

        # Confidence is higher in high-contrast textured areas
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        conf = (cv2.normalize(grad_mag, None, 0.5, 0.95, cv2.NORM_MINMAX)).astype(np.float32)

        return depth_map, conf

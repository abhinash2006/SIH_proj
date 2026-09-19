"""
SAM 2 Segmenter Wrapper for UAV 3D Reconstruction.
Provides pixel-accurate segmentation masks for dynamic objects using Meta's SAM 2.
Includes robust bounding-box fallback mode when SAM 2 weights or CUDA libraries are absent.
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
class MaskResult:
    mask: np.ndarray  # Binary mask [H, W], uint8, 255 for dynamic, 0 for static
    dynamic_pixel_count: int
    dynamic_pixel_ratio: float
    source: str  # "sam2" or "fallback_bbox"
    processing_time_ms: float


class SAM2Segmenter:
    """
    Modular wrapper around Meta SAM 2 (Segment Anything Model 2).
    Generates pixel-accurate binary masks from prompt boxes or points.
    """

    def __init__(
        self,
        checkpoint_path: str = "models/sam2/sam2.1_hiera_small.pt",
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_s.yaml",
        device: str = "auto",
        use_fallback_mask: bool = True,
    ):
        self.checkpoint_path = checkpoint_path
        self.model_cfg = model_cfg
        self.device = device
        self.use_fallback_mask = use_fallback_mask
        self.model = None
        self.predictor = None
        self._is_sam2_ready = False
        self._init_checked = False

    def _check_and_load(self):
        """Attempts to load SAM 2 model; handles missing weights gracefully."""
        if self._init_checked:
            return self._is_sam2_ready

        self._init_checked = True

        if not os.path.exists(self.checkpoint_path):
            logger.warning(
                f"[SAM2] SAM 2 checkpoint not found at '{self.checkpoint_path}' — "
                "dynamic-object segmentation falling back to bounding-box masks."
            )
            self._is_sam2_ready = False
            return False

        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            dev = self.device
            if dev == "auto":
                dev = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"[SAM2] Loading SAM 2 model from '{self.checkpoint_path}' on {dev}...")
            self.model = build_sam2(self.model_cfg, self.checkpoint_path, device=dev)
            self.predictor = SAM2ImagePredictor(self.model)
            self._is_sam2_ready = True
            logger.info("[SAM2] SAM 2 model loaded successfully.")
            return True
        except Exception as e:
            logger.warning(f"[SAM2] Failed to initialize SAM 2 ({e}). Using bounding-box fallback.")
            self._is_sam2_ready = False
            return False

    def segment(
        self,
        image: np.ndarray,
        prompts: Optional[Dict[str, Any]] = None,
    ) -> MaskResult:
        """
        Generic prompt-based segmentation interface.
        prompts can contain 'point_coords', 'point_labels', 'box'.
        """
        boxes = prompts.get("box", []) if prompts else []
        return self.generate_mask(image, boxes)

    def track(self, video_path: str) -> List[MaskResult]:
        """
        Video object tracking interface (SAM 2 Video Predictor placeholder/wrapper).
        """
        logger.info(f"[SAM2] Video tracking requested for '{video_path}'")
        return []

    def generate_mask(
        self,
        image: Union[str, np.ndarray],
        bounding_boxes: List[List[float]],
        dilation_kernel: int = 5,
    ) -> MaskResult:
        """
        Generates binary dynamic object mask [H, W] from list of bounding boxes [x1, y1, x2, y2].

        Args:
            image: Image path or numpy array (BGR or RGB).
            bounding_boxes: List of [x1, y1, x2, y2] bounding boxes.
            dilation_kernel: Kernel size to expand mask boundaries.

        Returns:
            MaskResult containing uint8 binary mask and stats.
        """
        import time

        start_t = time.perf_counter()

        if isinstance(image, (str, Path)):
            img_bgr = cv2.imread(str(image))
            if img_bgr is None:
                raise FileNotFoundError(f"Failed to load image: {image}")
        else:
            img_bgr = image

        h, w = img_bgr.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        if not bounding_boxes:
            # No dynamic objects detected
            return MaskResult(
                mask=combined_mask,
                dynamic_pixel_count=0,
                dynamic_pixel_ratio=0.0,
                source="empty",
                processing_time_ms=round((time.perf_counter() - start_t) * 1000.0, 2),
            )

        is_ready = self._check_and_load()

        if is_ready and self.predictor is not None:
            # Run official Meta SAM 2 prediction
            try:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                self.predictor.set_image(img_rgb)
                boxes_np = np.array(bounding_boxes, dtype=np.float32)

                masks, scores, _ = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=boxes_np,
                    multimask_output=False,
                )

                for m in masks:
                    # m is boolean [1, H, W] or [H, W]
                    m_uint = (m.squeeze() > 0).astype(np.uint8) * 255
                    combined_mask = cv2.bitwise_or(combined_mask, m_uint)

                source_name = "sam2"
            except Exception as e:
                logger.warning(f"[SAM2] Inference failed ({e}), switching to bounding-box mask.")
                combined_mask = self._generate_fallback_bbox_mask(h, w, bounding_boxes)
                source_name = "fallback_bbox"
        else:
            # Bounding box polygon fallback
            combined_mask = self._generate_fallback_bbox_mask(h, w, bounding_boxes)
            source_name = "fallback_bbox"

        # Apply dilation to ensure dynamic object boundary pixels are safely excluded
        if dilation_kernel > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel, dilation_kernel))
            combined_mask = cv2.dilate(combined_mask, kernel)

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        dyn_pixels = int(np.count_nonzero(combined_mask))
        ratio = float(dyn_pixels / (h * w))

        return MaskResult(
            mask=combined_mask,
            dynamic_pixel_count=dyn_pixels,
            dynamic_pixel_ratio=round(ratio, 4),
            source=source_name,
            processing_time_ms=round(elapsed_ms, 2),
        )

    @staticmethod
    def _generate_fallback_bbox_mask(h: int, w: int, boxes: List[List[float]]) -> np.ndarray:
        """Generates rectangular / rounded polygon masks from bounding boxes."""
        mask = np.zeros((h, w), dtype=np.uint8)
        for box in boxes:
            x1 = max(0, min(w - 1, int(round(box[0]))))
            y1 = max(0, min(h - 1, int(round(box[1]))))
            x2 = max(0, min(w, int(round(box[2]))))
            y2 = max(0, min(h, int(round(box[3]))))
            if x2 > x1 and y2 > y1:
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        return mask

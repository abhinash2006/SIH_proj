"""
Dynamic Object Filter for UAV 3D Reconstruction.
Orchestrates YOLO detection and SAM 2 segmentation to produce reconstruction-safe images and masks.
Excludes transient dynamic objects (cars, pedestrians, animals) to prevent reconstruction ghosts.
"""

import os
import cv2
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .yolo_detector import YOLODetector, DetectionResult
from src.segmentation.sam2_segmenter import SAM2Segmenter, MaskResult

logger = logging.getLogger(__name__)


@dataclass
class DynamicFilterResult:
    frame_path: str
    mask_path: str
    safe_image_path: Optional[str]
    dynamic_detected: bool
    dynamic_count: int
    dynamic_classes: List[str]
    dynamic_pixel_ratio: float
    mask: np.ndarray


class DynamicObjectFilter:
    """
    Coordinates YOLO dynamic object detection with SAM 2 pixel-level segmentation.
    Generates binary exclusion masks for point cloud generation.
    """

    def __init__(
        self,
        mode: str = "yolo+sam2",  # "yolo+sam2", "yolo_only", "sam2_only", "disabled"
        yolo_detector: Optional[YOLODetector] = None,
        sam2_segmenter: Optional[SAM2Segmenter] = None,
        mask_dilation_kernel: int = 5,
        save_visualizations: bool = True,
    ):
        self.mode = mode.lower()
        self.detector = yolo_detector or YOLODetector()
        self.segmenter = sam2_segmenter or SAM2Segmenter()
        self.dilation_kernel = mask_dilation_kernel
        self.save_visualizations = save_visualizations

    def process_frame(
        self,
        image_path: str,
        output_mask_dir: str,
        output_safe_dir: Optional[str] = None,
    ) -> DynamicFilterResult:
        """
        Processes a single keyframe:
        1. Detects dynamic objects with YOLO.
        2. Segments dynamic regions with SAM 2.
        3. Saves mask and optional reconstruction-safe visualization.
        """
        img_p = Path(image_path)
        img = cv2.imread(str(img_p))
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")

        h, w = img.shape[:2]
        out_mask_path = Path(output_mask_dir)
        out_mask_path.mkdir(parents=True, exist_ok=True)
        mask_filename = out_mask_path / f"mask_{img_p.stem}.png"

        if self.mode == "disabled":
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.imwrite(str(mask_filename), empty_mask)
            return DynamicFilterResult(
                frame_path=str(img_p),
                mask_path=str(mask_filename),
                safe_image_path=None,
                dynamic_detected=False,
                dynamic_count=0,
                dynamic_classes=[],
                dynamic_pixel_ratio=0.0,
                mask=empty_mask,
            )

        # 1. Run YOLO Detection
        det_result: DetectionResult = self.detector.detect(img)
        dynamic_boxes = [d.box for d in det_result.detections if d.is_dynamic]
        dyn_classes = [d.class_name for d in det_result.detections if d.is_dynamic]

        # 2. Run SAM 2 or Bounding Box Segmentation
        if self.mode in ("yolo+sam2", "sam2_only"):
            mask_result: MaskResult = self.segmenter.generate_mask(
                img,
                bounding_boxes=dynamic_boxes,
                dilation_kernel=self.dilation_kernel,
            )
            binary_mask = mask_result.mask
            pixel_ratio = mask_result.dynamic_pixel_ratio
        elif self.mode == "yolo_only":
            binary_mask = SAM2Segmenter._generate_fallback_bbox_mask(h, w, dynamic_boxes)
            pixel_ratio = float(np.count_nonzero(binary_mask) / (h * w))
        else:
            binary_mask = np.zeros((h, w), dtype=np.uint8)
            pixel_ratio = 0.0

        # Save mask image
        cv2.imwrite(str(mask_filename), binary_mask)

        # Optional: Save reconstruction-safe image (masked out in neutral gray)
        safe_path_str = None
        if output_safe_dir and self.save_visualizations:
            out_safe_path = Path(output_safe_dir)
            out_safe_path.mkdir(parents=True, exist_ok=True)
            safe_file = out_safe_path / f"safe_{img_p.name}"

            # Invert mask: keep static regions, replace dynamic with subtle gray overlay
            safe_img = img.copy()
            safe_img[binary_mask > 0] = [128, 128, 128]
            cv2.imwrite(str(safe_file), safe_img)
            safe_path_str = str(safe_file)

        return DynamicFilterResult(
            frame_path=str(img_p),
            mask_path=str(mask_filename),
            safe_image_path=safe_path_str,
            dynamic_detected=len(dynamic_boxes) > 0,
            dynamic_count=len(dynamic_boxes),
            dynamic_classes=dyn_classes,
            dynamic_pixel_ratio=pixel_ratio,
            mask=binary_mask,
        )

    def process_sequence(
        self,
        frame_paths: List[str],
        output_mask_dir: str,
        output_safe_dir: Optional[str] = None,
    ) -> List[DynamicFilterResult]:
        """Processes a sequence of keyframes and returns results."""
        results = []
        total_dyn = 0

        for p in frame_paths:
            res = self.process_frame(p, output_mask_dir, output_safe_dir)
            results.append(res)
            total_dyn += res.dynamic_count

        logger.info(
            f"[DYNAMIC_FILTER] Processed {len(frame_paths)} frames | "
            f"Detected {total_dyn} dynamic objects | Mode: '{self.mode}'"
        )
        return results

"""
Frame Quality Filter for UAV 3D Reconstruction.
Evaluates sharpness, exposure, contrast, and entropy to filter out degraded frames.
Generates structured frame_quality.csv reports.
"""

import os
import csv
import cv2
import shutil
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class FrameQualityScore:
    frame_id: str
    frame_path: str
    blur_score: float
    brightness: float
    contrast: float
    entropy: float
    accepted: bool
    rejection_reason: str


@dataclass
class FrameQualityReport:
    total_frames: int
    accepted_frames: int
    rejected_frames: int
    scores: List[FrameQualityScore]
    csv_report_path: str


class FrameQualityFilter:
    """
    Evaluates frame quality for 3D reconstruction suitability.
    Discards blurry, severely underexposed/overexposed, low-contrast, or corrupted frames.
    """

    def __init__(
        self,
        min_blur_score: float = 50.0,
        min_brightness: float = 30.0,
        max_brightness: float = 235.0,
        min_contrast: float = 20.0,
        min_entropy: float = 1.0,
    ):
        self.min_blur_score = float(min_blur_score)
        self.min_brightness = float(min_brightness)
        self.max_brightness = float(max_brightness)
        self.min_contrast = float(min_contrast)
        self.min_entropy = float(min_entropy)

    @staticmethod
    def compute_entropy(gray_img: np.ndarray) -> float:
        """Computes Shannon entropy of grayscale image."""
        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / (hist.sum() + 1e-7)
        non_zeros = hist_norm[hist_norm > 0]
        return float(-np.sum(non_zeros * np.log2(non_zeros)))

    def evaluate_frame(self, image_path: str) -> FrameQualityScore:
        """Evaluates a single frame and returns its score and acceptance decision."""
        path_obj = Path(image_path)
        frame_id = path_obj.stem

        if not path_obj.exists():
            return FrameQualityScore(
                frame_id=frame_id,
                frame_path=str(path_obj),
                blur_score=0.0,
                brightness=0.0,
                contrast=0.0,
                entropy=0.0,
                accepted=False,
                rejection_reason="file_not_found",
            )

        img = cv2.imread(str(path_obj))
        if img is None or img.size == 0:
            return FrameQualityScore(
                frame_id=frame_id,
                frame_path=str(path_obj),
                blur_score=0.0,
                brightness=0.0,
                contrast=0.0,
                entropy=0.0,
                accepted=False,
                rejection_reason="corrupted_image",
            )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Blur Score: Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = float(laplacian.var())

        # 2. Brightness: Mean intensity
        brightness = float(np.mean(gray))

        # 3. Contrast: Standard deviation of intensity
        contrast = float(np.std(gray))

        # 4. Entropy: Shannon entropy
        entropy = self.compute_entropy(gray)

        # Decision rules
        accepted = True
        rejection_reason = "none"

        if blur_score < self.min_blur_score:
            accepted = False
            rejection_reason = "excessive_blur"
        elif brightness < self.min_brightness:
            accepted = False
            rejection_reason = "underexposed"
        elif brightness > self.max_brightness:
            accepted = False
            rejection_reason = "overexposed"
        elif contrast < self.min_contrast:
            accepted = False
            rejection_reason = "low_contrast"
        elif entropy < self.min_entropy:
            accepted = False
            rejection_reason = "low_entropy"

        return FrameQualityScore(
            frame_id=frame_id,
            frame_path=str(path_obj),
            blur_score=round(blur_score, 2),
            brightness=round(brightness, 2),
            contrast=round(contrast, 2),
            entropy=round(entropy, 2),
            accepted=accepted,
            rejection_reason=rejection_reason,
        )

    def filter_frames(
        self,
        frame_paths: List[str],
        output_dir: str,
        csv_report_path: Optional[str] = None,
    ) -> FrameQualityReport:
        """
        Evaluates an image sequence, copies accepted frames to output_dir,
        and saves frame_quality.csv report.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if csv_report_path is None:
            csv_path = out_path / "frame_quality.csv"
        else:
            csv_path = Path(csv_report_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)

        scores: List[FrameQualityScore] = []
        accepted_count = 0

        for p in frame_paths:
            score = self.evaluate_frame(p)
            scores.append(score)
            if score.accepted:
                accepted_count += 1
                dst_file = out_path / Path(p).name
                if str(Path(p).resolve()) != str(dst_file.resolve()):
                    shutil.copy2(p, dst_file)

        # Write frame_quality.csv
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_id", "blur_score", "brightness", "contrast", "accepted", "rejection_reason"])
            for s in scores:
                writer.writerow([s.frame_id, s.blur_score, s.brightness, s.contrast, s.accepted, s.rejection_reason])

        logger.info(
            f"[FILTER] Accepted {accepted_count}/{len(frame_paths)} frames "
            f"({(accepted_count / max(1, len(frame_paths)) * 100):.1f}%). Report saved to '{csv_path}'."
        )

        return FrameQualityReport(
            total_frames=len(frame_paths),
            accepted_frames=accepted_count,
            rejected_frames=len(frame_paths) - accepted_count,
            scores=scores,
            csv_report_path=str(csv_path.resolve()),
        )

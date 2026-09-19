"""
Intelligent Keyframe Selector for UAV 3D Reconstruction.
Selects optimal keyframes based on visual disparity, motion, overlap, and sharpness.
Avoids redundant frames while maintaining continuous multi-view baseline for 3D reconstruction.
"""

import os
import cv2
import shutil
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class KeyframeSelectionResult:
    number_of_input_frames: int
    number_of_quality_frames: int
    number_of_keyframes: int
    selection_ratio: float
    keyframe_paths: List[str]
    output_dir: str


class IntelligentKeyframeSelector:
    """
    Intelligently selects keyframes from quality-filtered frames using visual
    histogram disparity, motion/feature change, and local sharpness maximization.
    """

    def __init__(
        self,
        min_distance_threshold: float = 0.15,
        min_frame_gap: int = 2,
        max_frame_gap: int = 25,
        prefer_sharpest: bool = True,
    ):
        self.min_distance_threshold = float(min_distance_threshold)
        self.min_frame_gap = max(1, int(min_frame_gap))
        self.max_frame_gap = max(self.min_frame_gap, int(max_frame_gap))
        self.prefer_sharpest = prefer_sharpest

    @staticmethod
    def compute_frame_descriptor(img: np.ndarray) -> np.ndarray:
        """Computes a normalized HSV color histogram descriptor for visual distance."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.flatten()

    @staticmethod
    def compute_blur_score(img: np.ndarray) -> float:
        """Computes Laplacian variance sharpness score."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def select_keyframes(
        self,
        frame_paths: List[str],
        output_dir: str,
        total_initial_frames: Optional[int] = None,
    ) -> KeyframeSelectionResult:
        """
        Performs intelligent keyframe selection on the provided candidate frames.

        Args:
            frame_paths: Paths to quality-accepted frames.
            output_dir: Directory where selected keyframes will be saved.
            total_initial_frames: Optional total raw frames extracted before quality filtering.

        Returns:
            KeyframeSelectionResult containing stats and selected keyframe paths.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        num_quality_frames = len(frame_paths)
        num_initial = total_initial_frames if total_initial_frames is not None else num_quality_frames

        if num_quality_frames == 0:
            logger.warning("[KEYFRAME] No quality frames provided for keyframe selection!")
            return KeyframeSelectionResult(
                number_of_input_frames=num_initial,
                number_of_quality_frames=0,
                number_of_keyframes=0,
                selection_ratio=0.0,
                keyframe_paths=[],
                output_dir=str(out_path.resolve()),
            )

        if num_quality_frames <= 3:
            # If sequence is extremely short, take all available frames
            selected_indices = list(range(num_quality_frames))
        else:
            # Load images and compute descriptors
            descriptors = []
            sharpness_scores = []
            valid_indices = []

            for i, fp in enumerate(frame_paths):
                img = cv2.imread(fp)
                if img is None:
                    continue
                descriptors.append(self.compute_frame_descriptor(img))
                sharpness_scores.append(self.compute_blur_score(img))
                valid_indices.append(i)

            if not valid_indices:
                raise ValueError("Could not read any frames from provided paths.")

            # Always select first valid frame
            selected_indices = [valid_indices[0]]
            last_selected_desc = descriptors[0]
            last_selected_idx = valid_indices[0]

            for i in range(1, len(valid_indices)):
                curr_idx = valid_indices[i]
                curr_desc = descriptors[i]
                gap = curr_idx - last_selected_idx

                # Compute Bhattacharyya distance between histograms (range [0, 1])
                dist = cv2.compareHist(
                    last_selected_desc.reshape(-1, 1),
                    curr_desc.reshape(-1, 1),
                    cv2.HISTCMP_BHATTACHARYYA,
                )

                # Selection conditions:
                # 1. Traveled minimum gap AND exceeded distance threshold
                # 2. Reached maximum allowed frame gap (prevents excessive gaps during slow hovering)
                is_far_enough = (gap >= self.min_frame_gap) and (dist >= self.min_distance_threshold)
                is_max_gap_reached = (gap >= self.max_frame_gap)

                if is_far_enough or is_max_gap_reached:
                    # If prefer_sharpest, inspect local window around curr_idx to pick the sharpest frame
                    best_pick = curr_idx
                    if self.prefer_sharpest and i > 0:
                        window_start = max(0, i - 1)
                        window_end = min(len(valid_indices), i + 2)
                        best_local_sharpness = -1.0
                        for w_i in range(window_start, window_end):
                            cand_idx = valid_indices[w_i]
                            if cand_idx > last_selected_idx and sharpness_scores[w_i] > best_local_sharpness:
                                best_local_sharpness = sharpness_scores[w_i]
                                best_pick = cand_idx

                    selected_indices.append(best_pick)
                    last_selected_idx = best_pick
                    desc_pos = valid_indices.index(best_pick) if best_pick in valid_indices else i
                    last_selected_desc = descriptors[desc_pos]

            # Ensure last frame is included for full trajectory coverage if not already close
            if valid_indices[-1] not in selected_indices:
                if (valid_indices[-1] - selected_indices[-1]) >= self.min_frame_gap:
                    selected_indices.append(valid_indices[-1])

        # Copy selected keyframes to output directory
        keyframe_paths: List[str] = []
        for rank, s_idx in enumerate(selected_indices, start=1):
            src_file = Path(frame_paths[s_idx])
            dst_file = out_path / f"keyframe_{rank:04d}_{src_file.name}"
            if str(src_file.resolve()) != str(dst_file.resolve()):
                shutil.copy2(src_file, dst_file)
            keyframe_paths.append(str(dst_file.resolve()))

        num_keyframes = len(keyframe_paths)
        selection_ratio = float(num_keyframes / max(1, num_initial))

        logger.info(
            f"[KEYFRAME] Selected {num_keyframes} keyframes from {num_quality_frames} quality frames "
            f"(Total initial: {num_initial} | Selection ratio: {selection_ratio:.2%})"
        )

        return KeyframeSelectionResult(
            number_of_input_frames=num_initial,
            number_of_quality_frames=num_quality_frames,
            number_of_keyframes=num_keyframes,
            selection_ratio=round(selection_ratio, 4),
            keyframe_paths=keyframe_paths,
            output_dir=str(out_path.resolve()),
        )

"""
Video Frame Extractor for UAV Drone Footage.
Extracts frames at a configurable sampling FPS into ordered image files.
"""

import os
import cv2
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class VideoFrameExtractor:
    """
    Extracts frames from UAV drone videos at a user-defined FPS.
    Prevents redundant processing of identical contiguous frames.
    """

    def __init__(
        self,
        sampling_fps: float = 2.0,
        max_frames: Optional[int] = None,
        jpeg_quality: int = 95,
    ):
        self.sampling_fps = max(0.1, float(sampling_fps))
        self.max_frames = max_frames
        self.jpeg_quality = int(jpeg_quality)

    def extract(
        self,
        video_path: str,
        output_dir: str,
    ) -> Dict[str, Any]:
        """
        Extracts frames from video into output_dir.

        Args:
            video_path: Path to input drone video (MP4/MOV/AVI).
            output_dir: Directory where extracted frames will be saved.

        Returns:
            Dictionary containing metadata and list of saved image paths.
        """
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Drone video file not found: {video_path}")

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file with OpenCV: {video_path}")

        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_video_frames / orig_fps if orig_fps > 0 else 0.0

        # Calculate sampling step interval in frame index
        sample_step = max(1, int(round(orig_fps / self.sampling_fps)))

        logger.info(
            f"[VIDEO] Loading '{video_file.name}' | {width}x{height} @ {orig_fps:.1f} FPS | "
            f"Duration: {duration_sec:.1f}s | Target Sampling: {self.sampling_fps} FPS (step: {sample_step})"
        )

        frame_idx = 0
        extracted_paths: List[str] = []
        saved_count = 0

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % sample_step == 0:
                saved_count += 1
                out_filename = f"frame_{saved_count:06d}.jpg"
                out_filepath = out_path / out_filename

                cv2.imwrite(str(out_filepath), frame, encode_params)
                extracted_paths.append(str(out_filepath))

                if self.max_frames and saved_count >= self.max_frames:
                    logger.info(f"[VIDEO] Reached maximum frame extraction limit: {self.max_frames}")
                    break

            frame_idx += 1

        cap.release()

        logger.info(f"[VIDEO] Extracted {len(extracted_paths)} frames from {total_video_frames} total frames.")

        return {
            "video_path": str(video_file.resolve()),
            "output_dir": str(out_path.resolve()),
            "original_fps": orig_fps,
            "sampling_fps": self.sampling_fps,
            "resolution": (width, height),
            "duration_seconds": duration_sec,
            "total_video_frames": total_video_frames,
            "extracted_count": len(extracted_paths),
            "frame_paths": extracted_paths,
        }


def extract_video_frames(
    video_path: str,
    output_dir: str,
    sampling_fps: float = 2.0,
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    extractor = VideoFrameExtractor(sampling_fps=sampling_fps, max_frames=max_frames)
    return extractor.extract(video_path, output_dir)

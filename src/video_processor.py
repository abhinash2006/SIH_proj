import os
import cv2
import glob
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

class VideoProcessor:
    """
    Handles video inspection, metadata extraction, and frame extraction
    with support for both video files and image directories.
    """
    
    @staticmethod
    def is_video_file(path: Union[str, Path]) -> bool:
        path = Path(path)
        return path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS

    @staticmethod
    def is_image_folder(path: Union[str, Path]) -> bool:
        path = Path(path)
        if not path.is_dir():
            return False
        for f in path.rglob("*"):
            if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                return True
        return False

    @staticmethod
    def process_image_folder(folder_path: Union[str, Path]) -> List[str]:
        folder_path = Path(folder_path)
        image_files = sorted([
            str(p) for p in folder_path.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ])
        return image_files

    @staticmethod
    def get_video_metadata(video_path: Union[str, Path]) -> Dict[str, Union[int, float, str]]:
        video_path = str(video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0.0
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        cap.release()
        
        metadata = {
            "fps": round(fps, 2),
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_seconds": round(duration, 2),
            "codec": codec,
            "resolution": f"{width}x{height}",
            "file_size_mb": round(os.path.getsize(video_path) / (1024 * 1024), 2)
        }
        return metadata

    @staticmethod
    def extract_frames(
        input_source: Union[str, Path],
        output_dir: Union[str, Path],
        target_fps: Optional[float] = 2.0,
        max_frames: Optional[int] = 100,
        min_frame_gap: int = 1
    ) -> List[str]:
        """
        Extract frames from a video or copy/index from an image folder.
        """
        input_source = Path(input_source)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_frame_paths = []
        
        if input_source.is_dir():
            # Image folder input (recursive search)
            logger.info(f"[INFO] Processing image folder input: {input_source}")
            image_files = sorted([
                p for p in input_source.rglob("*")
                if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ])
            if not image_files:
                raise ValueError(f"No valid images found in directory: {input_source}")
                
            # Sample if exceeding max_frames
            step = 1
            if max_frames and len(image_files) > max_frames:
                step = max(1, len(image_files) // max_frames)
            
            selected_files = image_files[::step]
            if max_frames:
                selected_files = selected_files[:max_frames]
                
            for idx, img_path in enumerate(selected_files):
                img = cv2.imread(str(img_path))
                if img is None:
                    logger.warning(f"[WARNING] Skipping unreadable image: {img_path}")
                    continue
                out_name = f"frame_{idx:05d}.jpg"
                out_path = output_dir / out_name
                cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved_frame_paths.append(str(out_path))
                
            logger.info(f"[SUCCESS] Prepared {len(saved_frame_paths)} frames from directory.")
            return saved_frame_paths

        elif VideoProcessor.is_video_file(input_source):
            # Video file input
            cap = cv2.VideoCapture(str(input_source))
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {input_source}")
                
            orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if target_fps and target_fps > 0 and target_fps < orig_fps:
                frame_interval = max(min_frame_gap, int(round(orig_fps / target_fps)))
            else:
                frame_interval = max(1, min_frame_gap)
                
            logger.info(
                f"[INFO] Extracting video: {input_source.name} | Total frames: {total_frames} | "
                f"FPS: {orig_fps:.1f} -> Target: {target_fps} FPS (interval: every {frame_interval} frames)"
            )
            
            frame_idx = 0
            saved_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_idx % frame_interval == 0:
                    out_name = f"frame_{saved_count:05d}.jpg"
                    out_path = output_dir / out_name
                    cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved_frame_paths.append(str(out_path))
                    saved_count += 1
                    
                    if max_frames and saved_count >= max_frames:
                        break
                        
                frame_idx += 1
                
            cap.release()
            logger.info(f"[SUCCESS] Extracted {len(saved_frame_paths)} frames to {output_dir}")
            return saved_frame_paths
        else:
            raise ValueError(f"Unsupported input source: {input_source}. Must be video file or folder with images.")

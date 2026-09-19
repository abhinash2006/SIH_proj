import os
import cv2
import numpy as np
import pytest
from pathlib import Path
from src.video.extractor import VideoFrameExtractor, extract_video_frames


@pytest.fixture
def dummy_video(tmp_path):
    video_path = tmp_path / "test_drone.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30.0
    width, height = 320, 240
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for i in range(60):  # 2 seconds at 30 fps
        frame = np.full((height, width, 3), i * 4, dtype=np.uint8)
        cv2.putText(frame, f"Frame {i}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)

    out.release()
    return str(video_path)


def test_extractor_sampling_fps(dummy_video, tmp_path):
    out_dir = tmp_path / "extracted_frames"
    extractor = VideoFrameExtractor(sampling_fps=2.0)
    result = extractor.extract(dummy_video, str(out_dir))

    assert result["extracted_count"] > 0
    assert result["extracted_count"] <= 6  # 2 seconds * 2 fps approx
    assert len(result["frame_paths"]) == result["extracted_count"]
    for fp in result["frame_paths"]:
        assert os.path.exists(fp)


def test_extractor_missing_file(tmp_path):
    extractor = VideoFrameExtractor()
    with pytest.raises(FileNotFoundError):
        extractor.extract(str(tmp_path / "non_existent.mp4"), str(tmp_path / "out"))

import os
import cv2
import numpy as np
import pytest
from pathlib import Path
from src.video.quality import FrameQualityFilter


@pytest.fixture
def sample_frames(tmp_path):
    frames_dir = tmp_path / "raw_frames"
    frames_dir.mkdir()

    # 1. Sharp frame
    sharp_img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(sharp_img, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.circle(sharp_img, (100, 100), 30, (0, 0, 255), -1)
    sharp_path = frames_dir / "sharp.jpg"
    cv2.imwrite(str(sharp_path), sharp_img)

    # 2. Extremely blurry frame
    blurry_img = cv2.GaussianBlur(sharp_img, (51, 51), 30)
    blurry_path = frames_dir / "blurry.jpg"
    cv2.imwrite(str(blurry_path), blurry_img)

    # 3. Severely underexposed frame
    dark_img = np.full((200, 200, 3), 10, dtype=np.uint8)
    dark_path = frames_dir / "dark.jpg"
    cv2.imwrite(str(dark_path), dark_img)

    return [str(sharp_path), str(blurry_path), str(dark_path)]


def test_quality_eval(sample_frames, tmp_path):
    qfilter = FrameQualityFilter(min_blur_score=50.0, min_brightness=25.0)
    out_dir = tmp_path / "selected"
    report = qfilter.filter_frames(sample_frames, str(out_dir))

    assert report.total_frames == 3
    assert report.accepted_frames >= 1
    assert os.path.exists(report.csv_report_path)

    # Check scores
    scores_by_id = {s.frame_id: s for s in report.scores}
    assert scores_by_id["sharp"].accepted is True
    assert scores_by_id["blurry"].accepted is False
    assert scores_by_id["blurry"].rejection_reason == "excessive_blur"
    assert scores_by_id["dark"].accepted is False
    assert scores_by_id["dark"].rejection_reason in ("underexposed", "excessive_blur")

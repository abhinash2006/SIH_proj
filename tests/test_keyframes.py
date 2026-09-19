import os
import cv2
import numpy as np
import pytest
from pathlib import Path
from src.video.keyframes import IntelligentKeyframeSelector


@pytest.fixture
def sequence_frames(tmp_path):
    seq_dir = tmp_path / "seq"
    seq_dir.mkdir()
    paths = []

    # Generate 15 frames with motion and visual variation
    for i in range(15):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Moving square
        x = int(i * 5)
        cv2.rectangle(img, (x, 20), (x + 30, 80), (0, 255, i * 15), -1)
        fpath = seq_dir / f"frame_{i:03d}.jpg"
        cv2.imwrite(str(fpath), img)
        paths.append(str(fpath))

    return paths


def test_intelligent_keyframe_selection(sequence_frames, tmp_path):
    selector = IntelligentKeyframeSelector(min_distance_threshold=0.05, min_frame_gap=2, max_frame_gap=6)
    out_dir = tmp_path / "keyframes"
    res = selector.select_keyframes(sequence_frames, str(out_dir), total_initial_frames=20)

    assert res.number_of_keyframes > 1
    assert res.number_of_keyframes < len(sequence_frames)
    assert res.selection_ratio > 0.0
    for kp in res.keyframe_paths:
        assert os.path.exists(kp)

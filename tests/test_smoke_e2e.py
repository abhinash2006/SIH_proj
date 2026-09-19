import os
import cv2
import csv
import numpy as np
import pytest
from pathlib import Path
from src.pipeline.runner import PipelineRunner, PipelineConfig, MissionResult


@pytest.fixture
def synthetic_mission_data(tmp_path):
    # 1. Generate 5 synthetic aerial frames
    frames_dir = tmp_path / "drone_images"
    frames_dir.mkdir()
    frame_paths = []
    for i in range(5):
        img = np.zeros((150, 150, 3), dtype=np.uint8)
        # Terrain background
        img[:, :] = [40, 100, 40]
        # Moving box simulating flight over terrain
        x_pos = int(20 + i * 15)
        cv2.rectangle(img, (x_pos, 40), (x_pos + 40, 90), (180, 180, 180), -1)
        fpath = frames_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(fpath), img)
        frame_paths.append(str(fpath))

    # 2. Generate GPS CSV
    gps_path = tmp_path / "flight_gps.csv"
    with open(gps_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "latitude", "longitude", "altitude"])
        for i in range(5):
            writer.writerow([i * 1.0, 12.9716 + i * 0.0001, 77.5946 + i * 0.0001, 50.0 + i * 0.2])

    return str(frames_dir), str(gps_path)


def test_end_to_end_smoke_pipeline(synthetic_mission_data, tmp_path):
    frames_dir, gps_csv = synthetic_mission_data

    runner = PipelineRunner(
        config=PipelineConfig(
            reconstruction_mode="hybrid",
            vggt_backend="mock",  # Use mock VGGT for lightning fast smoke testing
        )
    )

    result: MissionResult = runner.run(
        input_video_or_images=frames_dir,
        gps_csv_path=gps_csv,
        mission_id="smoke_test_run",
    )

    assert result.status in ("success", "completed_with_warnings")
    assert os.path.exists(result.artifacts.final_ply)
    assert os.path.exists(result.artifacts.final_obj)
    assert os.path.exists(result.artifacts.metrics_json)
    assert os.path.exists(result.artifacts.processing_report_json)
    assert result.metrics["num_points"] > 0
    assert result.metrics["num_triangles"] > 0

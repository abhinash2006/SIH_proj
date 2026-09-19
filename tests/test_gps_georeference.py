import os
import csv
import numpy as np
import pytest
from src.georeference.gps import GPSTelemetryParser, parse_gps_csv, GPSTrajectory
from src.georeference.alignment import GeoreferenceAligner, GeoreferenceTransform


@pytest.fixture
def sample_gps_csv(tmp_path):
    csv_file = tmp_path / "drone_gps.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "latitude", "longitude", "altitude"])
        # Generate 10 points moving north-east
        for i in range(10):
            writer.writerow([i * 1.0, 28.6139 + i * 0.0001, 77.2090 + i * 0.0001, 100.0 + i * 0.5])
    return str(csv_file)


def test_gps_parser(sample_gps_csv):
    traj: GPSTrajectory = parse_gps_csv(sample_gps_csv)
    assert len(traj.points) == 10
    assert traj.positions_enu.shape == (10, 3)
    assert traj.total_distance_m > 0.0
    # First point should be origin (0, 0, 0)
    np.testing.assert_allclose(traj.positions_enu[0], [0.0, 0.0, 0.0], atol=1e-3)


def test_umeyama_alignment():
    aligner = GeoreferenceAligner()

    # Synthetic source points (e.g. 5 camera centers)
    src = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [3.0, 1.0, 1.0],
        [4.0, 2.0, 1.0],
    ])

    # Ground truth transform: scale=5.0, shift=[10, 20, 30]
    true_s = 5.0
    true_t = np.array([10.0, 20.0, 30.0])
    dst = true_s * src + true_t

    s, R, t = aligner.umeyama_alignment(src, dst)

    assert np.isclose(s, true_s, atol=1e-3)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-3)
    np.testing.assert_allclose(t, true_t, atol=1e-3)

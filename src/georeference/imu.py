"""
IMU Telemetry Parser for UAV Flight Data.
Parses accelerometer, gyroscope, and orientation (roll, pitch, yaw) logs.
Provides independent trajectory and orientation streams.
"""

import os
import csv
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class IMUReading:
    timestamp: float
    accel: np.ndarray      # [ax, ay, az] in m/s^2
    gyro: np.ndarray       # [gx, gy, gz] in rad/s
    orientation: Optional[np.ndarray] # [roll, pitch, yaw] in degrees or radians


@dataclass
class IMUTrajectory:
    readings: List[IMUReading]
    timestamps: np.ndarray
    accelerations: np.ndarray  # [N, 3]
    angular_velocities: np.ndarray  # [N, 3]
    orientations: Optional[np.ndarray]  # [N, 3]


class IMUTelemetryParser:
    """
    Parses and standardizes IMU sensor logs from UAV flight controllers.
    """

    def parse_csv(self, csv_file_path: str) -> IMUTrajectory:
        csv_p = Path(csv_file_path)
        if not csv_p.exists():
            raise FileNotFoundError(f"IMU CSV file not found: {csv_file_path}")

        readings: List[IMUReading] = []
        with open(csv_p, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            field_map = {f.strip().lower(): f for f in reader.fieldnames or []}

            for idx, row in enumerate(reader):
                try:
                    ts = float(row.get(field_map.get("timestamp", "timestamp"), idx))

                    # Accelerometer
                    ax = float(row.get(field_map.get("ax", "accel_x"), 0.0))
                    ay = float(row.get(field_map.get("ay", "accel_y"), 0.0))
                    az = float(row.get(field_map.get("az", "accel_z"), 9.81))

                    # Gyroscope
                    gx = float(row.get(field_map.get("gx", "gyro_x"), 0.0))
                    gy = float(row.get(field_map.get("gy", "gyro_y"), 0.0))
                    gz = float(row.get(field_map.get("gz", "gyro_z"), 0.0))

                    # Orientation (roll, pitch, yaw)
                    roll = float(row.get(field_map.get("roll", "roll_deg"), 0.0))
                    pitch = float(row.get(field_map.get("pitch", "pitch_deg"), 0.0))
                    yaw = float(row.get(field_map.get("yaw", "yaw_deg"), 0.0))

                    readings.append(
                        IMUReading(
                            timestamp=ts,
                            accel=np.array([ax, ay, az], dtype=np.float32),
                            gyro=np.array([gx, gy, gz], dtype=np.float32),
                            orientation=np.array([roll, pitch, yaw], dtype=np.float32),
                        )
                    )
                except (ValueError, KeyError):
                    continue

        if not readings:
            raise ValueError(f"No valid IMU entries found in {csv_file_path}")

        readings.sort(key=lambda r: r.timestamp)
        N = len(readings)
        timestamps = np.array([r.timestamp for r in readings], dtype=np.float64)
        accels = np.stack([r.accel for r in readings], axis=0)
        gyros = np.stack([r.gyro for r in readings], axis=0)
        orientations = np.stack([r.orientation for r in readings], axis=0)

        logger.info(f"[IMU] Parsed {N} IMU samples spanning {timestamps[-1] - timestamps[0]:.1f}s")

        return IMUTrajectory(
            readings=readings,
            timestamps=timestamps,
            accelerations=accels,
            angular_velocities=gyros,
            orientations=orientations,
        )


def parse_imu_csv(csv_path: str) -> IMUTrajectory:
    parser = IMUTelemetryParser()
    return parser.parse_csv(csv_path)

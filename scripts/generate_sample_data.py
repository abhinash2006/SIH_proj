"""
Synthetic UAV Flight Dataset Generator.
Creates a sample aerial video, GPS telemetry CSV, IMU CSV, and camera.json
for testing the single-pass reconstruction pipeline without requiring hardware flight.
"""

import os
import cv2
import csv
import json
import numpy as np
from pathlib import Path

OUT_DIR = Path("data/raw/sample_mission")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_PATH = OUT_DIR / "synthetic_drone_flight.mp4"
GPS_PATH = OUT_DIR / "flight_gps.csv"
IMU_PATH = OUT_DIR / "flight_imu.csv"
CAMERA_PATH = OUT_DIR / "camera.json"


def generate_video(num_frames=60, width=640, height=480, fps=15):
    print(f"[GEN] Generating synthetic video ({width}x{height} @ {fps}fps, {num_frames} frames)...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(VIDEO_PATH), fourcc, fps, (width, height))

    for i in range(num_frames):
        # Base terrain: grassy green with grid texture
        frame = np.full((height, width, 3), [45, 95, 45], dtype=np.uint8)

        # Draw grid roads / terrain features
        for x in range(0, width, 80):
            cv2.line(frame, (x, 0), (x, height), (70, 70, 70), 2)
        for y in range(0, height, 80):
            cv2.line(frame, (y, 0), (width, y), (70, 70, 70), 2)

        # Simulated stationary building structure
        cv2.rectangle(frame, (200, 150), (320, 260), (140, 140, 150), -1)
        cv2.putText(frame, "BUILDING", (210, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Moving car along horizontal road (transient dynamic object)
        car_x = int((i * 12) % (width + 60) - 30)
        car_y = 280
        cv2.rectangle(frame, (car_x, car_y), (car_x + 50, car_y + 24), (0, 0, 220), -1)
        cv2.putText(frame, "CAR", (car_x + 8, car_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Simulated camera motion shift (drone forward flight)
        offset_y = int(i * 3)
        M = np.float32([[1, 0, 0], [0, 1, -offset_y % 80]])
        frame = cv2.warpAffine(frame, M, (width, height), borderMode=cv2.BORDER_WRAP)

        out.write(frame)

    out.release()
    print(f"[GEN] Video saved to '{VIDEO_PATH}'.")


def generate_gps(num_points=60, duration_sec=4.0):
    print(f"[GEN] Generating GPS log ({num_points} samples)...")
    base_lat = 28.613939
    base_lon = 77.209021
    base_alt = 45.0

    with open(GPS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "latitude", "longitude", "altitude"])
        for i in range(num_points):
            ts = round((i / num_points) * duration_sec, 2)
            lat = base_lat + (i * 0.00002)
            lon = base_lon + (i * 0.000015)
            alt = round(base_alt + np.sin(i * 0.1) * 1.5, 2)
            writer.writerow([ts, f"{lat:.7f}", f"{lon:.7f}", alt])

    print(f"[GEN] GPS log saved to '{GPS_PATH}'.")


def generate_imu(num_points=60, duration_sec=4.0):
    print(f"[GEN] Generating IMU sensor stream...")
    with open(IMU_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "ax", "ay", "az", "gx", "gy", "gz", "roll", "pitch", "yaw"])
        for i in range(num_points):
            ts = round((i / num_points) * duration_sec, 2)
            ax = round(np.sin(i * 0.2) * 0.3, 3)
            ay = round(0.5 + np.cos(i * 0.2) * 0.2, 3)
            az = round(9.81 + np.sin(i * 0.1) * 0.1, 3)
            writer.writerow([ts, ax, ay, az, 0.01, 0.02, 0.0, 1.2, -3.5, 45.0 + i * 0.5])

    print(f"[GEN] IMU log saved to '{IMU_PATH}'.")


def generate_camera():
    cam_data = {
        "camera_model": "PINHOLE",
        "width": 640,
        "height": 480,
        "fx": 580.0,
        "fy": 580.0,
        "cx": 320.0,
        "cy": 240.0,
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0],
    }
    with open(CAMERA_PATH, "w", encoding="utf-8") as f:
        json.dump(cam_data, f, indent=2)
    print(f"[GEN] Camera intrinsics saved to '{CAMERA_PATH}'.")


if __name__ == "__main__":
    generate_video()
    generate_gps()
    generate_imu()
    generate_camera()
    print("\n[SUCCESS] Sample UAV flight dataset generated successfully in 'data/raw/sample_mission/'!")

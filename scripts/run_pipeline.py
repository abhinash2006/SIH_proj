"""
Command-Line Interface (CLI) Runner for UAV 3D Reconstruction System.
Usage:
    python scripts/run_pipeline.py --input data/raw/videos/drone_flight_sample.mp4 --gps data/raw/sample_mission/flight_gps.csv --mode hybrid
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Ensure workspace root is in sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.pipeline.runner import PipelineRunner, PipelineConfig, MissionResult

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("AeroScan_CLI")


def main():
    parser = argparse.ArgumentParser(
        description="Single-Pass Drone Video to 3D Model Generation (AeroScan 3D / SIH UAV Reconstruction)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input drone video (MP4) or image directory")
    parser.add_argument("--gps", "-g", type=str, default=None, help="Path to GPS telemetry CSV (timestamp, latitude, longitude, altitude)")
    parser.add_argument("--imu", type=str, default=None, help="Path to IMU telemetry CSV")
    parser.add_argument("--camera", "-c", type=str, default=None, help="Path to camera calibration camera.json")
    parser.add_argument("--mode", "-m", type=str, default="hybrid", choices=["hybrid", "vggt", "colmap"], help="Reconstruction pipeline mode")
    parser.add_argument("--vggt-backend", type=str, default="pretrained", choices=["pretrained", "finetuned", "mock"], help="VGGT architecture backend")
    parser.add_argument("--vggt-checkpoint", type=str, default=None, help="Custom VGGT weights checkpoint path")
    parser.add_argument("--yolo-checkpoint", type=str, default="yolo11n.pt", help="YOLO weights checkpoint")
    parser.add_argument("--fps", type=float, default=2.0, help="Frame extraction sampling rate (frames per second)")
    parser.add_argument("--mission-id", type=str, default=None, help="Custom mission identifier")

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("AeroScan 3D -- Autonomous UAV Reconstruction Pipeline (SIH)")
    print("=" * 80)
    print(f"  * Input Source     : {args.input}")
    print(f"  * Telemetry (GPS)  : {args.gps or 'None (relative mode)'}")
    print(f"  * Telemetry (IMU)  : {args.imu or 'None'}")
    print(f"  * Pipeline Mode    : {args.mode.upper()}")
    print(f"  * VGGT Backend     : {args.vggt_backend}")
    print(f"  * Sampling Rate    : {args.fps} FPS")
    print("=" * 80 + "\n")

    cfg = PipelineConfig(
        reconstruction_mode=args.mode,
        vggt_backend=args.vggt_backend,
        vggt_checkpoint=args.vggt_checkpoint,
        yolo_checkpoint=args.yolo_checkpoint,
        frame_sampling_fps=args.fps,
    )

    runner = PipelineRunner(config=cfg)

    result: MissionResult = runner.run(
        input_video_or_images=args.input,
        gps_csv_path=args.gps,
        imu_csv_path=args.imu,
        camera_json_path=args.camera,
        mission_id=args.mission_id,
    )

    print("\n" + "=" * 80)
    print(f"[SUCCESS] RECONSTRUCTION COMPLETE -- Mission: '{result.mission_id}'")
    print("=" * 80)
    print(f"  * Status             : {result.status.upper()}")
    print(f"  * 3D Points          : {result.metrics.get('num_points', 0):,}")
    print(f"  * Mesh Triangles     : {result.metrics.get('num_triangles', 0):,}")
    print(f"  * Processing Time    : {result.metrics.get('total_processing_time_s', 0):.2f}s")
    print(f"  * Peak VRAM          : {result.metrics.get('peak_vram_mb', 0):.1f} MB")
    print(f"  * Web 3D Model (GLB) : {result.artifacts.final_glb}")
    print(f"  * Point Cloud (PLY)  : {result.artifacts.final_ply}")
    print(f"  * Full Report        : {result.artifacts.processing_report_json}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

import os
import sys
import time
import logging
from pathlib import Path

# Ensure local source directory is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from scripts.run_pipeline import run_pipeline
from app import create_ui

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("SIH_DisasterDemo")

def run_sih_demo():
    print("=" * 65)
    print(" 🛸 DRONE-VGGT: AI DISASTER INSPECTION & 3D RESCUE MAPPING SYSTEM")
    print(" SIH Automated Demonstration Mode")
    print("=" * 65)

    sample_dir = Path("data/raw/uav_sequence/1121222322212102-4/images")
    if not sample_dir.exists():
        print("[INFO] Fetching benchmark UAV dataset...")
        from scripts.download_dataset import download_uavlight_dataset
        download_uavlight_dataset()

    print(f"\n[STEP 1] Executing End-to-End Disaster Inspection Pipeline on: {sample_dir}")
    run_pipeline(
        input_path=str(sample_dir),
        output_dir="outputs/sih_demo",
        mode="all",
        max_frames=12,
        use_depth_anything=True,
        depth_model_size="small"
    )

    print("\n[STEP 2] Launching Interactive 3D Disaster Dashboard...")
    demo = create_ui()
    print("=" * 65)
    print(" 🚀 SIH DEMO READY: Open http://127.0.0.1:7860 in your browser")
    print(" HTML Report: outputs/sih_demo/experiment_D/disaster_report.html")
    print("=" * 65)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)

if __name__ == "__main__":
    run_sih_demo()

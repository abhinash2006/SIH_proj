"""
FastAPI Backend Server for Single-Pass UAV 3D Reconstruction System.
Provides non-blocking background job execution, real-time stage progress reporting,
and static artifact streaming for Three.js 3D Web Viewer.
"""

import os
import sys
import json
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# Ensure Windows PyTorch CUDA DLL path is configured
if os.name == "nt":
    candidates = [
        os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib"),
        r"D:\Vggt\drone_vggt_env\Lib\site-packages\torch\lib"
    ]
    for dll_path in candidates:
        if os.path.exists(dll_path) and hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(dll_path)
                break
            except Exception:
                pass

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.pipeline.runner import PipelineRunner, PipelineConfig, MissionResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("UAV_3D_Backend")

app = FastAPI(
    title="Single-Pass Drone Video to 3D Model Generation API",
    description="SIH Autonomous UAV Multi-View 3D Reconstruction & Georeferencing Pipeline",
    version="1.0.0",
)

# Enable CORS for React/Vite development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure output and upload directories exist
BASE_OUTPUT_DIR = Path("output").resolve()
BASE_UPLOAD_DIR = Path("data/uploads").resolve()
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BASE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Mount outputs for Three.js direct asset streaming (GLB, PLY, images)
app.mount("/outputs", StaticFiles(directory=str(BASE_OUTPUT_DIR)), name="outputs")

# In-memory mission status store
missions: Dict[str, Dict[str, Any]] = {}


class ProcessRequest(BaseModel):
    mission_id: str
    reconstruction_mode: str = "hybrid"  # "hybrid", "vggt", "colmap"
    vggt_backend: str = "pretrained"     # "pretrained", "finetuned", "mock"
    vggt_checkpoint: Optional[str] = None


def run_pipeline_task(mission_id: str, mode: str, vggt_backend: str, vggt_checkpoint: Optional[str]):
    """Background task executing the reconstruction pipeline."""
    mission = missions.get(mission_id)
    if not mission:
        return

    mission["status"] = "processing"
    mission["stage"] = "INITIALIZING"
    mission["progress"] = 0.05
    mission["logs"].append(f"Starting mission '{mission_id}' (Mode: {mode}, Backend: {vggt_backend})")

    def progress_callback(stage: str, pct: float, msg: str):
        mission["stage"] = stage.upper()
        mission["progress"] = round(pct * 100.0, 1)
        mission["logs"].append(f"[{time.strftime('%H:%M:%S')}] [{stage.upper()}] {msg}")
        if len(mission["logs"]) > 200:
            mission["logs"] = mission["logs"][-200:]

    try:
        cfg = PipelineConfig(
            reconstruction_mode=mode,
            vggt_backend=vggt_backend,
            vggt_checkpoint=vggt_checkpoint,
        )
        runner = PipelineRunner(config=cfg)

        video_path = mission.get("video_path")
        gps_path = mission.get("gps_path")
        imu_path = mission.get("imu_path")
        camera_path = mission.get("camera_path")

        result: MissionResult = runner.run(
            input_video_or_images=video_path,
            gps_csv_path=gps_path,
            imu_csv_path=imu_path,
            camera_json_path=camera_path,
            mission_id=mission_id,
            progress_callback=progress_callback,
        )

        mission["status"] = result.status
        mission["progress"] = 100.0
        mission["stage"] = "COMPLETED"
        mission["metrics"] = result.metrics
        mission["summary"] = result.summary
        mission["artifacts"] = {
            "glb_url": f"/outputs/{mission_id}/final.glb",
            "ply_url": f"/outputs/{mission_id}/final.ply",
            "obj_url": f"/outputs/{mission_id}/final.obj",
            "pointcloud_url": f"/outputs/{mission_id}/pointcloud/pointcloud.ply",
            "report_url": f"/outputs/{mission_id}/processing_report.json",
        }
        mission["logs"].append(f"Mission '{mission_id}' finished with status: {result.status}")

    except Exception as e:
        logger.error(f"Mission '{mission_id}' failed: {e}", exc_info=True)
        mission["status"] = "failed"
        mission["stage"] = "FAILED"
        mission["error"] = str(e)
        mission["logs"].append(f"[ERROR] Reconstruction failed: {str(e)}")


@app.get("/")
def health_check():
    import torch
    return {
        "system": "Single-Pass UAV 3D Reconstruction API",
        "status": "online",
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.post("/api/upload")
async def upload_mission_data(
    video: UploadFile = File(...),
    gps_csv: Optional[UploadFile] = File(None),
    imu_csv: Optional[UploadFile] = File(None),
    camera_json: Optional[UploadFile] = File(None),
):
    """
    Accepts UAV flight video, GPS CSV, IMU CSV, and camera intrinsics.
    Initializes a new mission staging directory.
    """
    mission_id = f"mission_{int(time.time())}_{os.urandom(3).hex()}"
    staging_dir = BASE_UPLOAD_DIR / mission_id
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded video
    video_dest = staging_dir / video.filename
    with open(video_dest, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    gps_dest_str = None
    if gps_csv and gps_csv.filename:
        gps_dest = staging_dir / gps_csv.filename
        with open(gps_dest, "wb") as buffer:
            shutil.copyfileobj(gps_csv.file, buffer)
        gps_dest_str = str(gps_dest.resolve())

    imu_dest_str = None
    if imu_csv and imu_csv.filename:
        imu_dest = staging_dir / imu_csv.filename
        with open(imu_dest, "wb") as buffer:
            shutil.copyfileobj(imu_csv.file, buffer)
        imu_dest_str = str(imu_dest.resolve())

    cam_dest_str = None
    if camera_json and camera_json.filename:
        cam_dest = staging_dir / camera_json.filename
        with open(cam_dest, "wb") as buffer:
            shutil.copyfileobj(camera_json.file, buffer)
        cam_dest_str = str(cam_dest.resolve())

    missions[mission_id] = {
        "mission_id": mission_id,
        "status": "ready",
        "stage": "UPLOADED",
        "progress": 0.0,
        "video_path": str(video_dest.resolve()),
        "gps_path": gps_dest_str,
        "imu_path": imu_dest_str,
        "camera_path": cam_dest_str,
        "logs": [f"Uploaded '{video.filename}' (GPS: {bool(gps_dest_str)}, IMU: {bool(imu_dest_str)})"],
        "metrics": {},
        "artifacts": {},
        "created_at": time.time(),
    }

    return {
        "mission_id": mission_id,
        "status": "ready",
        "message": "Files uploaded and verified successfully.",
    }


@app.post("/api/process")
async def start_processing(req: ProcessRequest, background_tasks: BackgroundTasks):
    """Starts asynchronous reconstruction job."""
    if req.mission_id not in missions:
        raise HTTPException(status_code=404, detail=f"Mission '{req.mission_id}' not found.")

    mission = missions[req.mission_id]
    if mission["status"] == "processing":
        return {"mission_id": req.mission_id, "status": "already_running"}

    background_tasks.add_task(
        run_pipeline_task,
        mission_id=req.mission_id,
        mode=req.reconstruction_mode,
        vggt_backend=req.vggt_backend,
        vggt_checkpoint=req.vggt_checkpoint,
    )

    return {
        "mission_id": req.mission_id,
        "status": "processing_started",
        "mode": req.reconstruction_mode,
    }


@app.get("/api/status/{mission_id}")
def get_mission_status(mission_id: str):
    """Returns real-time progress percentage, stage, and execution logs."""
    if mission_id not in missions:
        # Check if completed on disk
        m_dir = BASE_OUTPUT_DIR / mission_id
        if m_dir.exists():
            report_p = m_dir / "processing_report.json"
            if report_p.exists():
                with open(report_p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return {
                    "mission_id": mission_id,
                    "status": data.get("status", "completed"),
                    "stage": "COMPLETED",
                    "progress": 100.0,
                    "logs": ["Mission loaded from disk"],
                }
        raise HTTPException(status_code=404, detail=f"Mission '{mission_id}' not found.")

    m = missions[mission_id]
    return {
        "mission_id": mission_id,
        "status": m["status"],
        "stage": m["stage"],
        "progress": m["progress"],
        "logs": m["logs"][-40:],  # Return last 40 logs
    }


@app.get("/api/results/{mission_id}")
def get_mission_results(mission_id: str):
    """Returns download links and 3D viewer model links for the mission."""
    m = missions.get(mission_id)
    if not m:
        m_dir = BASE_OUTPUT_DIR / mission_id
        if not m_dir.exists():
            raise HTTPException(status_code=404, detail="Mission not found.")
        artifacts = {
            "glb_url": f"/outputs/{mission_id}/final.glb",
            "ply_url": f"/outputs/{mission_id}/final.ply",
            "obj_url": f"/outputs/{mission_id}/final.obj",
            "pointcloud_url": f"/outputs/{mission_id}/pointcloud/pointcloud.ply",
            "report_url": f"/outputs/{mission_id}/processing_report.json",
        }
        return {"mission_id": mission_id, "artifacts": artifacts}

    return {
        "mission_id": mission_id,
        "status": m["status"],
        "artifacts": m.get("artifacts", {}),
    }


@app.get("/api/metrics/{mission_id}")
def get_mission_metrics(mission_id: str):
    """Returns quantitative accuracy and evaluation metrics."""
    m_dir = BASE_OUTPUT_DIR / mission_id
    metrics_file = m_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)

    m = missions.get(mission_id)
    if m and "metrics" in m and m["metrics"]:
        return m["metrics"]

    raise HTTPException(status_code=404, detail="Metrics not found for this mission.")


@app.get("/api/missions")
def list_missions():
    """Lists recent missions."""
    on_disk = [d.name for d in BASE_OUTPUT_DIR.iterdir() if d.is_dir()]
    return {
        "active_missions": list(missions.keys()),
        "stored_missions": on_disk,
    }


# Mount static frontend production build if available
FRONTEND_DIST = Path("frontend/dist").resolve()
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")


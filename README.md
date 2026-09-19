# AeroScan 3D — Autonomous UAV Single-Pass 3D Reconstruction System
### Smart India Hackathon (SIH) Platform: Multi-View Aerial Drone Video to Georeferenced 3D Digital Twin

[![Python 3.10-3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.6.0+cu124](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-red.svg)](https://pytorch.org/)
[![Open3D 0.19.0](https://img.shields.io/badge/Open3D-0.19.0-green.svg)](https://www.open3d.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-11n%2Fv8n-yellow.svg)](https://github.com/ultralytics/ultralytics)
[![FastAPI Backend](https://img.shields.io/badge/FastAPI-0.110.0-teal.svg)](https://fastapi.tiangolo.com/)
[![React Three.js](https://img.shields.io/badge/Frontend-React%20%2B%20Three.js-purple.svg)](https://threejs.org/)

---

## 1. Project Overview & Objective

**AeroScan 3D** is an autonomous, single-pass visual geometry reconstruction platform designed for the Smart India Hackathon (SIH). It converts continuous aerial drone video (1080p/4K) along with flight telemetry (GPS, IMU, camera calibration) into metric, georeferenced 3D point clouds, textured surface meshes (GLB/OBJ/PLY), and interactive web visualizations.

### The UAV Reconstruction Challenge
In emergency response, infrastructure inspection, and disaster survey missions:
1. **Dynamic Transient Interference**: Moving vehicles, pedestrians, and animals corrupt multi-view triangulation, creating floating ghosts and distorted surface geometry.
2. **Computational Constraints**: High-resolution video processing must operate reliably on commodity hardware (e.g. laptop RTX 3050 with 4GB VRAM) without out-of-memory crashes.
3. **Metric Georeferencing**: Unscaled camera poses must align with real-world WGS84 GPS coordinates to enable accurate distance and elevation measurements.

---

## 2. System Architecture

```text
                     🚁 Aerial Drone Video (1080p / 4K)
                                   │
                     [1] Frame Extraction (FPS Sampling)
                                   │
                     [2] Frame Quality Filtering (Blur, Contrast, Exposure)
                                   │
                     [3] Intelligent Keyframe Selection
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
   [4] YOLO Detection (Dynamic Objects)      [5] Monocular Depth (Optional)
              │                                (Depth Anything V2)
   [5] SAM 2 Precise Segmentation Masks                 │
              │                                         │
              └────────────────────┬────────────────────┘
                                   │
                     [6] VGGT 3D Geometry Engine
                   (Modular Pretrained / Fine-Tuned)
                                   │
               Intrinsics, Extrinsics, Depth & Point Maps
                                   │
                     [7] Depth Comparison & Fusion
                                   │
               ┌───────────────────┴───────────────────┐
               ▼                                       ▼
     [8] Telemetry Ingestion                  [9] 3D Point Cloud Gen
     (GPS WGS84/ENU + IMU)                     (Filtered static pixels)
               │                                       │
               └───────────────────┬───────────────────┘
                                   │
                   [10] Similarity Georeferencing
                           (Umeyama / RANSAC)
                                   │
                   [11] Point Cloud Filtering (SOR / ROR / Voxel)
                                   │
                   [12] Surface Mesh Reconstruction (Poisson / BPA)
                                   │
                   [13] Sharp-Frame Texture Projection
                                   │
                   [14] Export (GLB / OBJ / PLY)
                                   │
               ┌───────────────────┴───────────────────┐
               ▼                                       ▼
     FastAPI Backend                          React + Three.js
   (Async Background Jobs)                 (Interactive 3D Viewer)
```

---

## 3. Dynamic Object Removal: YOLO + SAM 2 Workflow

Rather than treating SAM 2 as a replacement for YOLO, they operate in synergy:
1. **YOLO (`yolo11n.pt`)**: Fast 2D object detection identifies transient dynamic classes (`person`, `car`, `truck`, `bus`, `motorcycle`, `bicycle`, `animal`) and extracts bounding boxes at original frame resolution. Static infrastructure (buildings, roads, terrain, vegetation, poles) is strictly preserved.
2. **SAM 2 (`sam2_segmenter.py`)**: Takes YOLO bounding boxes as prompt anchors and generates pixel-accurate binary masks `[H, W]`.
3. **Point Cloud Exclusion**: Pixels flagged in the dynamic mask are excluded prior to 3D point cloud unprojection, eliminating phantom dynamic artifacts from the reconstructed 3D mesh.

---

## 4. Modular VGGT Checkpoint Design & Replacement Guide

> [!IMPORTANT]
> **Zero-Disruption Architecture for Fine-Tuned Model Integration**:
> The VGGT model is fine-tuned separately. The application is designed so that VGGT is an isolated, modular component behind a standardized abstract adapter interface (`BaseVGGTAdapter`).

### The Adapter Hierarchy
```text
                     BaseVGGTAdapter (Abstract Base Class)
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
  PretrainedVGGTAdapter  FineTunedVGGTAdapter  MockVGGTAdapter
  (Official Meta 1B)     (Custom UAV Weights)  (CI / Fast Testing)
```

### How to Load the Fine-Tuned Checkpoint:
When your fine-tuned VGGT checkpoint is ready:
1. Place the checkpoint file at `models/vggt/finetuned/best_checkpoint.pth`.
2. Update `configs/models.yaml`:
   ```yaml
   models:
     vggt:
       enabled: true
       backend: "finetuned"
       checkpoint: "models/vggt/finetuned/best_checkpoint.pth"
   ```
3. Alternatively, set the environment variable without editing any files:
   ```powershell
   $env:VGGT_CHECKPOINT="models/vggt/finetuned/best_checkpoint.pth"
   ```
The rest of the pipeline (YOLO, SAM 2, GPS georeferencing, Open3D reconstruction, Three.js viewer) continues to function identically without code modifications.

---

## 5. Reconstruction Modes

Selectable via `--mode` or Web UI:
- **Mode 3: Hybrid (Default Target)**: `Video → Keyframes → YOLO/SAM 2 → VGGT → GPS/IMU → Open3D → Mesh → Texturing → Three.js`.
- **Mode 1: Pure VGGT**: Bypasses dynamic masking and uses direct VGGT predicted point maps and geometry.
- **Mode 2: COLMAP Baseline**: Optional classical photogrammetry baseline for SfM feature matching and comparison.

---

## 6. Installation & Environment Setup

### Prerequisites
- Windows 10/11 or Ubuntu 22.04 LTS
- Python 3.10, 3.11, or 3.12
- NVIDIA GPU with CUDA 12.4 (RTX 3050 4GB tested and optimized)
- Node.js v18+ and npm

### 1. Python Environment Setup
```powershell
# Clone or enter directory
cd d:\Vggt

# Activate virtual environment
.\drone_vggt_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Frontend Setup
```powershell
cd frontend
npm install
npm run build
cd ..
```

---

## 7. Model Checkpoint Downloads (Rule 31 Compliant)

Check existing models and download safely without duplicates:
```powershell
# View model status
python scripts/download_models.py

# Download specific models when needed:
python scripts/download_models.py yolo11n
python scripts/download_models.py depth_anything_v2
python scripts/download_models.py sam2_1_small
```

---

## 8. Running the Application

### Option A: Complete Web GUI (FastAPI + React + Three.js)
```powershell
# Launch FastAPI backend with integrated React static app
.\drone_vggt_env\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
Open your browser at `http://localhost:8000` to access the AeroScan 3D Control Center:
- Upload drone flight video (MP4/MOV).
- Upload GPS telemetry log (CSV).
- Monitor live pipeline progress and console logs.
- Rotate, pan, zoom, measure distances, and toggle wireframes in the interactive Three.js 3D viewer.
- Download `final.glb`, `final.ply`, and `final.obj`.

### Option B: Command-Line Interface (CLI)
```powershell
# Generate sample test sequence
python scripts/generate_sample_data.py

# Run complete reconstruction pipeline
python scripts/run_pipeline.py \
    --input data/raw/sample_mission/synthetic_drone_flight.mp4 \
    --gps data/raw/sample_mission/flight_gps.csv \
    --mode hybrid \
    --vggt-backend mock \
    --mission-id sih_demo_run
```

---

## 9. Telemetry Input Formats

### GPS Telemetry CSV (`flight_gps.csv`)
```csv
timestamp,latitude,longitude,altitude
0.0,28.613939,77.209021,45.0
1.0,28.613959,77.209036,45.2
2.0,28.613979,77.209051,45.3
```

### IMU Telemetry CSV (`flight_imu.csv`, Optional)
```csv
timestamp,ax,ay,az,gx,gy,gz,roll,pitch,yaw
0.0,0.01,0.02,9.81,0.01,0.01,0.0,1.2,-3.5,45.0
```

### Camera Calibration JSON (`camera.json`, Optional)
```json
{
  "camera_model": "PINHOLE",
  "width": 1920,
  "height": 1080,
  "fx": 1420.5,
  "fy": 1420.5,
  "cx": 960.0,
  "cy": 540.0,
  "distortion_coefficients": [0.0, 0.0, 0.0, 0.0]
}
```

---

## 10. Automated Testing Suite

Run the full automated test suite:
```powershell
.\drone_vggt_env\Scripts\python.exe -m pytest tests/ -v
```

### Test Coverage:
- `tests/test_video_extractor.py`: Frame extraction and FPS sampling.
- `tests/test_frame_quality.py`: Laplacian blur, contrast, and exposure quality scoring.
- `tests/test_keyframes.py`: Intelligent keyframe selection and visual disparity.
- `tests/test_yolo_detection.py`: Dynamic object bounding box extraction and class filtering.
- `tests/test_sam2_segmenter.py`: SAM 2 mask generation and fallback mode.
- `tests/test_vggt_model.py`: Modular adapter loading, predictions, and factory switching.
- `tests/test_depth_anything.py`: Standardized DepthResult interface.
- `tests/test_depth_fusion.py`: Relative-to-metric affine scale and shift alignment.
- `tests/test_gps_georeference.py`: GPS CSV parsing, ENU projection, and Umeyama Sim(3) alignment.
- `tests/test_pointcloud.py`: Point cloud generation and Open3D filtering.
- `tests/test_mesh_export.py`: Surface mesh reconstruction and GLB/OBJ/PLY export.
- `tests/test_evaluation_metrics.py`: Chamfer distance, RMSE, and trajectory error calculations.
- `tests/test_smoke_e2e.py`: End-to-end mission reconstruction test.

---

## 11. Project Directory Structure

```text
SinglePass3D/
├── models/
│   ├── vggt/
│   │   ├── pretrained/           # Pretrained VGGT-1B checkpoint
│   │   └── finetuned/            # Future fine-tuned VGGT checkpoint
│   ├── depth_anything/           # Depth Anything V2 weights
│   ├── yolo/                     # YOLO11 / YOLOv8 weights
│   └── sam2/                     # SAM 2 weights
│
├── data/
│   ├── raw/                      # Drone videos & GPS logs
│   ├── frames/                   # Extracted video frames
│   ├── selected_frames/          # Quality-filtered frames
│   ├── keyframes/                # Intelligently selected keyframes
│   ├── masks/                    # Dynamic exclusion masks
│   └── output/                   # Mission artifacts
│
├── src/
│   ├── video/                    # Frame extraction, quality scoring, keyframes
│   ├── detection/                # Ultralytics YOLO & dynamic object filtering
│   ├── segmentation/             # Meta SAM 2 segmenter & fallback
│   ├── depth/                    # Depth Anything V2 estimator
│   ├── geometry/                 # Modular VGGT model adapter & depth fusion
│   ├── georeference/             # GPS ENU projection, IMU, Umeyama alignment
│   ├── reconstruction/           # Open3D point clouds, filtering, Poisson mesh, texturing
│   ├── evaluation/               # Chamfer distance, RMSE, trajectory metrics
│   ├── output/                   # Output manager & report serializing
│   └── pipeline/                 # End-to-end runner orchestrator
│
├── backend/                      # FastAPI asynchronous server
├── frontend/                     # React + Three.js 3D WebGL viewer
├── configs/
│   ├── models.yaml               # Centralized model registry
│   └── pipeline.yaml             # Pipeline filtering & reconstruction parameters
├── tests/                        # Comprehensive unit & end-to-end tests
├── scripts/
│   ├── run_pipeline.py           # CLI runner
│   ├── generate_sample_data.py   # Synthetic drone flight generator
│   └── download_models.py        # Safe model downloader
├── docker/                       # Dockerfile & docker-compose.yml
├── requirements.txt
└── README.md
```

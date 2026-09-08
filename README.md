# AI-Powered Drone Disaster Inspection & 3D Rescue Mapping System (Drone-VGGT + Depth Anything V2)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6.0+cu124](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-red.svg)](https://pytorch.org/)
[![License: CC BY-NC 4.0 / Apache 2.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0%20%2F%20Apache--2.0-lightgrey.svg)](MODEL_CARD.md)

An autonomous, production-grade 3D reconstruction and AI spatial rescue intelligence platform designed for **Smart India Hackathon (SIH)** disaster response applications. The system transforms continuous aerial drone video into metric 3D point clouds, surface meshes, monocular depth validation maps, 3D object/person locations, structural damage alerts, and automated HTML rescue reports.

---

## 1. SIH Problem Statement & Proposed Solution

### The Problem
During natural disasters (floods, earthquakes, landslides, fires), emergency responders receive extensive raw drone video footage. Manually scanning hundreds of 2D frames is time-consuming and fails to provide an immediate 3D spatial understanding of victims, hazardous terrain, damaged infrastructure, or blocked access routes.

### The Solution
Our system converts raw aerial drone video into a **Metric 3D Reconstruction** paired with **AI Disaster Inspection**:
1. **Multi-View Geometry**: Meta's **VGGT-1B** foundation transformer predicts camera poses, metric depth, and 3D point maps.
2. **Depth Validation**: **Depth Anything V2 Small** (Apache-2.0) estimates dense monocular depth for robust scale/shift alignment, exponential consistency scoring, and fused confidence filtering.
3. **AI Rescue Intelligence**: YOLOv8 + 2D-to-3D projection detects visible victims/objects, projects bounding boxes into 3D world coordinates $(X, Y, Z)$, evaluates structural damage & flood hazards, and compiles actionable rescue priorities.

---

## 2. Integrated System Architecture

```text
                               🚁 DRONE VIDEO
                                     │
                                     ▼
                            VIDEO PROCESSING
                                     │
                                     ▼
                           FRAME SELECTION (Laplacian)
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
         DEPTH ANYTHING V2 (Small)             META VGGT-1B
           Dense Monocular Depth          Multi-view Cameras, Depth
                    │                           & Point Maps
                    └────────────────┬────────────────┘
                                     ▼
                              DEPTH ALIGNMENT
                     (Robust Scale & Shift to Metric)
                                     │
                                     ▼
                            CONFIDENCE FUSION
                     (VGGT + Monocular Agreement)
                                     │
                                     ▼
                        DEPTH-AWARE 3D FILTERING
                                     │
                                     ▼
                         METRIC 3D RECONSTRUCTION
                                     │
                     ┌───────────────┼───────────────┐
                     ▼               ▼               ▼
                  PERSON          DAMAGE          HAZARD
                 DETECTION       DETECTION       DETECTION
                     │               │               │
                     └───────────────┼───────────────┘
                                     ▼
                           2D-to-3D GEO PROJECTION
                             (X, Y, Z Coordinates)
                                     │
                                     ▼
                          INCIDENT MANAGER (SQLite)
                                     │
                                     ▼
                           RISK SCORER & REPORT
                    (LOW / MEDIUM / HIGH / CRITICAL)
                                     │
                                     ▼
                       INTERACTIVE 3D RESCUE DASHBOARD
                             (Gradio Web UI)
```

---

## 3. System Directory Structure

```
d:/Vggt/
├── app.py                      # Interactive 7-Tab Gradio Web Dashboard
├── configs/
│   ├── config.yaml             # Main system hyperparameters
│   └── disaster_config.yaml    # Disaster & hazard inspection settings
├── data/
│   └── raw/uav_sequence/       # Benchmark UAV drone flight sequence
├── models/
│   ├── VGGT-1B/                # Meta VGGT-1B weights (model.pt, 4.8 GB)
│   └── depth_anything_v2/      # Depth Anything V2 Small (vits.pth, 99 MB, Apache-2.0)
├── outputs/                    # Experiment outputs (A, B, C, D), PLY, HTML reports
├── scripts/
│   ├── download_dataset.py     # Automated UAV benchmark downloader
│   ├── download_model.py       # Automated VGGT downloader
│   ├── download_depth_anything.py # Depth Anything V2 downloader
│   ├── run_pipeline.py         # Main 4-Experiment CLI runner
│   └── demo.py                 # One-command SIH Demonstration script
├── src/
│   ├── video_processor.py      # Frame extraction & video decoding
│   ├── frame_selector.py       # Laplacian blur & quality selection
│   ├── vggt_inference.py       # Meta VGGT inference engine
│   ├── depth_anything_v2.py    # Depth Anything V2 estimator engine
│   ├── depth_alignment.py      # Robust scale/shift depth aligner
│   ├── depth_consistency.py    # Exponential consistency analyzer
│   ├── depth_confidence.py     # Fused confidence calculator
│   ├── pointcloud_filter.py    # SOR + ROR + Depth-aware filter
│   └── disaster_inspection/
│       ├── object_detection.py # YOLOv8 person/vehicle/building detector
│       ├── tracking.py         # Multi-frame persistent object tracker
│       ├── geo_projection.py   # 2D-to-3D world coordinate projector
│       ├── damage_detection.py # Structural damage & roof disruption detector
│       ├── hazard_detection.py # Flood water, debris & smoke/fire hazard analyzer
│       ├── segmentation.py     # Water & debris color/texture segmentation
│       ├── incident_manager.py # SQLite & JSON disaster incident database
│       ├── risk_scoring.py     # Interpretable risk score (LOW-CRITICAL)
│       └── report_generator.py # HTML & JSON rescue report builder
├── tests/                      # Pytest automated test suite (16 tests)
├── MODEL_CARD.md               # Complete model inventory & licensing terms
└── README.md                   # System documentation
```

---

## 4. Installation & Environment Setup

### Prerequisites
- **OS**: Windows 10/11, Linux (Ubuntu 20.04+)
- **GPU**: NVIDIA GPU with >= 4 GB VRAM (Tested on RTX 3050 Laptop GPU & RTX 4090)
- **Python**: 3.10 - 3.12

```powershell
# 1. Activate environment
.\drone_vggt_env\Scripts\activate

# 2. Verify PyTorch CUDA & dependencies
pytest tests/ -v
```

---

## 5. Execution Modes

### A. SIH Demo Mode (For Judges & Evaluators)
Run the one-command demonstration script:
```powershell
python scripts/demo.py
```
This automatically loads the sample disaster drone sequence, runs all 4 Experiments (A, B, C, D), generates the 3D Point Cloud, builds the incident database, compiles `outputs/sih_demo/experiment_D/disaster_report.html`, and launches the interactive Gradio dashboard at `http://127.0.0.1:7860`.

### B. End-to-End CLI Experiments (A, B, C, D)
Run all 4 experiments from command line:
```powershell
# Run full Disaster Inspection & 3D Rescue Mapping:
python scripts/run_pipeline.py --input data/raw/uav_sequence/1121222322212102-4/images --mode all --max_frames 12

# Run Baseline VGGT only:
python scripts/run_pipeline.py --input data/raw/uav_sequence/1121222322212102-4/images --mode A --disable-depth-anything
```

### C. Interactive Gradio Web Dashboard
Launch the web UI:
```powershell
python app.py
```
Open browser at `http://127.0.0.1:7860`.

---

## 6. The 4 Benchmark Experiments

1. **Experiment A (Baseline)**: Raw selected frames $\rightarrow$ VGGT $\rightarrow$ 3D Point Cloud.
2. **Experiment B (Depth Validation)**: VGGT + Depth Anything V2 $\rightarrow$ Robust Scale/Shift Alignment $\rightarrow$ Exponential Consistency Heatmaps.
3. **Experiment C (Depth-Aware Reconstruction)**: VGGT + Depth Anything V2 $\rightarrow$ Fused Confidence Gating $\rightarrow$ Noise-free 3D Point Cloud.
4. **Experiment D (AI Disaster Inspection & Rescue Mapping)**: Experiment C 3D model + YOLOv8 Person/Object Detection + 2D-to-3D Geo-Projection + Structural Damage & Flood Hazard Analysis + SQLite Incident Database + HTML Rescue Report.

---

## 7. Model Inventory & Licensing

| Model | Purpose | Checkpoint | License |
| :--- | :--- | :--- | :--- |
| **VGGT-1B** | Multi-View Cameras, Depth & Point Maps | `facebook/VGGT-1B` (4.8 GB) | **CC BY-NC 4.0** |
| **Depth Anything V2 Small** | Monocular Depth Validation & Confidence | `depth_anything_v2_vits.pth` (99 MB) | **Apache 2.0** |
| **YOLOv8n** | Person, Vehicle & Structure Detection | `yolov8n.pt` (6.2 MB) | **AGPL-3.0 / Open** |
| **OpenCV / Open3D** | Image Refinement & Poisson Meshing | Classical Algorithms | **Apache 2.0 / MIT** |

---

## 8. Responsible AI & Disaster Disclaimer

> **IMPORTANT**: The AI predictions generated by this platform (person detections, structural damage alerts, flood hazard bounds) serve as **Decision Support** for search-and-rescue teams. All AI-flagged incidents require human responder field verification before initiating tactical operations.

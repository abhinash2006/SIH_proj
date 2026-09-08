# Model Card & Model Inventory

## Project: Single-Pass Drone Video to Accurate 3D Model Generation System

This document provides a comprehensive inventory of all neural models, algorithms, libraries, checkpoints, and licensing terms used in the Drone-VGGT 3D reconstruction system.

---

### Model & Component Inventory Table

| Component / Model | Purpose | Architecture / Type | Pretrained / Classical | Source / Repository | Checkpoint Name | License | Research / Commercial Restrictions | Military Restrictions |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **VGGT-1B** | Single-pass all-in-one 3D camera, depth, and point map estimation from multi-view sequence | Visual Geometry Grounded Transformer (DINOv2 backbone + alternating global/frame attention + DPT heads, ~1.26B params) | Pretrained | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) | `facebook/VGGT-1B` (`model.pt`, 4.8 GB) | **CC BY-NC 4.0** (Non-Commercial, Creative Commons) | Non-commercial research & education only. Commercial redistribution restricted. | Subject to Meta standard open-source policy & CC BY-NC. |
| **Depth Anything V2 Small** | Dense monocular depth estimation for depth consistency validation & confidence fusion | DINOv2 ViT-Small backbone + DPT Decoder (~24.8M params) | Pretrained | [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) | `depth_anything_v2_vits.pth` (99.2 MB) | **Apache 2.0** | Permissive, Commercial & Research allowed without restriction. | None. |
| **YOLOv8n** | Disaster object, person, vehicle, and structure detection | Ultralytics YOLO CNN/ViT (~3.2M params) | Pretrained | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | `yolov8n.pt` (6.2 MB) | **AGPL-3.0 / Open Source** | Permissive open source inference allowed. | None. |
| **DINOv2 (ViT-L/14)** | Feature extraction backbone inside VGGT & optional visual similarity filter | Vision Transformer (Self-supervised) | Pretrained | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | `dinov2_vitl14_reg` (Embedded in VGGT) | **Apache 2.0 / CC BY-NC 4.0** | Built into VGGT weights. | None beyond base license. |
| **OpenCV** | Video decoding, frame extraction, Laplacian blur detection, contrast/exposure analysis, LAB CLAHE refinement | Classical Computer Vision | Classical Algorithm | [opencv/opencv](https://github.com/opencv/opencv) | N/A | **Apache 2.0** | Commercial & Research allowed. | None. |
| **Open3D** | Statistical Outlier Removal (SOR), Radius Outlier Removal (ROR), Voxel Grid Downsampling, Normal estimation, Poisson Surface Reconstruction | Classical 3D Point Cloud & Mesh Library | Classical Algorithm | [isl-org/Open3D](https://github.com/isl-org/Open3D) | N/A | **MIT License** | Permissive, Commercial & Research allowed. | None. |
| **COLMAP Utils** | Camera pose, intrinsics, and 3D point serialization into COLMAP binary (`.bin`) and text (`.txt`) formats for bundle adjustment & downstream pipelines | Structure-from-Motion Geometry Serializer | Classical SfM Compatibility | [colmap/colmap](https://github.com/colmap/colmap) | N/A | **BSD 3-Clause** | Permissive, Commercial & Research allowed. | None. |
| **3D Gaussian Splatting Exporter** | Scene export format for real-time volumetric rendering | Scene Structure & Point Cloud Exporter | Classical Geometry / Splat Setup | [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) | N/A | Custom / Compatible export | Non-commercial research restrictions apply to original Inria software; export format is standard COLMAP+PLY. | None on output format. |

---

### Dataset Inventory Table

| Dataset Name | Source / URL | License | Sequence Type | Frames / Size | Ground Truth Available | Suitability for Drone 3D Reconstruction |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **UAVLight Drone Benchmark** | [HuggingFace / dukang92/UAVLight](https://huggingface.co/datasets/dukang92/UAVLight) | **Creative Commons Attribution 4.0 International (CC BY 4.0)** | Multi-view sequential drone flight over architectural/campus structures | 123 multi-view high-res images (1920x1080) + 2425-frame raw video (80.9 MB) | Yes (COLMAP sparse camera poses & triangulated 3D points) | High overlap, sequential drone trajectories, parallax, real outdoor illumination variation, architectural buildings. |

---

### Checkpoint Licensing and Redistribution Rules

1. **Meta VGGT Weights (`VGGT-1B`)**:
   - The weights are released under **CC BY-NC 4.0** by Meta Research.
   - **Commercial Use:** Requires direct authorization/licensing from Meta for commercial deployment.
   - **Redistribution:** Weights are downloaded dynamically via Hugging Face Hub (`facebook/VGGT-1B`) or local cache. They are **not bundled** directly into the git repository to respect redistribution terms and minimize repository size.
2. **UAVLight Dataset**:
   - Released under **CC BY 4.0** allowing academic and commercial use with attribution.

---

### Training & Fine-Tuning Strategy (Scientific Assessment)

- **Initial Prototype Strategy:** Pretrained Zero-Shot Inference.
  - The VGGT-1B model is trained on tens of millions of multi-view images (Co3D, RealEstate10K, ScanNet++, ACID, MegaDepth).
  - Fine-tuning from scratch was deliberately bypassed initially because VGGT demonstrates strong zero-shot generalization across diverse drone and aerial scenes without incurring catastrophic forgetting or heavy computational cost.
- **Fine-Tuning Roadmap (If Required in Future Work):**
  - **Target Dataset:** UAV123 / UAVLight / UrbanScene3D.
  - **Loss Functions:** L1 Depth Loss, Smooth L1 Point Loss, Camera Pose Geodesic Loss, and Confidence Cross-Entropy.
  - **Augmentation:** Random color jitter, solar glare simulation, motion blur, and aerial crop-scaling.
  - **Hardware Requirements:** Minimum 4x NVIDIA A100 (80GB VRAM) for multi-view batching.

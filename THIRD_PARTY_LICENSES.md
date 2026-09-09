# Third-Party Open Source Licensing Documentation

This project integrates third-party open-source libraries and models for 3D reconstruction, monocular depth estimation, and 2D object perception.

---

## 1. Ultralytics YOLO (Object Detection & Tracking)

- **Package**: `ultralytics`
- **Official Repository**: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **Official Documentation**: [https://docs.ultralytics.com/](https://docs.ultralytics.com/)
- **License**: **AGPL-3.0 (GNU Affero General Public License v3.0)** / Enterprise Commercial License
- **Usage Notice**:
  - Ultralytics YOLO models (including `yolo26n.pt`, `yolo11n.pt`, and `yolov8n.pt`) are used exclusively as an optional 2D perception layer.
  - Any derivative works or network-hosted deployments using Ultralytics software under AGPL-3.0 must make the complete source code available under AGPL-3.0.
  - For commercial applications requiring closed-source redistribution without AGPL-3.0 constraints, an official Enterprise Commercial License must be obtained directly from [Ultralytics Licensing](https://ultralytics.com/license).

---

## 2. Meta VGGT (Visual Geometry Grounding Transformer)

- **Official Repository**: [https://github.com/facebookresearch/vggt](https://github.com/facebookresearch/vggt)
- **License**: **CC-BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International)**
- **Usage Notice**: Restricted to non-commercial research and educational evaluation.

---

## 3. Depth Anything V2 (Monocular Depth Estimation)

- **Official Repository**: [https://github.com/DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
- **License**: **Apache-2.0 License**
- **Usage Notice**: Permissive open-source license supporting both non-commercial and commercial integration.

---

## Summary Table

| Component | Repository | Primary License | Permitted Commercial Use |
| :--- | :--- | :--- | :--- |
| **VGGT-1B** | facebookresearch/vggt | CC-BY-NC 4.0 | Non-Commercial Only |
| **Depth Anything V2** | DepthAnything/Depth-Anything-V2 | Apache-2.0 | Yes (Apache 2.0) |
| **Ultralytics YOLO** | ultralytics/ultralytics | AGPL-3.0 / Enterprise | AGPL-3.0 / Commercial License Required |

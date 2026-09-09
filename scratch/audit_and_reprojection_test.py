"""
Comprehensive Audit & Fix Verification for Tasks 1, 2, 4, 6, 7.
"""
import os
import sys
from pathlib import Path
import cv2
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

if os.name == "nt":
    candidates = [
        os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib"),
        r"D:\Vggt\drone_vggt_env\Lib\site-packages\torch\lib"
    ]
    for p in candidates:
        if os.path.exists(p) and hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(p)
                break
            except Exception:
                pass

from src.video_processor import VideoProcessor
from src.frame_selector import FrameSelector
from src.yolo_detector import YOLODetector
from src.coordinate_utils import CoordinateUtils, CoordinateConverter
from src.reprojection import ReprojectionValidator

video_path = "data/raw/videos/flooded_drone_sample.mp4"
out_dir = Path("outputs/investigation")
out_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# PART 1: FRAME SELECTION AUDIT
# ---------------------------------------------------------
print("=" * 80)
print("PART 1: FRAME SELECTION AUDIT (Why were only 4 frames selected?)")
print("=" * 80)

frames_dir = out_dir / "extracted_frames"
extracted = sorted(list(frames_dir.glob("*.jpg")))
if not extracted:
    extracted = VideoProcessor.extract_frames(video_path, frames_dir, target_fps=2.0)
print(f"Total extracted frames: {len(extracted)}")

# Test default FrameSelector vs relaxed FrameSelector
sel_default = FrameSelector({"frame_selection": {"sharpness_threshold": 30.0, "max_selected_frames": 16}})
res_def = sel_default.select_frames(extracted)
print(f"Default selector (sharpness=30, sim=0.985): {len(res_def['selected_paths'])} frames selected")
print(f"   Quality rejections: {res_def['rejected_quality_count']}")
print(f"   Redundant rejections: {res_def['rejected_redundant_count']}")
print("Sample rejections:")
for r in res_def['rejected_details'][:8]:
    print(f"   {os.path.basename(r[0])}: {r[1]}")

# Test with similarity_threshold=0.992 or 0.995 to keep more overlapping viewpoints
sel_fine = FrameSelector({
    "frame_selection": {
        "sharpness_threshold": 25.0,
        "similarity_threshold": 0.992,
        "max_selected_frames": 12
    }
})
res_fine = sel_fine.select_frames(extracted)
print(f"\nTuned selector (sharpness=25, sim=0.992, max=12): {len(res_fine['selected_paths'])} frames selected")
print(f"   Selected frames: {[os.path.basename(p) for p in res_fine['selected_paths']]}")

# ---------------------------------------------------------
# PART 2: YOLO PERSON TRACE ON SELECTED FRAMES
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("PART 2: YOLO PERSON TRACE ON SELECTED FRAMES")
print("=" * 80)

yolo = YOLODetector(
    model_name="yolo26n.pt",
    confidence=0.06,  # capture aerial people
    tiled_inference=True,
    tile_size=640,
    tile_overlap=0.30
)

# Test on the tuned 12 selected frames
selected_paths = res_fine['selected_paths']
annotated_dir = out_dir / "annotated_trace"
annotated_dir.mkdir(parents=True, exist_ok=True)

batch_dets = yolo.track_frames(selected_paths, conf=0.06)

total_person_dets = 0
for idx, (fpath, dets) in enumerate(zip(selected_paths, batch_dets)):
    fname = os.path.basename(fpath)
    p_dets = [d for d in dets if "person" in str(d.get("class_name", "")).lower()]
    v_dets = [d for d in dets if str(d.get("class_name", "")).lower() in ["car", "truck", "bus", "boat"]]
    total_person_dets += len(p_dets)
    
    print(f"\nFrame {idx:02d} ({fname}):")
    print(f"   Total Detections: {len(dets)} | Persons: {len(p_dets)} | Vehicles: {len(v_dets)}")
    for p in p_dets:
        print(f"      PERSON -> Conf: {p.get('confidence'):.3f}, BBox: {p.get('bbox')}, TrackID: {p.get('track_id')}")

    # Annotate frame
    img_bgr = cv2.imread(fpath)
    ann = yolo.annotate_frame(img_bgr, dets)
    cv2.imwrite(str(annotated_dir / f"trace_{idx:02d}_{fname}"), cv2.cvtColor(ann, cv2.COLOR_RGB2BGR))

print(f"\nTotal Person Detections across selected frames: {total_person_dets}")
print(f"Annotated frames saved to: {annotated_dir}")

# ---------------------------------------------------------
# PART 3: REPROJECTION ERROR MATHEMATICAL AUDIT
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("PART 3: REPROJECTION ERROR MATHEMATICAL AUDIT & SYNTHETIC TEST")
print("=" * 80)

# Simulate VGGT setup:
# Original image: 1920x1080
# VGGT input: 392x392
# Intrinsic matrix for 392x392:
fx_vggt, fy_vggt = 300.0, 300.0
cx_vggt, cy_vggt = 196.0, 196.0
K_vggt = np.array([
    [fx_vggt, 0.0, cx_vggt],
    [0.0, fy_vggt, cy_vggt],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

extrinsic = np.eye(4)[:3]  # camera at origin looking down +Z

# Choose a known 3D point in front of camera
# Suppose a person is at (X=1.5, Y=2.0, Z=10.0) meters
true_3d_point = np.array([1.5, 2.0, 10.0])

# True projection onto 392x392 VGGT pixel grid:
u_vggt_true = (fx_vggt * 1.5 / 10.0) + cx_vggt  # 30 + 196 = 226.0
v_vggt_true = (fy_vggt * 2.0 / 10.0) + cy_vggt  # 60 + 196 = 256.0

# Corresponding pixel on original 1920x1080 image:
# Scale factors:
scale_x = 1920.0 / 392.0  # 4.89796
scale_y = 1080.0 / 392.0  # 2.75510
u_orig_true = u_vggt_true * scale_x  # 1106.94 px
v_orig_true = v_vggt_true * scale_y  # 705.31 px

print(f"Synthetic Ground Truth:")
print(f"   3D Point (meters): {true_3d_point}")
print(f"   VGGT Pixel (392x392): ({u_vggt_true:.2f}, {v_vggt_true:.2f})")
print(f"   Original Pixel (1920x1080): ({u_orig_true:.2f}, {v_orig_true:.2f})")

# Case A: OLD CODE (scaled_shape=None)
val_old = ReprojectionValidator.validate_reprojection(
    point_3d=true_3d_point,
    intrinsics=K_vggt,
    extrinsics=extrinsic,
    original_pixel_uv=(u_orig_true, v_orig_true),
    original_image_shape=(1080, 1920),
    scaled_shape=None  # This was omitted in app.py!
)
print(f"\n[Case A] Old Code without scaled_shape:")
print(f"   Reprojection status: {val_old['reprojection_status']}")
print(f"   Pixel discrepancy: {val_old['pixel_discrepancy']:.2f} px")

# Case B: FIXED CODE with scaled_shape=(392, 392)
val_fixed = ReprojectionValidator.validate_reprojection(
    point_3d=true_3d_point,
    intrinsics=K_vggt,
    extrinsics=extrinsic,
    original_pixel_uv=(u_orig_true, v_orig_true),
    original_image_shape=(1080, 1920),
    scaled_shape=(392, 392)
)
print(f"\n[Case B] Fixed Code with scaled_shape=(392, 392):")
print(f"   Reprojection status: {val_fixed['reprojection_status']}")
print(f"   Pixel discrepancy: {val_fixed['pixel_discrepancy']:.2f} px")

# Case C: Scaled Intrinsics to (1080, 1920) directly
K_orig = CoordinateUtils.scale_intrinsics(K_vggt, (392, 392), (1080, 1920))
val_scaled_K = ReprojectionValidator.validate_reprojection(
    point_3d=true_3d_point,
    intrinsics=K_orig,
    extrinsics=extrinsic,
    original_pixel_uv=(u_orig_true, v_orig_true),
    original_image_shape=(1080, 1920),
    scaled_shape=(1080, 1920)
)
print(f"\n[Case C] With properly scaled K (original image space):")
print(f"   Reprojection status: {val_scaled_K['reprojection_status']}")
print(f"   Pixel discrepancy: {val_scaled_K['pixel_discrepancy']:.2f} px")

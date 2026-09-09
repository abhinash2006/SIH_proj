"""
Diagnostic Script: Tracing the Complete PERSON Detection Pipeline
Analyzes the real flooded drone video (data/raw/videos/flooded_drone_sample.mp4)
and measures the exact count of PERSON detections that survive EACH stage.
"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2

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
from src.aerial_classifier import AerialContextValidator, TrackClassHistory, ClassThresholdConfig
from src.flood_analyzer import FloodAnalyzer
from src.spatial_context import SpatialContextAnalyzer
from src.person_fusion import MultiFramePersonFuser

video_path = "data/raw/videos/flooded_drone_sample.mp4"
frames_dir = Path("outputs/gradio_session/extracted_frames")
selected_dir = Path("outputs/gradio_session/selected_frames")

print("="*80)
print(f"DIAGNOSING PERSON DETECTION PIPELINE ON: {video_path}")
print("="*80)

# Check extracted frames
if not frames_dir.exists() or len(list(frames_dir.glob("*.jpg"))) == 0:
    print(f"Extracting frames from {video_path}...")
    frame_paths = VideoProcessor.extract_frames(video_path, frames_dir, target_fps=2.0)
else:
    frame_paths = sorted([str(p) for p in frames_dir.glob("*.jpg")])

print(f"Total extracted frames: {len(frame_paths)}")

# Check selected frames (the 8 views used in active reconstruction)
if not selected_dir.exists() or len(list(selected_dir.glob("*.jpg"))) == 0:
    selector = FrameSelector({"frame_selection": {"sharpness_threshold": 20.0, "max_selected_frames": 8}})
    selected_paths, _ = selector.select_optimal_frames(frame_paths, selected_dir, max_frames=8)
else:
    selected_paths = sorted([str(p) for p in selected_dir.glob("*.jpg")])

print(f"Total selected keyframes: {len(selected_paths)}")

# Initialize YOLO model
yolo = YOLODetector(
    model_name="yolo26n.pt",
    confidence=0.01,  # Lowest raw threshold to capture ALL raw candidates
    tiled_inference=True,
    tile_size=640,
    tile_overlap=0.25
)
yolo.load_model()

# -------------------------------------------------------------
# STEP A: DIAGNOSE ON ALL 52 EXTRACTED FRAMES (To see all people across the flight)
# -------------------------------------------------------------
print("\nScanning all 52 extracted video frames with tiled inference (raw conf >= 0.01)...")
all_frames_raw_persons = []
frame_person_counts = {}

for f_idx, f_path in enumerate(frame_paths):
    img = cv2.imread(f_path)
    H, W = img.shape[:2]
    tiles = yolo._generate_tiles(W, H, yolo.tile_size, yolo.tile_overlap)
    
    frame_raw_persons = []
    # 1. Full-frame predictions
    res_full = yolo.model.predict(source=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), conf=0.01, imgsz=640, classes=[0], verbose=False)
    for r in res_full:
        for b in r.boxes:
            conf = float(b.conf[0].item())
            xyxy = b.xyxy[0].cpu().numpy()
            frame_raw_persons.append({
                "frame_idx": f_idx,
                "frame_id": os.path.basename(f_path),
                "conf": conf,
                "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                "tile": "full_frame"
            })
            
    # 2. Tiled predictions
    for tx1, ty1, tx2, ty2 in tiles:
        crop = cv2.cvtColor(img[ty1:ty2, tx1:tx2], cv2.COLOR_BGR2RGB)
        res_tile = yolo.model.predict(source=crop, conf=0.01, imgsz=640, classes=[0], verbose=False)
        for r in res_tile:
            for b in r.boxes:
                conf = float(b.conf[0].item())
                xyxy = b.xyxy[0].cpu().numpy()
                frame_raw_persons.append({
                    "frame_idx": f_idx,
                    "frame_id": os.path.basename(f_path),
                    "conf": conf,
                    "bbox": [int(xyxy[0] + tx1), int(xyxy[1] + ty1), int(xyxy[2] + tx1), int(xyxy[3] + ty1)],
                    "tile": f"tile_{tx1}_{ty1}"
                })
    
    all_frames_raw_persons.extend(frame_raw_persons)
    if frame_raw_persons:
        frame_person_counts[f_idx] = len(frame_raw_persons)

print(f"Total raw PERSON detections across all 52 frames (conf >= 0.01): {len(all_frames_raw_persons)}")
print(f"Frames with raw person detections: {len(frame_person_counts)} / {len(frame_paths)}")
print("Sample frame detection distribution (frame_idx: count):")
for f_i in sorted(frame_person_counts.keys())[:15]:
    print(f"  Frame {f_i:02d} ({os.path.basename(frame_paths[f_i])}): {frame_person_counts[f_i]} raw persons")

# Specifically check frames 34 to 45 (mentioned in prompt)
dense_range_persons = [p for p in all_frames_raw_persons if 34 <= p["frame_idx"] <= 45]
print(f"\nRaw PERSON detections in dense region (Frames 34-45): {len(dense_range_persons)}")
if dense_range_persons:
    confs = [p["conf"] for p in dense_range_persons]
    print(f"  Confidence range in frames 34-45: min={min(confs):.3f}, max={max(confs):.3f}, mean={np.mean(confs):.3f}")

# -------------------------------------------------------------
# STEP B: TRACE THE PIPELINE ON THE 8 ACTIVE SELECTED KEYFRAMES
# -------------------------------------------------------------
print("\n" + "="*80)
print("TRACING PIPELINE STAGES ON THE 8 ACTIVE RECONSTRUCTION KEYFRAMES")
print("="*80)

# Stage 1: Raw tiled detections at conf=0.01 on 8 keyframes
stage_counts = {}

raw_keyframe_persons = []
for f_idx, f_path in enumerate(selected_paths):
    img = cv2.imread(f_path)
    H, W = img.shape[:2]
    tiles = yolo._generate_tiles(W, H, yolo.tile_size, yolo.tile_overlap)
    for tx1, ty1, tx2, ty2 in tiles:
        crop = cv2.cvtColor(img[ty1:ty2, tx1:tx2], cv2.COLOR_BGR2RGB)
        res_tile = yolo.model.predict(source=crop, conf=0.01, imgsz=640, classes=[0], verbose=False)
        for r in res_tile:
            for b in r.boxes:
                conf = float(b.conf[0].item())
                xyxy = b.xyxy[0].cpu().numpy()
                raw_keyframe_persons.append({
                    "frame_idx": f_idx,
                    "frame_id": os.path.basename(f_path),
                    "raw_class_name": "person",
                    "class_name": "person",
                    "raw_detector_confidence": conf,
                    "confidence": conf,
                    "bbox_xyxy": [int(xyxy[0] + tx1), int(xyxy[1] + ty1), int(xyxy[2] + tx1), int(xyxy[3] + ty1)],
                    "bbox": [int(xyxy[0] + tx1), int(xyxy[1] + ty1), int(xyxy[2] + tx1), int(xyxy[3] + ty1)]
                })

stage_counts["Raw YOLO PERSON detections (conf >= 0.01)"] = len(raw_keyframe_persons)

# Stage 2: After Confidence Threshold (e.g. 0.08)
conf_filtered_persons = [p for p in raw_keyframe_persons if p["confidence"] >= 0.08]
stage_counts["After confidence threshold (>= 0.08)"] = len(conf_filtered_persons)

# Stage 3: After NMS (tiled duplicate suppression)
nms_persons = yolo._apply_nms(conf_filtered_persons, iou_thresh=0.45)
stage_counts["After NMS duplicate suppression"] = len(nms_persons)

# Stage 4: Using yolo.track_frames(selected_paths, conf=0.08)
tracked_batch = yolo.track_frames(selected_paths, conf=0.08)
all_tracked_dets = [d for frame_dets in tracked_batch for d in frame_dets]
tracked_persons = [d for d in all_tracked_dets if d.get("class_name") == "person" or d.get("raw_class_name") == "person"]
stage_counts["After multi-frame tracking"] = len(tracked_persons)

# Count unique tracks
unique_tracks = set(d.get("track_id") for d in tracked_persons if d.get("track_id") is not None)
stage_counts["Persistent person tracks"] = len(unique_tracks)

# Stage 5: Context Validation
val_cfg = ClassThresholdConfig(person_thresh=0.08)
context_val = AerialContextValidator(val_cfg)
validated_persons = []
uncertain_persons = []
for p in tracked_persons:
    res = context_val.validate_detection(p)
    if res["final_class"] == "PERSON":
        validated_persons.append(p)
    else:
        uncertain_persons.append((p, res))

stage_counts["After context validation (class == PERSON)"] = len(validated_persons)
stage_counts["Filtered to UNCERTAIN_OBJECT"] = len(uncertain_persons)

# Stage 6: MultiFramePersonFuser
fused_targets = MultiFramePersonFuser.fuse_person_tracks(
    person_detections=validated_persons,
    total_selected_frames=len(selected_paths)
)
stage_counts["Fused SAR Targets"] = len(fused_targets)

# Breakdown of targets
n_3d = sum(1 for t in fused_targets if t.localization_status == "LOCALIZED_3D")
n_reproj = sum(1 for t in fused_targets if t.reprojection_error is not None and t.reprojection_error <= 25.0)
n_stranded = sum(1 for t in fused_targets if t.evidence.get("detection_state") == "POTENTIAL STRANDED PERSON")
n_validated_rescue = sum(1 for t in fused_targets if t.evidence.get("detection_state") == "VALIDATED RESCUE TARGET")

stage_counts["2D→3D localized persons"] = n_3d
stage_counts["Reprojection-valid persons (<= 25px)"] = n_reproj
stage_counts["Potential stranded persons"] = n_stranded
stage_counts["Validated rescue targets"] = n_validated_rescue

print("\n" + "="*80)
print("DIAGNOSTIC PIPELINE STAGES TABLE (SECTION 1 REQUIRED TABLE)")
print("="*80)
print(f"{'Stage':<45} | {'PERSON COUNT':<15}")
print("-" * 65)
for stage_name, count in stage_counts.items():
    print(f"{stage_name:<45} | {count:<15}")

print("\n" + "="*80)
print("DETAILS ON TRACKED PERSON DETECTIONS")
print("="*80)
for p in tracked_persons:
    bbox = p.get("bbox_xyxy", [0, 0, 0, 0])
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    print(f"Track #{p.get('track_id')}: Frame {p.get('frame_id')} | Conf: {p.get('confidence'):.3f} | BBox: {bbox} ({w}x{h}, area={w*h})")

yolo.release()
print("\nDiagnostic complete.")

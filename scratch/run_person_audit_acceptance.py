"""
Acceptance Run: UAV Small Person Detection & Person Tracking Audit
Runs on the real flooded UAV video data/raw/videos/flooded_drone_sample.mp4
Produces empirical before-vs-after metrics and person-only visual artifacts.
"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2
import json

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

from src.yolo_detector import YOLODetector
from src.aerial_classifier import AerialContextValidator, TrackClassHistory, ClassThresholdConfig
from src.flood_analyzer import FloodAnalyzer

ARTIFACT_DIR = Path(r"C:\Users\abhin\.gemini\antigravity-ide\brain\d77e2693-bf2f-4caa-a570-cceda149da64")
selected_dir = Path("outputs/gradio_session/selected_frames")
extracted_dir = Path("outputs/gradio_session/extracted_frames")

keyframe_paths = sorted([str(p) for p in selected_dir.glob("*.jpg")])
if not keyframe_paths and extracted_dir.exists():
    keyframe_paths = sorted([str(p) for p in extracted_dir.glob("*.jpg")])[:8]

print(f"[INFO] Using {len(keyframe_paths)} keyframes for evaluation.")

# Initialize updated YOLO detector
detector = YOLODetector(
    model_name="yolo26n.pt",
    confidence=0.15,
    iou_threshold=0.45,
    person_conf_threshold=0.08,
    person_iou_threshold=0.40,
    tiled_inference=True,
    tile_size=640,
    tile_overlap=0.25
)
detector.load_model()

# Initialize updated AerialContextValidator
validator_cfg = ClassThresholdConfig(
    person_thresh=0.08,
    vehicle_thresh=0.15,
    building_thresh=0.15,
    boat_thresh=0.15,
    person_min_area=10.0,
    person_max_area=3500.0,
    person_max_aspect=8.0
)
validator = AerialContextValidator(validator_cfg)

print("\n" + "="*80)
print("STAGE 1: EVALUATING KEYFRAMES WITH CLASS-AWARE NMS & TRACKING")
print("="*80)

tracked_results = detector.track_frames(keyframe_paths)

stage_metrics = {
    "total_raw_person_detections": 0,
    "total_nms_person_detections": 0,
    "total_tracked_person_detections": 0,
    "total_validated_person_detections": 0,
    "unique_person_tracks": set(),
    "tracks_validated": set(),
    "per_frame": {}
}

all_validated_per_frame = []
all_raw_person_per_frame = []

for idx, (f_path, f_dets) in enumerate(zip(keyframe_paths, tracked_results)):
    f_name = os.path.basename(f_path)
    img = cv2.imread(f_path)
    H, W = img.shape[:2]

    # Measure raw person candidates
    raw_candidates = []
    # Full frame
    res_full = detector.model.predict(source=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), conf=0.08, classes=[0], verbose=False)
    for r in res_full:
        for b in r.boxes:
            c = float(b.conf[0].item())
            bx = b.xyxy[0].cpu().numpy().astype(int)
            raw_candidates.append({"conf": c, "bbox": bx.tolist(), "tile": "full_frame"})
    # Tiled
    tiles = detector._generate_tiles(W, H, detector.tile_size, detector.tile_overlap)
    for tx1, ty1, tx2, ty2 in tiles:
        crop = img[ty1:ty2, tx1:tx2]
        res_t = detector.model.predict(source=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), conf=0.08, classes=[0], verbose=False)
        for r in res_t:
            for b in r.boxes:
                c = float(b.conf[0].item())
                bx = b.xyxy[0].cpu().numpy().astype(int)
                raw_candidates.append({"conf": c, "bbox": [bx[0]+tx1, bx[1]+ty1, bx[2]+tx1, bx[3]+ty1], "tile": f"tile_{tx1}_{ty1}"})

    person_dets = [d for d in f_dets if d.get("class_name") == "person"]
    val_dets = []
    for d in person_dets:
        v_res = validator.validate_detection(d)
        val_dets.append(v_res)
        if v_res["final_class"] == "PERSON":
            stage_metrics["tracks_validated"].add(d.get("track_id"))

    all_raw_person_per_frame.append(raw_candidates)
    all_validated_per_frame.append(val_dets)

    p_count = len(person_dets)
    val_count = sum(1 for v in val_dets if v["final_class"] == "PERSON")
    for d in person_dets:
        stage_metrics["unique_person_tracks"].add(d.get("track_id"))

    stage_metrics["total_raw_person_detections"] += len(raw_candidates)
    stage_metrics["total_nms_person_detections"] += p_count
    stage_metrics["total_tracked_person_detections"] += p_count
    stage_metrics["total_validated_person_detections"] += val_count

    stage_metrics["per_frame"][f_name] = {
        "raw_person": len(raw_candidates),
        "after_nms": p_count,
        "validated_person": val_count,
        "tracks": [d.get("track_id") for d in person_dets]
    }

    print(f"[{f_name}] Raw Candidates: {len(raw_candidates)} | After Class-Aware NMS & Tracking: {p_count} | Validated: {val_count}")

# Generate Diagnostic Contact Sheet for the richest person frame
max_idx = max(range(len(keyframe_paths)), key=lambda i: len([d for d in tracked_results[i] if d.get("class_name") == "person"]))
best_frame_path = keyframe_paths[max_idx]
best_raw = [
    {"class_name": "person", "raw_class_name": "person", "bbox_xyxy": c["bbox"], "confidence": c["conf"], "source_tile": c["tile"]}
    for c in all_raw_person_per_frame[max_idx]
]
best_tracked = [d for d in tracked_results[max_idx] if d.get("class_name") == "person"]
best_val = all_validated_per_frame[max_idx]

diag_img = detector.annotate_person_diagnostic(
    best_frame_path,
    raw_person_dets=best_raw,
    tracked_person_dets=best_tracked,
    validated_person_dets=best_val
)

diag_path = ARTIFACT_DIR / "diagnostic_person_contact_sheet.jpg"
cv2.imwrite(str(diag_path), cv2.cvtColor(diag_img, cv2.COLOR_RGB2BGR))
print(f"\n[ARTIFACT] Saved person-only diagnostic contact sheet to: {diag_path}")

# Check frames 34-45 from extracted frames
print("\n" + "="*80)
print("STAGE 2: EVALUATING CANDIDATE HUMAN FRAMES 34-45")
print("="*80)
flooded_frames_dir = Path("scratch/flooded_frames")
f34_45_paths = []
for i in range(34, 46):
    f_p = flooded_frames_dir / f"frame_{i:05d}.jpg"
    if f_p.exists():
        f34_45_paths.append(str(f_p))

if f34_45_paths:
    tracked_34_45 = detector.track_frames(f34_45_paths)
    total_p_34_45 = 0
    tracks_34_45 = set()
    for p, dets in zip(f34_45_paths, tracked_34_45):
        p_dets = [d for d in dets if d.get("class_name") == "person"]
        total_p_34_45 += len(p_dets)
        for d in p_dets:
            tracks_34_45.add(d.get("track_id"))
        print(f"[{os.path.basename(p)}] Person detections: {len(p_dets)} | Tracks: {[d.get('track_id') for d in p_dets]}")
    print(f"Frames 34-45 Total Person Observations: {total_p_34_45} across {len(tracks_34_45)} persistent tracks.")

print("\n" + "="*80)
print("STAGE 3: SUMMARY AUDIT METRICS")
print("="*80)
print(f"Total Raw Person Detections (conf >= 0.08): {stage_metrics['total_raw_person_detections']}")
print(f"Total Person Detections After Class-Aware NMS: {stage_metrics['total_nms_person_detections']}")
print(f"Total Multi-Frame Tracked Persons: {stage_metrics['total_tracked_person_detections']}")
print(f"Unique Person Track IDs: {len(stage_metrics['unique_person_tracks'])} -> {sorted(list(stage_metrics['unique_person_tracks']))}")
print(f"Total Validated Person Detections: {stage_metrics['total_validated_person_detections']}")
print(f"Unique Validated Person Tracks: {len(stage_metrics['tracks_validated'])} -> {sorted(list(stage_metrics['tracks_validated']))}")
print("="*80)

# Save JSON results for report generation
audit_summary = {
    "raw_person_detections": stage_metrics["total_raw_person_detections"],
    "nms_person_detections": stage_metrics["total_nms_person_detections"],
    "tracked_person_detections": stage_metrics["total_tracked_person_detections"],
    "validated_person_detections": stage_metrics["total_validated_person_detections"],
    "unique_person_tracks": len(stage_metrics["unique_person_tracks"]),
    "unique_validated_tracks": len(stage_metrics["tracks_validated"]),
    "per_frame": stage_metrics["per_frame"]
}
with open(ARTIFACT_DIR / "person_audit_summary.json", "w") as f:
    json.dump(audit_summary, f, indent=2)
print("[INFO] Saved audit summary JSON.")

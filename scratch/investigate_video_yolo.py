"""
Investigation Script for Task 1 & Task 2:
Inspects data/raw/videos/flooded_drone_sample.mp4.
Extracts frames, tests raw YOLO across a range of confidences,
logs all detections per frame, generates an annotated contact sheet,
and traces detections stage-by-stage.
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add workspace root
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

# Set up Windows DLLs if needed
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
get_video_metadata = VideoProcessor.get_video_metadata
from src.frame_selector import FrameSelector
from ultralytics import YOLO

video_path = "data/raw/videos/flooded_drone_sample.mp4"
out_dir = Path("outputs/investigation")
out_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("TASK 1: VIDEO METADATA & FRAME INSPECTION")
print("=" * 80)

meta = get_video_metadata(video_path)
print(f"Video File: {video_path}")
print(f"Resolution: {meta.get('resolution')}")
print(f"FPS: {meta.get('fps')}")
print(f"Duration: {meta.get('duration_s')}s")
print(f"Total Frames in File: {meta.get('total_frames')}")

# Extract frames at 2.0 FPS
extracted_dir = out_dir / "extracted_frames"
extracted_frames = VideoProcessor.extract_frames(video_path, extracted_dir, target_fps=2.0)
print(f"Extracted {len(extracted_frames)} frames at 2.0 FPS.")

# Run frame selector for 4, 8, 12, 16 keyframes
config = {
    "frame_selection": {
        "sharpness_threshold": 30.0,
        "max_selected_frames": 16
    },
    "pointcloud_filtering": {"enabled": True}
}
selector = FrameSelector(config)
selected_16, _ = selector.select_optimal_frames(extracted_frames, out_dir / "selected_16", max_frames=16)
print(f"Selected {len(selected_16)} keyframes (target=16).")

# Also get the 4 frames that were previously selected
selected_4, _ = selector.select_optimal_frames(extracted_frames, out_dir / "selected_4", max_frames=4)
print(f"Selected {len(selected_4)} keyframes (target=4).")

print("\n" + "=" * 80)
print("RUNNING RAW YOLO DETECTION (ULTRA-LOW CONFIDENCE = 0.05)")
print("Checking ALL extracted frames for any person or vehicle...")
print("=" * 80)

model_path = "models/yolo26n.pt"
if not os.path.exists(model_path):
    model_path = "yolov8n.pt"  # fallback if needed
print(f"Using YOLO weights: {model_path}")

yolo_raw = YOLO(model_path)

all_frame_detections = {}
frames_with_person = []
frames_with_any_obj = []

for idx, fpath in enumerate(extracted_frames):
    img = cv2.imread(fpath)
    h, w = img.shape[:2]
    # Run at conf=0.05 to see ALL raw candidates
    results = yolo_raw.predict(img, conf=0.05, verbose=False)[0]
    
    dets = []
    has_person = False
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = yolo_raw.names[cls_id]
        conf = float(box.conf[0])
        xyxy = [float(c) for c in box.xyxy[0]]
        bw = xyxy[2] - xyxy[0]
        bh = xyxy[3] - xyxy[1]
        
        det_info = {
            "class": cls_name,
            "conf": round(conf, 3),
            "bbox": [round(c, 1) for c in xyxy],
            "size": (round(bw, 1), round(bh, 1)),
            "area": round(bw * bh, 1)
        }
        dets.append(det_info)
        if cls_name == "person":
            has_person = True
            
    all_frame_detections[idx] = dets
    if has_person:
        frames_with_person.append((idx, fpath, [d for d in dets if d["class"] == "person"]))
    if len(dets) > 0:
        frames_with_any_obj.append((idx, fpath, dets))

print(f"\nTotal extracted frames scanned: {len(extracted_frames)}")
print(f"Frames with ANY raw detection (conf >= 0.05): {len(frames_with_any_obj)}")
print(f"Frames with RAW PERSON detections (conf >= 0.05): {len(frames_with_person)}")

if len(frames_with_person) > 0:
    print("\n--- RAW PERSON DETECTIONS FOUND! ---")
    for f_idx, fpath, p_dets in frames_with_person:
        print(f"Frame {f_idx:02d} ({os.path.basename(fpath)}): {len(p_dets)} person(s)")
        for p in p_dets:
            print(f"   Conf: {p['conf']}, BBox: {p['bbox']}, Size: {p['size']} px")
else:
    print("\n--- ZERO RAW PERSON DETECTIONS FOUND AT CONF=0.05 ACROSS ALL 52 FRAMES ---")
    print("Listing summary of detected classes across all frames:")
    class_counts = {}
    for idx, dets in all_frame_detections.items():
        for d in dets:
            c = d["class"]
            class_counts[c] = class_counts.get(c, 0) + 1
    for c, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"   {c}: {cnt} detections across sequence")

print("\n" + "=" * 80)
print("TESTING TILED YOLO (SLICING) TO DETECT SMALL AERIAL OBJECTS/PERSONS")
print("=" * 80)

# Run tiled inference with 640x640 windows and overlap
from src.yolo_detector import YOLODetector
yolo_tiled = YOLODetector(
    model_name="yolo26n.pt",
    confidence=0.10,
    tiled_inference=True,
    tile_size=640,
    tile_overlap=0.30
)

tiled_person_frames = []
for idx, fpath in enumerate(extracted_frames):
    dets = yolo_tiled.detect_frame(fpath)
    p_dets = [d for d in dets if d.get("class_name") == "person" or d.get("raw_class_name") == "person"]
    if len(p_dets) > 0:
        tiled_person_frames.append((idx, fpath, p_dets))

print(f"Tiled YOLO (conf=0.10, tile=640) detected PERSON in {len(tiled_person_frames)} frames.")
if len(tiled_person_frames) > 0:
    for f_idx, fpath, p_dets in tiled_person_frames:
        print(f"Frame {f_idx:02d}: {len(p_dets)} person(s)")
        for p in p_dets:
            print(f"   Conf: {p.get('confidence')}, BBox: {p.get('bbox')}")
else:
    print("Zero persons detected even with tiled slicing at conf=0.10.")

# Create an inspection contact sheet of the selected keyframes
print("\nCreating inspection contact sheet of selected keyframes...")
contact_sheet_imgs = []
for fpath in selected_16[:8]:
    img_bgr = cv2.imread(fpath)
    # Resize for contact sheet
    thumb = cv2.resize(img_bgr, (480, 270))
    # Annotate frame_id and timestamp
    fname = os.path.basename(fpath)
    cv2.putText(thumb, fname, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    contact_sheet_imgs.append(thumb)

# Stitch 2x4 grid
row1 = np.hstack(contact_sheet_imgs[:4])
row2 = np.hstack(contact_sheet_imgs[4:8]) if len(contact_sheet_imgs) > 4 else np.zeros_like(row1)
contact_sheet = np.vstack([row1, row2])
cv2.imwrite(str(out_dir / "selected_frames_contact_sheet.png"), contact_sheet)
print(f"Contact sheet saved to: {out_dir / 'selected_frames_contact_sheet.png'}")

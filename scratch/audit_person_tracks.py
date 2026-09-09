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
from src.coordinate_utils import CoordinateConverter
from src.point_lookup import RobustPointLookup
from src.reprojection import ReprojectionValidator

video_path = "data/raw/videos/flooded_drone_sample.mp4"
frames_dir = Path("outputs/investigation/extracted_frames")
extracted = sorted(list(frames_dir.glob("*.jpg")))
if not extracted:
    extracted = VideoProcessor.extract_frames(video_path, frames_dir, target_fps=2.0)

selector = FrameSelector({
    "frame_selection": {
        "sharpness_threshold": 20.0,
        "similarity_threshold": 0.994,
        "max_selected_frames": 8
    }
})
res_sel = selector.select_frames(extracted)
selected_paths = res_sel["selected_paths"]
print(f"Selected {len(selected_paths)} frames: {[os.path.basename(p) for p in selected_paths]}")

# Run VGGT
from src.vggt_inference import VGGTInferenceEngine
vggt = VGGTInferenceEngine()
predictions = vggt.run_inference(selected_paths)
vggt.unload_model()

# Run YOLO tracking
yolo = YOLODetector(confidence=0.08, tiled_inference=True, tile_size=640, tile_overlap=0.25)
yolo_dets_batch = yolo.track_frames(selected_paths, conf=0.08)
yolo.release()

vggt_H, vggt_W = predictions["point_maps"].shape[1:3]

all_person_records = []
all_candidates_summary = []

for i, f_path in enumerate(selected_paths):
    img_bgr = cv2.imread(str(f_path))
    H_orig, W_orig = img_bgr.shape[:2]
    depth_m = predictions["depth_maps"][i]
    ext = predictions["extrinsics"][i]
    intri = predictions["intrinsics"][i]
    pts_map = predictions["point_maps"][i]
    frame_dets = yolo_dets_batch[i]

    for det in frame_dets:
        cls_name = str(det.get("class_name", "")).lower()
        bbox_orig = det["bbox_xyxy"]
        cx, cy = det["pixel_center"]

        u_vggt, v_vggt = CoordinateConverter.point_to_vggt(cx, cy, (H_orig, W_orig), (vggt_H, vggt_W))
        pos_3d, meta = RobustPointLookup.lookup_3d_point(
            point_map=pts_map,
            u_vggt=u_vggt,
            v_vggt=v_vggt,
            depth_map=depth_m,
            intrinsics=intri,
            extrinsics=ext
        )

        reproj_val = None
        reproj_status = "UNAVAILABLE"
        if pos_3d is not None:
            val_res = ReprojectionValidator.validate_reprojection(
                point_3d=pos_3d,
                intrinsics=intri,
                extrinsics=ext,
                original_pixel_uv=(cx, cy),
                original_image_shape=(H_orig, W_orig),
                scaled_shape=(vggt_H, vggt_W)
            )
            reproj_val = val_res.get("pixel_discrepancy")
            reproj_status = val_res.get("reprojection_status")

        rec = {
            "frame_idx": i,
            "frame_id": os.path.basename(f_path),
            "track_id": det.get("track_id"),
            "class_name": cls_name,
            "conf": det.get("confidence"),
            "bbox": bbox_orig,
            "center": (cx, cy),
            "pos_3d": pos_3d,
            "reproj_error": reproj_val,
            "reproj_status": reproj_status,
            "is_valid_reproj": bool(reproj_val is not None and reproj_val <= 25.0)
        }
        all_candidates_summary.append(rec)
        if "person" in cls_name:
            all_person_records.append(rec)

print("\n" + "="*80)
print(f"AUDIT SUMMARY: Total Detections: {len(all_candidates_summary)} | Person Detections: {len(all_person_records)}")
print("="*80)

# Group by track_id for persons
tracks = {}
for p in all_person_records:
    tid = p["track_id"]
    if tid not in tracks:
        tracks[tid] = []
    tracks[tid].append(p)

print(f"Unique Person Tracks: {len(tracks)}")
for tid, obs in tracks.items():
    valid_reprojs = [o["reproj_error"] for o in obs if o["is_valid_reproj"]]
    all_reprojs = [o["reproj_error"] for o in obs if o["reproj_error"] is not None]
    best_err = min(all_reprojs) if all_reprojs else None
    mean_err = np.mean(all_reprojs) if all_reprojs else None
    
    print(f"\n--- TRACK #{tid} ({len(obs)} observations) ---")
    print(f"  Frames: {[o['frame_idx'] for o in obs]}")
    print(f"  3D Candidates: {sum(1 for o in obs if o['pos_3d'] is not None)}/{len(obs)}")
    print(f"  Valid Reprojections (<25px): {len(valid_reprojs)}/{len(all_reprojs)}")
    print(f"  Best Reproj Error: {best_err}")
    print(f"  Mean Reproj Error: {mean_err}")
    for o in obs:
        print(f"    Frame {o['frame_idx']} ({o['frame_id']}): Conf={o['conf']:.2f}, BBox={o['bbox']}, Reproj={o['reproj_error']}, Status={o['reproj_status']}")

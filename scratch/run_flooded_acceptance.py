"""
Final Acceptance Verification Script:
Executes the actual flooded drone video through the upgraded end-to-end pipeline:
MP4
→ frame extraction (52 frames)
→ intelligent frame selection (8 views with tuned similarity)
→ VGGT 3D foundation transformer
→ 3D reconstruction with adaptive voxel filtering
→ Depth Anything V2 monocular validation
→ Tiled YOLO aerial perception (conf=0.08)
→ Multi-frame person tracking & fusion
→ 2D→3D localization with scaled intrinsic coordinate frame
→ Reprojection error gate (<25 px validated)
→ Multi-evidence flood analyzer (VALIDATED_INUNDATION)
→ Spatial context & SAR triage scoring
→ Validated rescue targets
→ Interactive 3D rescue visualization.
"""

import os
import sys
from pathlib import Path

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

import app

video_path = "data/raw/videos/flooded_drone_sample.mp4"
if not os.path.exists(video_path):
    print(f"Error: {video_path} does not exist!")
    sys.exit(1)

print(f"Starting Real End-to-End Pipeline Acceptance on: {video_path}")

results = app.process_disaster_drone_pipeline(
    video_file=video_path,
    image_folder_str=None,
    target_fps=2.0,
    max_frames=8,
    sharpness_thresh=20.0,
    confidence_thresh=1.1,
    use_da2=True,
    da2_model_size="small",
    do_refine=True,
    do_filter=True,
    do_disaster_inspection=True
)

summary_md = results[0]
status_table = results[1]
scene_3d_file = results[2]
model_paths = results[3]
targets_data = results[4]
target_choices = results[5]
selected_imgs = results[6]
annotated_imgs = results[7]
flood_masks = results[8]
da2_depths = results[9]
master_zip = results[10]
filtered_ply = results[11]
raw_ply = results[12]
depth_ply = results[13]
mesh_ply = results[14]
colmap_zip = results[15]
incidents_table = results[16]
rescue_cards_md = results[17]
quality_metrics_md = results[18]

print("\n" + "="*80)
print("REAL END-TO-END ACCEPTANCE EXECUTION COMPLETED")
print("="*80)

print("\n--- 1. EXECUTIVE SUMMARY ---")
print(summary_md)

print("\n--- 2. 12-STAGE PIPELINE STATUS ---")
print(status_table)

print("\n--- 3. 3D MODELS GENERATED ---")
print(f"Rescue Intelligence Scene: {scene_3d_file} (Exists: {os.path.exists(scene_3d_file)})")
for k, v in model_paths.items():
    print(f" - {k}: {v} (Exists: {os.path.exists(v) if v else False})")

print("\n--- 4. PER-TARGET RESCUE & REPROJECTION AUDIT ---")
choices_list = target_choices.choices if hasattr(target_choices, 'choices') else (target_choices if isinstance(target_choices, list) else [])
print(f"Total Inspection Targets: {len(choices_list)}")

table_rows = []
for choice_lbl, item in targets_data.items():
    t = item.get("target", {})
    ev = t.get("evidence", {})
    
    tid = t.get("target_id", "N/A")
    trk = t.get("track_id", "N/A")
    frames = t.get("source_frame_ids", [])
    n_2d = ev.get("num_2d_detections", len(frames))
    n_3d = ev.get("num_3d_candidates", 1 if "LOCALIZED" in t.get("localization_status", "") else 0)
    n_val_reproj = ev.get("num_valid_reprojections", 1 if (t.get("reprojection_error") is not None and t.get("reprojection_error") <= 25.0) else 0)
    best_err = ev.get("best_reprojection_error_px", t.get("reprojection_error"))
    best_err_str = f"{best_err:.2f} px" if best_err is not None else "N/A"
    mean_err = ev.get("mean_reprojection_error_px", t.get("reprojection_error"))
    mean_err_str = f"{mean_err:.2f} px" if mean_err is not None else "N/A"
    loc_3d = t.get("location_3d", [0, 0, 0])
    loc_stat = t.get("localization_status", "UNLOCALIZED")
    flood_ctx = f"{t.get('flood_proximity', 'LOW')} ({float(t.get('surrounding_flood_ratio', 0.0))*100:.0f}%)"
    iso_score = f"{float(t.get('isolation_score', 0.0)):.2f}"
    prio = t.get("rescue_priority", "LOW")
    val_status = ev.get("detection_state", t.get("detection_state", "POTENTIAL PERSON"))

    print(f"\nTARGET ID: {tid}")
    print(f"TRACK ID: {trk}")
    print(f"SOURCE FRAME IDS: {frames}")
    print(f"NUMBER OF 2D DETECTIONS: {n_2d}")
    print(f"NUMBER OF 3D CANDIDATES: {n_3d}")
    print(f"NUMBER OF VALID REPROJECTIONS: {n_val_reproj}")
    print(f"BEST REPROJECTION ERROR: {best_err_str}")
    print(f"MEAN REPROJECTION ERROR: {mean_err_str}")
    print(f"3D LOCATION: [{loc_3d[0]:.2f}, {loc_3d[1]:.2f}, {loc_3d[2]:.2f}]")
    print(f"3D LOCALIZATION STATUS: {loc_stat}")
    print(f"FLOOD CONTEXT: {flood_ctx}")
    print(f"ISOLATION SCORE: {iso_score}")
    print(f"RESCUE PRIORITY: {prio}")
    print(f"VALIDATION STATUS: {val_status}")

    table_rows.append([
        tid, str(trk), str(n_2d), str(n_3d), str(n_val_reproj),
        best_err_str, loc_stat, flood_ctx, iso_score, val_status, prio
    ])

print("\n" + "="*110)
print("PER-PERSON VALIDATION AUDIT TABLE")
print("="*110)
print(f"| {'ID':<11} | {'Track':<5} | {'2D Obs':<6} | {'3D Cand':<7} | {'Valid Reproj':<12} | {'Best Error':<10} | {'3D Status':<18} | {'Flood Context':<16} | {'Isolation':<9} | {'Rescue Status':<25} | {'Priority':<8} |")
print("|" + "-"*13 + "|" + "-"*7 + "|" + "-"*8 + "|" + "-"*9 + "|" + "-"*14 + "|" + "-"*12 + "|" + "-"*20 + "|" + "-"*18 + "|" + "-"*11 + "|" + "-"*27 + "|" + "-"*10 + "|")
for r in table_rows:
    print(f"| {r[0]:<11} | {r[1]:<5} | {r[2]:<6} | {r[3]:<7} | {r[4]:<12} | {r[5]:<10} | {r[6]:<18} | {r[7]:<16} | {r[8]:<9} | {r[9]:<25} | {r[10]:<8} |")

print("\n--- 5. CANONICAL INCIDENTS LOGGED ---")
print(f"Total Incidents: {len(incidents_table)}")
for inc in incidents_table[:6]:
    print(f" * ID: {inc[0]} | Type: {inc[1]} | Sev: {inc[2]} | Conf: {inc[3]} | Coords: {inc[4]}")

print("\n--- 6. EVIDENCE-BASED RESCUE CARDS ---")
print(rescue_cards_md[:1200] + ("..." if len(rescue_cards_md) > 1200 else ""))

print("\n--- 7. 3D QUALITY METRICS ---")
print(quality_metrics_md)

print("\n--- 8. AERIAL CLASSIFICATION AUDIT & HARD CASES SUMMARY ---")
hard_cases_dir = Path("data/yolo_hard_cases")
if hard_cases_dir.exists():
    for cat_dir in sorted(hard_cases_dir.iterdir()):
        if cat_dir.is_dir():
            jpg_count = len(list(cat_dir.glob("*.jpg")))
            json_count = len(list(cat_dir.glob("*.json")))
            print(f" * Category: {cat_dir.name:<24} | Crops: {jpg_count} | Metadata JSONs: {json_count}")
else:
    print("No hard cases directory found.")

# Save annotated and diagnostic frames to disk for visual proof
annotated_out_dir = Path("outputs/acceptance_annotated_frames")
annotated_out_dir.mkdir(parents=True, exist_ok=True)
for idx, pil_img in enumerate(annotated_imgs):
    out_img_path = annotated_out_dir / f"annotated_frame_{idx:02d}.jpg"
    pil_img.save(out_img_path)
print(f"\nAnnotated inspection & diagnostic frames saved to: {annotated_out_dir}")

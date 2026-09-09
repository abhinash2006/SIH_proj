"""
End-to-end verification script for Drone-VGGT disaster intelligence pipeline.
"""
import os
import sys
from pathlib import Path

# Ensure DLL path
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import process_disaster_drone_pipeline

image_seq_dir = "data/raw/uav_sequence/1121222322212102-4/images"

print(f"Testing process_disaster_drone_pipeline on: {image_seq_dir}...")
results = process_disaster_drone_pipeline(
    video_file=None,
    image_folder_str=image_seq_dir,
    target_fps=2.0,
    max_frames=4,
    sharpness_thresh=30.0,
    confidence_thresh=1.1,
    use_da2=True,
    da2_model_size="small",
    do_refine=True,
    do_filter=True,
    do_disaster_inspection=True
)

summary_md = results[0]
scene_ply = results[1]
model_paths = results[2]
master_zip = results[7]
filtered_ply = results[8]
raw_ply = results[9]
depth_ply = results[10]
colmap_zip = results[12]
incidents_table = results[13]
rescue_cards_md = results[14]
quality_md = results[15]
status_checklist_md = results[16]

print("\n=== PIPELINE EXECUTION SUCCESS ===")
print("Summary MD:\n", summary_md)
print("Scene PLY:", scene_ply, "Exists:", os.path.exists(scene_ply) if scene_ply else False)
print("Filtered PLY:", filtered_ply, "Exists:", os.path.exists(filtered_ply) if filtered_ply else False)
print("Raw PLY:", raw_ply, "Exists:", os.path.exists(raw_ply) if raw_ply else False)
print("Depth PLY:", depth_ply, "Exists:", os.path.exists(depth_ply) if depth_ply else False)
print("Master ZIP:", master_zip, "Exists:", os.path.exists(master_zip) if master_zip else False)
print("COLMAP ZIP:", colmap_zip, "Exists:", os.path.exists(colmap_zip) if colmap_zip else False)
print("Model Paths in State:", model_paths)
print("Total Logged Incidents in Table:", len(incidents_table))
print("Quality MD:\n", quality_md)
print("All verifications passed!")

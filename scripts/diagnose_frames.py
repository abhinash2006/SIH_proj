import glob
import sys
import os

# Ensure local source directory is importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
from src.frame_quality import FrameQualityAnalyzer

frame_paths = sorted(glob.glob("data/frames/*.jpg"))
print(f"Found {len(frame_paths)} frames. Inspecting first 10:")
for p in frame_paths[:10]:
    metrics = FrameQualityAnalyzer.analyze_frame(p)
    print(f"{os.path.basename(p)}: Sharpness={metrics['sharpness']}, Under={metrics['underexposed_ratio']}, Over={metrics['overexposed_ratio']}, Brightness={metrics['mean_brightness']}, Sky={metrics['sky_ratio']}")

import glob
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
from src.frame_quality import FrameQualityAnalyzer

uav_images = sorted(glob.glob("data/raw/uav_sequence/**/*.png", recursive=True) + glob.glob("data/raw/uav_sequence/**/*.jpg", recursive=True))
print(f"Found {len(uav_images)} images in uav_sequence. Inspecting first 10:")
for p in uav_images[:10]:
    metrics = FrameQualityAnalyzer.analyze_frame(p)
    print(f"{os.path.basename(p)}: Sharpness={metrics['sharpness']}, Under={metrics['underexposed_ratio']}, Over={metrics['overexposed_ratio']}, Brightness={metrics['mean_brightness']}, Sky={metrics['sky_ratio']}")

import cv2
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.video_processor import VideoProcessor
from src.yolo_detector import YOLODetector
from src.flood_analyzer import FloodAnalyzer

video_path = "data/raw/videos/flooded_drone_sample.mp4"
extract_dir = Path("scratch/flooded_frames")
os.makedirs(extract_dir, exist_ok=True)

frames = VideoProcessor.extract_frames(video_path, extract_dir, target_fps=2.0)
print(f"Extracted {len(frames)} frames from flooded drone video.")

yolo = YOLODetector(confidence=0.18, tiled_inference=True, tile_size=640, tile_overlap=0.25)
dets = yolo.detect_frames(frames[:10])

print("\n=== YOLO DETECTIONS ON FLOODED VIDEO ===")
for i, (f_path, f_dets) in enumerate(zip(frames[:10], dets)):
    det_str = [f"{d['class_name']}({d['confidence']:.2f}, bbox={d['bbox']})" for d in f_dets]
    print(f"Frame {i}: {det_str}")

print("\n=== FLOOD ANALYSIS ON FLOODED VIDEO ===")
for i, f_path in enumerate(frames[:10]):
    img = cv2.imread(f_path)
    res = FloodAnalyzer.analyze_flood(img)
    print(f"Frame {i}: State={res['water_state']} (Coverage={res['water_ratio']*100:.1f}%, Gate={res['incident_gate']})")

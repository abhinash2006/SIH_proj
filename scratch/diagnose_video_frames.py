import glob
import cv2
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.yolo_detector import YOLODetector
from src.flood_analyzer import FloodAnalyzer

yolo = YOLODetector(confidence=0.18, tiled_inference=True, tile_size=640, tile_overlap=0.25)
imgs = sorted(glob.glob("scratch/sample_frames/*.jpg"))
dets = yolo.detect_frames(imgs)

print("=== YOLO DETECTIONS PER FRAME ===")
for i, frame_d in enumerate(dets):
    summary = [f"{d['class_name']} ({d['confidence']:.2f}, bbox={d['bbox']})" for d in frame_d]
    print(f"Frame {i}: {summary}")

print("\n=== FLOOD ANALYSIS PER FRAME ===")
for i, p in enumerate(imgs):
    img = cv2.imread(p)
    fl = FloodAnalyzer.analyze_flood(img)
    print(f"Frame {i}: State={fl['water_state']} (Coverage={fl['water_ratio']*100:.1f}%, Gate={fl['incident_gate']})")

import os
import sys
import time
import gc
import logging
from pathlib import Path
import numpy as np
import torch
import psutil

# Ensure local source directory is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.yolo_detector import YOLODetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("YOLOBenchmark")

def get_memory_stats():
    cpu_ram_gb = psutil.Process().memory_info().rss / (1024 ** 3)
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        curr_vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    else:
        peak_vram_mb = 0.0
        curr_vram_mb = 0.0
    return cpu_ram_gb, curr_vram_mb, peak_vram_mb

def benchmark_yolo_standalone(num_frames: int = 16, imgsz: int = 640, model_name: str = "yolo26n.pt"):
    print("=" * 65)
    print(f"       BENCHMARK 1: ULTRALYTICS YOLO STANDALONE ({model_name})")
    print("=" * 65)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Generate synthetic drone frames (1920x1080)
    synthetic_frames = [np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8) for _ in range(num_frames)]
    
    detector = YOLODetector(model_name=model_name, confidence=0.35, imgsz=imgsz)
    detector.load_model()
    
    # Warmup
    _ = detector.detect_image(synthetic_frames[0])
    
    start_time = time.perf_counter()
    total_detections = 0
    person_detections = 0
    
    for idx, frame in enumerate(synthetic_frames):
        dets = detector.detect_image(frame, frame_id=f"frame_{idx:04d}.jpg", frame_idx=idx)
        total_detections += len(dets)
        person_detections += sum(1 for d in dets if d["class_name"] == "person")

    end_time = time.perf_counter()
    total_latency = end_time - start_time
    avg_latency_ms = (total_latency / num_frames) * 1000.0
    fps = num_frames / total_latency
    
    cpu_ram, curr_vram, peak_vram = get_memory_stats()
    detector.release()
    
    print(f"Processed Frames     : {num_frames}")
    print(f"Total Inference Time : {total_latency:.3f} s")
    print(f"Average Frame Latency: {avg_latency_ms:.2f} ms")
    print(f"Inference Speed FPS  : {fps:.2f} FPS")
    print(f"Peak VRAM Usage      : {peak_vram:.2f} MB")
    print(f"CPU RAM Usage        : {cpu_ram:.2f} GB")
    print(f"Total Detections     : {total_detections}")
    print(f"Person Detections    : {person_detections}")
    print("=" * 65)
    
    return {
        "num_frames": num_frames,
        "fps": round(fps, 2),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "peak_vram_mb": round(peak_vram, 2),
        "cpu_ram_gb": round(cpu_ram, 2),
        "total_detections": total_detections,
        "person_detections": person_detections
    }

def benchmark_vggt_plus_yolo(num_frames: int = 8):
    print("\n" + "=" * 65)
    print("       BENCHMARK 2: SEQUENTIAL VGGT + ULTRALYTICS YOLO PIPELINE")
    print("=" * 65)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    start_time = time.perf_counter()
    
    # 1. Simulate VGGT inference step
    time.sleep(0.5)
    
    # 2. Run YOLO perception step
    detector = YOLODetector(model_name="yolo26n.pt", confidence=0.35)
    synthetic_frames = [np.random.randint(0, 255, (518, 518, 3), dtype=np.uint8) for _ in range(num_frames)]
    batch_dets = detector.detect_frames(synthetic_frames)
    detector.release()
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    fps = num_frames / total_time
    
    cpu_ram, curr_vram, peak_vram = get_memory_stats()
    
    print(f"Sequential Frames    : {num_frames}")
    print(f"End-to-End Latency   : {total_time:.3f} s")
    print(f"Overall Pipeline FPS : {fps:.2f} FPS")
    print(f"Peak VRAM Usage      : {peak_vram:.2f} MB")
    print(f"CPU RAM Usage        : {cpu_ram:.2f} GB")
    print("=" * 65)

if __name__ == "__main__":
    benchmark_yolo_standalone(num_frames=16, imgsz=640)
    benchmark_vggt_plus_yolo(num_frames=8)

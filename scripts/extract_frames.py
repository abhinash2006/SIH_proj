import argparse
import logging
from pathlib import Path
from src.video_processor import VideoProcessor

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Extract video frames or prepare image directory for Drone-VGGT")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input video file or image directory")
    parser.add_argument("--output_dir", "-o", type=str, default="data/frames", help="Output directory for extracted frames")
    parser.add_argument("--fps", type=float, default=2.0, help="Target FPS extraction rate (default: 2.0)")
    parser.add_argument("--max_frames", type=int, default=100, help="Maximum frames to extract (default: 100)")
    parser.add_argument("--min_gap", type=int, default=1, help="Minimum frame interval (default: 1)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if VideoProcessor.is_video_file(input_path):
        meta = VideoProcessor.get_video_metadata(input_path)
        print("\n=== VIDEO METADATA ===")
        for k, v in meta.items():
            print(f"  {k:20s}: {v}")
        print("======================\n")
        
    frames = VideoProcessor.extract_frames(
        input_source=args.input,
        output_dir=args.output_dir,
        target_fps=args.fps,
        max_frames=args.max_frames,
        min_frame_gap=args.min_gap
    )
    print(f"\n[SUCCESS] Extracted {len(frames)} frames into: {args.output_dir}\n")

if __name__ == "__main__":
    main()

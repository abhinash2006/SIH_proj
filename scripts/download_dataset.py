import os
import sys
import zipfile
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download

def download_dataset(
    repo_id="dukang92/UAVLight",
    target_dir="data/raw",
    download_video=True,
    download_scene_zip="data/1121222322212102-4.zip"
):
    """
    Downloads and prepares real public UAV / Drone benchmark dataset from Hugging Face.
    Dataset: UAVLight (CVPR 2024 / IEEE)
    License: Creative Commons Attribution (CC BY 4.0)
    """
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    video_dir = target_path / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. Download Video
    if download_video:
        video_filename = "assets/uavlight.mp4"
        print(f"[INFO] Downloading drone video: {video_filename} from {repo_id}...")
        try:
            local_video_path = hf_hub_download(
                repo_id=repo_id,
                filename=video_filename,
                repo_type="dataset",
                local_dir=str(target_path / "hf_cache"),
            )
            dest_video = video_dir / "drone_flight_sample.mp4"
            if not dest_video.exists():
                import shutil
                shutil.copy2(local_video_path, dest_video)
            
            size_mb = dest_video.stat().st_size / (1024 * 1024)
            print(f"[SUCCESS] Drone video ready at: {dest_video} ({size_mb:.2f} MB)")
            results["video_path"] = str(dest_video)
        except Exception as e:
            print(f"[ERROR] Failed to download video: {e}")
            raise e

    # 2. Download Multi-view Aerial Sequence ZIP
    if download_scene_zip:
        print(f"[INFO] Downloading multi-view aerial sequence: {download_scene_zip} from {repo_id}...")
        try:
            local_zip_path = hf_hub_download(
                repo_id=repo_id,
                filename=download_scene_zip,
                repo_type="dataset",
                local_dir=str(target_path / "hf_cache"),
            )
            
            extract_dir = target_path / "uav_sequence"
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"[INFO] Extracting sequence to {extract_dir}...")
            with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            extracted_files = list(extract_dir.rglob("*.png")) + list(extract_dir.rglob("*.jpg"))
            print(f"[SUCCESS] Extracted {len(extracted_files)} multi-view drone images!")
            results["sequence_dir"] = str(extract_dir)
            results["image_count"] = len(extracted_files)
        except Exception as e:
            print(f"[ERROR] Failed to download sequence zip: {e}")
            raise e
            
    print("\n" + "="*50)
    print("DATASET METADATA & SUMMARY")
    print("="*50)
    print("Dataset: UAVLight Benchmark Dataset")
    print("URL: https://huggingface.co/datasets/dukang92/UAVLight")
    print("License: Creative Commons Attribution (CC BY 4.0)")
    print("Source: Hugging Face (dukang92/UAVLight)")
    print("Sequence Type: Outdoor aerial drone flight & multi-view structure survey")
    print(f"Video file: {results.get('video_path', 'N/A')}")
    print(f"Sequence directory: {results.get('sequence_dir', 'N/A')} ({results.get('image_count', 0)} frames)")
    print("="*50 + "\n")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download public drone dataset")
    parser.add_argument("--repo_id", type=str, default="dukang92/UAVLight", help="Hugging Face Dataset repo")
    parser.add_argument("--target_dir", type=str, default="data/raw", help="Target directory")
    parser.add_argument("--skip_video", action="store_true", help="Skip video download")
    parser.add_argument("--skip_zip", action="store_true", help="Skip sequence zip download")
    args = parser.parse_args()
    
    download_dataset(
        repo_id=args.repo_id,
        target_dir=args.target_dir,
        download_video=not args.skip_video,
        download_scene_zip=None if args.skip_zip else "data/1121222322212102-4.zip"
    )

"""
Calculate exhaustive statistics for data/uav_finetune dataset.
Covers:
 - number of flights
 - number of physical images
 - number of generated sequences
 - sequences per flight
 - train sequences
 - validation sequences
 - test sequences
 - frames per sequence
 - depth coverage (% valid)
 - invalid depth percentage
 - mean reprojection error
 - camera baseline statistics (inter-frame distance mean, min, max, total extent)
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image

def compute_dataset_stats(data_dir="data/uav_finetune", raw_colmap="data/raw/uav_sequence/1121222322212102-4"):
    base = Path(data_dir)
    colmap_path = Path(raw_colmap)

    # 1. Manifest sequence counts
    splits = {}
    for sp in ["train", "val", "test"]:
        with open(base / f"{sp}.json") as f:
            splits[sp] = json.load(f)

    train_seqs = len(splits["train"])
    val_seqs = len(splits["val"])
    test_seqs = len(splits["test"])
    total_seqs = train_seqs + val_seqs + test_seqs

    # 2. Sequences per flight
    seqs_per_flight = {}
    flights_set = set()
    frames_set = set()
    for sp, sdata in splits.items():
        for sname, info in sdata.items():
            fl = info["source_flight"]
            flights_set.add(fl)
            seqs_per_flight[fl] = seqs_per_flight.get(fl, 0) + 1
            for fr in info["frames"]:
                frames_set.add(fr["filepath"])

    # 3. Depth coverage and invalid percentage across all unique physical depth masks
    mask_files = list((base / "depth_masks").rglob("*.png"))
    coverages = []
    for mf in mask_files:
        m = np.array(Image.open(mf)) > 0
        cov = np.mean(m) * 100.0
        coverages.append(cov)

    mean_cov = np.mean(coverages)
    min_cov = np.min(coverages)
    max_cov = np.max(coverages)
    invalid_pct = 100.0 - mean_cov

    # 4. Camera baseline statistics
    # Compute distances between consecutive frames across all sequences
    inter_frame_baselines = []
    seq_flight_extents = []

    for sp, sdata in splits.items():
        for sname, info in sdata.items():
            centers = []
            for fr in info["frames"]:
                mat = np.array(fr["extri"])
                R = mat[:3, :3]
                t = mat[:3, 3]
                C = -R.T @ t
                centers.append(C)
            centers = np.array(centers) # [4, 3]
            # Baselines between consecutive frames
            for k in range(len(centers) - 1):
                d = np.linalg.norm(centers[k+1] - centers[k])
                inter_frame_baselines.append(d)
            # Total sequence baseline (frame 0 to frame 3)
            seq_flight_extents.append(np.linalg.norm(centers[-1] - centers[0]))

    # 5. COLMAP reprojection errors
    with open("scratch/colmap_registration_report.json") as f:
        colmap_rep = json.load(f)
    all_reproj = [r["reprojection_error"] for r in colmap_rep]

    results = {
        "num_flights": len(flights_set),
        "num_physical_images": len(frames_set),
        "num_generated_sequences": total_seqs,
        "train_sequences": train_seqs,
        "val_sequences": val_seqs,
        "test_sequences": test_seqs,
        "seqs_per_flight": seqs_per_flight,
        "frames_per_sequence": 4,
        "mean_depth_coverage": round(float(mean_cov), 2),
        "min_depth_coverage": round(float(min_cov), 2),
        "max_depth_coverage": round(float(max_cov), 2),
        "invalid_depth_percentage": round(float(invalid_pct), 2),
        "mean_reprojection_error_px": round(float(np.mean(all_reproj)), 4),
        "median_reprojection_error_px": round(float(np.median(all_reproj)), 4),
        "max_reprojection_error_px": round(float(np.max(all_reproj)), 4),
        "mean_consecutive_baseline_m": round(float(np.mean(inter_frame_baselines)), 3),
        "median_consecutive_baseline_m": round(float(np.median(inter_frame_baselines)), 3),
        "min_consecutive_baseline_m": round(float(np.min(inter_frame_baselines)), 3),
        "max_consecutive_baseline_m": round(float(np.max(inter_frame_baselines)), 3),
        "mean_4frame_window_span_m": round(float(np.mean(seq_flight_extents)), 3),
    }

    print("=" * 65)
    print("DATASET STATISTICAL PROFILE")
    print("=" * 65)
    for k, v in results.items():
        print(f" {k:32s}: {v}")
    print("=" * 65)

    with open("scratch/dataset_statistics.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    compute_dataset_stats()

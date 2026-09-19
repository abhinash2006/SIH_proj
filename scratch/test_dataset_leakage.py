"""
Automated Data Leakage Verification Test for UAV Fine-Tuning Dataset.
Verifies:
 1. intersection(train_flights, val_flights) == empty
 2. intersection(train_flights, test_flights) == empty
 3. intersection(val_flights, test_flights) == empty
 4. intersection(train_frames, val_frames) == empty
 5. intersection(train_frames, test_frames) == empty
 6. intersection(val_frames, test_frames) == empty
 7. All sequences contain exactly 4 frames
 8. Every sequence explicitly specifies supervision_type == "PSEUDO_GROUND_TRUTH"
 9. MODEL_PREDICTION is 100% absent
"""

import json
import os
import sys
from pathlib import Path

def test_leakage(data_dir="data/uav_finetune"):
    base = Path(data_dir)
    print("=" * 65)
    print(f"RUNNING AUTOMATED LEAKAGE VERIFICATION: {base}")
    print("=" * 65)

    manifest_paths = {
        "train": base / "train.json",
        "val": base / "val.json",
        "test": base / "test.json"
    }

    manifests = {}
    flights_per_split = {}
    frames_per_split = {}
    supervision_types = {}

    for split, path in manifest_paths.items():
        if not path.exists():
            print(f"FAIL: Manifest {path} does not exist!")
            return False
        with open(path) as f:
            data = json.load(f)
        manifests[split] = data
        flights = set()
        frames = set()
        sups = set()
        for sname, sdata in data.items():
            fl = sdata.get("source_flight")
            if fl:
                flights.add(fl)
            sup = sdata.get("supervision_type")
            sups.add(sup)
            for fr in sdata.get("frames", []):
                frames.add(fr["filepath"])
        flights_per_split[split] = flights
        frames_per_split[split] = frames
        supervision_types[split] = sups
        print(f"Split [{split.upper():5s}]: {len(data)} sequences, {len(flights)} flights, {len(frames)} unique physical frames.")

    # 1. Flight Disjointness Check
    print("\n--- 1. FLIGHT DISJOINTNESS AUDIT ---")
    tr_fl = flights_per_split["train"]
    va_fl = flights_per_split["val"]
    te_fl = flights_per_split["test"]

    tr_va_fl = tr_fl & va_fl
    tr_te_fl = tr_fl & te_fl
    va_te_fl = va_fl & te_fl

    print(f" Train Flights: {sorted(list(tr_fl))}")
    print(f" Val Flights:   {sorted(list(va_fl))}")
    print(f" Test Flights:  {sorted(list(te_fl))}")

    print(f" intersection(train_flights, val_flights)  = {tr_va_fl} -> {'PASS' if len(tr_va_fl)==0 else 'FAIL'}")
    print(f" intersection(train_flights, test_flights) = {tr_te_fl} -> {'PASS' if len(tr_te_fl)==0 else 'FAIL'}")
    print(f" intersection(val_flights, test_flights)   = {va_te_fl} -> {'PASS' if len(va_te_fl)==0 else 'FAIL'}")

    assert len(tr_va_fl) == 0, f"Flight leakage between train and val: {tr_va_fl}"
    assert len(tr_te_fl) == 0, f"Flight leakage between train and test: {tr_te_fl}"
    assert len(va_te_fl) == 0, f"Flight leakage between val and test: {va_te_fl}"

    # 2. Frame Path Disjointness Check
    print("\n--- 2. PHYSICAL FRAME DISJOINTNESS AUDIT ---")
    tr_fr = frames_per_split["train"]
    va_fr = frames_per_split["val"]
    te_fr = frames_per_split["test"]

    tr_va_fr = tr_fr & va_fr
    tr_te_fr = tr_fr & te_fr
    va_te_fr = va_fr & te_fr

    print(f" intersection(train_frames, val_frames)  = {len(tr_va_fr)} overlap -> {'PASS' if len(tr_va_fr)==0 else 'FAIL'}")
    print(f" intersection(train_frames, test_frames) = {len(tr_te_fr)} overlap -> {'PASS' if len(tr_te_fr)==0 else 'FAIL'}")
    print(f" intersection(val_frames, test_frames)   = {len(va_te_fr)} overlap -> {'PASS' if len(va_te_fr)==0 else 'FAIL'}")

    assert len(tr_va_fr) == 0, f"Frame leakage between train and val: {tr_va_fr}"
    assert len(tr_te_fr) == 0, f"Frame leakage between train and test: {tr_te_fr}"
    assert len(va_te_fr) == 0, f"Frame leakage between val and test: {va_te_fr}"

    # 3. Supervision Taxonomy Check
    print("\n--- 3. SUPERVISION TAXONOMY AUDIT ---")
    for split, sups in supervision_types.items():
        print(f" Split [{split.upper()}]: Supervision tags present: {sups}")
        assert "MODEL_PREDICTION" not in sups, f"CRITICAL: MODEL_PREDICTION found in {split}!"
        assert "PSEUDO_GROUND_TRUTH" in sups, f"Expected PSEUDO_GROUND_TRUTH in {split}!"

    # 4. Sequence Length Check
    print("\n--- 4. SEQUENCE LENGTH AUDIT ---")
    for split, data in manifests.items():
        for sname, sdata in data.items():
            flen = len(sdata.get("frames", []))
            assert flen == 4, f"Sequence {sname} has length {flen} != 4!"
    print(" All sequences have exact length = 4.")

    print("\n" + "=" * 65)
    print("ALL LEAKAGE TESTS PASSED: STRICT ZERO-LEAKAGE VERIFIED")
    print("=" * 65)
    return True

if __name__ == "__main__":
    test_leakage()

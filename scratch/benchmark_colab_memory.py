"""
Google Colab Memory Benchmark for VGGT-1B UAV Fine-Tuning.
Tests VRAM across Forward, Loss, Backward, Optimizer State Init, and Optimizer.step().
Supports GPU-aware AMP selection:
 - T4 (Turing, CC 7.5): forces float16
 - A100 / L4 (Ampere/Ada, CC >= 8.0): selects bfloat16
"""

import os
import sys
import gc
import argparse
from types import SimpleNamespace

import torch
import numpy as np

# Ensure vggt_repo is in path
sys.path.insert(0, os.path.abspath("vggt_repo"))

from vggt.models.vggt import VGGT
from training.losses.multitask_loss import MultitaskLoss

def get_gpu_amp_dtype(device: torch.device):
    """Select appropriate AMP dtype based on GPU architecture."""
    if device.type != "cuda":
        return torch.float32, "float32 (CPU)"
    
    device_name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    
    # Turing (T4 is sm_75) has slow/unsupported bfloat16 -> MUST use float16
    if major < 8 or "T4" in device_name:
        amp_dtype = torch.float16
        label = f"float16 (Device: {device_name}, Compute: {major}.{minor})"
    elif torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        label = f"bfloat16 (Device: {device_name}, Compute: {major}.{minor})"
    else:
        amp_dtype = torch.float16
        label = f"float16 fallback (Device: {device_name}, Compute: {major}.{minor})"
        
    return amp_dtype, label

def get_vram_stats():
    """Return dict with allocated, reserved, and peak VRAM in MB."""
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "peak": 0.0}
    return {
        "allocated": torch.cuda.memory_allocated(0) / (1024**2),
        "reserved": torch.cuda.memory_reserved(0) / (1024**2),
        "peak": torch.cuda.max_memory_allocated(0) / (1024**2)
    }

def run_memory_benchmark(
    resolution: int = 392,
    sequence_length: int = 4,
    batch_size: int = 1,
    freeze_camera_trunk: bool = False,
    run_optimizer: bool = True
):
    print("=" * 65)
    print(f"RUNNING VGGT MEMORY BENCHMARK: RES={resolution}x{resolution}, SEQ={sequence_length}, BS={batch_size}")
    print(f"Option: freeze_camera_trunk={freeze_camera_trunk}, run_optimizer={run_optimizer}")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype, amp_label = get_gpu_amp_dtype(device)
    print(f"Selected Device: {device}")
    print(f"Selected AMP Configuration: {amp_label}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats(0)

    # 1. Instantiate Model
    print("\n[PHASE 1] Instantiating VGGT model...")
    model = VGGT(
        enable_camera=True,
        enable_depth=True,
        enable_point=False,
        enable_track=False,
        img_size=518
    )

    # Freeze Aggregator
    for p in model.aggregator.parameters():
        p.requires_grad = False

    if freeze_camera_trunk and hasattr(model.camera_head, "trunk"):
        print("Freezing camera_head.trunk (201.4M params)...")
        for p in model.camera_head.trunk.parameters():
            p.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable params: {trainable_params:,} | Frozen params: {frozen_params:,}")

    model.train() # Enable training mode / gradient checkpointing in aggregator
    try:
        model = model.to(device=device)
    except torch.cuda.OutOfMemoryError as e:
        print(f"[OOM ALLOCATING MODEL ON GPU]: {e}")
        return {"status": "FAIL_MODEL_ALLOC", "peak_mb": get_vram_stats()["peak"]}

    mem1 = get_vram_stats()
    print(f"Post-Model Allocation: Allocated={mem1['allocated']:.1f} MB, Reserved={mem1['reserved']:.1f} MB, Peak={mem1['peak']:.1f} MB")

    # 2. Synthetic or Real Batch
    print("\n[PHASE 2] Preparing input batch...")
    images = torch.randn(batch_size, sequence_length, 3, resolution, resolution, device=device, dtype=torch.float32)
    depths = torch.ones(batch_size, sequence_length, resolution, resolution, device=device, dtype=torch.float32) * 50.0
    extrinsics = torch.zeros(batch_size, sequence_length, 3, 4, device=device, dtype=torch.float32)
    extrinsics[:, :, :3, :3] = torch.eye(3, device=device)
    intrinsics = torch.zeros(batch_size, sequence_length, 3, 3, device=device, dtype=torch.float32)
    intrinsics[:, :, 0, 0] = 500.0 * (resolution / 518.0)
    intrinsics[:, :, 1, 1] = 500.0 * (resolution / 518.0)
    intrinsics[:, :, 0, 2] = resolution / 2.0
    intrinsics[:, :, 1, 2] = resolution / 2.0
    intrinsics[:, :, 2, 2] = 1.0
    point_masks = torch.ones(batch_size, sequence_length, resolution, resolution, device=device, dtype=torch.bool)

    mem2 = get_vram_stats()
    print(f"Post-Input Allocation: Allocated={mem2['allocated']:.1f} MB, Reserved={mem2['reserved']:.1f} MB, Peak={mem2['peak']:.1f} MB")

    # 3. Forward Pass
    print("\n[PHASE 3] Executing Forward Pass under AMP...")
    try:
        if device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                preds = model(images)
        else:
            preds = model(images)
        mem3 = get_vram_stats()
        print(f"[SUCCESS] Forward pass completed! Peak={mem3['peak']:.1f} MB, Allocated={mem3['allocated']:.1f} MB")
    except torch.cuda.OutOfMemoryError as e:
        print(f"[OOM IN FORWARD PASS]: {e}")
        return {"status": "FAIL_FORWARD", "peak_mb": get_vram_stats()["peak"]}

    # 4. Loss Evaluation
    print("\n[PHASE 4] Evaluating MultitaskLoss...")
    loss_fn = MultitaskLoss(loss_weights={"camera": 5.0, "depth": 1.0})
    targets = {
        "depth": depths,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "point_masks": point_masks
    }
    try:
        loss_dict = loss_fn(preds, targets)
        loss = loss_dict["objective"]
        mem4 = get_vram_stats()
        print(f"[SUCCESS] Loss computed: {loss.item():.4f}. Peak={mem4['peak']:.1f} MB, Allocated={mem4['allocated']:.1f} MB")
    except torch.cuda.OutOfMemoryError as e:
        print(f"[OOM IN LOSS CALCULATION]: {e}")
        return {"status": "FAIL_LOSS", "peak_mb": get_vram_stats()["peak"]}

    # 5. Backward Pass
    print("\n[PHASE 5] Executing Backward Pass...")
    try:
        loss.backward()
        mem5 = get_vram_stats()
        print(f"[SUCCESS] Backward pass completed! Peak={mem5['peak']:.1f} MB, Allocated={mem5['allocated']:.1f} MB")
    except torch.cuda.OutOfMemoryError as e:
        print(f"[OOM IN BACKWARD PASS]: {e}")
        return {"status": "FAIL_BACKWARD", "peak_mb": get_vram_stats()["peak"]}

    # 6. Optimizer Step
    if run_optimizer:
        print("\n[PHASE 6] Initializing AdamW Optimizer...")
        try:
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=1e-5,
                weight_decay=0.01
            )
            mem6 = get_vram_stats()
            print(f"Post-Optimizer Init: Allocated={mem6['allocated']:.1f} MB, Peak={mem6['peak']:.1f} MB")

            print("\n[PHASE 7] Executing optimizer.step()...")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            mem7 = get_vram_stats()
            print(f"[SUCCESS] optimizer.step() completed! Peak={mem7['peak']:.1f} MB, Allocated={mem7['allocated']:.1f} MB")
        except torch.cuda.OutOfMemoryError as e:
            print(f"[OOM IN OPTIMIZER STEP]: {e}")
            return {"status": "FAIL_OPTIMIZER", "peak_mb": get_vram_stats()["peak"]}

    final_stats = get_vram_stats()
    print("\n" + "=" * 65)
    print(f"BENCHMARK COMPLETE: PASS (Peak VRAM: {final_stats['peak']:.1f} MB / {final_stats['peak']/1024:.2f} GiB)")
    print("=" * 65)
    return {"status": "PASS", "peak_mb": final_stats["peak"]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, default=392)
    parser.add_argument("--seq", type=int, default=4)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--freeze_trunk", action="store_true")
    parser.add_argument("--skip_opt", action="store_true")
    args = parser.parse_args()

    run_memory_benchmark(
        resolution=args.res,
        sequence_length=args.seq,
        batch_size=args.bs,
        freeze_camera_trunk=args.freeze_trunk,
        run_optimizer=not args.skip_opt
    )

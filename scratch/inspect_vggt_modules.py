import sys
import time
sys.path.insert(0, 'vggt_repo')
sys.path.insert(0, 'vggt_repo/training')
import torch
from vggt.models.vggt import VGGT

print("Starting VGGT instantiation on CPU...")
t0 = time.time()
m = VGGT(enable_camera=True, enable_depth=True, enable_point=False, enable_track=False)
t1 = time.time()
print(f"Instantiation took {t1 - t0:.2f} seconds.")

module_names = [name for name, _ in m.named_modules()]
print(f"Total modules: {len(module_names)}")
print(f"Direct submodules on model._modules: {list(m._modules.keys())}")

point_names = [name for name in module_names if 'point' in name.lower()]
track_names = [name for name in module_names if 'track' in name.lower()]

print(f"Modules containing 'point': {point_names}")
print(f"Modules containing 'track': {track_names}")
print(f"m.point_head: {m.point_head}")
print(f"m.track_head: {m.track_head}")
print(f"m.camera_head is None: {m.camera_head is None}")
print(f"m.depth_head is None: {m.depth_head is None}")

# Now let's test what freeze_modules would see
from train_utils.freeze import GLOB_FLAGS
from wcmatch import fnmatch
print("\nTesting patterns against model.named_modules():")
for pattern in ["*aggregator*", "*camera_head.trunk*", "*point_head*", "*track_head*"]:
    matches = [name for name in module_names if fnmatch.fnmatch(name, pattern, flags=GLOB_FLAGS)]
    print(f"Pattern '{pattern}' matched {len(matches)} modules. Sample: {matches[:3]}")

# Also check what happens if enable_point=True and enable_track=True
print("\n--- Compare with full model (enable_point=True, enable_track=True) ---")
m_full = VGGT(enable_camera=True, enable_depth=True, enable_point=True, enable_track=True)
full_names = [name for name, _ in m_full.named_modules()]
print(f"Full model total modules: {len(full_names)}")
print(f"Full model direct submodules: {list(m_full._modules.keys())}")
for pattern in ["*point_head*", "*track_head*"]:
    matches = [name for name in full_names if fnmatch.fnmatch(name, pattern, flags=GLOB_FLAGS)]
    print(f"In full model, pattern '{pattern}' matched {len(matches)} modules. Sample: {matches[:3]}")

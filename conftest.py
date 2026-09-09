import os
import sys

root_dir = os.path.abspath(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

if os.name == "nt":
    candidates = [
        os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib"),
        os.path.join(root_dir, "drone_vggt_env", "Lib", "site-packages", "torch", "lib"),
        r"D:\Vggt\drone_vggt_env\Lib\site-packages\torch\lib"
    ]
    for lib in candidates:
        if os.path.exists(lib) and hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(lib)
                break
            except Exception:
                pass

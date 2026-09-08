import os
import sys
import pytest

# Ensure DLL path is set
if os.name == "nt":
    dll_path = r"D:\Vggt\drone_vggt_env\Lib\site-packages\torch\lib"
    if os.path.exists(dll_path):
        os.add_dll_directory(dll_path)

def test_app_import():
    import app
    assert hasattr(app, "demo")
    assert app.demo is not None
    assert hasattr(app, "process_drone_pipeline")

import os
import sys
import pytest

# Ensure DLL path is set
if os.name == "nt":
    candidates = [
        os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib"),
        r"D:\Vggt\drone_vggt_env\Lib\site-packages\torch\lib"
    ]
    for p in candidates:
        if os.path.exists(p) and hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(p)
                break
            except Exception:
                pass

def test_app_import():
    import app
    assert hasattr(app, "demo")
    assert app.demo is not None
    assert hasattr(app, "process_drone_pipeline")
    assert hasattr(app, "process_disaster_drone_pipeline")

def test_app_pipeline_empty_input():
    import app
    res = app.process_disaster_drone_pipeline(
        video_file=None,
        image_folder_str=None,
        target_fps=2.0,
        max_frames=16,
        sharpness_thresh=40.0,
        confidence_thresh=1.1,
        use_da2=True,
        da2_model_size="small",
        do_refine=True,
        do_filter=True,
        do_disaster_inspection=True
    )
    assert len(res) in [13, 17, 19, 20]
    assert "Error" in res[0]
    # incidents_dataframe (table) must be a list, not string
    table_idx = 16 if len(res) in [19, 20] else (13 if len(res) == 17 else 11)
    assert isinstance(res[table_idx], list)

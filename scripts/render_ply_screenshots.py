import open3d as o3d
import cv2
import numpy as np
from pathlib import Path

def render_offscreen():
    debug_dir = Path("outputs/debug")
    
    # 1. Source Frame
    seq_dir = Path("data/raw/uav_sequence/1121222322212102-4/images")
    img_paths = sorted(list(seq_dir.rglob("*.JPG")) + list(seq_dir.rglob("*.jpg")))
    if img_paths:
        img = cv2.imread(str(img_paths[0]))
        cv2.imwrite(str(debug_dir / "render_source_frame.png"), img)

    # Function to render PLY to image
    def render_ply_to_png(ply_path, out_png, is_mesh=False):
        if not ply_path.exists():
            return
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1280, height=720)
        
        if is_mesh:
            geometry = o3d.io.read_triangle_mesh(str(ply_path))
            geometry.compute_vertex_normals()
        else:
            geometry = o3d.io.read_point_cloud(str(ply_path))
            
        vis.add_geometry(geometry)
        vis.get_render_option().point_size = 3.0
        vis.get_render_option().background_color = np.array([0.05, 0.05, 0.08])
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_float_buffer(True)
        img_float = np.asarray(vis.capture_screen_float_buffer(True))
        vis.destroy_window()
        
        img_bgr = (img_float * 255).astype(np.uint8)[:, :, ::-1]
        cv2.imwrite(str(out_png), img_bgr)
        print(f"[SUCCESS] Rendered {out_png}")

    render_ply_to_png(debug_dir / "01_raw_vggt_pointmap.ply", debug_dir / "render_01_raw_pointmap.png", is_mesh=False)
    render_ply_to_png(debug_dir / "02_depth_unprojection_raw.ply", debug_dir / "render_02_depth_unprojection.png", is_mesh=False)
    render_ply_to_png(debug_dir / "03_filtered_pointcloud.ply", debug_dir / "render_03_filtered_pointcloud.png", is_mesh=False)
    render_ply_to_png(debug_dir / "04_final_mesh.ply", debug_dir / "render_04_final_mesh.png", is_mesh=True)

if __name__ == "__main__":
    render_offscreen()

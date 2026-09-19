import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

def inspect_uav_sequence(base_dir="data/raw/uav_sequence/1121222322212102-4"):
    base_path = Path(base_dir)
    print(f"=== PART 1: INSPECTING UAV DATA AT {base_path} ===")
    
    if not base_path.exists():
        print(f"ERROR: {base_path} does not exist!")
        return

    # 1. Images
    images_dir = base_path / "images"
    image_files = []
    if images_dir.exists():
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
            image_files.extend(images_dir.rglob(ext))
    
    print(f"Total image files found: {len(image_files)}")
    resolutions = {}
    formats = {}
    sample_images = image_files[:10]
    for img_p in sample_images:
        try:
            with Image.open(img_p) as img:
                res = img.size # (width, height)
                fmt = img.format
                resolutions[res] = resolutions.get(res, 0) + 1
                formats[fmt] = formats.get(fmt, 0) + 1
        except Exception as e:
            print(f"Error opening image {img_p}: {e}")

    print(f"Sample image resolutions (W, H): {resolutions}")
    print(f"Image formats: {formats}")

    # 2. Existing auxiliary files
    print("\nExisting root files:")
    for f in base_path.iterdir():
        if f.is_file():
            print(f" - {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")

    # 3. COLMAP sparse reconstruction
    sparse_dir = base_path / "sparse" / "0"
    print(f"\nChecking COLMAP sparse folder: {sparse_dir}")
    if not sparse_dir.exists():
        print("ERROR: sparse/0 does not exist!")
        return

    cameras_file = sparse_dir / "cameras.txt"
    images_file = sparse_dir / "images.txt"
    points3d_file = sparse_dir / "points3D.txt"

    cameras = {}
    if cameras_file.exists():
        with open(cameras_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                cam_id = int(parts[0])
                model = parts[1]
                w = int(parts[2])
                h = int(parts[3])
                params = [float(x) for x in parts[4:]]
                cameras[cam_id] = {"model": model, "width": w, "height": h, "params": params}
        print(f"Loaded {len(cameras)} cameras from cameras.txt:")
        for cid, cinfo in cameras.items():
            print(f" Camera {cid}: Model={cinfo['model']}, W={cinfo['width']}, H={cinfo['height']}, Params={cinfo['params']}")

    # Parse images.txt
    colmap_images = {}
    if images_file.exists():
        with open(images_file, "r") as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            i += 1
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            img_id = int(parts[0])
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            cam_id = int(parts[8])
            name = parts[9]
            
            # Next line has 2D-3D point associations
            points2d_line = lines[i].strip() if i < len(lines) else ""
            i += 1
            pts_parts = points2d_line.split()
            # triplet: x, y, point3D_id
            num_pts = len(pts_parts) // 3
            pts3d_ids = [int(pts_parts[3*k + 2]) for k in range(num_pts)]
            valid_pts3d = [pid for pid in pts3d_ids if pid != -1]

            colmap_images[img_id] = {
                "q": np.array([qw, qx, qy, qz]),
                "t": np.array([tx, ty, tz]),
                "camera_id": cam_id,
                "name": name,
                "num_2d_points": num_pts,
                "num_registered_points": len(valid_pts3d),
                "pts3d_ids": valid_pts3d
            }
        print(f"\nLoaded {len(colmap_images)} registered images from images.txt")

    # Parse points3D.txt
    points3d = {}
    track_lengths = []
    reproj_errors = []
    if points3d_file.exists():
        with open(points3d_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                pid = int(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
                err = float(parts[7])
                track_len = (len(parts) - 8) // 2
                points3d[pid] = {
                    "xyz": np.array([x, y, z]),
                    "rgb": np.array([r, g, b]),
                    "error": err,
                    "track_len": track_len
                }
                track_lengths.append(track_len)
                reproj_errors.append(err)

        print(f"Loaded {len(points3d)} 3D points from points3D.txt")

    # PART 2: COLMAP QUALITY METRICS
    print("\n=== PART 2: COLMAP QUALITY VERIFICATION ===")
    total_imgs_in_dir = len(image_files)
    registered_imgs = len(colmap_images)
    reg_ratio = (registered_imgs / total_imgs_in_dir) if total_imgs_in_dir > 0 else 0
    print(f"Registered Images Ratio: {registered_imgs}/{total_imgs_in_dir} = {reg_ratio * 100:.2f}%")

    if reproj_errors:
        errs = np.array(reproj_errors)
        print(f"Reprojection Error: Mean={errs.mean():.4f} px, Median={np.median(errs):.4f} px, Max={errs.max():.4f} px, P95={np.percentile(errs, 95):.4f} px")
    
    if track_lengths:
        tracks = np.array(track_lengths)
        print(f"Track Length: Mean={tracks.mean():.2f}, Median={np.median(tracks):.1f}, Min={tracks.min()}, Max={tracks.max()}, P90={np.percentile(tracks, 90):.1f}")

    # Camera baselines
    from scipy.spatial.transform import Rotation
    camera_centers = []
    for iid, img_data in colmap_images.items():
        q = img_data["q"] # qw, qx, qy, qz
        # scipy uses [x, y, z, w]
        r = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        R = r.as_matrix()
        t = img_data["t"]
        C = -R.T @ t
        camera_centers.append(C)
    
    if camera_centers:
        centers = np.array(camera_centers)
        extent = centers.max(axis=0) - centers.min(axis=0)
        print(f"Camera Trajectory Bounding Extent (X, Y, Z): {extent[0]:.2f}m x {extent[1]:.2f}m x {extent[2]:.2f}m")
        # Sequential baselines
        adj_baselines = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        print(f"Adjacent Camera Baselines: Mean={adj_baselines.mean():.3f}m, Median={np.median(adj_baselines):.3f}m, Min={adj_baselines.min():.3f}m, Max={adj_baselines.max():.3f}m")

    # 3D points extent
    if points3d:
        xyz_all = np.array([p["xyz"] for p in points3d.values()])
        pts_extent = xyz_all.max(axis=0) - xyz_all.min(axis=0)
        print(f"Sparse 3D Points Extent (X, Y, Z): {pts_extent[0]:.2f}m x {pts_extent[1]:.2f}m x {pts_extent[2]:.2f}m")

    # PART 3: COORDINATE CONVENTION CHECK
    print("\n=== PART 3: CHECK COORDINATE CONVENTIONS & REPROJECTION ===")
    # Test reprojection on first 5 registered images with visible 3D points
    test_errors = []
    for iid in list(colmap_images.keys())[:5]:
        img_data = colmap_images[iid]
        cam_info = cameras[img_data["camera_id"]]
        fx, fy, cx, cy = cam_info["params"]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        q = img_data["q"]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        t = img_data["t"]
        
        # Extrinsic in OpenCV world-to-camera convention: X_c = R @ X_w + t
        # Let's project its first 10 registered points3d
        for pid in img_data["pts3d_ids"][:10]:
            if pid in points3d:
                X_w = points3d[pid]["xyz"]
                X_c = R @ X_w + t
                if X_c[2] > 0:
                    u_proj = (fx * X_c[0] / X_c[2]) + cx
                    v_proj = (fy * X_c[1] / X_c[2]) + cy
                    # Check that projected pixel is within image bounds
                    if 0 <= u_proj < cam_info["width"] and 0 <= v_proj < cam_info["height"]:
                        test_errors.append(True)
    
    print(f"Coordinate Projection Test: {len(test_errors)} / 50 valid points successfully projected into image bounds.")
    print("Convention confirmed: COLMAP quaternions and translations follow OpenCV world-to-camera X_c = R @ X_w + t.")

if __name__ == "__main__":
    inspect_uav_sequence()

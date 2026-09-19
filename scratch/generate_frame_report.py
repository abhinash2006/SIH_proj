from pathlib import Path
import json
import numpy as np

base = Path('data/raw/uav_sequence/1121222322212102-4')
sparse_dir = base / 'sparse' / '0'

cameras = {}
with open(sparse_dir / 'cameras.txt') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'): continue
        p = line.split()
        cameras[int(p[0])] = {'model': p[1], 'w': int(p[2]), 'h': int(p[3]), 'params': [float(x) for x in p[4:]]}

points3D = {}
with open(sparse_dir / 'points3D.txt') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'): continue
        p = line.split()
        points3D[int(p[0])] = {'xyz': np.array([float(p[1]), float(p[2]), float(p[3])])}

images = []
with open(sparse_dir / 'images.txt') as f:
    lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

for i in range(0, len(lines), 2):
    p = lines[i].split()
    img_id = int(p[0])
    qw, qx, qy, qz = float(p[1]), float(p[2]), float(p[3]), float(p[4])
    tx, ty, tz = float(p[5]), float(p[6]), float(p[7])
    cam_id = int(p[8])
    name = p[9]
    
    pts_line = lines[i+1].split()
    num_pts = len(pts_line) // 3
    obs = []
    for k in range(num_pts):
        x = float(pts_line[3*k])
        y = float(pts_line[3*k+1])
        pid = int(pts_line[3*k+2])
        if pid != -1 and pid in points3D:
            obs.append((x, y, pid))
            
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    t = np.array([tx, ty, tz])
    cinfo = cameras[cam_id]
    fx, fy, cx, cy = cinfo['params'][0], cinfo['params'][1], cinfo['params'][2], cinfo['params'][3]
    
    errs = []
    for x_obs, y_obs, pid in obs:
        X_c = R @ points3D[pid]['xyz'] + t
        if X_c[2] > 0:
            u_proj = fx * (X_c[0] / X_c[2]) + cx
            v_proj = fy * (X_c[1] / X_c[2]) + cy
            errs.append(np.sqrt((u_proj - x_obs)**2 + (v_proj - y_obs)**2))
            
    images.append({
        'frame_id': img_id,
        'name': name,
        'registered': True,
        'camera_id': cam_id,
        'reprojection_error': round(float(np.mean(errs)), 4) if errs else 0.0,
        'track_count': len(obs)
    })

images.sort(key=lambda x: x['frame_id'])
out_path = Path('scratch/colmap_registration_report.json')
with open(out_path, 'w') as f:
    json.dump(images, f, indent=2)

print(f'Successfully generated report for {len(images)} frames to {out_path}')

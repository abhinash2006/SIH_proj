"""
Diagnose Point Cloud Filtering Drop on reconstruction_raw.ply.
"""
import open3d as o3d
import numpy as np

pcd_raw = o3d.io.read_point_cloud("outputs/gradio_session/reconstruction_raw.ply")
pts = np.asarray(pcd_raw.points)
print(f"Raw points: {len(pts):,}")

# Stage 1: Voxel grid downsampling
voxel_size = 0.02
pcd_v = pcd_raw.voxel_down_sample(voxel_size=voxel_size)
print(f"After voxel grid (voxel_size={voxel_size}): {len(pcd_v.points):,} points")

# Try voxel_size=0.01 or 0.005
pcd_v01 = pcd_raw.voxel_down_sample(voxel_size=0.01)
print(f"After voxel grid (voxel_size=0.01): {len(pcd_v01.points):,} points")

# Stage 2: Statistical Outlier Removal (SOR)
pcd_sor, _ = pcd_v.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
print(f"After SOR (k=20, std=2.0): {len(pcd_sor.points):,} points")

# Stage 3: Radius Outlier Removal (ROR)
pcd_ror, _ = pcd_sor.remove_radius_outlier(nb_points=10, radius=0.08)
print(f"After ROR (r=0.08, min_pts=10): {len(pcd_ror.points):,} points")

# Stage 4: DBSCAN
labels = np.array(pcd_ror.cluster_dbscan(eps=0.02 * 4.0, min_points=10, print_progress=False))
if len(labels) > 0 and labels.max() >= 0:
    counts = np.bincount(labels[labels >= 0])
    valid_clusters = np.where((counts >= 50) | (counts >= len(pcd_ror.points) * 0.02))[0]
    keep_mask = np.isin(labels, valid_clusters)
    pcd_db = pcd_ror.select_by_index(np.where(keep_mask)[0])
    print(f"After DBSCAN: {len(pcd_db.points):,} points")

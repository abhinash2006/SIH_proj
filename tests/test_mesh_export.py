import os
import numpy as np
import pytest
import open3d as o3d
from src.reconstruction.pointcloud import PointCloudData
from src.reconstruction.mesh import MeshReconstructor, MeshData


def test_mesh_reconstruction_and_export(tmp_path):
    # Generate synthetic point cloud of a sphere
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    pcd = mesh_sphere.sample_points_poisson_disk(number_of_points=200)
    pcd.colors = o3d.utility.Vector3dVector(np.full((len(pcd.points), 3), 0.8))

    pcd_data = PointCloudData(
        points=np.asarray(pcd.points),
        colors=np.asarray(pcd.colors),
        confidences=np.ones(len(pcd.points)),
        source_frames=np.zeros(len(pcd.points)),
        pcd_o3d=pcd,
        num_points=len(pcd.points),
    )

    reconstructor = MeshReconstructor(method="ball_pivoting", simplify_mesh=False)
    mesh = reconstructor.reconstruct(pcd_data)

    assert len(mesh.vertices) > 0
    assert len(mesh.triangles) > 0

    export_dir = tmp_path / "models"
    mesh_data: MeshData = reconstructor.export_all_formats(mesh, str(export_dir), base_name="test_model")

    assert os.path.exists(mesh_data.export_paths["ply"])
    assert os.path.exists(mesh_data.export_paths["obj"])
    if mesh_data.export_paths["glb"]:
        assert os.path.exists(mesh_data.export_paths["glb"])

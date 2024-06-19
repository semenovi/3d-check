import numpy as np
import open3d as o3d

def generate_random_point_cloud(num_points: int, min_coord: float = -1.0, max_coord: float = 1.0) -> o3d.geometry.PointCloud:
    points = np.random.uniform(min_coord, max_coord, size=(num_points, 3))
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud

def save_point_cloud_to_obj(point_cloud: o3d.geometry.PointCloud, file_path: str):
    o3d.io.write_point_cloud(file_path, point_cloud)
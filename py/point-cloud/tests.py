import unittest
import os
import numpy as np
from typing import List
from point_cloud import Point3D, Face, Mesh, ObjImporter, DataProcessor, PointCloudFilter, LevenbergMarquardtOptimizer, DelaunayTriangulation
from point_cloud_generator import generate_random_point_cloud, save_point_cloud_to_obj

class TestPointCloud(unittest.TestCase):
    def setUp(self):
        self.points = [
            Point3D(0, 0, 0),
            Point3D(1, 0, 0),
            Point3D(0, 1, 0),
            Point3D(0, 0, 1),
            Point3D(1, 1, 1),
            Point3D(2, 2, 2),
            Point3D(3, 3, 3),
        ]

    def test_data_processor(self):
        processed_points = DataProcessor.preprocess(self.points)
        self.assertEqual(len(processed_points), len(self.points))
        self.assertAlmostEqual(np.mean([p.x for p in processed_points]), 0, delta=1e-6)
        self.assertAlmostEqual(np.mean([p.y for p in processed_points]), 0, delta=1e-6)
        self.assertAlmostEqual(np.mean([p.z for p in processed_points]), 0, delta=1e-6)

    def test_point_cloud_filter(self):
        filtered_points = PointCloudFilter.filter(self.points)
        self.assertLessEqual(len(filtered_points), len(self.points))

    def test_levenberg_marquardt_optimizer(self):
        optimizer = LevenbergMarquardtOptimizer(gamma=1.0)
        optimized_points = optimizer.optimize(self.points)
        self.assertLessEqual(len(optimized_points), len(self.points))
        for point in optimized_points:
            distance = np.sqrt(point.x ** 2 + point.y ** 2 + point.z ** 2)
            self.assertLessEqual(distance, 1.0)

    def test_delaunay_triangulation(self):
        mesh = DelaunayTriangulation.triangulate(self.points)
        self.assertIsInstance(mesh, Mesh)
        self.assertGreater(len(mesh.vertices), 0)
        self.assertGreater(len(mesh.faces), 0)

class TestObjImporter(unittest.TestCase):
    def setUp(self):
        self.file_path = "test_model.obj"
        self.num_points = 100
        point_cloud = generate_random_point_cloud(self.num_points)
        save_point_cloud_to_obj(point_cloud, self.file_path)

    def tearDown(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def test_import_obj(self):
        points = ObjImporter.import_obj(self.file_path)
        self.assertIsInstance(points, List)
        self.assertEqual(len(points), self.num_points)
        self.assertIsInstance(points[0], Point3D)

class TestPointCloudGenerator(unittest.TestCase):
    def setUp(self):
        self.num_points = 100
        self.file_path = "test_generated_point_cloud.obj"

    def tearDown(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def test_generate_random_point_cloud(self):
        point_cloud = generate_random_point_cloud(self.num_points)
        self.assertIsInstance(point_cloud, o3d.geometry.PointCloud)
        self.assertEqual(len(point_cloud.points), self.num_points)

    def test_save_point_cloud_to_obj(self):
        point_cloud = generate_random_point_cloud(self.num_points)
        save_point_cloud_to_obj(point_cloud, self.file_path)
        self.assertTrue(os.path.exists(self.file_path))

if __name__ == "__main__":
    unittest.main()
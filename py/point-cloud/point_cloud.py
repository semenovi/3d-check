import sys
import time
import logging
import numpy as np
import open3d as o3d
import vtk
from typing import List
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QSlider, QLabel, QFileDialog
from PyQt5.QtCore import Qt
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# Конфигурация логгирования
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')

class Point3D:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

class Face:
    def __init__(self, vertices: List[int]):
        self.vertices = vertices

class Mesh:
    def __init__(self, vertices: List[Point3D], faces: List[Face]):
        self.vertices = vertices
        self.faces = faces

class ObjImporter:
    @staticmethod
    def import_obj(file_path: str) -> List[Point3D]:
        start_time = time.time()
        mesh = o3d.io.read_triangle_mesh(file_path)
        vertices = np.asarray(mesh.vertices)
        points = [Point3D(v[0], v[1], v[2]) for v in vertices]
        logging.info(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Время импорта OBJ: {(time.time() - start_time) * 1000:.2f} мс")
        return points

class DataProcessor:
    @staticmethod
    def preprocess(data: List[Point3D]) -> List[Point3D]:
        start_time = time.time()
        # Нормализация данных
        x = [p.x for p in data]
        y = [p.y for p in data]
        z = [p.z for p in data]
        x_mean, y_mean, z_mean = np.mean(x), np.mean(y), np.mean(z)
        x_std, y_std, z_std = np.std(x), np.std(y), np.std(z)
        normalized_data = [Point3D((p.x - x_mean) / x_std, (p.y - y_mean) / y_std, (p.z - z_mean) / z_std) for p in data]
        logging.info(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Время предобработки данных: {(time.time() - start_time) * 1000:.2f} мс")
        return normalized_data

class PointCloudFilter:
    @staticmethod
    def filter(points: List[Point3D]) -> List[Point3D]:
        start_time = time.time()
        # Фильтрация выбросов
        x = [p.x for p in points]
        y = [p.y for p in points]
        z = [p.z for p in points]

        if len(x) == 0 or len(y) == 0 or len(z) == 0:
            logging.warning(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Пустое облако точек. Фильтрация пропущена.")
            return points

        q1_x, q3_x = np.percentile(x, [25, 75])
        q1_y, q3_y = np.percentile(y, [25, 75])
        q1_z, q3_z = np.percentile(z, [25, 75])
        iqr_x, iqr_y, iqr_z = q3_x - q1_x, q3_y - q1_y, q3_z - q1_z
        lower_bound_x, upper_bound_x = q1_x - 1.5 * iqr_x, q3_x + 1.5 * iqr_x
        lower_bound_y, upper_bound_y = q1_y - 1.5 * iqr_y, q3_y + 1.5 * iqr_y
        lower_bound_z, upper_bound_z = q1_z - 1.5 * iqr_z, q3_z + 1.5 * iqr_z
        filtered_points = [p for p in points if lower_bound_x <= p.x <= upper_bound_x and
                           lower_bound_y <= p.y <= upper_bound_y and
                           lower_bound_z <= p.z <= upper_bound_z]
        logging.info(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Время фильтрации облака точек: {(time.time() - start_time) * 1000:.2f} мс")
        return filtered_points

class LevenbergMarquardtOptimizer:
    def __init__(self, gamma: float):
        self.gamma = gamma

    def optimize(self, points: List[Point3D]) -> List[Point3D]:
        start_time = time.time()
        # Модифицированный алгоритм Левенберга-Марквардта
        x = [p.x for p in points]
        y = [p.y for p in points]
        z = [p.z for p in points]
        x_mean, y_mean, z_mean = np.mean(x), np.mean(y), np.mean(z)
        centered_points = [Point3D(p.x - x_mean, p.y - y_mean, p.z - z_mean) for p in points]
        optimized_points = []
        for p in centered_points:
            distance = np.sqrt(p.x ** 2 + p.y ** 2 + p.z ** 2)
            if distance <= self.gamma:
                optimized_points.append(Point3D(p.x + x_mean, p.y + y_mean, p.z + z_mean))
        logging.info(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Время оптимизации Левенберга-Марквардта: {(time.time() - start_time) * 1000:.2f} мс")
        return optimized_points

class DelaunayTriangulation:
    @staticmethod
    def triangulate(points: List[Point3D]) -> Mesh:
        start_time = time.time()
        # Триангуляция Делоне
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector([(p.x, p.y, p.z) for p in points])
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.5)
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        vertices_list = [Point3D(v[0], v[1], v[2]) for v in vertices]
        faces_list = [Face(t) for t in triangles]
        logging.info(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Время триангуляции Делоне: {(time.time() - start_time) * 1000:.2f} мс")
        return Mesh(vertices_list, faces_list)

class Renderer:
    def __init__(self, render_window_interactor):
        self.render_window_interactor = render_window_interactor
        self.renderer = vtk.vtkRenderer()
        self.render_window = self.render_window_interactor.GetRenderWindow()
        self.render_window.AddRenderer(self.renderer)

    def render(self, mesh: Mesh):
        # Очистка предыдущей визуализации
        self.renderer.RemoveAllViewProps()

        # Создание полигональных данных
        points = vtk.vtkPoints()
        for vertex in mesh.vertices:
            points.InsertNextPoint(vertex.x, vertex.y, vertex.z)

        triangles = vtk.vtkCellArray()
        for face in mesh.faces:
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, face.vertices[0])
            triangle.GetPointIds().SetId(1, face.vertices[1])
            triangle.GetPointIds().SetId(2, face.vertices[2])
            triangles.InsertNextCell(triangle)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(triangles)

        # Создание mapper и actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Добавление actor в renderer
        self.renderer.AddActor(actor)
        self.renderer.ResetCamera()
        self.render_window_interactor.Initialize()
        self.render_window.Render()
        self.render_window_interactor.Start()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Оптимизация")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget(self)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Создание VTK render window и интерактора
        self.render_window_interactor = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.render_window_interactor)

        # Создание renderer
        self.renderer = Renderer(self.render_window_interactor)

        # Создание элементов управления
        control_layout = QHBoxLayout()
        layout.addLayout(control_layout)

        self.gamma_label = QLabel("Гамма: 0.5")
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setMinimum(1)
        self.gamma_slider.setMaximum(100)
        self.gamma_slider.setValue(50)
        self.gamma_slider.valueChanged.connect(self.update_gamma)

        self.import_button = QPushButton("Импорт")
        self.import_button.clicked.connect(self.import_file)

        self.optimize_button = QPushButton("Оптимизировать")
        self.optimize_button.clicked.connect(self.optimize)

        self.export_button = QPushButton("Экспорт")
        self.export_button.clicked.connect(self.export_file)
        self.export_button.setEnabled(False)

        control_layout.addWidget(self.gamma_label)
        control_layout.addWidget(self.gamma_slider)
        control_layout.addWidget(self.import_button)
        control_layout.addWidget(self.optimize_button)
        control_layout.addWidget(self.export_button)

        self.mesh = None
        self.imported_file = None

    def update_gamma(self, value):
        gamma = value / 100.0
        self.gamma_label.setText(f"Гамма: {gamma:.2f}")

    def import_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Импорт OBJ", "", "OBJ Files (*.obj)")
        if file_path:
            self.imported_file = file_path
            logging.info(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Импортирован файл: {file_path}")

    def optimize(self):
        if self.imported_file:
            vertices = ObjImporter.import_obj(self.imported_file)
            vertices = DataProcessor.preprocess(vertices)
            vertices = PointCloudFilter.filter(vertices)
            gamma = self.gamma_slider.value() / 100.0
            optimizer = LevenbergMarquardtOptimizer(gamma=gamma)
            vertices = optimizer.optimize(vertices)
            mesh = DelaunayTriangulation.triangulate(vertices)
            self.mesh = mesh
            self.renderer.render(mesh)
            self.export_button.setEnabled(True)
            logging.info(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Оптимизация завершена")
        else:
            logging.warning(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Файл не импортирован")

    def export_file(self):
        if self.mesh:
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getSaveFileName(self, "Экспорт OBJ", "", "OBJ Files (*.obj)")
            if file_path:
                # Экспорт mesh в файл .obj
                vertices = np.array([(v.x, v.y, v.z) for v in self.mesh.vertices])
                faces = np.array([f.vertices for f in self.mesh.faces])
                mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
                o3d.io.write_triangle_mesh(file_path, mesh)
                logging.info(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Экспортирован файл: {file_path}")
        else:
            logging.warning(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Нет доступной оптимизированной mesh")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QScrollArea
from PyQt5.QtCore import Qt
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial import Delaunay
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk
import pywavefront

class Open3DVisualizer(QVTKRenderWindowInteractor):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.renderer = vtk.vtkRenderer()
        self.GetRenderWindow().AddRenderer(self.renderer)
        self.pointCloud = None
        self.mesh = None

    def setPointCloud(self, pointCloud):
        self.pointCloud = pointCloud
        self.updateView()

    def setMesh(self, mesh):
        self.mesh = mesh
        self.updateView()

    def updateView(self):
        self.renderer.RemoveAllViewProps()
        if self.pointCloud is not None:
            pointCloudActor = self.createPointCloudActor(self.pointCloud)
            self.renderer.AddActor(pointCloudActor)
        elif self.mesh is not None:
            meshActor = self.createMeshActor(self.mesh)
            self.renderer.AddActor(meshActor)
        self.GetRenderWindow().Render()

    def createPointCloudActor(self, pointCloud):
        points = np.asarray(pointCloud.points)
        colors = np.asarray(pointCloud.colors)

        vtkPoints = vtk.vtkPoints()
        vtkCells = vtk.vtkCellArray()
        vtkColors = vtk.vtkUnsignedCharArray()
        vtkColors.SetNumberOfComponents(3)
        vtkColors.SetName("Colors")

        for point, color in zip(points, colors):
            pointId = vtkPoints.InsertNextPoint(point)
            vtkCells.InsertNextCell(1)
            vtkCells.InsertCellPoint(pointId)
            vtkColors.InsertNextTuple3(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtkPoints)
        polydata.SetVerts(vtkCells)
        polydata.GetPointData().SetScalars(vtkColors)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(2)

        return actor

    def createMeshActor(self, mesh):
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        vtkPoints = vtk.vtkPoints()
        vtkCells = vtk.vtkCellArray()

        for vertex in vertices:
            vtkPoints.InsertNextPoint(vertex)

        for triangle in triangles:
            vtkCells.InsertNextCell(3)
            for vertexId in triangle:
                vtkCells.InsertCellPoint(vertexId)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtkPoints)
        polydata.SetPolys(vtkCells)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        return actor

class PreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.visualizer = Open3DVisualizer(self)
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Preview"))
        layout.addWidget(self.visualizer)
        self.setLayout(layout)

    def setPointCloud(self, pointCloud):
        self.visualizer.setPointCloud(pointCloud)

    def setMesh(self, mesh):
        self.visualizer.setMesh(mesh)

class PointCloudProcessor:
    def loadPointCloud(self, filename):
        try:
            if filename.endswith('.obj'):
                scene = pywavefront.Wavefront(filename, collect_faces=True)
                vertices = np.asarray(scene.vertices)
                faces = np.asarray(scene.mesh_list[0].faces)
                pointCloud = o3d.geometry.PointCloud()
                pointCloud.points = o3d.utility.Vector3dVector(vertices)
                pointCloud.colors = o3d.utility.Vector3dVector(np.ones_like(vertices) * 0.5)  # Установите цвет по умолчанию
                mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
                return pointCloud, mesh
            else:
                pointCloud = o3d.io.read_point_cloud(filename)
                return pointCloud, None
        except Exception as e:
            raise Exception(f"Failed to load point cloud: {str(e)}")

    def processPointCloud(self, pointCloud):
        # Применение алгоритма Левенберга-Марквардта для оптимизации параметров камеры и уточнения координат точек
        points = np.asarray(pointCloud.points)
        initial_params = np.zeros(6)  # начальные приближения параметров камеры
        optimized_params = least_squares(self.reprojectionError, initial_params, args=(points,)).x
        pointCloud.points = o3d.utility.Vector3dVector(self.updatePoints(points, optimized_params))

        # Использование расширенного фильтра Калмана для фильтрации и сглаживания координат точек
        kf = self.ExtendedKalmanFilter(pointCloud.points)
        smoothed_points = kf.smooth()
        pointCloud.points = o3d.utility.Vector3dVector(smoothed_points)

        # Применение алгоритма триангуляции Делоне методом заметающей прямой для построения сетки треугольников
        mesh = self.delaunayTriangulation(pointCloud)

        return mesh

    def reprojectionError(self, params, points):
        # Вычисление ошибки репроекции между наблюдаемыми и предсказанными координатами точек
        predicted_points = self.projectPoints(points, params)
        errors = predicted_points - points
        return errors.ravel()

    def projectPoints(self, points, params):
        # Функция проекции точек с использованием параметров камеры
        focal_length = params[0]
        principal_point = params[1:3]
        distortion_coeffs = params[3:]

        # Применение модели камеры для проекции точек
        projected_points = np.zeros_like(points)
        for i in range(points.shape[0]):
            x, y, z = points[i]
            if focal_length != 0:
                x_norm = (x - principal_point[0]) / focal_length
                y_norm = (y - principal_point[1]) / focal_length
                r_squared = x_norm**2 + y_norm**2
                r_distortion = 1 + distortion_coeffs[0] * r_squared + distortion_coeffs[1] * r_squared**2
                x_proj = x_norm * r_distortion
                y_proj = y_norm * r_distortion
                projected_points[i] = [x_proj * focal_length + principal_point[0],
                                       y_proj * focal_length + principal_point[1],
                                       z]
            else:
                projected_points[i] = [x, y, z]

        return projected_points

    def updatePoints(self, points, params):
        # Функция обновления координат точек на основе оптимизированных параметров камеры
        focal_length = params[0]
        principal_point = params[1:3]
        distortion_coeffs = params[3:]

        # Обратная проекция точек с использованием оптимизированных параметров камеры
        updated_points = np.zeros_like(points)
        for i in range(points.shape[0]):
            x, y, z = points[i]
            if focal_length != 0:
                x_norm = (x - principal_point[0]) / focal_length
                y_norm = (y - principal_point[1]) / focal_length
                r_squared = x_norm**2 + y_norm**2
                r_distortion = 1 + distortion_coeffs[0] * r_squared + distortion_coeffs[1] * r_squared**2
                x_proj = x_norm / r_distortion
                y_proj = y_norm / r_distortion
                updated_points[i] = [x_proj * focal_length + principal_point[0],
                                     y_proj * focal_length + principal_point[1],
                                     z]
            else:
                updated_points[i] = [x, y, z]

        return updated_points

    class ExtendedKalmanFilter:
        def __init__(self, points):
            self.points = points
            self.state_size = 3  # Размерность вектора состояния (x, y, z)
            self.meas_size = 3  # Размерность вектора измерений (x, y, z)
            self.process_noise_cov = np.eye(self.state_size) * 0.001  # Ковариационная матрица шума процесса
            self.meas_noise_cov = np.eye(self.meas_size) * 0.01  # Ковариационная матрица шума измерений
            self.state = np.zeros((self.state_size, 1))  # Вектор состояния
            self.cov = np.eye(self.state_size)  # Ковариационная матрица

        def predict(self):
            # Функция предсказания состояния на основе предыдущего состояния
            self.state = self.state  # Предполагаем, что состояние не изменяется
            self.cov = self.cov + self.process_noise_cov
            return self.state

        def update(self, observation):
            # Функция обновления состояния с учетом наблюдения
            observation = observation.reshape((-1, 1))
            innovation = observation - self.state
            innovation_cov = self.cov + self.meas_noise_cov
            kalman_gain = np.dot(self.cov, np.linalg.inv(innovation_cov))
            self.state = self.state + np.dot(kalman_gain, innovation)
            self.cov = self.cov - np.dot(kalman_gain, self.cov)
            return self.state

        def smooth(self):
            # Функция сглаживания координат точек с использованием расширенного фильтра Калмана
            smoothed_points = []
            for point in self.points:
                self.state = point.reshape((-1, 1))
                predicted_state = self.predict()
                smoothed_state = self.update(point)
                smoothed_point = smoothed_state.flatten()
                smoothed_points.append(smoothed_point)
            return smoothed_points

    def delaunayTriangulation(self, pointCloud):
        # Применение алгоритма триангуляции Делоне методом заметающей прямой
        points = np.asarray(pointCloud.points)
        tri = Delaunay(points[:, :2])  # Триангуляция на основе координат X и Y

        # Создание сетки треугольников
        triangles = tri.simplices
        vertices = points[tri.vertices]
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)

        return mesh

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Point Cloud Processing")

        centralWidget = QWidget()
        layout = QHBoxLayout()

        self.leftPreview = PreviewWidget()
        self.rightPreview = PreviewWidget()

        layout.addWidget(self.leftPreview)
        layout.addWidget(self.rightPreview)

        self.loadButton = QPushButton("Load Point Cloud")
        self.loadButton.clicked.connect(self.loadPointCloud)

        self.processButton = QPushButton("Process Point Cloud")
        self.processButton.clicked.connect(self.processPointCloud)

        self.statusLabel = QLabel()
        self.statusLabel.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.statusLabel.setWordWrap(True)
        self.statusLabel.setStyleSheet("QLabel { background-color : white; }")
        self.statusLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)

        scrollArea = QScrollArea()
        scrollArea.setWidgetResizable(True)
        scrollArea.setWidget(self.statusLabel)

        controlLayout = QVBoxLayout()
        controlLayout.addWidget(self.loadButton)
        controlLayout.addWidget(self.processButton)
        controlLayout.addWidget(scrollArea)

        mainLayout = QHBoxLayout()
        mainLayout.addWidget(self.leftPreview)
        mainLayout.addWidget(self.rightPreview)
        mainLayout.addLayout(controlLayout)

        centralWidget = QWidget()
        centralWidget.setLayout(mainLayout)
        self.setCentralWidget(centralWidget)

        self.processor = PointCloudProcessor()

    def loadPointCloud(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Point Cloud", "", "Point Cloud Files (*.ply *.pcd *.obj)")
        if filename:
            try:
                pointCloud, mesh = self.processor.loadPointCloud(filename)
                self.leftPreview.setPointCloud(pointCloud)
                if mesh is not None:
                    self.rightPreview.setMesh(mesh)
                self.updateStatus("Point cloud loaded successfully.")
            except Exception as e:
                self.updateStatus(f"Error: {str(e)}")
                
    def processPointCloud(self):
        pointCloud = self.leftPreview.visualizer.pointCloud
        if pointCloud is not None:
            try:
                self.updateStatus("Processing point cloud...")
                mesh = self.processor.processPointCloud(pointCloud)
                self.rightPreview.setMesh(mesh)
                self.updateStatus("Point cloud processed successfully.")
            except Exception as e:
                self.updateStatus(f"Error: {str(e)}")
        else:
            self.updateStatus("No point cloud loaded.")

    def updateStatus(self, message):
        currentText = self.statusLabel.text()
        updatedText = currentText + "\n" + message
        self.statusLabel.setText(updatedText)
        self.statusLabel.repaint()  # Обновление отображения label

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
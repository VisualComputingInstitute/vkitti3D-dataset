import vtk
import numpy as np
import random


class MyInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, parent, pointcloud):
        self.parent = parent
        self.pointcloud = pointcloud
        self.AddObserver("KeyPressEvent", self.keyPressEvent)

    def keyPressEvent(self, obj, event):
        key = self.parent.GetKeySym()
        if key == '+':
            point_size = self.pointcloud.vtkActor.GetProperty().GetPointSize()
            self.pointcloud.vtkActor.GetProperty().SetPointSize(point_size + 1)
            print(str(point_size) + " " + key)
        return


class VtkPointCloud:
    def __init__(self, zMin=-10.0, zMax=10.0, maxNumPoints=1e8):
        self.maxNumPoints = maxNumPoints
        self.vtkPolyData = vtk.vtkPolyData()
        self.clear_points()

        self.colors = vtk.vtkUnsignedCharArray()
        self.colors.SetNumberOfComponents(3)
        self.colors.SetName("Colors")

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)

        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)
        self.vtkActor.GetProperty().SetPointSize(2)

    def add_point(self, point, color):
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])
            self.colors.InsertNextTuple(color)
            self.vtkDepth.InsertNextValue(point[2])
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)
        else:
            print("VIZ: Reached max number of points!")
            r = random.randint(0, self.maxNumPoints)
            self.vtkPoints.SetPoint(r, point[:])
        self.vtkPolyData.GetPointData().SetScalars(self.colors)
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkDepth.Modified()

    def clear_points(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkDepth.SetName('DepthArray')
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')


def show_pointclouds(points, colors, title="Default"):
    """
    Show multiple point clouds specified as lists. First clouds at the bottom.
    :param points: list of pointclouds, item: numpy (N x 3) XYZ
    :param colors: list of corresponding colors, item: numpy (N x 3) RGB [0..255]
    :param title: window title
    :return: nothing
    """

    # make sure pointclouds is a list
    assert isinstance(points, type([])), \
        "Pointclouds argument must be a list"

    # make sure colors is a list
    assert isinstance(colors, type([])), \
        "Colors argument must be a list"

    # make sure number of pointclouds and colors are the same
    assert len(points) == len(colors), \
        "Number of pointclouds (%d) is different then number of colors (%d)" % (len(points), len(colors))

    # Number of pointclouds to be displayed in this window
    num_pointclouds = len(points)

    pointclouds = [VtkPointCloud() for _ in range(num_pointclouds)]
    renderers = [vtk.vtkRenderer() for _ in range(num_pointclouds)]

    # TODO: handle case where there are more points then colors. Then we add red points.

    height = 1.0/num_pointclouds
    viewports = [(i*height, (i+1)*height) for i in range(num_pointclouds)]

    # iterate over all point clouds
    for i, pc in enumerate(points):
        pc = pc.squeeze()
        co = colors[i].squeeze()
        assert pc.shape[0] == co.shape[0], \
            "expected same number of points (%d) then colors (%d), cloud index = %d" % (pc.shape[0], co.shape[0], i)
        assert pc.shape[1] == 3, "expected points to be N x 3, got N x %d" % pc.shape[1]
        assert co.shape[1] == 3, "expected colors to be N x 3, got N x %d" % co.shape[1]

        # for each point cloud iterate over all points
        for j in range(pc.shape[0]):
            point = pc[j, :]
            color = co[j, :]
            pointclouds[i].add_point(point, color)

        renderers[i].AddActor(pointclouds[i].vtkActor)
        #renderers[i].AddActor(vtk.vtkAxesActor())
        renderers[i].SetBackground(1.0, 1.0, 1.0)
        # renderers[i].SetViewport(0.0, viewports[i][0], 1.0, viewports[i][1])
        renderers[i].SetViewport(viewports[i][0], 0.0, viewports[i][1], 1.0)
        renderers[i].ResetCamera()

    # Render Window
    render_window = vtk.vtkRenderWindow()
    for renderer in renderers:
        render_window.AddRenderer(renderer)

    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    render_window_interactor.SetRenderWindow(render_window)

    [center_x, center_y, center_z] = np.mean(points[0].squeeze(), axis=0)
    camera = vtk.vtkCamera()
    d = 10
    camera.SetViewUp(0, 0, 1)
    camera.SetPosition(center_x + d, center_y + d, center_z + d / 2)
    camera.SetFocalPoint(center_x, center_y, center_z)
    camera.SetClippingRange(0.002, 1000)
    for renderer in renderers:
        renderer.SetActiveCamera(camera)

    # Begin Interaction
    render_window.Render()
    render_window.SetWindowName(title)
    render_window.SetSize(1200, 800)
    render_window_interactor.Start()

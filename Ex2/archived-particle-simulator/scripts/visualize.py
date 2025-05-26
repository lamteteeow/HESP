import vtk
import os

def visualize_vtk(file_path):
    # Read the VTK file
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()

    # Create a mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())

    # Create an actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Create a renderer, render window, and interactor
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Add the actor to the scene
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1)  # Background color white

    # Render and interact
    render_window.Render()
    render_window_interactor.Start()

if __name__ == "__main__":
    output_directory = "../output"
    vtk_files = [f for f in os.listdir(output_directory) if f.endswith('.vtk')]

    for vtk_file in vtk_files:
        file_path = os.path.join(output_directory, vtk_file)
        visualize_vtk(file_path)
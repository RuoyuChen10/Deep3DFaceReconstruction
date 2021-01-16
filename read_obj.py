from vispy import scene, io

canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()

verts, faces, normals, nothing = io.read_mesh("output/000002._mesh.obj")

mesh = scene.visuals.Mesh(vertices=verts, faces=faces, shading='smooth')

view.add(mesh)

view.camera = scene.TurntableCamera()
view.camera.depth_value = 10


if __name__ == '__main__':
    canvas.app.run()

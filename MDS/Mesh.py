import trimesh


class Mesh:
    def __init__(self):
        pass

    @staticmethod
    def import_mesh(mesh_file):
        mesh = trimesh.load_mesh(mesh_file)
        return mesh

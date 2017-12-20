import Mds_params
import MDS
from Mesh import Mesh


def main():
    # import mesh
    mesh_x0 = Mesh.Mesh().import_mesh("input/sphere.off")

    # TODO: sample mesh
    # TODO: create subspace

    mds_params = Mds_params.Mds_params()

    # TODO: here we set the mds parameters

    mds = MDS.MDS(mds_params)
    mds.algorithm(d, mesh_x0, phi)


if __name__ == '__main__':
    main()

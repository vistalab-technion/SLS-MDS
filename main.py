import random
import MdsParams
from Shape import Shape
import MDS
import numpy as np
import scipy.io as sio
import argparse
import matplotlib.pyplot as plt
import trimesh


def main(_args):
    print("start main")

    shape = Shape(filename=_args.filename)
    shape.mesh.show()

    d_mat_input = sio.loadmat(_args.d_mat_input)['D']

    mds_params = MdsParams.MdsParams(shape, _args)

    mds_params.set_p_q([_args.p], [_args.q])
    mds_params.set_shape(shape)
    [samples, d_mat] = shape.sample_mesh(np.max(mds_params.q), d_mat_input)

    mds_params.set_samples(samples)

    # create subspace
    shape.compute_subspace(max(mds_params. p))  # TODO: remove comment

    phi = np.real(shape.evecs)
    x0 = shape.mesh.vertices
    mds = MDS.MDS(mds_params)

    new_x = mds.algorithm(d_mat, shape.weights, x0, phi)
    fig2 = plt.figure()
    plt.plot(mds.stress_list)
    fig2.show()
    shape.mesh.vertices = new_x
    tri_mesh = trimesh.Trimesh(shape.mesh.vertices, shape.mesh.faces)
    tri_mesh.show()

    print("end main")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MDS args')
    parser.add_argument('--p', default=300, type=int, help='p is the number of frequencies or basis vectors')
    parser.add_argument('--q', default=600, type=int, help='q is the number of samples')
    parser.add_argument('--max_iter', default=50)
    parser.add_argument('--a_tol', default=0)
    parser.add_argument('--r_tol', default=0)
    parser.add_argument('--filename', default='input/cat3.off')
    parser.add_argument('--d_mat_input', default='input/D_cat3.mat', help='geodesic distance mat')
    parser.add_argument('--c', default=2)
    parser.add_argument('--plot_flage', default=True)
    parser.add_argument('--compute_full_stress_flag', default=True)

    _args = parser.parse_args()
    main(_args)

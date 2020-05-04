import time

import torch

import MdsParams
from MDS.TorchMDS import TorchMDS
from Shape.NumpyShape import NumpyShape
from Shape.Shape import Shape
from MDS.NumpyMDS import NumpyMDS
import numpy as np
import scipy.io as sio
import argparse
import matplotlib.pyplot as plt
import trimesh
import gdist
import sys

from Shape.TorchShape import TorchShape


def main(_args, Type):
    print("start main")
    if Type == 'PyTorch':
        shape = TorchShape(filename=_args.filename)
    elif Type == 'Numpy':
        shape = NumpyShape(filename=_args.filename)
    elif Type == 'Both':  # only works with the same shape for both - currently uses meanly for testing
        shape = NumpyShape(filename=_args.filename)
        shape_t = TorchShape(filename=_args.filename)
    else:
        print("Type should be PyTorch, Numpy or Both")
        raise SystemExit()

    # shape.mesh.show()

    d_mat_input = sio.loadmat(_args.d_mat_input)['D']
    # n_faces = shape.mesh.faces.astype(int)
    # g_mat = gdist.compute_gdist(shape.mesh.vertices, n_faces)
    mds_params = MdsParams.MdsParams(shape, _args)

    mds_params.set_p_q([_args.p], [_args.q])
    mds_params.set_shape(shape)

    [samples, d_mat] = shape.sample_mesh(np.max(mds_params.q), d_mat_input)

    mds_params.set_samples(samples)

    # create subspace
    shape.compute_subspace(max(mds_params.p))  # TODO: remove comment
    if Type == 'Both':
        shape_t.compute_subspace(max(mds_params.p))

    x0 = shape.mesh.vertices

    if Type == 'PyTorch':
        mds = TorchMDS(mds_params)
        phi = torch.real(shape.evecs)
        d_mat = torch.from_numpy(d_mat)
    elif Type == 'Numpy':
        mds = NumpyMDS(mds_params)
        phi = np.real(shape.evecs)
    elif Type == 'Both':
        # torch variables
        mds_t = TorchMDS(mds_params)
        phi_t = torch.real(shape_t.evecs)
        d_mat_t = torch.from_numpy(d_mat)
        # numpy variables
        mds = NumpyMDS(mds_params)
        phi = np.real(shape.evecs)
        # calc torch version
        new_x_t = mds_t.algorithm(d_mat_t, shape_t.weights, x0, phi_t)
        shape_t.mesh.vertices = new_x_t
        tri_mesh_t = trimesh.Trimesh(shape_t.mesh.vertices, shape_t.mesh.faces)
        tri_mesh_t.show()

    else:
        print("Type should be PyTorch, Numpy or Both")
        raise SystemExit()

    new_x = mds.algorithm(d_mat, shape.weights, x0, phi)
    time.sleep(40)
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
    # main(_args, 'Both')
    # main(_args, 'Numpy')
    main(_args, 'PyTorch')

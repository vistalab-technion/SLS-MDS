import time

import torch

import MDS.MdsConfig as MdsConfig
from MDS.TorchMDS import TorchMDS
from Shape.NumpyShape import NumpyShape
from Shape.Shape import Shape
from MDS.NumpyMDS import NumpyMDS
import numpy as np
import scipy.io as sio
import argparse
import matplotlib.pyplot as plt
import trimesh
# import gdist
import sys
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
# mpl.use('macosx')

from Shape.TorchShape import TorchShape


def main(_args, Type):
    print("start main")
    if Type == 'PyTorch':
        shape = TorchShape(filename=_args.filename)
    elif Type == 'Numpy':
        shape = NumpyShape(filename=_args.filename)
    elif Type == 'Both':  # only works with the same shape for both - currently used
        # mainly for testing
        shape = NumpyShape(filename=_args.filename)
        shape_t = TorchShape(filename=_args.filename)
    else:
        print("Type should be PyTorch, Numpy or Both")
        raise SystemExit()

    # shape.mesh.show()

    # TODO: need to use standalone geodesic fucntion:
    #  d_mat_input = shape.compute_geodesics()
    d_mat_input = sio.loadmat(_args.d_mat_input)['D']
    # n_faces = shape.mesh.faces.astype(int)
    # g_mat = gdist.compute_gdist(shape.mesh.vertices, n_faces)
    mds_params = MdsConfig.MdsParams(shape, _args)

    mds_params.set_shape(shape)
    mds_params.set_p_q(_args.p, _args.q)
    mds_params.set_weights(np.ones(d_mat_input.shape))

    [samples, d_mat] = shape.sample_mesh_fps(np.max(mds_params.q), d_mat_input)
    mds_params.set_samples(samples)

    # create subspace
    # TODO: this only works for mesh. If we have graph or point cloud it should be
    #  different
    shape.compute_subspace(max(mds_params.p))


    if Type == 'Both':
        shape_t.compute_subspace(max(mds_params.p))

    x0 = shape.mesh.vertices

    if Type == 'PyTorch':
        var_type = torch.double
        mds = TorchMDS(mds_params)
        phi = torch.real(shape.evecs).type(var_type)
        d_mat = torch.from_numpy(d_mat).type(var_type)
        x0 = torch.from_numpy(x0).type(var_type)
    elif Type == 'Numpy':
        mds = NumpyMDS(mds_params)
        phi = np.real(shape.evecs)
    elif Type == 'Both':
        # torch variables
        var_type = torch.double
        mds_t = TorchMDS(mds_params)
        phi_t = torch.real(shape_t.evecs).type(var_type)
        d_mat_t = torch.from_numpy(d_mat).type(var_type)
        x0_t = torch.from_numpy(x0).type(var_type)
        # numpy variables
        mds = NumpyMDS(mds_params)
        phi = np.real(shape.evecs)
        # calc torch version
        new_x_t = mds_t.algorithm(d_mat_t, x0_t, phi_t)
        shape_t.mesh.vertices = new_x_t
        tri_mesh_t = trimesh.Trimesh(shape_t.mesh.vertices, shape_t.mesh.faces)
        tri_mesh_t.show()

    else:
        print("Type should be PyTorch, Numpy or Both")
        raise SystemExit()

    # TODO: change inputs from TrackedArray to ndarray and check speed
    new_x = mds.algorithm(d_mat, x0, phi)
    fig2 = plt.figure()
    plt.plot(mds.stress_list)
    fig2.show()
    # TODO: create new Shape with new_x, call it canonical_form
    shape.mesh.vertices = new_x
    tri_mesh = trimesh.Trimesh(shape.mesh.vertices, shape.mesh.faces)
    # tri_mesh.show()
    mds.plot_embedding(shape.mesh.vertices)
    print("end main")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MDS args')
    parser.add_argument('--p', default=[100, 200], help='p is the number of frequencies or '
                                                   'basis vectors')
    parser.add_argument('--q', default=[200, 400], help='q is the number of samples')
    parser.add_argument('--max_iter', default=500)
    parser.add_argument('--a_tol', default=0.001, help="absolute tolerance")
    parser.add_argument('--r_tol', default=0.00001, help="relative tolerance")
    parser.add_argument('--filename', default='input/cat3.off', help="file name")
    parser.add_argument('--d_mat_input', default='input/D_cat3.mat',
                        help='geodesic distance mat')
    parser.add_argument('--c', default=2, help="c = q/p, i.e. Nyquist ratio")
    parser.add_argument('--plot_flag', default=True)
    parser.add_argument('--compute_full_stress_flag', default=True)
    parser.add_argument('--display_every', default=10, help='display every n iterations')
    parser.add_argument('--max_size_for_pinv', default=1000, help='display every n iterations')

    _args = parser.parse_args()
    # main(_args, 'Both')
    main(_args, 'Numpy')
    # main(_args, 'PyTorch')

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
import os

from Shape.TorchShape import TorchShape

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main(_args, Type):
    print("start main")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if Type == 'PyTorch':
        shape = TorchShape(filename=_args.filename, device=device)
        print(device)

    elif Type == 'Numpy':
        shape = NumpyShape(filename=_args.filename)
    elif Type == 'Both':  # only works with the same shape for both - currently used
        # mainly for testing
        shape = NumpyShape(filename=_args.filename)
        shape_t = TorchShape(device=device, filename=_args.filename)
    else:
        print("Type should be PyTorch, Numpy or Both")
        raise SystemExit()

    # shape.mesh.show()
    shape.plot_embedding(shape.mesh.vertices)

    # TODO: need to use standalone geodesic fucntion:
    #  d_mat_input = shape.compute_geodesics()
    d_mat_input = sio.loadmat(_args.d_mat_input)['D']
    mds_params = MdsConfig.MdsParams(shape, _args)

    mds_params.set_shape(shape)
    mds_params.set_p_q(_args.p, _args.q)
    if Type == 'Both':
        mds_params.set_weights(shape.weights, shape_t.weights)
    else:
        mds_params.set_weights(shape.weights)

    [samples, d_mat] = shape.sample_mesh_fps(np.max(mds_params.q), d_mat_input)
    mds_params.set_samples(samples)

    # create subspace
    # TODO: this only works for mesh. If we have graph or point cloud it should be
    #  different
    shape.compute_subspace(max(mds_params.p))

    if Type == 'Both':
        shape_t.compute_subspace(max(mds_params.p))

    x0 = shape.mesh.vertices

    var_type = torch.float64
    if Type == 'PyTorch':
        mds = TorchMDS(mds_params, device=device)
        phi = shape.evecs.type(var_type).to(device)
        d_mat = torch.tensor(d_mat, dtype=var_type, device=device)
        x0 = torch.tensor(x0, dtype=var_type, device=device)
    elif Type == 'Numpy':
        mds = NumpyMDS(mds_params)
        phi = np.real(shape.evecs)
    elif Type == 'Both':
        # torch variables
        mds_t = TorchMDS(mds_params, device=device)
        phi_t = shape_t.evecs.type(var_type).to(device)
        d_mat_t = torch.tensor(d_mat, dtype=var_type, device=device)
        x0_t = torch.tensor(x0, dtype=var_type, device=device)
        # numpy variables
        mds = NumpyMDS(mds_params)
        phi = np.real(shape.evecs)
        # calc torch version
        new_x_t = mds_t.algorithm(d_mat_t, x0_t, phi_t)
        shape_t.mesh.vertices = new_x_t.cpu()
        tri_mesh_t = trimesh.Trimesh(shape_t.mesh.vertices, shape_t.mesh.faces)
        fig1 = plt.figure()
        plt.plot(mds_t.stress_list)
        fig1.show()
    else:
        print("Type should be PyTorch, Numpy or Both")
        raise SystemExit()

    # TODO: change inputs from TrackedArray to ndarray and check speed
    opt_x = mds.algorithm(d_mat, x0, phi)
    fig2 = plt.figure()
    plt.plot(mds.stress_list)
    fig2.show()

    if Type == 'PyTorch':
        canonical_form = Shape(vertices=opt_x.cpu(), faces=shape.mesh.faces)
    elif Type == 'Numpy':
        canonical_form = Shape(vertices=opt_x, faces=shape.mesh.faces)
    else:
        canonical_form_t = Shape(vertices=new_x_t.cpu(), faces=shape.mesh.faces)
        canonical_form = Shape(vertices=opt_x, faces=shape.mesh.faces)
        # shape.plot_embedding(canonical_form_t.mesh.vertices)

    # canonical_form.mesh.show()
    shape.plot_embedding(canonical_form.mesh.vertices)

    print("end main")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MDS args')
    parser.add_argument('--p', default=[100, 200], help='p is the number of frequencies or '
                                                   'basis vectors')
    parser.add_argument('--q', default=[200, 400], help='q is the number of samples')
    parser.add_argument('--max_iter', default=500)
    parser.add_argument('--a_tol', default=0.001, help="absolute tolerance")
    parser.add_argument('--r_tol', default=0.00001, help="relative tolerance")
    parser.add_argument('--filename', default='input/dog0.off', help="file name")
    parser.add_argument('--d_mat_input', default='input/D_dog0.mat',
                        help='geodesic distance mat')
    parser.add_argument('--c', default=2, help="c = q/p, i.e. Nyquist ratio")
    parser.add_argument('--plot_flag', default=True)
    parser.add_argument('--compute_full_stress_flag', default=False)
    parser.add_argument('--display_every', default=100, help='display every n iterations')
    parser.add_argument('--max_size_for_pinv', default=1000, help='display every n iterations')

    _args = parser.parse_args()
    # main(_args, 'Both')
    # main(_args, 'Numpy')
    main(_args, 'PyTorch')

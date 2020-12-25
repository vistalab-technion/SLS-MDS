import argparse
import numpy as np
import torch
import scipy.io as sio
from torch import sum
from torch.nn import Module
from typing import Any
from torch.autograd import Function
from torch.autograd.functional import jacobian, hessian
import MDS.MdsConfig as MdsConfig
from MDS.TorchMDS import TorchMDS
from Shape.NumpyShape import NumpyShape
from Shape.Shape import Shape
from MDS.NumpyMDS import NumpyMDS


class MDSLayer(Function):

    @staticmethod
    def forward(ctx, xn, d_mat, x0, phi, weights, samples):
        # solve MDS
        print("Forward")
        print(xn, d_mat, x0, phi, weights, samples)
        ctx.d_mat = torch.tensor(d_mat, requires_grad=True)
        ctx.phi = torch.tensor(phi, requires_grad=True)
        ctx.xn = xn
        ctx.x0 = torch.tensor(x0, requires_grad=True)
        ctx.weights = torch.tensor(weights, requires_grad=True)
        ctx.samples = torch.tensor(samples, requires_grad=False)
        return xn

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        print("Backward")

        with torch.enable_grad():
            phi = ctx.phi
            xn = ctx.xn
            x0 = ctx.x0
            weights = ctx.weights
            d_mat = ctx.d_mat

            def f(X):

                def V(w_mat):
                    mat_v = -w_mat + torch.diag(torch.diag(w_mat))
                    mat_v -= torch.diag(torch.sum(mat_v, 1))
                    return mat_v

                pass

                # TODO: transform into torch
                def B(d_mat, dx_mat, w_mat):
                    b_mat = np.zeros(d_mat.shape)
                    try:
                        tmp = -np.multiply(w_mat, d_mat)
                        b_mat[dx_mat != 0] = np.divide(tmp[dx_mat != 0],
                                                       dx_mat[dx_mat != 0])

                    except ZeroDivisionError:
                        print("divided by zero")

                    diag_mat_b = -np.diag(np.sum(b_mat, 1))
                    b_mat += diag_mat_b
                    return b_mat

            # TODO: write fixed point iteration for Eq. 18 in terms of X
            alpha = phi.T

        dxn_dd_mat = None
        dxn_x0 = None
        dxn_dphi = None
        dxn_dweights = None
        print(grad_outputs)
        return None, dxn_dd_mat, dxn_x0, dxn_dphi, dxn_dweights, None


class DifferentiableMDS(Module):
    def __init__(self, mds, x0, phi, d_mat):
        super().__init__()
        self.mds = mds
        self.weights = mds.mds_params.weights
        self.d_mat = d_mat
        self.x0 = np.array(x0)
        self.phi = phi
        self.samples = mds.mds_params.samples_array

    def forward(self):
        # SOLVING MDS
        xn = torch.tensor(self.mds.algorithm(self.d_mat, self.x0, self.phi),
                           requires_grad=True)
        return MDSLayer.apply(xn, self.d_mat, self.x0, self.phi, self.weights,
                              self.samples)


# downstream application
def main(_args):
    print("start main")
    shape = NumpyShape(filename=_args.filename)
    # shape.mesh.show()
    # shape.plot_embedding(shape.mesh.vertices)

    d_mat_input = sio.loadmat(_args.d_mat_input)['D']

    mds_params = MdsConfig.MdsParams(shape, _args)
    mds_params.set_shape(shape)
    mds_params.set_p_q(_args.p, _args.q)
    mds_params.set_weights(np.ones(d_mat_input.shape))

    [samples, d_mat] = shape.sample_mesh_fps(np.max(mds_params.q), d_mat_input)
    mds_params.set_samples(samples)

    # create subspace
    shape.compute_subspace(max(mds_params.p))
    mds = NumpyMDS(mds_params)
    phi = np.real(shape.evecs)
    x0 = shape.mesh.vertices

    diff_mds = DifferentiableMDS(mds, x0, phi, d_mat)
    xn = diff_mds.forward()
    loss = sum(xn)**2
    loss.backward()

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
    parser.add_argument('--plot_flag', default=False)
    parser.add_argument('--compute_full_stress_flag', default=True)
    parser.add_argument('--display_every', default=10, help='display every n iterations')
    parser.add_argument('--max_size_for_pinv', default=1000,
                        help='display every n iterations')

    _args = parser.parse_args()
    main(_args)

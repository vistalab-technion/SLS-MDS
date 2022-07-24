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
from Shape.TorchShape import TorchShape


class MDSLayer(Function):
    @staticmethod
    def forward(ctx, xn, d_mat, x0, phi, weights, samples, mds):
        # TODO: here you need to solve MDS

        print("Forward")
        print(xn, d_mat, x0, phi, weights, samples)
        phi.requires_grad = True
        x0.requires_grad = True
        d_mat.requires_grad = True

        ctx.d_mat = d_mat
        ctx.x0 = x0
        ctx.phi = phi
        ctx.xn = xn

        ctx.weights = torch.tensor(weights, requires_grad=True)
        ctx.samples = torch.tensor(samples, requires_grad=False)
        ctx.mds = mds
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
            samples = ctx.samples
            mds_object = ctx.mds

            def f(x1, x2):
                # q = mds_object.mds_params.q
                # p = mds_object.mds_params.p
                p = 100
                q = 200

                # extract samples from all variables
                x0_s = x0[samples[0:q], :]
                x_s = x1[samples[0:q], :]
                w_s = TorchMDS.compute_sub(weights, samples[0:q])
                phi_s = phi[samples[0:q], 0:p]
                d_s = TorchMDS.compute_sub(d_mat, samples[0:q])

                # v_s is the matrix v constructed from the sampled weights
                v_s = TorchMDS.compute_v(w_s)
                v_s_p = (torch.transpose(phi_s, 0, 1) @ v_s) @ phi_s  # projection of v_s on phi_s

                if (v_s_p.shape[0] < x0.shape[0]) or \
                        (p == mds_object.mds_params.shape.size):
                    v_s_p_inv = torch.pinverse(v_s_p)
                else:
                    print('"size too large for using pinv."')
                    raise SystemExit

                z = v_s_p_inv @ torch.transpose(phi_s, 0, 1)

                d_euc_s_mat_t = torch.cdist(x_s, x_s)
                b_s = mds_object.compute_mat_b(d_s, d_euc_s_mat_t, w_s)

                y = torch.sub(b_s @ x_s, v_s @ x0_s)
                alpha = z @ y
                return x0 + (phi[:, 0:p] @ alpha)

                # TODO: write fixed point equation for Eq. 18 in terms of X
                # alpha_{k+1} = g(alpha_k, X_0, D, W) - >  f(X_0, Phi, alpha,D,W,S)=0

        # return:
        # (-(df/dD)^{-1} @ (df/dX) ) @ dl/dX - > d loss /dD
        # (-(df/dW)^{-1} @ (df/dX) ) @ dl/dX - > d loss /dW
        # etc...

        dxn_dd_mat = jacobian(f(xn), (xn, d_mat))
        dxn_x0 = jacobian(f(xn), (xn, x0))
        dxn_dphi = jacobian(f(xn), (xn, phi))
        dxn_dweights = jacobian(f(xn), (xn, weights))
        dxn_dsamples = jacobian(f(xn), (xn, samples))
        print(grad_outputs)
        return None, dxn_dd_mat, dxn_x0, dxn_dphi, dxn_dweights, dxn_dsamples


class DifferentiableMDS(Module):
    def __init__(self, mds, x0, phi, d_mat):
        super().__init__()
        self.mds = mds
        self.weights = mds.mds_params.weights
        self.d_mat = d_mat
        self.phi = phi
        self.x0 = x0
        self.samples = mds.mds_params.samples_array

    def forward(self):
        # SOLVING MDS
        opt_x = torch.tensor(self.mds.algorithm(self.d_mat, self.x0, self.phi), requires_grad=True).clone()
        return MDSLayer.apply(opt_x, self.d_mat, self.x0, self.phi, self.weights,
                              self.samples, self.mds)


# downstream application
def main(_args):
    print("start main")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape = TorchShape(filename=_args.filename, device=device)
    # shape.mesh.show()
    # shape.plot_embedding(shape.mesh.vertices)

    d_mat_input = sio.loadmat(_args.d_mat_input)['D']

    mds_params = MdsConfig.MdsParams(shape, _args)
    mds_params.set_shape(shape)
    mds_params.set_p_q(_args.p, _args.q)
    mds_params.set_weights(shape.weights)

    [samples, d_mat] = shape.sample_mesh_fps(np.max(mds_params.q), d_mat_input)
    mds_params.set_samples(samples)

    # create subspace
    shape.compute_subspace(max(mds_params.p))
    mds = TorchMDS(mds_params, device)
    x0 = shape.mesh.vertices
    var_type = torch.float64

    phi = shape.evecs.type(var_type).to(device)
    d_mat = torch.tensor(d_mat, dtype=var_type, device=device)
    x0 = torch.tensor(x0, dtype=var_type, device=device)

    diff_mds = DifferentiableMDS(mds, x0, phi, d_mat)
    xn = diff_mds.forward()
    loss = sum(xn) ** 2
    loss.backward()


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
    parser.add_argument('--compute_full_stress_flag', default=True)
    parser.add_argument('--display_every', default=40, help='display every n iterations')
    parser.add_argument('--max_size_for_pinv', default=1000,
                        help='display every n iterations')

    _args = parser.parse_args()
    main(_args)

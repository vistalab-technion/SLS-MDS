import argparse
import numpy as np
import torch
from torch import sum
from torch.nn import Module
from typing import Any
from torch.autograd import Function
from torch.autograd.functional import jacobian, jvp, vjp
import MDS.MdsConfig as MdsConfig
from MDS.TorchMDS import TorchMDS
from Shape.TorchShape import TorchShape


class MDSLayer(Function):
    @staticmethod
    def forward(ctx, opt_x, x0, d, phi, w, mds, samples):
        print("Forward")

        ctx.mds = mds
        ctx.opt_x = opt_x
        ctx.samples = samples

        ctx.save_for_backward(opt_x, x0, d, phi, w)

        return opt_x

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        print("Backward")
        with torch.enable_grad():
            opt_x, x0, d, phi, w = ctx.saved_tensors
            mds_object = ctx.mds
            p = mds_object.mds_params.p[0]
            q = mds_object.mds_params.p[0]

            # # extract samples from all variables
            # opt_x_s = opt_x[ctx.samples[0:q], :]
            # x0_s = x0[ctx.samples[0:q], :]
            # w_s = TorchMDS.compute_sub(w, ctx.samples[0:q])
            # d_s = TorchMDS.compute_sub(d, ctx.samples[0:q])
            # phi_s = phi[ctx.samples[0:q], 0:p]

            def f_xn(opt_x_):
                # extract samples from all variables
                opt_x_s = opt_x_[ctx.samples[0:q], :]
                x0_s = x0[ctx.samples[0:q], :]
                w_s = TorchMDS.compute_sub(w, ctx.samples[0:q])
                d_s = TorchMDS.compute_sub(d, ctx.samples[0:q])
                phi_s = phi[ctx.samples[0:q], 0:p]

                v_s = TorchMDS.compute_v(w_s)
                v_s_p = (torch.transpose(phi_s, 0, 1) @ v_s) @ phi_s  # projection of v_s on phi_s

                if (v_s_p.shape[0] < opt_x.shape[0]) or \
                        (p == mds_object.mds_params.shape.size):
                    v_s_p_inv = torch.pinverse(v_s_p)
                else:
                    print('"size too large for using pinv."')
                    raise SystemExit

                z = v_s_p_inv @ torch.transpose(phi_s, 0, 1)

                d_euc_s_mat_t = torch.cdist(opt_x_s, opt_x_s)
                b_s = mds_object.compute_mat_b(d_s, d_euc_s_mat_t, w_s)

                y = torch.sub(b_s @ opt_x_s, v_s @ x0_s)
                alpha = z @ y
                return opt_x_ + (phi[:, 0:p] @ alpha)

            def f_phi(phi_):
                # extract samples from all variables
                opt_x_s = opt_x[ctx.samples[0:q], :]
                x0_s = x0[ctx.samples[0:q], :]
                w_s = TorchMDS.compute_sub(w, ctx.samples[0:q])
                d_s = TorchMDS.compute_sub(d, ctx.samples[0:q])
                phi_s = phi_[ctx.samples[0:q], 0:p]

                v_s = TorchMDS.compute_v(w_s)
                v_s_p = (torch.transpose(phi_s, 0, 1) @ v_s) @ phi_s  # projection of v_s on phi_s

                if (v_s_p.shape[0] < opt_x.shape[0]) or \
                        (p == mds_object.mds_params.shape.size):
                    v_s_p_inv = torch.pinverse(v_s_p)
                else:
                    print('"size too large for using pinv."')
                    raise SystemExit

                z = v_s_p_inv @ torch.transpose(phi_s, 0, 1)

                d_euc_s_mat_t = torch.cdist(opt_x_s, opt_x_s)
                b_s = mds_object.compute_mat_b(d_s, d_euc_s_mat_t, w_s)

                y = torch.sub(b_s @ opt_x_s, v_s @ x0_s)
                alpha = z @ y
                return opt_x + (phi_[:, 0:p] @ alpha)

            def f_x0(x0_):
                # extract samples from all variables
                opt_x_s = opt_x[ctx.samples[0:q], :]
                x0_s = x0_[ctx.samples[0:q], :]
                w_s = TorchMDS.compute_sub(w, ctx.samples[0:q])
                d_s = TorchMDS.compute_sub(d, ctx.samples[0:q])
                phi_s = phi[ctx.samples[0:q], 0:p]

                v_s = TorchMDS.compute_v(w_s)
                v_s_p = (torch.transpose(phi_s, 0, 1) @ v_s) @ phi_s  # projection of v_s on phi_s

                if (v_s_p.shape[0] < opt_x.shape[0]) or \
                        (p == mds_object.mds_params.shape.size):
                    v_s_p_inv = torch.pinverse(v_s_p)
                else:
                    print('"size too large for using pinv."')
                    raise SystemExit

                z = v_s_p_inv @ torch.transpose(phi_s, 0, 1)

                d_euc_s_mat_t = torch.cdist(opt_x_s, opt_x_s)
                b_s = mds_object.compute_mat_b(d_s, d_euc_s_mat_t, w_s)

                y = torch.sub(b_s @ opt_x_s, v_s @ x0_s)
                alpha = z @ y
                return opt_x + (phi[:, 0:p] @ alpha)

            def f_d(d_):
                # extract samples from all variables
                opt_x_s = opt_x[ctx.samples[0:q], :]
                x0_s = x0[ctx.samples[0:q], :]
                w_s = TorchMDS.compute_sub(w, ctx.samples[0:q])
                d_s = TorchMDS.compute_sub(d_, ctx.samples[0:q])
                phi_s = phi[ctx.samples[0:q], 0:p]

                v_s = TorchMDS.compute_v(w_s)
                v_s_p = (torch.transpose(phi_s, 0, 1) @ v_s) @ phi_s  # projection of v_s on phi_s

                if (v_s_p.shape[0] < opt_x.shape[0]) or \
                        (p == mds_object.mds_params.shape.size):
                    v_s_p_inv = torch.pinverse(v_s_p)
                else:
                    print('"size too large for using pinv."')
                    raise SystemExit

                z = v_s_p_inv @ torch.transpose(phi_s, 0, 1)

                d_euc_s_mat_t = torch.cdist(opt_x_s, opt_x_s)
                b_s = mds_object.compute_mat_b(d_s, d_euc_s_mat_t, w_s)

                y = torch.sub(b_s @ opt_x_s, v_s @ x0_s)
                alpha = z @ y
                return opt_x + (phi[:, 0:p] @ alpha)

            def f_w(w_):
                # extract samples from all variables
                opt_x_s = opt_x[ctx.samples[0:q], :]
                x0_s = x0[ctx.samples[0:q], :]
                w_s = TorchMDS.compute_sub(w_, ctx.samples[0:q])
                d_s = TorchMDS.compute_sub(d, ctx.samples[0:q])
                phi_s = phi[ctx.samples[0:q], 0:p]

                v_s = TorchMDS.compute_v(w_s)
                v_s_p = (torch.transpose(phi_s, 0, 1) @ v_s) @ phi_s  # projection of v_s on phi_s

                if (v_s_p.shape[0] < opt_x.shape[0]) or \
                        (p == mds_object.mds_params.shape.size):
                    v_s_p_inv = torch.pinverse(v_s_p)
                else:
                    print('"size too large for using pinv."')
                    raise SystemExit

                z = v_s_p_inv @ torch.transpose(phi_s, 0, 1)

                d_euc_s_mat_t = torch.cdist(opt_x_s, opt_x_s)
                b_s = mds_object.compute_mat_b(d_s, d_euc_s_mat_t, w_s)

                y = torch.sub(b_s @ opt_x_s, v_s @ x0_s)
                alpha = z @ y
                return opt_x + (phi[:, 0:p] @ alpha)

        # TODO: write fixed point equation for Eq. 18 in terms of X
        # alpha_{k+1} = g(alpha_k, X_0, D, W) - >  f(X_0, Phi, alpha,D,W,S)=0

        # return:
        # -(df/dD)^{-1} @ ((df/dX)  @ dl/dX) - > d loss /dD
        # -(df/dW)^{-1} @ ((df/dX)  @ dl/dX) - > d loss /dW
        # etc...
        # grad_outputs = grad_outputs
        print(f'grad_outputs= {grad_outputs}')

        df_dxn = jacobian(f_xn, opt_x)[0][0]
        dxn_df = torch.linalg.pinv(df_dxn)

        df_dx0 = jacobian(f_x0, x0)[0][0]
        df_dx0_grad = df_dx0 @ grad_outputs[0].t()
        dxn_dx0 = -(dxn_df @ df_dx0_grad)
        dxn_dx0 = dxn_dx0.t()

        # df_dphi = jacobian(f_phi, phi[:, :50])[0][0]

        grad_outputs_mat_d_w = torch.ones((d.shape[0], d.shape[1]), device=_args.device)
        grad_outputs_phi = torch.ones((phi.shape[0], phi.shape[1]), device=_args.device)

        grad_outputs_mat_d_w = torch.mul(grad_outputs_mat_d_w, grad_outputs[0][0][0])
        grad_outputs_phi = torch.mul(grad_outputs_phi, grad_outputs[0][0][0])
        # grad_outputs_x0 = grad_outputs[0].clone()

        df_dd_grad_jvp = jvp(f_d, d, grad_outputs_mat_d_w)
        dxn_dd = - (df_dd_grad_jvp[1] @ dxn_df)

        df_dw_grad_jvp = jvp(f_w, w, grad_outputs_mat_d_w)
        dxn_dw = - (df_dw_grad_jvp[1] @ dxn_df)

        df_dphi_grad_jvp = jvp(f_phi, phi, grad_outputs_phi)
        dxn_dphi = - (dxn_df @ df_dphi_grad_jvp[1])  # shape should be [size (3400), 200]

        # df_dx0_grad_jvp = jvp(f_x0, x0, grad_outputs_x0)
        # dxn_dx0 = dxn_df @ df_dx0_grad_jvp[1]
        yop

        return None, dxn_dx0, dxn_dd, dxn_dphi, dxn_dw, None, None

class MDSLayer_s(Function):
    @staticmethod
    def forward(ctx, opt_x, x0_s, d_s, phi_s, w_s, mds, samples):
        print("Forward")
        opt_x_s = opt_x[samples[0:mds.mds_params.q[0]], :]

        ctx.mds = mds
        ctx.opt_x = opt_x

        ctx.save_for_backward(opt_x_s, x0_s, d_s, phi_s, w_s)

        return opt_x_s

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        print("Backward")
        with torch.enable_grad():
            x_s, x0_s, d_s, phi_s, w_s = ctx.saved_tensors
            mds_object = ctx.mds
            opt_x = ctx.opt_x
            p = mds_object.mds_params.p[0]

            def f_x_x0_phi(xs, x0s, phis):
                v_s = TorchMDS.compute_v(w_s)
                v_s_p = (torch.transpose(phis, 0, 1) @ v_s) @ phis  # projection of v_s on phi_s

                if (v_s_p.shape[0] < opt_x.shape[0]) or \
                        (p == mds_object.mds_params.shape.size):
                    v_s_p_inv = torch.pinverse(v_s_p)
                else:
                    print('"size too large for using pinv."')
                    raise SystemExit

                z = v_s_p_inv @ torch.transpose(phis, 0, 1)

                d_euc_s_mat_t = torch.cdist(xs, xs)
                b_s = mds_object.compute_mat_b(d_s, d_euc_s_mat_t, w_s)

                y = torch.sub(b_s @ xs, v_s @ x0s)
                alpha = z @ y
                return xs + (phis[:, 0:p] @ alpha)

            def f_d(ds):
                v_s = TorchMDS.compute_v(w_s)
                v_s_p = (torch.transpose(phi_s, 0, 1) @ v_s) @ phi_s  # projection of v_s on phi_s

                if (v_s_p.shape[0] < opt_x.shape[0]) or \
                        (p == mds_object.mds_params.shape.size):
                    v_s_p_inv = torch.pinverse(v_s_p)
                else:
                    print('"size too large for using pinv."')
                    raise SystemExit

                z = v_s_p_inv @ torch.transpose(phi_s, 0, 1)

                d_euc_s_mat_t = torch.cdist(x_s, x_s)
                b_s = mds_object.compute_mat_b(ds, d_euc_s_mat_t, w_s)

                y = torch.sub(b_s @ x_s, v_s @ x0_s)
                alpha = z @ y
                return x_s + (phi_s[:, 0:p] @ alpha)

            def f_w(ws):
                v_s = TorchMDS.compute_v(ws)
                v_s_p = (torch.transpose(phi_s, 0, 1) @ v_s) @ phi_s  # projection of v_s on phi_s

                if (v_s_p.shape[0] < opt_x.shape[0]) or \
                        (p == mds_object.mds_params.shape.size):
                    v_s_p_inv = torch.pinverse(v_s_p)
                else:
                    print('"size too large for using pinv."')
                    raise SystemExit

                z = v_s_p_inv @ torch.transpose(phi_s, 0, 1)

                d_euc_s_mat_t = torch.cdist(x_s, x_s)
                b_s = mds_object.compute_mat_b(d_s, d_euc_s_mat_t, ws)

                y = torch.sub(b_s @ x_s, v_s @ x0_s)
                alpha = z @ y
                return x_s + (phi_s[:, 0:p] @ alpha)

        # TODO: write fixed point equation for Eq. 18 in terms of X
        # alpha_{k+1} = g(alpha_k, X_0, D, W) - >  f(X_0, Phi, alpha,D,W,S)=0

        # return:
        # -(df/dD)^{-1} @ ((df/dX)  @ dl/dX) - > d loss /dD
        # -(df/dW)^{-1} @ ((df/dX)  @ dl/dX) - > d loss /dW
        # etc...

        df_d_x_x0_phi_s = jacobian(f_x_x0_phi, (x_s, x0_s, phi_s))
        #
        for index, mat in enumerate(df_d_x_x0_phi_s):
            print(f'df_d_x_x0_phi_s[{index}]: max={torch.max(mat)}, min={torch.min(mat)}')
        df_dxn_s = df_d_x_x0_phi_s[0][0][0]
        df_dx0_s = df_d_x_x0_phi_s[1][0][0]
        df_dphi_s = df_d_x_x0_phi_s[2][0][0]
        dxn_df = torch.linalg.pinv(df_dxn_s)  # check

        grad_outputs_mat_d_w = torch.ones((d_s.shape[0], d_s.shape[1]), device=_args.device)
        grad_outputs = grad_outputs[0]

        df_dx0_grad = df_dx0_s @ grad_outputs.t()
        dxn_dx0 = -(dxn_df @ df_dx0_grad)
        dxn_dx0 = dxn_dx0.t()

        grad_outputs_mat_d_w = torch.mul(grad_outputs_mat_d_w, grad_outputs[0][0])

        df_dphi_grad = df_dphi_s.t() @ grad_outputs
        dxn_dphi = -(dxn_df.t() @ df_dphi_grad.t())

        _, df_dd_grad = jvp(f_d, d_s, grad_outputs_mat_d_w)
        _, df_dw_grad = jvp(f_w, w_s, grad_outputs_mat_d_w)

        dxn_dd_mat = - (df_dd_grad @ dxn_df)
        dxn_dweights = - (df_dw_grad @ dxn_df)

        return None, dxn_dx0, dxn_dd_mat, dxn_dphi, dxn_dweights, None, None


class DifferentiableMDS(Module):
    def __init__(self, mds, x0, phi, d_mat):
        super().__init__()
        self.mds = mds
        self.d_mat = d_mat
        self.x0 = x0
        self.phi = phi
        self.weights = mds.mds_params.weights.clone().detach().requires_grad_(True)
        self.samples = torch.tensor(mds.mds_params.samples_array, requires_grad=False)

    def forward(self):
        # SOLVING MDS
        opt_x = self.mds.algorithm(self.d_mat, self.x0, self.phi).requires_grad_()
        p = self.mds.mds_params.p[0]
        q = self.mds.mds_params.q[0]

        if _args.is_samples:
            # extract samples from all variables
            x0_s = self.x0[self.samples[0:q], :]
            w_s = TorchMDS.compute_sub(self.weights, self.samples[0:q])
            d_s = TorchMDS.compute_sub(self.d_mat, self.samples[0:q])
            phi_s = self.phi[self.samples[0:q], 0:p]

            return MDSLayer_s.apply(opt_x, x0_s, d_s, phi_s, w_s, self.mds, self.samples)
        else:
            return MDSLayer.apply(opt_x, self.x0, self.d_mat, self.phi, self.weights, self.mds, self.samples)


def main(_args):
    print("start main")
    torch.cuda.set_device(_args.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shape = TorchShape(args=_args, device=device)

    mds_params = MdsConfig.MdsParams(shape, _args)
    mds_params.set_shape(shape)
    mds_params.set_weights(shape.weights)

    [samples, d_mat] = shape.sample_mesh_fps(np.max(mds_params.q), d_mat=_args.d_mat_input)
    mds_params.set_samples(samples)

    # create subspace
    shape.compute_subspace(max(mds_params.p))
    mds = TorchMDS(mds_params, device)
    x0 = shape.mesh.vertices
    var_type = torch.float64

    phi = shape.evecs.type(var_type).to(device).requires_grad_()
    d_mat = torch.tensor(d_mat, dtype=var_type, device=device, requires_grad=True)
    x0 = torch.tensor(x0, dtype=var_type, device=device, requires_grad=True)

    diff_mds = DifferentiableMDS(mds, x0, phi, d_mat)
    xn = diff_mds()
    loss = sum(xn) ** 2
    loss.backward()

    print("end")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MDS args')
    parser.add_argument('--p', default=[100, 200], help='p is the number of frequencies or '
                                                        'basis vectors')
    parser.add_argument('--q', default=[200, 400], help='q is the number of samples')
    parser.add_argument('--max_iter', default=500)
    parser.add_argument('--a_tol', default=0.001, help='absolute tolerance')
    parser.add_argument('--r_tol', default=0.00001, help='relative tolerance')
    parser.add_argument('--filename', default='input/cat3.off', help='file name')
    parser.add_argument('--d_mat_input', default='input/cat3_geo_dist_s.npy',
                        help='geodesic distance mat')
    parser.add_argument('--c', type=int, default=2, help='c = q/p, i.e. Nyquist ratio')
    parser.add_argument('--plot_flag', type=bool, default=False)
    parser.add_argument('--compute_full_stress_flag', type=bool,  default=True)
    parser.add_argument('--display_every', type=int,  default=20, help='display every n iterations')
    parser.add_argument('--max_size_for_pinv', type=int,  default=1000,
                        help='display every n iterations')
    parser.add_argument('--device', type=int, default='0',  help='cuda device')
    parser.add_argument('--is_samples', type=bool, default=False,
                        help='if true use samples for DifferentiableMDS - MDSLayer')

    _args = parser.parse_args()
    main(_args)

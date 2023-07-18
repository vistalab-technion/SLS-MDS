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

        ctx.save_for_backward(opt_x, x0, d, phi, w, samples)

        return opt_x

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        print("Backward")
        print(grad_outputs)
        print(torch.min(*grad_outputs))
        print(torch.max(*grad_outputs))
        with torch.enable_grad():
            opt_x, x0, d, phi, w, samples = ctx.saved_tensors

            mds_object = ctx.mds
            x_s = opt_x[samples[0:mds_object.mds_params.q[0]], :]
            p = mds_object.mds_params.p[0]
            q = mds_object.mds_params.q[0]

            # extract samples from all variables
            x0_s = x0[samples[0:q], :]
            w_s = TorchMDS.compute_sub(w, samples[0:q])
            d_s = TorchMDS.compute_sub(d, samples[0:q])
            phi_s = phi[samples[0:q], 0:p]

            def f_xn(x):
                xs = x[samples[0:mds_object.mds_params.q[0]], :]
                v_s = TorchMDS.compute_v(w_s)
                v_s_p = (torch.transpose(phi_s, 0, 1) @ v_s) @ phi_s  # projection of v_s on phi_s

                if (v_s_p.shape[0] < opt_x.shape[0]) or \
                        (p == mds_object.mds_params.shape.size):
                    v_s_p_inv = torch.pinverse(v_s_p)
                else:
                    print('"size too large for using pinv."')
                    raise SystemExit

                z = v_s_p_inv @ torch.transpose(phi_s, 0, 1)

                d_euc_s_mat_t = torch.cdist(xs, xs)
                b_s = mds_object.compute_mat_b(d_s, d_euc_s_mat_t, w_s)

                y = torch.sub(b_s @ xs, v_s @ x0_s)
                alpha = z @ y
                return x + (phi[:, 0:p] @ alpha)

            def f_x0(x_0):
                x0_s = x_0[samples[0:q], :]
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
                b_s = mds_object.compute_mat_b(d_s, d_euc_s_mat_t, w_s)

                y = torch.sub(b_s @ x_s, v_s @ x0_s)
                alpha = z @ y
                return opt_x + (phi[:, 0:p] @ alpha)

            def f_phi(_phi):
                phi_s = _phi[samples[0:q], 0:p]
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
                b_s = mds_object.compute_mat_b(d_s, d_euc_s_mat_t, w_s)

                y = torch.sub(b_s @ x_s, v_s @ x0_s)
                alpha = z @ y
                return opt_x + (_phi[:, 0:p] @ alpha)

            def f_d(_d):
                d_s = TorchMDS.compute_sub(_d, samples[0:q])
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
                b_s = mds_object.compute_mat_b(d_s, d_euc_s_mat_t, w_s)

                y = torch.sub(b_s @ x_s, v_s @ x0_s)
                alpha = z @ y
                return opt_x + (phi[:, 0:p] @ alpha)

            def f_w(_w):
                w_s = TorchMDS.compute_sub(_w, samples[0:q])
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
                b_s = mds_object.compute_mat_b(d_s, d_euc_s_mat_t, w_s)

                y = torch.sub(b_s @ x_s, v_s @ x0_s)
                alpha = z @ y
                return opt_x + (phi[:, 0:p] @ alpha)

        # TODO: write fixed point equation for Eq. 18 in terms of X
        # alpha_{k+1} = g(alpha_k, X_0, D, W) - >  f(X_0, Phi, alpha,D,W,S)=0

        # return:
        # -(df/dD)^{-1} @ ((df/dX)  @ dl/dX) - > d loss /dD
        # -(df/dW)^{-1} @ ((df/dX)  @ dl/dX) - > d loss /dW
        # etc...

        df_dx = jacobian(f_xn, opt_x)
        df_dx = df_dx.reshape(df_dx.shape[0]*df_dx.shape[1], df_dx.shape[2]*df_dx.shape[3])
        dxn_df = torch.linalg.pinv(df_dx)  # pinv(df/dx) = (df/dx)^-1
        dxn_df_grad = dxn_df @ grad_outputs[0].reshape(1200)
        dxn_df_grad = dxn_df_grad.reshape(400, 3)

        _, df_dx0_vjp = vjp(f_x0, x0, dxn_df_grad)

        _, df_dd_vjp = vjp(f_d, d, dxn_df_grad)

        _, df_dw_vjp = vjp(f_w, w, dxn_df_grad)

        _, df_dphi_vjp = vjp(f_phi, phi, dxn_df_grad)

        return None, df_dx0_vjp, df_dd_vjp, df_dphi_vjp, df_dw_vjp, None, None


class MDSLayer_s(Function):
    @staticmethod
    def forward(ctx, opt_x, x0_s, phi_s, d_s, w_s, mds, samples):
        print("Forward")
        # opt_x, x0_s, d_s, phi_s, w_s, mds, samples = inputs
        opt_x_s = opt_x[samples[0:mds.mds_params.q[0]], :]

        ctx.mds = mds
        ctx.opt_x = opt_x

        ctx.save_for_backward(opt_x_s, x0_s, phi_s, d_s, w_s)

        return opt_x_s

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        print("Backward")
        print(grad_outputs)
        print(torch.min(*grad_outputs))
        print(torch.max(*grad_outputs))
        with torch.enable_grad():
            x_s, x0_s, phi_s, d_s, w_s = ctx.saved_tensors
            mds_object = ctx.mds
            opt_x = ctx.opt_x
            p = mds_object.mds_params.p[0]

            def f_xn(xs):
                v_s = TorchMDS.compute_v(w_s)
                v_s_p = (torch.transpose(phi_s, 0, 1) @ v_s) @ phi_s  # projection of v_s on phi_s

                if (v_s_p.shape[0] < opt_x.shape[0]) or \
                        (p == mds_object.mds_params.shape.size):
                    v_s_p_inv = torch.pinverse(v_s_p)
                else:
                    print('"size too large for using pinv."')
                    raise SystemExit

                z = v_s_p_inv @ torch.transpose(phi_s, 0, 1)

                d_euc_s_mat_t = torch.cdist(xs, xs)
                b_s = mds_object.compute_mat_b(d_s, d_euc_s_mat_t, w_s)

                y = torch.sub(b_s @ xs, v_s @ x0_s)
                alpha = z @ y
                return xs + (phi_s[:, 0:p] @ alpha)

            def f_x0(x0s):
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
                b_s = mds_object.compute_mat_b(d_s, d_euc_s_mat_t, w_s)

                y = torch.sub(b_s @ x_s, v_s @ x0s)
                alpha = z @ y
                return x_s + (phi_s[:, 0:p] @ alpha)

            def f_phi(phis):
                v_s = TorchMDS.compute_v(w_s)
                v_s_p = (torch.transpose(phis, 0, 1) @ v_s) @ phis  # projection of v_s on phi_s

                if (v_s_p.shape[0] < opt_x.shape[0]) or \
                        (p == mds_object.mds_params.shape.size):
                    v_s_p_inv = torch.pinverse(v_s_p)
                else:
                    print('"size too large for using pinv."')
                    raise SystemExit

                z = v_s_p_inv @ torch.transpose(phis, 0, 1)

                d_euc_s_mat_t = torch.cdist(x_s, x_s)
                b_s = mds_object.compute_mat_b(d_s, d_euc_s_mat_t, w_s)

                y = torch.sub(b_s @ x_s, v_s @ x0_s)
                alpha = z @ y
                return x_s + (phis[:, 0:p] @ alpha)

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

        df_dx = jacobian(f_xn, x_s)
        df_dx_s = df_dx.reshape(df_dx.shape[0] * df_dx.shape[1], df_dx.shape[2] * df_dx.shape[3])
        dxn_df = torch.linalg.pinv(df_dx_s)  # pinv(df/dx) = (df/dx)^-1
        dxn_df_grad = dxn_df @ grad_outputs[0].reshape(grad_outputs[0].shape[0] * grad_outputs[0].shape[1])
        dxn_df_grad = - dxn_df_grad.reshape(df_dx.shape[0], df_dx.shape[1])

        _, df_dx0_vjp = vjp(f_x0, x0_s, dxn_df_grad)
        _, df_dx0_jvp = jvp(f_x0, x0_s, dxn_df_grad)

        _, df_dd_vjp = vjp(f_d, d_s, dxn_df_grad)
        # _, df_dd_jvp = jvp(f_d, d_s, dxn_df_grad)

        _, df_dw_vjp = vjp(f_w, w_s, dxn_df_grad)

        _, df_dphi_vjp = vjp(f_phi, phi_s, dxn_df_grad)

        return None, df_dx0_vjp, df_dphi_vjp, df_dd_vjp, df_dw_vjp, None, None


class DifferentiableMDS(Module):
    def __init__(self, mds, x0, phi, d_mat, weights, is_samples):
        super().__init__()
        self.mds = mds
        self.d_mat = d_mat
        self.x0 = x0
        self.phi = phi
        self.weights = weights
        self.samples = torch.tensor(mds.mds_params.samples_array, requires_grad=False)
        self.is_samples = is_samples

    def forward(self):
        # SOLVING MDS
        opt_x = self.mds.algorithm(self.d_mat, self.x0, self.phi)
        p = self.mds.mds_params.p[0]
        q = self.mds.mds_params.q[0]

        if not self.is_samples:
            return MDSLayer.apply(opt_x, self.x0, self.d_mat, self.phi, self.weights, self.mds, self.samples)

        else:
            # extract samples from all variables
            x0_s = self.x0[self.samples[0:q], :]
            w_s = TorchMDS.compute_sub(self.weights, self.samples[0:q])
            d_s = TorchMDS.compute_sub(self.d_mat, self.samples[0:q])
            phi_s = self.phi[self.samples[0:q], 0:p]
            return MDSLayer_s.apply(opt_x, x0_s, phi_s, d_s, w_s, self.mds, self.samples)


def main(_args):
    print("start main")
    torch.cuda.set_device(_args.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    var_type = torch.float64

    shape_target = TorchShape(args=_args, filename=_args.filename_target, device=device)

    mds_params_target = MdsConfig.MdsParams(shape_target, _args)
    mds_params_target.set_shape(shape_target)
    mds_params_target.set_weights(shape_target.weights)

    [samples_target, d_mat_target] = shape_target.sample_mesh_fps(np.max(mds_params_target.q),
                                                                  d_mat=_args.d_mat_input_target)
    mds_params_target.set_samples(samples_target)

    # create subspace
    shape_target.compute_subspace(max(mds_params_target.p))
    mds_target = TorchMDS(mds_params_target, device)
    x0_target = shape_target.get_vertices()

    phi_target = shape_target.get_evecs().type(var_type).to(device)
    d_mat_target = torch.tensor(d_mat_target, dtype=var_type, device=device)
    x0_target = torch.tensor(x0_target, dtype=var_type, device=device)
    weights_target = mds_params_target.weights.clone().detach().requires_grad_()

    diff_mds = DifferentiableMDS(mds_target, x0_target, phi_target, d_mat_target, weights_target)
    xn = diff_mds.forward()
    loss = torch.sum(xn)**2
    loss.backward()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MDS args')
    parser.add_argument('--p', default=[100, 200], help='p is the number of frequencies or '
                                                        'basis vectors')
    parser.add_argument('--q', default=[200, 400], help='q is the number of samples')
    parser.add_argument('--max_iter', default=500)
    parser.add_argument('--a_tol', default=0.001, help='absolute tolerance')
    parser.add_argument('--r_tol', default=0.00001, help='relative tolerance')
    parser.add_argument('--filename_target', default='input/cat3.off', help='file name')
    parser.add_argument('--d_mat_input_target', default='input/D_cat3.mat',
                        help='geodesic distance mat')
    parser.add_argument('--c', type=int, default=2, help='c = q/p, i.e. Nyquist ratio')
    parser.add_argument('--plot_flag', type=bool, default=False)
    parser.add_argument('--compute_full_stress_flag', type=bool, default=True)
    parser.add_argument('--display_every', type=int, default=20, help='display every n iterations')
    parser.add_argument('--max_size_for_pinv', type=int, default=1000,
                        help='display every n iterations')
    parser.add_argument('--device', type=int, default='0', help='cuda device')
    parser.add_argument('--is_samples', type=bool, default=True,
                        help='if true use samples for DifferentiableMDS - MDSLayer')

    _args = parser.parse_args(args=[])
    main(_args)

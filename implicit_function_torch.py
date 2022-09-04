import argparse
import numpy as np
import torch
from torch import sum
from torch.nn import Module
from typing import Any
from torch.autograd import Function
from torch.autograd.functional import jacobian
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

        df_dxn = jacobian(f_xn, (opt_x))[0][0]  # error: cuda out of memory
        print(f'df_dxn: max={torch.max(df_dxn)}, min={torch.min(df_dxn)}')

        df_dx0 = jacobian(f_x0, x0)[0][0]
        print(f'df_dx0: max={torch.max(df_dx0)}, min={torch.min(df_dx0)}')

        # todo: # error: cuda out of memory, fix: add for loop that calculate jacobian for 400(+,-) iter.
        df_dphi = jacobian(f_phi, (phi))[0][0]
        print(f'df_dphi: max={torch.max(df_dphi)}, min={torch.min(df_dphi)}')

        df_dd = jacobian(f_d, d)[0][0]
        print(f'df_dd: max={torch.max(df_dd)}, min={torch.min(df_dd)}')

        df_dw = jacobian(f_w, w)[0][0]
        print(f'df_dw: max={torch.max(df_dw)}, min={torch.min(df_dw)}')
        # end todo
        df_dxn_grad = df_dxn @ grad_outputs[0].t()

        dxn_dx0 = torch.true_divide(1, df_dx0.t()) @ df_dxn_grad
        dxn_dx0 = dxn_dx0.t()
        torch.save(dxn_dx0, "output/dxn_dx0.pt")

        dxn_dd_mat = torch.true_divide(1, df_dd.t()) @ df_dxn_grad
        torch.save(dxn_dd_mat, "output/dxn_dd_mat.pt")

        dxn_dweights = torch.true_divide(1, df_dw.t()) @ df_dxn_grad
        torch.save(dxn_dweights, "output/dxn_dweights.pt")

        dxn_dphi = torch.true_divide(1, df_dphi.t()) @ df_dxn_grad
        dxn_dphi = dxn_dphi.t()
        torch.save(dxn_dphi, "output/dxn_dphi.pt")

        return None, dxn_dx0, dxn_dd_mat, dxn_dphi, dxn_dweights, None, None

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

        # torch.save(df_dx0, "output/df_dx0.pt")
        # df_dxn = torch.load("output/df_dxn.pt")
        # print(f'df_dxn: max={torch.max(df_dxn)}, min={torch.min(df_dxn)}')
        #
        # df_dx0 = torch.load("output/df_dx0.pt")
        # print(f'df_dx0: max={torch.max(df_dx0)}, min={torch.min(df_dx0)}')
        df_dd_s = jacobian(f_d, d_s)[0][0]
        df_dw_s = jacobian(f_w, w_s)[0][0]
        print(f'df_dd_s: max={torch.max(df_dd_s)}, min={torch.min(df_dd_s)}')
        print(f'df_dw_s: max={torch.max(df_dw_s)}, min={torch.min(df_dw_s)}')

        df_d_x_x0_phi_s = jacobian(f_x_x0_phi, (x_s, x0_s, phi_s))  # error: cuda out of memory

        for index, mat in enumerate(df_d_x_x0_phi_s):
            print(f'df_d_x_x0_phi_s[{index}]: max={torch.max(mat)}, min={torch.min(mat)}')
        df_dxn_s = df_d_x_x0_phi_s[0][0][0]
        df_dx0_s = df_d_x_x0_phi_s[1][0][0]
        df_dphi_s = df_d_x_x0_phi_s[2][0][0]

        df_dxn_grad = df_dxn_s @ grad_outputs[0].t()

        dxn_dx0 = torch.true_divide(1, df_dx0_s.t()) @ df_dxn_grad
        dxn_dx0 = dxn_dx0.t()
        torch.save(dxn_dx0, "output/dxn_dx0.pt")

        dxn_dd_mat = torch.true_divide(1, df_dd_s.t()) @ df_dxn_grad
        torch.save(dxn_dd_mat, "output/dxn_dd_mat.pt")

        dxn_dweights = torch.true_divide(1, df_dw_s.t()) @ df_dxn_grad
        torch.save(dxn_dweights, "output/dxn_dweights.pt")

        dxn_dphi = torch.true_divide(1, df_dphi_s.t()) @ df_dxn_grad
        dxn_dphi = dxn_dphi.t()
        torch.save(dxn_dphi, "output/dxn_dphi.pt")

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
    parser.add_argument('--is_samples', type=bool, default=True,
                        help='if true use samples for DifferentiableMDS - MDSLayer')

    _args = parser.parse_args()
    main(_args)

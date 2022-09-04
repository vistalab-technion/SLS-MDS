import numpy as np
import torch
import warnings
from MDS.MDS import MDS


class TorchMDS(MDS):

    def __init__(self, params, device):
        MDS.__init__(self, params)
        self.device = device

    def algorithm(self, distances, x0, phi):
        """
        _s stands for sampled, _p stands for projected on subspace
        :param distances: n x n - geodesic distance matrix. Should at least have distances
                        between points in sample set
        :param weights: n x n - symmetric weights for mds objective
        :param x0: n x embedding_dim - initial euclidean embedding
        :param phi: n x p_max - subspace
        :return: xn: x0 + (phi * alpha_star) - the optimal solution of the mds algorithm
        """
        print("start algorithm")
        # extract parameters
        q_array = self.mds_params.q  # array of q values, i.e., q = number of samples
        p_array = self.mds_params.p  # array of p values, i.e., p = size of subspace

        samples = torch.from_numpy(np.asarray(self.mds_params.samples_array)).clone().to(self.device)
        weights = self.mds_params.weights.to(self.device)

        intermediate_results_list = []

        old_stress = torch.tensor([])

        p = 0
        q = 0

        xk = x0
        for i in range(len(p_array)):
            converged = False

            q = q_array[i]  # q is the number of samples
            p = p_array[i]  # p is the number of frequencies\ basis vectors
            assert q > p, "q should be larger than p"
            if q < 2*p:
                warnings.warn('"It is recommended that q will be least 2p"')

            # alpha is the representation of the displacement field in the subspace
            alpha = torch.zeros([p, self.mds_params.shape.dim], dtype=torch.float64, device=self.device)

            # extract samples from all variables
            xk_s = xk[samples[0:q], :]
            w_s = self.compute_sub(weights, samples[0:q])
            phi_s = phi[samples[0:q], 0:p]
            d_s = self.compute_sub(distances, samples[0:q])

            # v_s is the matrix v constructed from the sampled weights
            v_s = self.compute_v(w_s)
            v_s_p = (torch.transpose(phi_s, 0, 1) @ v_s) @ phi_s  # projection of v_s on phi_s

            if (v_s_p.shape[0] < xk.shape[0]) or \
                    (p == self.mds_params.shape.size):
                v_s_p_inv = torch.pinverse(v_s_p)
            else:
                print('"size too large for using pinv."')
                raise SystemExit

            z = v_s_p_inv @ torch.transpose(phi_s, 0, 1)
            v_s_xk_s = v_s @ xk_s
            x_s = xk_s

            d_euc_s_mat_t = torch.cdist(x_s, x_s)

            old_stress = self.compute_stress(d_s, d_euc_s_mat_t, w_s)
            iter_count = 1
            self.stress_list.append((1/(q*q))*old_stress.cpu().detach().numpy())
            while not converged:
                # --------------------------  plotting --------------------------------
                if self.mds_params.plot_flag and (iter_count % self.mds_params.display_every) == 0:
                    if self.device.type == 'cuda':
                        self.plot_embedding(xk.cpu().detach().numpy() + torch.matmul(
                            phi[:, 0:p], alpha).cpu().detach().numpy())
                        print(f'iter : {iter_count}, stress : {old_stress}')

                    else:
                        self.plot_embedding(xk + torch.from_numpy(np.matmul(
                            phi[:, 0:p].numpy(), alpha.numpy())))
                        print(f'iter : {iter_count}, stress : {old_stress}')
                # --------------------------------------------------------------------

                b_s = self.compute_mat_b(d_s, d_euc_s_mat_t, w_s)
                # this is B from equation 5 in [1] computed on the sample set

                y = torch.sub(b_s @ x_s, v_s_xk_s)
                alpha = z @ y

                x_s = torch.add(xk_s, torch.matmul(phi_s, alpha))
                d_euc_s_mat_t = torch.cdist(x_s, x_s)

                # TODO: replace with dedicated function

                # check convergence
                new_stress = self.compute_stress(d_s, d_euc_s_mat_t, w_s)
                converged = (new_stress <= self.mds_params.a_tol) or \
                            (1 - (new_stress / old_stress) <= self.mds_params.r_tol) or \
                            (self.mds_params.max_iter <= iter_count)
                old_stress = new_stress
                self.stress_list.append((1 / (q * q)) * old_stress.cpu().detach().numpy())
                iter_count += 1

                if self.mds_params.compute_full_embedding_flag:
                    x = xk + torch.matmul(phi[:, 0:p], alpha)
                    if self.mds_params.compute_full_stress_flag:
                        intermediate_results_list.append(x)
            xk = xk + torch.matmul(phi[:, 0:p], alpha)

        self.mds_params.set_p_q([p], [q])
        print(f'final stress : {old_stress.data}')
        return xk + torch.matmul(phi[:, 0:p], alpha)

    @staticmethod
    def compute_v(w_mat):
        mat_v = -w_mat + torch.diag(torch.diag(w_mat))
        mat_v -= torch.diag(torch.sum(mat_v, 1))

        return mat_v

    def compute_mat_b(self, d_mat, dx_mat, w_mat):
        b_mat = torch.zeros(d_mat.shape, dtype=torch.float64).to(device=self.device)
        try:
            tmp = -torch.mul(w_mat, d_mat)
            b_mat[dx_mat != 0] = torch.true_divide(tmp[dx_mat != 0], dx_mat[dx_mat != 0])

        except ZeroDivisionError:
            print("divided by zero")

        diag_mat_b = -torch.diag(torch.sum(b_mat, 1))
        b_mat += diag_mat_b
        return b_mat

    @staticmethod
    def compute_stress(d_mat, d_euc_mat, w_mat):
        """
        computed stress = sum w_i*(d_mat_i-dx_mat_i)^2
        :param d_mat: geodesic distance matrix
        :param d_euc_mat: euclidean distance matrix
        :param w_mat: weights matrix
        :return: stress
        """
        tmp0 = torch.sub(torch.triu(d_euc_mat), torch.triu(d_mat))
        tmp = torch.pow(tmp0, 2)
        return torch.sum(torch.mul(torch.triu(w_mat), tmp))

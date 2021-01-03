import numpy as np
import torch
from scipy import linalg
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import warnings
from MDS.MDS import MDS
from SignalType import SignalType
import logging


class TorchMdsNp(MDS):

    def __init__(self, params):
        MDS.__init__(self, params)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.basicConfig(filename='../TorchMDS.log', level=logging.INFO)
        self.logger = logging.getLogger('TorchMDS')
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - '
                                      '%(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.info("PyTorch Logger")

    # _s stands for sampled, _p stands for projected on subspace
    def algorithm(self, distances, x0, phi):
        """
        :param distances: n x n - geodesic distance matrix. Should at least have distances
                        between points in sample set
        :param weights: n x n - symmetric weights for mds objective
        :param x0: n x embedding_dim - initial euclidean embedding
        :param phi: n x p_max - subspace
        :return:
        """
        print("start algorithm")
        # extract parameters
        q_array = self.mds_params.q  # array of q values, i.e., q = number of samples
        p_array = self.mds_params.p  # array of p values, i.e., p = size of subspace

        samples = torch.from_numpy(np.asarray(self.mds_params.samples_array)).\
            to(self.device)
        distances.to(self.device)
        #x0_torch = torch.from_numpy(x0).to(self.device)

        # alpha is the representation of the displacement field in the subspace
        alpha = torch.zeros([p_array[0], self.mds_params.shape.dim]).to(self.device)
        converged_flag = False

        intermediate_results_list = []

        for i in range(len(p_array)):
            # TODO: remove numpy dependencies. Optionally, we should normalize d_mat to\
            #  be between 0 and 1

            q = q_array[i]  # q is the number of samples
            p = p_array[i]  # p is the number of frequencies\ basis vectors
            assert q > p, "q should be larger than p"
            if q < 2*p:
                warnings.warn('"It is recommended that q will be least 2p"')

            # extract samples from all variables
            x0_s = x0[samples[0:q], :]
            w_s = self.compute_sub(self.mds_params.weights, samples[0:q]).to(self.device)
            phi_s = phi[samples[0:q], 0:p]
            d_s = self.compute_sub(distances, samples[0:q]).to(self.device)
            # v_s is the matrix v constructed from the sampled weights
            v_s = self.compute_v(w_s).to(self.device)
            # v_s_p = torch.matmul(torch.matmul(torch.transpose(phi_s, 0, 1), v_s),
            # phi_s)
            v_s_p = np.matmul(np.matmul(np.transpose(phi_s.cpu().numpy()), v_s.cpu().numpy()),
                              phi_s.cpu().numpy())  # projection of v_s on phi_s
            v_s_p = torch.from_numpy(v_s_p)

            v_s_p_inv = torch.pinverse(v_s_p)
            # TODO: check res - last time gave worng res vs linalg.pinv2(v_s_p)
            z = torch.from_numpy(np.matmul(v_s_p_inv.cpu().numpy(), torch.transpose(
                phi_s.data, 0, 1).cpu().numpy()))  # z = pinv(phi_s'*v_s*phi_s)*phi_s'
            v_s_x0_s = torch.from_numpy(np.matmul(v_s.cpu().numpy(), x0_s.cpu().numpy()))
            x_s = x0_s

            self.logger.info("index = {}:\nx0_s = {}\nw_s = {}\nphi_s = {}\nd_s = {}\n"
                             "v_s = {}\nv_s_p = {}\nv_s_P_i = {}\nz = {}\nv_s_x0_s = {}\n"
                             .format(i, x0_s, w_s, phi_s, d_s, v_s, v_s_p, v_s_p_inv,
                                     z, v_s_x0_s))
            d_euc_s_mat_t = torch.cdist(x_s, x_s)

            # d_euc_s_mat_np = squareform(pdist(x_s, 'euclidean'))
            # TODO: squareform is actually redundant, we can use an upper triangular mat
            # d_euc_s_mat = torch.from_numpy(d_euc_s_mat_t).type('float64')
            old_stress = self.compute_stress(d_s, d_euc_s_mat_t, w_s)
            iter_count = 0
            self.stress_list.append(old_stress)
            while not converged_flag:
                # --------------------------  plotting --------------------------------
                if self.mds_params.plot_flag and (iter_count % 10) == 0:
                    if self.device == 'cuda':
                        self.plot_embedding(x0.cpu().numpy() + torch.matmul(
                            phi[:, 0:p], alpha).cpu().numpy())
                        print(f'iter : {iter_count}, stress : {old_stress}')

                    else:
                        self.plot_embedding(x0.cpu() + torch.from_numpy(np.matmul(
                            phi[:, 0:p].cpu().numpy(), alpha.cpu().numpy())))
                        print(f'iter : {iter_count}, stress : {old_stress}')
                # --------------------------------------------------------------------

                b_s = self.compute_mat_b(d_s.cpu().numpy(), d_euc_s_mat_t.cpu().numpy(), w_s.cpu().numpy())
                # this is B from equation 5 in [1] computed on the sample set
                # TODO: check if possible to remove numpy
                b_s = torch.from_numpy(b_s)
                y = torch.from_numpy(np.subtract(np.matmul(
                    b_s.numpy(), x_s.cpu().numpy()), v_s_x0_s.cpu().numpy()))
                # alpha = torch.matmul(z, y)

                alpha = torch.from_numpy(np.matmul(z.numpy(), y.numpy()))
                x_s = torch.add(x0_s.cpu(), torch.from_numpy(np.matmul(
                    phi_s.cpu().numpy(), alpha.cpu().numpy())))

                d_euc_s_mat_t = torch.cdist(x_s, x_s).to(self.device)

                # d_euc_s_mat_np = squareform(pdist(x_s, 'euclidean'))
                # TODO: replace with dedicated function
                # d_euc_s_mat = torch.from_numpy(d_euc_s_mat_t).type(torch.double)

                # check convergence
                new_stress = self.compute_stress(d_s, d_euc_s_mat_t, w_s)
                # converged = (new_stress <= self.mds_params.a_tol) or \
                #             (1 - (new_stress / old_stress) <= self.mds_params.r_tol) or \
                #             (self.mds_params.max_iter <= iter_count)
                converged_flag = self.mds_params.max_iter <= iter_count
                old_stress = new_stress
                self.stress_list.append(old_stress)
                iter_count += 1

                # TODO: fix convergence criterion
                # converged = (new_stress <= self.mds_params.a_tol) or \
                #             (1 - (new_stress / old_stress) <= self.mds_params.r_tol) or \
                #             (self.mds_params.max_iter <= iter_count)

                if self.mds_params.compute_full_embedding_flag:
                    if self.device.type == 'cuda':
                        x = (x0.cpu() + torch.matmul(phi[:, 0:p].cpu(), alpha.cpu())).numpy()
                    else:
                        x = x0 + np.matmul(phi[:, 0:p], alpha)
                    if self.mds_params.compute_full_stress_flag:
                        intermediate_results_list.append(x)
        return x0 + torch.from_numpy(np.matmul(phi[:, 0:p].numpy(), alpha.numpy()))

    @staticmethod
    def compute_v(w_mat):
        mat_v = -w_mat + torch.diag(torch.diag(w_mat))
        mat_v -= torch.diag(torch.sum(mat_v, 1))

        return mat_v

    # @staticmethod
    # def compute_mat_b(d_mat, dx_mat, w_mat):
    #     b_mat = torch.zeros(d_mat.shape)
    #     try:
    #         tmp = -torch.matmul(w_mat, d_mat)
    #         b_mat[dx_mat != 0] = torch.div(tmp[dx_mat != 0], dx_mat[dx_mat != 0])
    #
    #     except ZeroDivisionError:
    #         print("divided by zero")
    #
    #     diag_mat_b = -torch.diag(torch.sum(b_mat, 1))
    #     b_mat += diag_mat_b
    #     return b_mat

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
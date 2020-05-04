import numpy as np
import torch
from scipy import linalg
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

from MDS.MDS import MDS
from SignalType import SignalType
import logging


class TorchMDS(MDS):

    def __init__(self, params):
        MDS.__init__(self, params)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.basicConfig(filename='TorchMDS.log', level=logging.INFO)
        self.logger = logging.getLogger('TorchMDS')
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.info("PyTorch Logger")

    # _s stands for sampled, _p stands for projected on subspace
    def algorithm(self, distances, weights, x0, phi):
        print("start algorithm")
        # extract parameters
        q_array = self.mds_params.q
        p_array = self.mds_params.p
        samples = torch.from_numpy(np.asarray(self.mds_params.samples_array)).to(self.device)
        distances.to(self.device)
        t_x0 = torch.from_numpy(x0).to(self.device)

        alpha = torch.zeros([p_array[0], self.mds_params.shape.dim]).to(self.device)
        converged = False

        intermediate_results_list = []

        for i in range(len(p_array)):
            q = q_array[i]  # q is the number of samples
            p = p_array[i]  # p is the number of frequencies\ basis vectors

            x0_s = t_x0[samples[0:q], :].type(torch.double)
            w_s = self.compute_sub(weights, samples).to(self.device)
            phi_s = phi[samples[0:q], 0:p]
            d_s = self.compute_sub(distances, samples).to(self.device).type(torch.double)
            v_s = self.compute_v(w_s).to(self.device)
            # v_s_p = torch.matmul(torch.matmul(torch.transpose(phi_s, 0, 1), v_s), phi_s)
            v_s_p = np.matmul(np.matmul(np.transpose(phi_s.numpy()), v_s.numpy()), phi_s.numpy())
            v_s_p = torch.from_numpy(v_s_p)

            v_s_p_i = torch.pinverse(v_s_p)  # todo: check res - last time gave worng res vs linalg.pinv2(v_s_p)
            z = torch.from_numpy(np.matmul(v_s_p_i.numpy(), torch.transpose(phi_s.data, 0, 1).numpy()))
            v_s_x0_s = torch.from_numpy(np.matmul(v_s.numpy(), x0_s.numpy()))
            x_s = x0_s

            self.logger.info("index = {}:\nx0_s = {}\nw_s = {}\nphi_s = {}\nd_s = {}\n"
                             "v_s = {}\nv_s_p = {}\nv_s_P_i = {}\nz = {}\nv_s_x0_s = {}\n"
                             .format(i, x0_s, w_s, phi_s, d_s, v_s, v_s_p, v_s_p_i, z, v_s_x0_s))

            dx_s_mat_n = squareform(pdist(x_s, 'euclidean'))
            dx_s_mat = torch.from_numpy(dx_s_mat_n).type(torch.double)
            old_stress = self.compute_stress(d_s, dx_s_mat, w_s)
            iter_count = 1
            self.stress_list.append(old_stress)
            while not converged:
                # if count == 50:
                #     break
                if self.mds_params.plot_flag:
                    if self.device == 'cuda':
                        self.plot_embedding(t_x0.cpu().numpy() + torch.matmul(phi[:, 0:p], alpha).cpu().numpy())
                    else:
                        self.plot_embedding(t_x0 + torch.from_numpy(np.matmul(phi[:, 0:p].numpy(), alpha.numpy())))
                b_s = self.compute_mat_b(d_s.numpy(), dx_s_mat.numpy(), w_s.numpy())
                b_s = torch.from_numpy(b_s)
                y = torch.from_numpy(np.subtract(np.matmul(b_s.numpy(), x_s.numpy()), v_s_x0_s.numpy()))
                # alpha = torch.matmul(z, y)

                alpha = torch.from_numpy(np.matmul(z.numpy(), y.numpy()))
                x_s = torch.add(x0_s, torch.from_numpy(np.matmul(phi_s.numpy(), alpha.numpy())))

                dx_s_mat_n = squareform(pdist(x_s, 'euclidean'))  # TODO: replace with dedicated function
                dx_s_mat = torch.from_numpy(dx_s_mat_n).type(torch.double)

                # check convergence
                new_stress = self.compute_stress(d_s, dx_s_mat, w_s)
                # converged = (new_stress <= self.mds_params.a_tol) or \
                #             (1 - (new_stress / old_stress) <= self.mds_params.r_tol) or \
                #             (self.mds_params.max_iter <= iter_count)
                converged = self.mds_params.max_iter <= iter_count
                old_stress = new_stress
                self.stress_list.append(old_stress)
                iter_count += 1

                # converged = (new_stress <= self.mds_params.a_tol) or \
                #             (1 - (new_stress / old_stress) <= self.mds_params.r_tol) or \
                #             (self.mds_params.max_iter <= iter_count)

                if self.mds_params.compute_full_embedding_flag:
                    if self.device == 'cuda':
                        x = (t_x0 + torch.matmul(phi[:, 0:p], alpha)).cpu().numpy()
                    else:
                        x = t_x0 + np.matmul(phi[:, 0:p], alpha)
                    if self.mds_params.compute_full_stress_flag:
                        intermediate_results_list.append(x)
        return t_x0 + torch.from_numpy(np.matmul(phi[:, 0:p].numpy(), alpha.numpy()))

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
    def compute_stress(d_mat, dx_mat, w_mat):
        print("start: compute_stress")
        tmp0 = torch.sub(torch.triu(dx_mat), torch.triu(d_mat))
        tmp = torch.pow(tmp0, 2)
        return torch.sum(torch.mul(torch.triu(w_mat), tmp))

import logging

import numpy as np
from scipy import linalg
from scipy.spatial.distance import pdist, squareform
import warnings
import torch
import datetime
import time

from MDS.MDS import MDS
from SignalType import SignalType


class BothMDS(MDS):

    def __init__(self, params, device):
        MDS.__init__(self, params)
        self.device = device
        logging.basicConfig(filename='BothMDSLog', level=logging.INFO)

        self.logger = logging.getLogger('Numpy_Torch_MDS')
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        now = datetime.datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        self.logger.info(dt_string)

    # _s stands for sampled, _p stands for projected on subspace
    def algorithm(self, distances, x0, phi, distances_t, x0_t, phi_t):
        print("start algorithm")
        # extract parameters
        q_array = self.mds_params.q
        p_array = self.mds_params.p
        samples = self.mds_params.samples_array
        samples_t = torch.tensor(self.mds_params.samples_array, device=self.device)
        weights_t = torch.from_numpy(self.mds_params.weights).to(self.device)

        # alpha = np.zeros([p_array[0], self.mds_params.shape.dim])
        intermediate_results_list = []

        for i in range(len(p_array)):
            converged = False
            print(i)
            q = q_array[i]  # q is the number of samples
            p = p_array[i]  # p is the number of frequencies\ basis vectors
            assert (q > p) or ((p == x0.shape[0]) and (q == x0.shape[0])),\
                "q should be larger than p or full size"
            if q < 2*p and ((p < x0.shape[0]) and (q < x0.shape[0])):
                warnings.warn('"It is recommended that q will be least 2p"')

            alpha = np.zeros([p, self.mds_params.shape.dim])
            x0_s = x0[samples[0:q], :]
            w_s = self.compute_sub(self.mds_params.weights, samples[0:q])
            phi_s = phi[samples[0:q], 0:p]
            d_s = self.compute_sub(distances, samples[0:q])
            v_s = self.compute_v(w_s)
            v_s_p = np.matmul(np.matmul(np.transpose(phi_s), v_s), phi_s)

            alpha_t = torch.zeros([p, self.mds_params.shape.dim])
            x0_s_t = x0_t[samples_t[0:q], :]
            w_s_t = self.compute_sub(weights_t, samples_t[0:q])
            phi_s_t = phi_t[samples_t[0:q], 0:p]
            d_s_t = self.compute_sub(distances_t, samples_t[0:q])
            v_s_t = self.compute_v_t(w_s_t)
            v_s_p_t = torch.matmul(torch.matmul(torch.transpose(phi_s_t, 0, 1), v_s_t), phi_s_t)

            if (v_s_p.shape[0] < x0.shape[0]) or \
                    (p == self.mds_params.shape.size):
                v_s_p_i = linalg.pinv2(v_s_p)
                v_s_p_i_t = torch.pinverse(v_s_p_t)
            else:
                print('"size too large for using pinv."')
                raise SystemExit
                # TODO: change such that it will be possible to work without pinv, i.e.,
                #  solve update equation via linear system solution

            z = np.matmul(v_s_p_i, np.transpose(phi_s))
            v_s_x0_s = np.matmul(v_s, x0_s)
            x_s = x0_s

            z_t = torch.matmul(v_s_p_i_t, torch.transpose(phi_s_t, 0, 1))
            v_s_x0_s_t = torch.matmul(v_s_t, x0_s_t)
            x_s_t = x0_s_t

            # self.logger.info("\nindex = {}\nx0_s = {}\nw_s = {}\nphi_s = {}\nd_s = {}"
            #                  "\nv_s = {}\nv_s_p = {}\nv_s_P_i = {}\nz = {}\nv_s_x0_s = {}\n".format(
            #                         i, x0_s - x0_s_t.numpy(), w_s - w_s_t.numpy(), phi_s - phi_s_t.numpy(),
            #                         d_s - d_s_t.numpy(), v_s - v_s_t.numpy(), v_s_p - v_s_p_t.numpy(),
            #                         v_s_p_i - v_s_p_i_t.numpy(), z - z_t.numpy(), v_s_x0_s - v_s_x0_s_t.numpy()))

            # self.logger.info("torch: index = {}:\nx0_s = {}\nw_s = {}\nphi_s = {}\nd_s = {}\n"
            #                 "v_s = {}\nv_s_p = {}\nv_s_P_i = {}\nz = {}\nv_s_x0_s = {}\n"
            #                 .format(i, x0_s_t, w_s_t, phi_s_t, d_s_t, v_s_t, v_s_p_t, v_s_p_i_t, z_t, v_s_x0_s_t))

            dx_s_mat = squareform(pdist(x_s, 'euclidean'))
            old_stress = self.compute_stress(d_s, dx_s_mat, w_s)
            iter_count = 1
            self.stress_list.append((1/(q*q))*old_stress)

            dx_s_mat_t = torch.cdist(x_s_t, x_s_t)
            old_stress_t = self.compute_stress_t(d_s_t, dx_s_mat_t, w_s_t)
            self.stress_list_t.append((1/(q*q))*old_stress_t)

            self.logger.info("\nold_stress_np = {}, old_stress_torch = {}".format(old_stress, old_stress_t))

            while not converged:
                # --------------------------  plotting --------------------------------
                if self.mds_params.plot_flag and \
                        (iter_count % self.mds_params.display_every) == 0:
                    if self.mds_params.plot_flag:
                        self.mds_params.shape.plot_embedding(x0 + np.matmul(phi[:, 0:p], alpha))
                        self.mds_params.shape.plot_embedding(x0_t + torch.matmul(phi_t[:, 0:p], alpha_t))
                        #TODO: change plot_embedding to work from shape
                    print(f'iter : {iter_count}, stress : np - {old_stress}, torch - {old_stress_t}')
                # --------------------------------------------------------------------

                # b_s = self.compute_mat_b(d_s, dx_s_mat, w_s)
                # b_s_t = torch.from_numpy(b_s).to(self.device)

                b_s, b_s_t = self.compute_mat_b(d_s, dx_s_mat, w_s)

                y = np.subtract(np.matmul(b_s, x_s), v_s_x0_s)

                alpha = np.matmul(z, y)
                x_s = np.add(x0_s, np.matmul(phi_s, alpha))

                # b_s_t = self.compute_mat_b_t(d_s_t, dx_s_mat_t, w_s_t)
                y_t = torch.sub(torch.matmul(b_s_t, x_s_t), v_s_x0_s_t)

                alpha_t = torch.matmul(z_t, y_t)
                x_s_t = torch.add(x0_s_t, torch.matmul(phi_s_t, alpha_t))

                dx_s_mat = squareform(pdist(x_s, 'euclidean'))  # TODO: replace with dedicated function

                # check convergence
                new_stress = self.compute_stress(d_s, dx_s_mat, w_s)

                # dx_s_mat_np = squareform(pdist(x_s.numpy(), 'euclidean'))  # TODO: replace with dedicated function
                dx_s_mat_t = torch.cdist(x_s_t, x_s_t)

                # check convergence
                new_stress_t = self.compute_stress_t(d_s_t, dx_s_mat_t, w_s_t)

                converged = (new_stress <= self.mds_params.a_tol) or \
                            (1 - (new_stress / old_stress) <= self.mds_params.r_tol) or \
                            (self.mds_params.max_iter <= iter_count)
                # converged = self.mds_params.max_iter <= iter_count
                old_stress = new_stress
                self.stress_list.append((1/(q*q))*old_stress)

                old_stress_t = new_stress_t
                self.stress_list_t.append((1/(q*q))*old_stress_t)

                iter_count += 1

                if self.mds_params.compute_full_embedding_flag:
                    x = x0 + np.matmul(phi[:, 0:p], alpha)
                    if self.mds_params.compute_full_stress_flag:
                        intermediate_results_list.append(x)


            #if i<(len(p_array)-1):
                # extra_zeros = np.zeros([p_array[i + 1]-p, self.mds_params.shape.dim])
                # alpha = np.concatenate((alpha, extra_zeros), 0)
            x0 = x0 + np.matmul(phi[:, 0:p], alpha)

            x0_t = x0_t + torch.matmul(phi_t[:, 0:p], alpha_t)

            # self.logger.info("\nend of while loop:\nx0 = {}\nx0_t = {}\nx0 - x0_t = {}"
            #                  .format(x0, x0_t, x0 - x0_t.numpy()))


        print(f'final stress : np - {old_stress}, torch - {old_stress_t}')

        return x0 + np.matmul(phi[:, 0:p], alpha) , x0_t + torch.matmul(phi_t[:, 0:p], alpha_t)

    @staticmethod
    def compute_v(w_mat):
        mat_v = -w_mat + np.diag(np.diag(w_mat))
        mat_v -= np.diag(np.sum(mat_v, 1))

        return mat_v

    @staticmethod
    def compute_stress(d_mat, dx_mat, w_mat):
        tmp0 = np.subtract(np.triu(dx_mat), np.triu(d_mat))
        tmp = np.power(tmp0, 2)
        return np.sum(np.multiply(np.triu(w_mat), tmp))

    @staticmethod
    def compute_v_t(w_mat):
        mat_v = -w_mat + torch.diag(torch.diag(w_mat))
        mat_v -= torch.diag(torch.sum(mat_v, 1))

        return mat_v

    @staticmethod
    def compute_stress_t(d_mat, dx_mat, w_mat):
        tmp0 = torch.sub(torch.triu(dx_mat), torch.triu(d_mat))
        tmp = torch.pow(tmp0, 2)
        return torch.sum(torch.mul(torch.triu(w_mat), tmp)) # change of 0.0005 from numpy

    def compute_mat_b(self, d_mat, dx_mat, w_mat):
        d_mat_t = torch.from_numpy(d_mat).to(self.device)
        dx_mat_t = torch.from_numpy(dx_mat).to(self.device)
        w_mat_t = torch.from_numpy(w_mat).to(self.device)

        b_mat = np.zeros(d_mat.shape)
        b_mat_t = torch.zeros(d_mat.shape, dtype=torch.float64, device=self.device)

        try:
            tmp = -np.multiply(w_mat, d_mat)
            b_mat[dx_mat != 0] = np.divide(tmp[dx_mat != 0], dx_mat[dx_mat != 0])

            tmp_t = -torch.mul(w_mat_t, d_mat_t)
            b_mat_t[dx_mat_t != 0] = torch.true_divide(tmp_t[dx_mat_t != 0], dx_mat_t[dx_mat_t != 0])

        except ZeroDivisionError:
            print("divided by zero")

        diag_mat_b = -np.diag(np.sum(b_mat, 1))
        b_mat += diag_mat_b

        diag_mat_b = -torch.diag(torch.sum(b_mat_t, 1))
        b_mat_t += diag_mat_b

        return b_mat, b_mat_t




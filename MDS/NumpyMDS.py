import logging

import numpy as np
from scipy import linalg
from scipy.spatial.distance import pdist, squareform
import warnings

from MDS.MDS import MDS
from SignalType import SignalType


class NumpyMDS(MDS):

    def __init__(self, params):
        MDS.__init__(self, params)
        logging.basicConfig(filename='NumpyMDS.log', level=logging.INFO)

        self.logger = logging.getLogger('NumpyMDS')
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.info("Numpy Logger")

    # _s stands for sampled, _p stands for projected on subspace
    def algorithm(self, distances, x0, phi):
        print("start algorithm")
        # extract parameters
        q_array = self.mds_params.q
        p_array = self.mds_params.p
        samples = self.mds_params.samples_array

        alpha = np.zeros([p_array[0], self.mds_params.shape.dim])
        converged = False
        intermediate_results_list = []

        for i in range(len(p_array)):
            q = q_array[i]  # q is the number of samples
            p = p_array[i]  # p is the number of frequencies\ basis vectors
            assert (q > p) or ((p == x0.shape[0]) and (q == x0.shape[0])),\
                "q should be larger than p or full size"
            if q < 2*p and ((p < x0.shape[0]) and (q < x0.shape[0])):
                warnings.warn('"It is recommended that q will be least 2p"')

            x0_s = x0[samples[0:q], :]
            w_s = self.compute_sub(self.mds_params.weights, samples)
            phi_s = phi[samples[0:q], 0:p]
            d_s = self.compute_sub(distances, samples)
            v_s = self.compute_v(w_s)
            v_s_p = np.matmul(np.matmul(np.transpose(phi_s), v_s), phi_s)
            if (v_s_p.shape[0] < x0.shape[0]) or \
                    (p == self.mds_params.shape.size):
                v_s_p_i = linalg.pinv2(v_s_p)
            else:
                print('"size too large for using pinv."')
                raise SystemExit
                # TODO: change such that it will be possible to work without pinv, i.e.,
                #  solve update equation via linear system solution

            z = np.matmul(v_s_p_i, np.transpose(phi_s))
            v_s_x0_s = np.matmul(v_s, x0_s)
            x_s = x0_s

            self.logger.info("index = {}:\nx0_s = {}\nw_s = {}\nphi_s = {}\nd_s = {}\n"
                             "v_s = {}\nv_s_p = {}\nv_s_P_i = {}\nz = {}\nv_s_x0_s = {}\n"
                             .format(i, x0_s, w_s, phi_s, d_s, v_s, v_s_p, v_s_p_i, z, v_s_x0_s))

            dx_s_mat = squareform(pdist(x_s, 'euclidean'))
            old_stress = self.compute_stress(d_s, dx_s_mat, w_s)
            iter_count = 1
            self.stress_list.append(old_stress)
            while not converged:
                # --------------------------  plotting --------------------------------
                if self.mds_params.plot_flag and \
                        (iter_count % self.mds_params.display_every) == 0:
                    if self.mds_params.plot_flag:
                        self.plot_embedding(x0 + np.matmul(phi[:, 0:p], alpha))
                        #TODO: change plot_embedding to work from shape
                    print(f'iter : {iter_count}, stress : {old_stress}')
                # --------------------------------------------------------------------

                b_s = self.compute_mat_b(d_s, dx_s_mat, w_s)
                y = np.subtract(np.matmul(b_s, x_s), v_s_x0_s)

                alpha = np.matmul(z, y)
                x_s = np.add(x0_s, np.matmul(phi_s, alpha))
                dx_s_mat = squareform(pdist(x_s, 'euclidean'))  # TODO: replace with dedicated function

                # check convergence
                new_stress = self.compute_stress(d_s, dx_s_mat, w_s)
                converged = (new_stress <= self.mds_params.a_tol) or \
                            (1 - (new_stress / old_stress) <= self.mds_params.r_tol) or \
                            (self.mds_params.max_iter <= iter_count)
                # converged = self.mds_params.max_iter <= iter_count
                old_stress = new_stress
                self.stress_list.append(old_stress)
                iter_count += 1

                if self.mds_params.compute_full_embedding_flag:
                    x = x0 + np.matmul(phi[:, 0:p], alpha)
                    if self.mds_params.compute_full_stress_flag:
                        intermediate_results_list.append(x)
        print(f'final stress : {old_stress}')
        return x0 + np.matmul(phi[:, 0:p], alpha)

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



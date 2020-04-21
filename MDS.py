import numpy as np
from SignalType import SignalType
from scipy import linalg
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class MDS:

    def __init__(self, params):
        self.mds_params = params
        self.stress_list = []

    # _s stands for sampled, _p stands for projected on subspace
    def algorithm(self, distances, weights, x0, phi):
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

            x0_s = x0[samples[0:q], :]
            w_s = self.compute_sub(weights, samples)
            phi_s = phi[samples[0:q], 0:p]
            d_s = self.compute_sub(distances, samples)
            v_s = self.compute_v(w_s)
            v_s_p = np.matmul(np.matmul(np.transpose(phi_s), v_s), phi_s)
            v_s_p_i = linalg.pinv2(v_s_p)
            z = np.matmul(v_s_p_i, np.transpose(phi_s))

            v_s_x0_s = np.matmul(v_s, x0_s)
            x_s = x0_s

            dx_s_mat = squareform(pdist(x_s, 'euclidean'))
            old_stress = self.compute_stress(d_s, dx_s_mat, w_s)
            iter_count = 1
            self.stress_list.append(old_stress)

            while not converged:

                if self.mds_params.plot_flag:
                    self.plot_embedding(x0 + np.matmul(phi[:, 0:p], alpha))
                    # TODO: plot full embedding

                b_s = self.compute_mat_b(d_s, dx_s_mat, w_s)
                y = np.subtract(np.matmul(b_s, x_s), v_s_x0_s)

                alpha = np.matmul(z, y)
                x_s = np.add(x0_s, np.matmul(phi_s, alpha))
                dx_s_mat = squareform(pdist(x_s, 'euclidean'))  # TODO: replace with dedicated function

                # check convergence
                new_stress = self.compute_stress(d_s, dx_s_mat, w_s)
                converged = self.mds_params.max_iter <= iter_count
                old_stress = new_stress
                self.stress_list.append(old_stress)
                iter_count += 1

                if self.mds_params.compute_full_embedding_flag:
                    x = x0 + np.matmul(phi[:, 0:p], alpha)
                    if self.mds_params.compute_full_stress_flag:
                        intermediate_results_list.append(x)

        return x0 + np.matmul(phi[:, 0:p], alpha)

    @staticmethod
    def compute_v(w_mat):
        mat_v = -w_mat + np.diag(np.diag(w_mat))
        mat_v -= np.diag(np.sum(mat_v, 1))

        return mat_v

    @staticmethod
    def compute_mat_b(d_mat, dx_mat, w_mat):
        try:
            tmp = -np.multiply(w_mat, d_mat)
            b_mat = np.zeros(d_mat.shape)
            b_mat[dx_mat != 0] = np.divide(tmp[dx_mat != 0], dx_mat[dx_mat != 0])

        except ZeroDivisionError:
            print("divided by zero")

        diag_mat_b = -np.diag(np.sum(b_mat, 1))
        b_mat += diag_mat_b
        return b_mat

    @staticmethod
    def compute_stress(d_mat, dx_mat, w_mat):
        print("start: compute_stress")
        tmp0 = np.subtract(np.triu(dx_mat), np.triu(d_mat))
        tmp = np.power(tmp0, 2)
        return np.sum(np.multiply(np.triu(w_mat), tmp))

    def plot_embedding(self, new_x):
        if self.mds_params.shape.signal_type == SignalType.MESH:
            x = new_x[:, 0]
            y = new_x[:, 1]
            z = new_x[:, 2]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, y, z)
            fig.show()

        elif self.mds_params.signal_type == SignalType.POINT_CLOUD:
            pass

    @staticmethod
    def compute_sub(mat, vec):
        tmp_mat = mat[vec, :]
        return tmp_mat[:, vec]


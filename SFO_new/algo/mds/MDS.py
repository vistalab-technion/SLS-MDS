import numpy as np
import torch
import trimesh
from algo.mds.SignalType import SignalType
from scipy.spatial.distance import pdist, squareform


class MDS:
    # mds_params = MdsParams.MdsParams()

    def __init__(self, params):
        self.mds_params = params

    # _s stands for sampled, _p stands for projected on subspace
    def algorithm(self, distances, weights, x0, phi):
        # extract parameters
        q_array = self.mds_params.q
        p_array = self.mds_params.p
        samples = self.mds_params.samples_array
        dim = np.size(x0, 1)
        alpha = torch.Tensor().cuda()
        converged = False
        stress_values = []
        intermediate_results_list = []

        for i in range(len(p_array)):
            q = q_array[i]  # q is the number of samples
            p = p_array[i]  # p is the number of frequencies\ basis vectors

            tmp = torch.zeros((p - len(alpha))).cuda()
            alpha = torch.cat([alpha, tmp])  # from_numpy(np.concatenate(alpha, tmp))
            x0_s = x0[samples[0:q], :]
            w_s = self.compute_s(weights, samples, q)
            phi_s = phi[samples[0:q], :]
            d_s = self.compute_s(distances, samples, q)# distances[samples[0:q], samples[0:q]]
            v_s = self.compute_v(w_s)
            v_s_p = np.matmul(np.matmul(np.transpose(phi_s), v_s), phi_s)
            v_s_p_i = np.linalg.pinv(v_s_p)
            z = np.matmul(v_s_p_i, np.transpose(phi_s))
            v_s_x0_s = np.matmul(v_s, x0_s)
            x_s = x0_s
            dx_s_mat = squareform(pdist(x_s, 'euclidean'))  # TODO: replace with dedicated function
            old_stress = self.compute_stress(d_s, dx_s_mat, w_s)
            iter_count = 1

            while not converged:
                b_s = self.compute_mat_b(d_s, dx_s_mat, w_s)
                y = np.matmul(b_s, x_s) - v_s_x0_s
                alpha = np.matmul(z, y)
                x_s = x0_s + np.matmul(phi_s, alpha)
                print(x_s)
                dx_s_mat = squareform(pdist(x_s, 'euclidean'))  # TODO: replace with dedicated function
                # check convergence
                new_stress = self.compute_stress(d_s, dx_s_mat, w_s)
                converged = (new_stress <= self.mds_params.a_tol) or \
                            (1 - (old_stress / new_stress) <= self.mds_params.r_tol) or \
                            (self.mds_params.max_iter <= iter_count)

                old_stress = new_stress
                iter_count += 1

                if self.mds_params.compute_full_embedding_flag:
                    x = x0 + np.matmul(phi, alpha)
                    if self.mds_params.compute_full_stress:
                        intermediate_results_list.append(x)

                if self.mds_params.plot_flag:
                    pass
                    # TODO: plot full embedding

        if self.mds_params.compute_full_stress_flag:
            for intermediate_x in len(intermediate_results_list):
                full_stress = self.compute_stress(dim, intermediate_x, weights)
                stress_values.append(full_stress)

        return x0 + np.matmul(phi, alpha)

    @staticmethod
    def compute_v(w_mat):
        mat_v = -w_mat
        tmp = w_mat.sum(axis=1)
        mat_v += np.diag(tmp)
        return mat_v

    @staticmethod
    def compute_mat_b(d_mat, dx_mat, w_mat):
        try:
            b_mat = np.zeros(d_mat.shape)
            idx_non_zeros = np.nonzero(dx_mat)
            b_mat[idx_non_zeros] = np.divide(  # dx_mat not supposed to have a zero element
                np.multiply(w_mat[idx_non_zeros], d_mat[idx_non_zeros]), dx_mat[idx_non_zeros])
        except ZeroDivisionError:
            print("divided by zero")

        b_mat += np.diag(-np.sum(b_mat, 1))
        return b_mat

    @staticmethod
    def compute_stress(d_mat, dx_mat, w_mat):
        # print(np.triu(dx_mat).shape, np.triu(d_mat).shape)
        tmp0 = np.subtract(np.triu(dx_mat), np.triu(d_mat))
        tmp = np.power(tmp0, 2)
        return np.sum(np.multiply(np.triu(w_mat), tmp))

    def plot_embedding(self, x):
        if self.mds_params.signal_type == SignalType.MESH:
            x_mesh = trimesh.Trimesh(self.mds_params.shape.mesh)
            x_mesh.vertices = x
            x_mesh.show()
        elif self.mds_params.signal_type == SignalType.POINT_CLOUD:
            pass

    @staticmethod
    def compute_s(mat, vec, size):
        mat_s = np.zeros((size, size))
        for index, var in enumerate(vec[0:size]):
            mat_s[index] = mat[var, vec[0:size]]
        return mat_s

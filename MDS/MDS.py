import numpy as np
from scipy.spatial.distance import pdist, squareform
import Mds_params
import torch

class MDS:
    num_of_iterations = 5
    converged = 0
    mds_params = Mds_params.MdsParams()

    def __init__(self, params):
        self.mds_params = params  # Todo: maybe need a copy constructor




    # _s for sampled, _p for projected on subspace
    def algorithm(self, mat_d, mat_w, mesh_x0, phi):

        #extract parameters
        q_array = self.mds_params.q
        p_array = self.mds_params.p
        samples = self.mds_params.samples_array
        d = 3 #Todo: get dimension
        alpha = [[]]

        for i in len(p_array):
            q = q_array[i]
            p = p_array[i]

            mesh_x = mesh_x0
            alpha = torch.from_numpy(np.concatenate(alpha, np.zeros(p-len(alpha))))

            x0_s = mesh_x0.vertices[samples[1:q], :]
            w_s = mat_w[samples[1:q], samples[1:q]]
            phi_s = phi[samples[1:q], :]
            d_s = mat_d[samples[1:q], samples[1:q]]
            v_s = self.compute_v(w_s)
            v_s_p = np.matmul(np.matmul(np.transpose(phi_s), v_s), phi_s)
            v_s_p_i = np.linalg.pinv(v_s_p)
            z = np.matmul(v_s_p_i, np.transpose(phi_s))
            v_s_x0_s = np.matmul(v_s, x0_s)
            x_s = x0_s
            while not self.converged:
                b_s = self.compute_mat_b(x_s)
                y = np.matmul(b_s, x_s) - v_s_x0_s
                alpha = np.matmul(z, y)
                x_s = x0_s + np.matmul(phi_s, alpha)
            mesh_x.vertices = mesh_x0.vertices + np.matmul(phi, alpha)
        return mesh_x

    @staticmethod
    def compute_v(mat_w_mds):
        mat_v = -mat_w_mds
        mat_v += np.diag(np.sum(mat_w_mds, 1))
        return mat_v

    def compute_mat_b(self,d_s, x_s, mat_w_mds):
        mat = 0
        squareform(pdist(x_s))
        return mat

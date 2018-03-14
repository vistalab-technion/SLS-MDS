import numpy as np
import torch
from Calculations import Calculations
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from SignalType import SignalType
from torch.autograd import Variable
from scipy import linalg


class MDS:
    # mds_params = MdsParams.MdsParams()

    def __init__(self, params):
        self.mds_params = params
        self.calc = Calculations()
        self.stress_list = []

    # _s stands for sampled, _p stands for projected on subspace
    def algorithm(self, distances, weights, x0, phi):
        print("start algorithm")
        # extract parameters
        q_array = self.mds_params.q
        p_array = self.mds_params.p
        samples = self.mds_params.samples_array

        dim = np.size(x0, 1)
        # alpha = torch.Tensor().cuda()
        converged = False
        stress_values = []
        intermediate_results_list = []

        for i in range(len(p_array)):
            q = q_array[i]  # q is the number of samples
            p = p_array[i]  # p is the number of frequencies\ basis vectors

            # tmp = torch.zeros((p - len(alpha))).cuda()
            # alpha = torch.cat([alpha, tmp])  # from_numpy(np.concatenate(alpha, tmp))
            x0_s = x0[samples[0:q], :]
            w_s = self.compute_sub(weights, samples)
            phi_s = phi[samples[0:q], :]
            d_s = self.compute_sub(distances, samples)  # distances[samples[0:q], samples[0:q]]
            v_s = self.compute_v(w_s)
            # v_s_p = torch.matmul(torch.matmul(torch.transpose(phi_s, 0, 1), v_s), phi_s)
            v_s_p = np.matmul(np.matmul(np.transpose(phi_s), v_s), phi_s)
            # v_s_p_i = self.calc.compute_mat_pinv(v_s_p, p)
            v_s_p_i = linalg.pinv(v_s_p)
            # v_s_p_i = torch.FloatTensor(v_s_p_i_np).cuda()
            # z = torch.matmul(torch.matmul(v_s_p_i, torch.transpose(phi_s, 0, 1).data), torch.transpose(mat_s, 0, 1))
            # v_s_x0_s = torch.matmul(torch.matmul(v_s, mat_s), x0_s)
            # z = torch.matmul(v_s_p_i, torch.transpose(phi_s.data, 0, 1))
            z = np.matmul(v_s_p_i, np.transpose(phi_s))
            # v_s_x0_s = torch.matmul(v_s, x0_s)
            v_s_x0_s = np.matmul(v_s, x0_s)
            x_s = x0_s
            # tmp_x_s = x_s.data.cpu().numpy()
            # dx_s_mat_t = torch.from_numpy(dx_s_mat).cuda()
            dx_s_mat = squareform(pdist(x_s, 'euclidean'))
            old_stress = self.compute_stress(d_s, dx_s_mat, w_s)
            iter_count = 1
            self.stress_list.append(old_stress)

            while not converged:
                b_s = self.compute_mat_b(d_s, dx_s_mat, w_s)
                # y = torch.matmul(torch.matmul(b_s, mat_s), x_s.data).sub(v_s_x0_s.data)
                # y = torch.matmul(b_s, x_s.data).sub(v_s_x0_s.data)
                y = np.subtract(np.matmul(b_s, x_s), v_s_x0_s)

                # alpha = torch.matmul(z, y)
                alpha = np.matmul(z, y)
                # x_s.data = torch.add(x0_s.data, torch.matmul(phi_s.data, alpha))
                x_s = np.add(x0_s, np.matmul(phi_s, alpha))
                # tmp_x_s = x_s.data.cpu().numpy()
                # dx_s_mat = torch.FloatTensor(squareform(pdist(tmp_x_s, 'euclidean'))).cuda()
                dx_s_mat = squareform(pdist(x_s, 'euclidean'))  # TODO: replace with dedicated function

                # check convergence
                new_stress = self.compute_stress(d_s, dx_s_mat, w_s)
                # converged = (new_stress <= self.mds_params.a_tol) or \
                #             (1 - (new_stress / old_stress) <= self.mds_params.r_tol) or \
                #             (self.mds_params.max_iter <= iter_count)
                converged = self.mds_params.max_iter <= iter_count
                old_stress = new_stress
                self.stress_list.append(old_stress)
                iter_count += 1

                plt.plot(self.stress_list)
                plt.show()

                # converged = (new_stress <= self.mds_params.a_tol) or \
                #             (1 - (new_stress / old_stress) <= self.mds_params.r_tol) or \
                #             (self.mds_params.max_iter <= iter_count)

                if self.mds_params.compute_full_embedding_flag:
                    # x = (x0.data + torch.matmul(phi.data, alpha)).cpu().numpy()
                    x = x0 + np.matmul(phi, alpha)
                    if self.mds_params.compute_full_stress_flag:
                        intermediate_results_list.append(x)

                if self.mds_params.plot_flag:
                    # self.plot_embedding(x0.data + torch.matmul(phi.data, alpha))
                    self.plot_embedding(x0 + np.matmul(phi, alpha))
                    # TODO: plot full embedding

        if self.mds_params.compute_full_stress_flag:
            for intermediate_x in intermediate_results_list:
                # dx_mat = torch.FloatTensor(squareform(pdist(intermediate_x, 'euclidean'))).cuda()  # TODO: replace with dedicated function
                dx_s_mat = squareform(pdist(intermediate_x, 'euclidean'))
                # check convergence
                full_stress = self.compute_stress(distances, dx_mat, weights)
                stress_values.append(full_stress)
        plt.plot(stress_values)
        plt.show()

        # return x0.data + torch.matmul(phi.data, alpha)
        return x0 + np.matmul(phi, alpha)

    @staticmethod
    def compute_v(w_mat):
        # mat_v = -w_mat.data + torch.diag(torch.diag(w_mat.data))
        # mat_v += torch.diag(mat_v.sum(dim=1))
        mat_v = -w_mat + np.diag(np.diag(w_mat))
        mat_v -= np.diag(np.sum(mat_v, 1))

        # tmp2 = torch.diag(tmp) + torch.diag(w_mat.data)
        # mat_v.data += tmp2
        # return Variable(mat_v).cuda()
        return mat_v

    @staticmethod
    def compute_mat_b(d_mat, dx_mat, w_mat):
        try:

            # tmp = -torch.mul(w_mat.data, d_mat.data)
            tmp = np.matmul(w_mat, d_mat)
        #     dx_mat[dx_mat == 0] = float('inf')
        #     b_mat = torch.zeros(d_mat.data.shape).cuda()
        #     b_mat[dx_mat != 0] = torch.div(tmp[dx_mat != 0], dx_mat[dx_mat != 0])
        #
        # except ZeroDivisionError:
        #     print("divided by zero")
        # #b_mat = torch.add(b_mat, torch.mul(torch.eye(b_mat.shape), torch.diag(-torch.sum(b_mat, 1))))
        # b_mat = b_mat - torch.diag(torch.sum(b_mat, 1))
        # return b_mat

            # b_mat = torch.zeros(d_mat.data.shape).cuda()
            # b_mat[dx_mat != 0] = torch.div(tmp[dx_mat != 0], dx_mat[dx_mat != 0])
            b_mat = np.zeros(d_mat.shape)
            b_mat[dx_mat != 0] = np.divide(tmp[dx_mat != 0], dx_mat[dx_mat != 0])

        except ZeroDivisionError:
            print("divided by zero")
        #  # b_mat = torch.add(b_mat, torch.mul(torch.eye(b_mat.shape), torch.diag(-torch.sum(b_mat, 1))))

        # diag_b_mat = - torch.diag(torch.sum(b_mat.type(torch.DoubleTensor), 0))
        # b_mat = b_mat.type(torch.DoubleTensor) + diag_b_mat
        # return b_mat.type(torch.FloatTensor).cuda()
        diag_mat_b = -np.diag(np.sum(b_mat, 1))
        b_mat += diag_mat_b
        return b_mat

    # def compute_mat_s(self, samples, row_size, column_size):
    #     mat_s = torch.zeros(row_size, column_size)
    #     mat_s[:, samples.cpu()] == 1
    #     return mat_s

    @staticmethod
    def compute_stress(d_mat, dx_mat, w_mat):
        print("start: compute_stress")
        # tmp0 = torch.triu(dx_mat).sub(torch.triu(d_mat.data))
        # tmp = torch.pow(tmp0, 2)
        tmp0 = np.subtract(np.triu(dx_mat), np.triu(d_mat))
        tmp = np.power(tmp0, 2)
        # return torch.sum(torch.mul(torch.triu(w_mat.data), tmp))
        return np.sum(np.matmul(np.triu(w_mat), tmp))

    def plot_embedding(self, x):
        if self.mds_params.shape.signal_type == SignalType.MESH:
            x_mesh = self.mds_params.shape.mesh
            # x_mesh.vertices = x.cpu().numpy()
            x_mesh.vertices = x
            x_mesh.show()
        elif self.mds_params.signal_type == SignalType.POINT_CLOUD:
            pass

    @staticmethod
    def compute_sub(mat, vec):
        tmp_mat = mat[vec, :]
        return tmp_mat[:, vec]

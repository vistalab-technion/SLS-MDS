import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from SignalType import SignalType


class MDS:

    def __init__(self, params):
        self.mds_params = params
        self.stress_list = []
        self.converged = False

    @staticmethod
    def compute_sub(mat, vec):
        tmp_mat = mat[vec, :]
        return tmp_mat[:, vec]

    def plot_embedding(self, new_x):
        if self.mds_params.shape.signal_type == SignalType.MESH:
            x = new_x[:, 0]
            y = new_x[:, 1]
            z = new_x[:, 2]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, y, z)
            fig.show()
            # TODO: we should change to mesh plotting

        elif self.mds_params.signal_type == SignalType.POINT_CLOUD:
            x = new_x[:, 0]
            y = new_x[:, 1]
            z = new_x[:, 2]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, y, z)
            fig.show()

    @staticmethod
    def compute_mat_b(d_mat, dx_mat, w_mat):
        b_mat = np.zeros(d_mat.shape)
        try:
            tmp = -np.multiply(w_mat, d_mat)
            b_mat[dx_mat != 0] = np.divide(tmp[dx_mat != 0], dx_mat[dx_mat != 0])

        except ZeroDivisionError:
            print("divided by zero")

        diag_mat_b = -np.diag(np.sum(b_mat, 1))
        b_mat += diag_mat_b
        return b_mat

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

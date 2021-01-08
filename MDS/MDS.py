import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from SignalType import SignalType


class MDS:

    def __init__(self, params):
        self.mds_params = params
        self.stress_list = []
        self.stress_list_t = []  # only used in "Both" type
        self.converged = False

    @staticmethod
    def compute_sub(mat, vec):
        tmp_mat = mat[vec, :]
        return tmp_mat[:, vec]

    # def plot_embedding(self, new_x):
    #     if self.mds_params.shape.signal_type == SignalType.MESH:
    #         x = new_x[:, 0]
    #         y = new_x[:, 1]
    #         z = new_x[:, 2]
    #         fig = plt.figure()
    #         ax = fig.add_subplot(111, projection='3d')
    #         ax.scatter(x, y, z)
    #         self.set_axes_equal(ax)
    #         fig.show()
    #
    #         # TODO: we should change to mesh plotting
    #
    #     elif self.mds_params.signal_type == SignalType.POINT_CLOUD:
    #         x = new_x[:, 0]
    #         y = new_x[:, 1]
    #         z = new_x[:, 2]
    #         fig = plt.figure()
    #         ax = fig.add_subplot(111, projection='3d')
    #         ax.scatter(x, y, z)
    #         fig.show()
    #
    # @staticmethod
    # def set_axes_radius(ax, origin, radius):
    #     ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    #     ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    #     ax.set_zlim3d([origin[2] - radius, origin[2] + radius])
    #
    # def set_axes_equal(self, ax):
    #     '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    #     cubes as cubes, etc..  This is one possible solution to Matplotlib's
    #     ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    #
    #     Input
    #       ax: a matplotlib axis, e.g., as output from plt.gca().
    #     '''
    #
    #     limits = np.array([
    #         ax.get_xlim3d(),
    #         ax.get_ylim3d(),
    #         ax.get_zlim3d(),
    #     ])
    #
    #     origin = np.mean(limits, axis=1)
    #     radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    #     self.set_axes_radius(ax, origin, radius)
import scipy.sparse.linalg as sp
import trimesh
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import sparse

from SignalType import SignalType
random.seed(10)


class Shape:

    def __init__(self, filename=None, vertices=None, faces=None):
        self.signal_type = SignalType.MESH

        if filename is not None:
            self.mesh = trimesh.load_mesh(filename, process=False)

        elif vertices is not None and faces is not None:
            self.mesh = trimesh.Trimesh(vertices, faces)
        else:
            print("Shape should get 'filename' or 'vertices and faces'")
            raise SystemError()

        self.size = len(self.mesh.vertices)
        self.dim = len(self.mesh.vertices[0])
        self.mass_mat, self.stiffness_mat = self.compute_laplacian()
        self.eigs = np.array([])
        self.evecs = np.array([])
        self.weights = np.ones((self.size, self.size))

    def set_signal_type(self, signal_type):
        self.signal_type = signal_type

    def compute_laplacian(self):
        nv = np.shape(self.mesh.vertices)[0]
        nf = np.shape(self.mesh.faces)[0]
        V = self.mesh.vertices
        F = self.mesh.faces

        l1 = np.sqrt(np.sum(np.square(V[F[:, 1], :] - V[F[:, 2], :]), 1))
        l2 = np.sqrt(np.sum(np.square(V[F[:, 2], :] - V[F[:, 0], :]), 1))
        l3 = np.sqrt(np.sum(np.square(V[F[:, 0], :] - V[F[:, 1], :]), 1))

        i1 = F[:, 0]
        i2 = F[:, 1]
        i3 = F[:, 2]

        s = (l1 + l2 + l3) * 0.5

        # triangle - wise area
        fA = np.sqrt(np.multiply(np.multiply(np.multiply(s, s - l1), s - l2), s - l3))

        # cotangent weight
        cot12 = (l1 ** 2 + l2 ** 2 - l3 ** 2) / fA / 4.0
        cot23 = (l2 ** 2 + l3 ** 2 - l1 ** 2) / fA / 4.0
        cot31 = (l1 ** 2 + l3 ** 2 - l2 ** 2) / fA / 4.0

        diag1 = -cot12 - cot31
        diag2 = -cot12 - cot23
        diag3 = -cot31 - cot23

        i = np.concatenate((i1, i2, i2, i3, i3, i1, i1, i2, i3), 0)
        j = np.concatenate((i2, i1, i3, i2, i1, i3, i1, i2, i3), 0)

        # values corresponding to pairs form(i, j)

        v = np.concatenate((cot12, cot12, cot23, cot23, cot31, cot31, diag1, diag2, diag3), 0)

        stiffness = sparse.coo_matrix((v, (i, j)), (nv, nv))

        tri2ver = sparse.coo_matrix((np.ones(nf), (F[:, 0], np.arange(0, nf))), (nv, nf))
        tri2ver += sparse.coo_matrix((np.ones(nf), (F[:, 1], np.arange(0, nf))), (nv, nf))
        tri2ver += sparse.coo_matrix((np.ones(nf), (F[:, 2], np.arange(0, nf))), (nv, nf))

        tri2ver[tri2ver.nonzero()[0], tri2ver.nonzero()[1]] = 1

        mass = sparse.spdiags(tri2ver * fA / 3, 0, nv, nv)

        return mass.tocsc(), stiffness.tocsc()

    def sample_mesh_fps(self, k, d_mat=None):
        """
        The function samples points from the mesh according to farthest point sampling
         strategy

        :param k: number of sample points.
        :param d_mat: geodesic distance mat between all points on the mesh.
        :return: set_c: set of sampled points
                 d_mat:  geodesic distance matrix between points in set_c
        """

        print("sample_mesh\n")

        if k == len(self.mesh.vertices):
            set_c = range(0, len(self.mesh.vertices))
            # TODO: compute geodesic distances
            if d_mat is None:
                raise NotImplementedError('Compute geodesic in not implemented yet')
                # todo: compute dist func
                d = self.compute_geodesics()
        else:
            compute_d_mat_flag = False
            if d_mat is None:
                compute_d_mat_flag = True

            x_idx = random.randint(0, self.size)  # choose index of vertex from 0 to sizeof(vertices)
            set_c = [x_idx]
            distances_from_set_c = np.empty(self.size, dtype=np.float32)
            distances_from_set_c.fill(np.inf)
            # r = 200
            i = 0
            if compute_d_mat_flag:
                d_mat = np.zeros((self.size, self.size))

            while i < k - 1:
                if compute_d_mat_flag:
                    raise NotImplementedError('Compute geodesic in not implemented yet')
                    # todo: compute dist func
                    d = self.compute_geodesics(x_idx)
                    d_mat[:, x_idx] = d  # saving distances ,if distance map is
                    # available, no need to compute distances from points to mesh
                else:
                    d = d_mat[:, x_idx] # distance of all points on the mesh from x

                distances_from_set_c = np.minimum(distances_from_set_c, d)
                r = max(distances_from_set_c)
                x_idx = np.where(distances_from_set_c == r)[0][0]
                set_c.append(x_idx)
                i += 1

        return [set_c, d_mat]

    def compute_subspace(self, k):
        print('start compute subspace')
        if k == self.size:
            self.eigs = np.ones(self.size)
            self.evecs = np.eye(self.size)

        else:
            if (self.mass_mat.size == 0) or (self.stiffness_mat.size == 0):
                self.compute_laplacian()
            [eigs, evecs] = sp.eigs(self.stiffness_mat, k, self.mass_mat, sigma=0,
                                    which='LM')  # sigma=0 gives 1/lambda -> LM gives smallest eigs
            self.eigs = -np.real(eigs) # code above gives -eigs, so we invert the signs
            self.evecs = np.real(evecs)
            # TODO fix eigs (put '-' and 1/)


    def plot_embedding(self, vertices):
            x = vertices[:, 0]
            y = vertices[:, 1]
            z = vertices[:, 2]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, y, z)
            self.set_axes_equal(ax)
            fig.show()


    @staticmethod
    def set_axes_radius(ax, origin, radius):
        ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
        ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
        ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

    def set_axes_equal(self, ax):
        '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Input
          ax: a matplotlib axis, e.g., as output from plt.gca().
        '''

        limits = np.array([
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ])

        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        self.set_axes_radius(ax, origin, radius)
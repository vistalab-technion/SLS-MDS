import trimesh
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import sparse
import pygeodesic.geodesic as geodesic
from SignalType import SignalType
# import gdist
# from torch_geometric.utils.geodesic import geodesic_distance
import scipy.io as sio


random.seed(10)


class Shape:
    def __init__(self, args):
        self.signal_type = SignalType.MESH
        self.args = args
        if args.filename is not None:
            self.filename = args.filename.split('.')[0]

            self.mesh = trimesh.load_mesh(args.filename, process=False)
            # self.mesh.show()
        elif args.vertices is not None and args.faces is not None:
            self.mesh = trimesh.Trimesh(args.vertices, args.faces)
        else:
            print("Shape should get 'filename' or 'vertices and faces'")
            raise SystemError()

        self.size = len(self.mesh.vertices)
        self.dim = len(self.mesh.vertices[0])

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
        The function samples points from the mesh according to the farthest point sampling
         strategy

        :param k: number of sample points.
        :param d_mat: geodesic distance mat between all points on the mesh.
        :return: set_c: set of sampled points
                 d_mat:  geodesic distance matrix between points in set_c
        """

        print("sample_mesh\n")
        if d_mat:
            if d_mat.endswith('.npy'):
                d_mat = np.load(d_mat)
            elif d_mat.endswith('.mat'):
                d_mat = sio.loadmat(d_mat)['D']
            else:
                print('file must endwith .npy or .mat')


        if k == len(self.mesh.vertices):
            set_c = range(0, len(self.mesh.vertices))
            if d_mat is None:
                d_mat = self.compute_geodesics()
                np.save(f"{self.filename}_geo_dist_full", d_mat)
        else:
            compute_d_mat_flag = False
            if d_mat is None:
                compute_d_mat_flag = True
                d_mat = np.zeros((self.size, self.size))

            # choose index of vertex from 0 to sizeof(vertices)
            x_idx = random.randint(0, self.size)
            set_c = [x_idx]
            distances_from_set_c = np.empty(self.size, dtype=np.float32)
            distances_from_set_c.fill(np.inf)
            i = 0

            while i < k - 1:
                # saving distances ,if distance map is available, no need to compute distances from points to mesh
                # d = distance from x to all points on the mesh
                if compute_d_mat_flag:
                    d = self.compute_geodesics(x_idx)
                    d_mat[:, x_idx] = d
                    d_mat[x_idx, :] = d
                else:
                    d = d_mat[:, x_idx]
                # np.save(f"{self.filename}/geo_dist_vector_{x_idx}", d)
                distances_from_set_c = np.minimum(distances_from_set_c, d)
                r = max(distances_from_set_c)
                x_idx = np.where(distances_from_set_c == r)[0][0]
                set_c.append(x_idx)
                i += 1
            np.save(f"{self.filename}_geo_dist_s", d_mat)

        return [set_c, d_mat]

    def compute_geodesics(self, source_index=None):
        print("compute geodesic distance")
        vertices = np.array(self.mesh.vertices).astype(np.float64)
        faces = np.array(self.mesh.faces).astype(np.int32)
        geo_alg = geodesic.PyGeodesicAlgorithmExact(vertices, faces)

        if source_index is None:
            dist = np.zeros((vertices.shape[0], vertices.shape[0]))
            for index1, var1 in enumerate(vertices):
                for index2, var2 in enumerate(vertices):
                    source_index = np.array(index1)
                    target_index = np.array(index2)

                    if source_index < target_index:
                        distance, path = geo_alg.geodesicDistance(source_index, target_index)

                        # distance = gdist.compute_gdist(vertices, faces, source_index, target_index)
                        dist[source_index][target_index] = distance
                        dist[target_index][source_index] = distance

        else:
            source_index = np.array([source_index], dtype=np.int32)
            dist = np.zeros(vertices.shape[0])
            for target_index, val in enumerate(vertices):
                if dist[target_index] == 0:
                    target_index = np.array([target_index], dtype=np.int32)
                    # distance = gdist.compute_gdist(vertices, faces, source_index, target_index)
                    distance, path = geo_alg.geodesicDistance(source_index, target_index)

                    if target_index % 2000 == 0:
                        print(f"target index = {target_index}, source index = {source_index}")
                    dist[target_index] = distance

        return dist

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
        """Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Input
          ax: a matplotlib axis, e.g., as output from plt.gca().
        """

        limits = np.array([
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ])

        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        self.set_axes_radius(ax, origin, radius)

import scipy.sparse.linalg as sp
import trimesh
import numpy as np
from Calculations import Calculations
import random
from scipy import sparse

from SignalType import SignalType


class Shape:

    signal_type = SignalType.MESH

    def __init__(self, filename=None):
        if filename is not None:
            self.mesh = trimesh.load_mesh(filename, process=False)
            self.size = len(self.mesh.vertices)
            self.dim = len(self.mesh.vertices[0])
            self.mass_mat, self.stiffness_mat = self.compute_laplacian()
            #self.compute_adjacency_mat()
            self.weights = np.ones([self.size, self.size]) #Variable(torch.ones((self.size, self.size))).type(torch.FloatTensor).cuda()

    def compute_laplacian(self, ):
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

            # cot12 = np.expand_dims(cot12,1); cot23 = np.expand_dims(cot23,1); cot31 = np.expand_dims(cot31,1);

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

    def compute_subspace(self, k):
        print('start compute subspace')
        if (self.mass_mat.size == 0) or (self.stiffness_mat.size == 0):
            self.compute_laplacian()
        [self.eigs, self.evecs] = sp.eigs(self.stiffness_mat, k, self.mass_mat, sigma=0, which='LM')  # gisma=0 gives 1/lambda -> LM gives smallest eigs
        # TODO fix eigs (put '-' and 1/)

    def sample_mesh(self, k,  d_mat=None):
        print("sample_mesh\n")

        if k == len(self.mesh.vertices):
            set_c = range(0, len(self.mesh.vertices))
            # TODO: compute geodesic distances
            if d_mat is None:
                print('You need to provide distance matrix')

        else:

            compute_d_mat_flag = False
            if d_mat is None:
                compute_d_mat_flag = True

            x_idx = random.randint(0, self.size)  # choose index of vertex from 0 to sizeof(vertices)
            set_c = [x_idx]
            distances_from_c = np.empty(self.size, dtype=np.float32)
            distances_from_c.fill(np.inf)
            # r = 200
            i = 0
            if compute_d_mat_flag:
                d_mat = np.zeros((self.size, self.size))

            while i < k-1:
                if compute_d_mat_flag:
                    d = self.compute_dist(x_idx, self.mesh)
                    d_mat[:, x_idx] = d  # saving distances ,if distance map is available, no need to compute distances from points to mesh
                else:
                    d = d_mat[:, x_idx]

                distances_from_c = np.minimum(distances_from_c, d)
                r = max(distances_from_c)
                x_idx = np.where(distances_from_c == r)[0][0]
                set_c.append(x_idx)

                i += 1

        return [set_c, d_mat]

    # @staticmethod
    # def compute_dist(v0, set_c, distances_from_c,  vertices):
    #     for i, v in enumerate(vertices):
    #         if v.all in set_c:
    #             continue
    #         d = vincenty(v, v0).km  # geodesic distance between v and v0
    #         if distances_from_c[i] > d:  # update the distances from c with the new v0
    #             distances_from_c[i] = d

    # def compute_dist(self, v0, mesh):
    #     d = []
    #     return d

    def set_signal_type(self, signal_type):
        self.shape.signal_type = signal_type

    def compute_adjacency_mat(self):
        adjacency_mat = np.zeros((self.size, self.size))
        for e in self.mesh.edges:
            adjacency_mat[e[0]][e[1]] = 1
        self.adjacency_mat = adjacency_mat

    def read_off(file):
        if 'OFF' != file.readline().strip():
            raise ValueError('Not a valid OFF header')
        n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
        verts = []
        for i_vert in range(n_verts):
            verts.append([float(s) for s in file.readline().strip().split(' ')])
        faces = []
        for i_face in range(n_faces):
            faces.append([int(s) for s in file.readline().strip().split(' ')][1:])
        return verts, faces




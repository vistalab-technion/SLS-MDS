import scipy.sparse.linalg as sp
#import torch
import trimesh
import numpy as np
#from torch.autograd import Variable
from Calculations import Calculations
import random
from scipy.spatial.distance import pdist

from SignalType import SignalType


class Shape:
    #mesh = trimesh.Trimesh()
    mesh = []
    mass_mat = np.array([])
    stiffness_mat = np.array([])
    adjacency_mat = np.array([])
    dist_mat = np.array([])
    weights = []
    size = 0
    dim = 0
    eigs = []
    evecs = []
    signal_type = SignalType.MESH

    # mesh class
    class Mesh:
        faces = []
        vertices = []

        def __init__(self, filename=None):
            [vertices, faces] = Shape.read_off(open(filename, 'r'))
            self.vertices = np.asarray(vertices)
            self.faces = np.asarray(faces)

    def __init__(self, filename=None):
        if filename is not None:
            #self.mesh = trimesh.load_mesh(filename)
            self.mesh = self.Mesh(filename)
            self.calc = Calculations()

            self.size = len(self.mesh.vertices)
            self.dim = len(self.mesh.vertices[0])
            self.compute_laplacian()
            #self.compute_adjacency_mat()
            self.weights = np.ones([self.size, self.size]) #Variable(torch.ones((self.size, self.size))).type(torch.FloatTensor).cuda()

    def compute_laplacian(self, l=None):
        # if l is None:
        #     self.dist_mat = self.calc.dist_mat(self.mesh, self.size)
        # else:
        #     self.dist_mat = l
        # self.mass_mat = self.calc.mass_mat(self.dist_mat, self.mesh, self.size)
        # self.stiffness_mat = self.calc.stiffness_mat(self.dist_mat, self.mesh, self.size)
        self.mass_mat, self.stiffness_mat = self.calc.laplacian(self.mesh, self.size)

    def compute_subspace(self, k):
        print('start compute subspace')
        if (self.mass_mat.size == 0) or (self.stiffness_mat.size == 0):
            self.compute_laplacian()
        # stiffness_mat_cpu = self.stiffness_mat.data.cpu().numpy()
        # mass_mat_cpu = self.mass_mat.data.cpu().numpy()
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




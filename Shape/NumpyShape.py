import scipy.sparse.linalg as sp
import trimesh
import numpy as np
import random
from scipy import sparse

from Shape.Shape import Shape


class NumpyShape(Shape):
    def __init__(self, filename=None):
        Shape.__init__(self, filename)
        if filename is not None:
            self.weights = np.ones((self.size, self.size))
            self.mass_mat, self.stiffness_mat = self.compute_laplacian()
            self.eigs = np.array([])
            self.evecs = np.array([])
            self.adjacency_mat = self.compute_adjacency_mat()

    def compute_subspace(self, k):
        print('start compute subspace')
        if (self.mass_mat.size == 0) or (self.stiffness_mat.size == 0):
            self.compute_laplacian()
        [eigs, evecs] = sp.eigs(self.stiffness_mat, k, self.mass_mat, sigma=0,
                                which='LM')  # gisma=0 gives 1/lambda -> LM gives smallest eigs
        self.eigs = eigs
        self.evecs = evecs
        # TODO fix eigs (put '-' and 1/)

    def compute_adjacency_mat(self):
        adjacency_mat = np.zeros((self.size, self.size))
        for e in self.mesh.edges:
            adjacency_mat[e[0]][e[1]] = 1
        return adjacency_mat


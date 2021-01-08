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

            self.adjacency_mat = self.compute_adjacency_mat()
            self.weights = np.ones((self.size, self.size))


    def compute_adjacency_mat(self):
        adjacency_mat = np.zeros((self.size, self.size))
        for e in self.mesh.edges:
            adjacency_mat[e[0]][e[1]] = 1
        return adjacency_mat


import scipy.sparse.linalg as sp
import torch
import trimesh
import numpy as np
import random
from scipy import sparse

from Shape.Shape import Shape


class TorchShape(Shape):
    def __init__(self, filename=None):
        Shape.__init__(self, filename)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mass_mat, self.stiffness_mat = self.compute_laplacian()
        self.weights = torch.ones((self.size, self.size)).to(self.device)
        self.eigs = torch.tensor
        self.evecs = torch.tensor
        self.adjacency_mat = self.compute_adjacency_mat()

    def compute_subspace(self, k):
        # TODO: move to Shape class
        # TODO: if k = n return identity

        print('start compute subspace')
        if (self.mass_mat.size == 0) or (self.stiffness_mat.size == 0):
            self.compute_laplacian()
        [eigs, evecs] = sp.eigs(self.stiffness_mat, k, self.mass_mat, sigma=0,
                                which='LM')  # gisma=0 gives 1/lambda -> LM gives smallest eigs
        self.eigs = torch.from_numpy(eigs.astype(np.float32)).to(self.device)
        self.evecs = torch.from_numpy(evecs.astype(np.float32)).to(self.device)

    def compute_adjacency_mat(self):
        adjacency_mat = torch.zeros((self.size, self.size))
        # TODO: move to Shape class

        for e in self.mesh.edges:
            adjacency_mat[e[0]][e[1]] = 1
        return adjacency_mat


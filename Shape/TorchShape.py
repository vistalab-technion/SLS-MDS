import scipy.sparse.linalg as sp
import torch
import trimesh
import numpy as np
import random
from scipy import sparse

from Shape.Shape import Shape


class TorchShape(Shape):
    def __init__(self, device, filename=None):
        Shape.__init__(self, filename)
        self.adjacency_mat = self.compute_adjacency_mat()

    def compute_adjacency_mat(self):
        adjacency_mat = torch.zeros((self.size, self.size))
        # TODO: move to Shape class

        for e in self.mesh.edges:
            adjacency_mat[e[0]][e[1]] = 1
        return adjacency_mat


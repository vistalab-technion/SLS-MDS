import scipy.sparse.linalg as sp
import torch
import numpy as np


from Shape.Shape import Shape


class TorchShape(Shape):
    def __init__(self, args, device=None):
        Shape.__init__(self, args)
        self.args = args
        if args.filename is not None:
            self.mass_mat, self.stiffness_mat = self.compute_laplacian()
            self.weights = torch.ones((self.size, self.size), dtype=torch.float64, device=device)
            self.eigs = torch.tensor
            self.evecs = torch.tensor
            self.adjacency_mat = self.compute_adjacency_mat()

    def compute_subspace(self, k):
        # TODO: if k = n return identity

        print('start compute subspace')
        if (self.mass_mat.size == 0) or (self.stiffness_mat.size == 0):
            self.compute_laplacian()
        [eigs, evecs] = sp.eigs(self.stiffness_mat, k, self.mass_mat, sigma=0,
                                which='LM')  # gisma=0 gives 1/lambda -> LM gives smallest eigs
        self.eigs = torch.from_numpy(eigs.real.astype(np.float64)).clone()
        self.evecs = torch.from_numpy(evecs.real.astype(np.float64)).clone()

    def compute_adjacency_mat(self):
        adjacency_mat = torch.zeros((self.size, self.size))

        for e in self.mesh.edges:
            adjacency_mat[e[0]][e[1]] = 1
        return adjacency_mat


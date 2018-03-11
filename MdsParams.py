import numpy as np
import torch
import trimesh
from torch.autograd import Variable
from Shape import Shape
from SignalType import SignalType


class MdsParams:
    p = torch.FloatTensor
    c = 2
    q = torch.FloatTensor
    max_iter = 10000
    a_tol = 0
    r_tol = 1e-8
    samples_array = torch.LongTensor
    plot_flag = False
    compute_full_stress_flag = False
    compute_full_embedding_flag = compute_full_stress_flag

    def __init__(self, shape):
        self.shape = shape

    def set_p_q(self, p, q=None):
        self.p = p
        if q is None:
            self.q = np.multiply(self.c,p)
        else:
            self.q = q

    def set_optim_param(self, max_iter, a_tol, r_tol):
        self.max_iter = max_iter
        self.a_tol = a_tol
        self.r_tol = r_tol

    def samples(self, samples_array):
        self.samples_array = torch.from_numpy(np.array(samples_array)).type(torch.LongTensor).cuda()

    def set_shape(self, shape):
        self.shape = shape

    def set_compute_full_stress_flag(self, compute_full_stress_flag):
        self.compute_full_stress_flag = compute_full_stress_flag

    def set_compute_full_embedding_flag(self, compute_full_embedding_flag):
        self.compute_full_embedding_flag = compute_full_embedding_flag

    def set_plot_flag(self, plot_flag):
        self.plot_flag = plot_flag






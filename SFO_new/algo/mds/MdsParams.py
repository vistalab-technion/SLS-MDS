import numpy as np


class MdsParams:
    p = np.array([])
    c = 2
    q = np.array([])
    max_iter = 10000
    a_tol = 0
    r_tol = 1e-8
    samples_array = np.array([])
    plot_flag = False
    compute_full_stress_flag = False
    compute_full_embedding_flag = compute_full_stress_flag

    def __init__(self, shape):
        self.shape = shape

    def set_p_q(self, p):
        self.p = p
        self.q = np.multiply(self.c, p)

    def set_optim_param(self, max_iter, a_tol, r_tol):
        self.max_iter = max_iter
        self.a_tol = a_tol
        self.r_tol = r_tol

    def samples(self, samples_array):
        self.samples_array = samples_array

    def set_shape(self, shape):
        self.shape = shape

    def set_compute_full_stress_flag(self, compute_full_stress_flag):
        self.compute_full_stress_flag = compute_full_stress_flag

    def set_compute_full_embedding_flag(self, compute_full_embedding_flag):
        self.compute_full_embedding_flag = compute_full_embedding_flag

    def set_plot_flag(self, plot_flag):
        self.plot_flag = plot_flag






import numpy as np
import torch


class MdsParams:
    p = []
    q = []
    c = 2
    max_iter = 10000
    a_tol = 0
    r_tol = 1e-8
    samples_array = []
    plot_flag = False

    def __init__(self):
        pass

    def set_p_q(self, p):
        self.p = p
        self.q = self.c * p


    def set_optim_param(self, max_iter, a_tol, r_tol):
        self.max_iter = max_iter
        self.a_tol = a_tol
        self.r_tol = r_tol

    def samples(self, samples_array):
        self.samples_array = samples_array

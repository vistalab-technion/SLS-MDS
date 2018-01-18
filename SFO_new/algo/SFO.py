import numpy as np
import torch
from torch.autograd import Variable
import Calculations
from algo import MFO



class SFO:
    num_of_iterations = 1
    mfo = MFO.MFO()
    calc = Calculations.Calculations()

    def __init__(self):
        self.first_time_flag = 1

    def algorithm(self, shape, shape_a, shape_b):
        curr_shape = shape
        const_a_inv = np.array([])
        const_w_pinv = np.array([])
        v_ab = np.array([])
        r_ab = np.array([])
        curr_shape = shape

        for i in range(self.num_of_iterations):
            curr_l = curr_shape.adjacency_mat  # compute mat L from embedding X

            if self.first_time_flag:  # calc the const Ac^-1 and the pinv of Wc on the original mesh
                self.first_time_flag = 0
                [const_a_inv, const_w_pinv] = self.calc_const_matrices(shape)
                [v_ab, r_ab] = self.calc_difference_operators(shape_a, shape_b)

            # improve mat L using mfo algorithm
            curr_l = self.mfo.algorithm(curr_shape, const_a_inv, const_w_pinv, v_ab, r_ab)
            curr_shape.compute_laplacian(curr_l)
            # compute embedding X from mat_l using mds algorithm
            # curr_shape = self.mds.algorithm(curr_shape)

        return curr_shape

    def calc_difference_operators(self, shape_a, shape_b):
        v_ab = Variable(torch.matmul(
            self.calc.compute_diag_mat_inv(shape_a.mass_mat), shape_b.mass_mat).data).type(
            torch.FloatTensor).cuda(self.calc.gpu_a)
        r_ab = Variable(torch.matmul(self.calc.compute_mat_pinv(shape_a.stiffness_mat, shape_a.size)
                                     , shape_b.stiffness_mat.data)).type(torch.FloatTensor).cuda(self.calc.gpu_w)

        return [v_ab, r_ab]

    def calc_const_matrices(self, shape):
        res_a = Variable(self.calc.compute_diag_mat_inv(shape.mass_mat).data.type(
            torch.FloatTensor).cuda(self.calc.gpu_a))
        res_w = Variable(self.calc.compute_mat_pinv(shape.stiffness_mat, shape.size).type(
            torch.FloatTensor).cuda(self.calc.gpu_w))

        return [res_a, res_w]

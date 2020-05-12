import Calculations
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from algo.mds import MdsParams
from algo.mds import MDS
from scipy.spatial.distance import pdist, squareform


class MFO:
    num_of_iterations = 60
    step_size = 0.0005
    lam = 0.2

    def __init__(self):
        self.calc = Calculations.Calculations()

    def algorithm(self, shape, const_a_inv, const_w_pinv, v_ab, r_ab):
        print("start mfo algorithm")
        index = 0
        loss_array = []
        curr_l = self.calc.dist_mat(shape.mesh, shape.size)
        mesh_x = shape.mesh
        num_of_vertices = shape.size
        loss = self.loss_fn(curr_l, mesh_x, num_of_vertices, const_a_inv, const_w_pinv, v_ab, r_ab)
        loss_array.append(loss.data[0])
        index += 1
        for k in range(self.num_of_iterations):

            # compute the gradient of loss with respect to all Variables with requires_grad = True.
            torch.autograd.backward(loss, retain_graph=True)

            curr_l.data -= (self.step_size * curr_l.grad.data)  # improve curr_l using gradient descent.

            next_loss = self.loss_fn(
                curr_l, mesh_x, num_of_vertices, const_a_inv, const_w_pinv, v_ab, r_ab)

            if (not self.calc.is_valid(curr_l, mesh_x.faces)) or (loss.data[0] < next_loss.data[0]):
                print("matrix not valid or not decreasing - main algo")
                # TODO: replace with geodesic distance later
                normal = np.random.normal(0, 1, size=(shape.size, 3))
                # print(normal)
                tmp = shape.mesh.vertices + normal
                d_mat_input = squareform(pdist(tmp, metric='euclidean'))  # TODO: replace with dedicated function

                mds_params = MdsParams.MdsParams(shape)

                mds_params.set_p_q([300])
                mds_params.set_optim_param(1000, 0, 1e-8)
                mds_params.set_shape(shape)
                mds_params.set_compute_full_stress_flag(False)
                mds_params.set_compute_full_stress_flag(False)
                mds_params.set_plot_flag(False)
                [samples, d_mat] = shape.sample_mesh_fps(np.max(mds_params.q), d_mat_input)
                mds_params.samples(samples)
                # create subspace
                shape.compute_subspace(max(mds_params.p))

                phi = shape.evecs
                mds = MDS.MDS(mds_params)
                new_x = mds.algorithm(d_mat, shape.weights, shape.mesh.vertices, phi)
                print(new_x)

            loss = next_loss
            loss_array.append(loss.data[0])
            index += 1
            curr_l.grad.data.zero_()
        print("loss_array = ")
        print(loss_array)
        plt.plot(loss_array)
        plt.show()
        return curr_l

    def loss_fn(self, mat, mesh_x, num_of_vertices, const_a_inv, const_w_pinv, v_ab, r_ab):

        if not self.calc.is_valid(mat, mesh_x.faces):
            self.step_size /= 2.0
            print("matrix not valid - loss_fn")

        loss = self.lam * torch.norm(
            torch.mm(const_a_inv, self.calc.stiffness_mat(mat, mesh_x, num_of_vertices)) - v_ab) \
            + (1 - self.lam) * torch.norm(
            torch.mm(const_w_pinv, self.calc.mass_mat(mat, mesh_x, num_of_vertices)) - r_ab)

        return loss

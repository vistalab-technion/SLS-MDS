import Calculations
import torch
import time


class MFO:
    num_of_iterations = 50
    step_size = 0.001
    lam = 0.2

    def __init__(self):
        self.calc = Calculations.Calculations()

    def algorithm(self, mesh_x, mat_l, const_a_inv, const_w_pinv, v_ab, r_ab):
        num_of_vertices = len(mesh_x.vertices)
        curr_l = mat_l
        loss = self.loss_fn(curr_l, mesh_x, num_of_vertices, const_a_inv, const_w_pinv, v_ab, r_ab)
        print("first loss data:", loss.data[0])

        for k in range(self.num_of_iterations):
            start_1 = time.time()

            print("mfo start of", k, "iteration")

            # compute the gradient of loss with respect to all Variables with requires_grad = True.
            torch.autograd.backward(loss, retain_graph=True)

            curr_l.data -= (self.step_size * curr_l.grad.data)  # improve curr_l using gradient descent.

            next_loss = self.loss_fn(
                curr_l, mesh_x, num_of_vertices, const_a_inv, const_w_pinv, v_ab, r_ab)

            while (not self.calc.is_valid(curr_l, mesh_x.faces)) or (loss.data[0] < next_loss.data[0]):
                print("step_size: ", self.step_size)
                self.step_size /= 2.0
                curr_l.data -= torch.mul(curr_l.grad.data, self.step_size)  # improve curr_l using gradient descent.
                next_loss = self.loss_fn(
                    curr_l, mesh_x, num_of_vertices, const_a_inv, const_w_pinv, v_ab, r_ab)

            loss = next_loss
            curr_l.grad.data.zero_()

            print("end of", k, "iteration, loss.data[0]: ", loss.data[0])
            end_1 = time.time()
            print("time = ", end_1 - start_1)

        return curr_l

    def loss_fn(self, mat, mesh_x, num_of_vertices, const_a_inv, const_w_pinv, v_ab, r_ab):

        while not self.calc.is_valid(mat, mesh_x.faces):
            self.step_size /= 2.0

        loss = self.lam * torch.norm(
            torch.mm(const_a_inv, self.calc.compute_mat_a(mesh_x, mat, num_of_vertices)) - v_ab) \
            + (1 - self.lam) * torch.norm(
            torch.mm(const_w_pinv, self.calc.compute_mat_w(mesh_x, mat, num_of_vertices).data) - r_ab)

        return loss

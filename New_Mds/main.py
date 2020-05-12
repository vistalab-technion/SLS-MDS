import random

import MdsParams
from Shape import Shape
import MDS
import numpy as np
from scipy.spatial.distance import pdist, squareform
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt


# def compute_s(mat, vec):
#     print(mat.data[0])
#     print(vec)
#     return mat[vec, :]


def main():
    print("start main\n")

    # import mesh
    print('Hi')

    # tmp = np.array([[2, 0], [1, 2], [0, 1]])
    # tmp2 = np.array([[2, 3], [0, 1], [1, 1], [6, 7]])
    # # print(tmp2[tmp, :])
    # new_temp = tmp2[tmp[:, 0]]
    # print(tmp2, "\n new tmp = ", new_temp)

    # tmp = torch.FloatTensor([[2, 0], [1, 2], [0, 1], [2, 5]])
    # tmp2 = torch.FloatTensor([[2, 3], [0, 1], [1, 1], [6, 7]])
    # # tmp_non_zeros_idx = torch.nonzero(tmp)
    # # print("tmp_non_zeros_idx: ", tmp_non_zeros_idx)
    # # print(tmp[tmp_non_zeros_idx])
    # # print(tmp2[tmp, :])
    # tmp[tmp == 0] = float('inf')
    # new_temp = torch.div(tmp2, tmp)
    # print(tmp, tmp2, "\n new tmp = ", new_temp)
    #
    # # new_temp[new_temp == float('inf')] = 0
    # print(new_temp)

    shape = Shape("input/sphere.off")
    # shape.mesh.show()
    # TODO: replace with geodesic distance later
    normal = np.random.normal(0, 1, size=(shape.size, 3))
    # print(normal)
    tmp = shape.mesh.vertices + normal
    d_mat_input = squareform(pdist(tmp, metric='euclidean'))  # TODO: replace with dedicated function

    mds_params = MdsParams.MdsParams(shape)

    mds_params.set_p_q([300])
    mds_params.set_optim_param(1000, 0, 0)
    mds_params.set_shape(shape)
    mds_params.set_compute_full_stress_flag(True)
    mds_params.set_compute_full_embedding_flag(True)
    mds_params.set_plot_flag(True)
    [samples, d_mat] = shape.sample_mesh_fps(np.max(mds_params.q), d_mat_input)

    # samples_t = torch.from_numpy(np.array(samples))
    mds_params.samples(samples)
    # create subspace
    shape.compute_subspace(max(mds_params.p))

    phi = np.real(shape.evecs)
    phi_t = Variable(torch.from_numpy(phi).type(torch.FloatTensor)).cuda()
    x0 = Variable(torch.from_numpy(shape.mesh.vertices).type(torch.FloatTensor)).cuda()
    d_mat_t = Variable(torch.FloatTensor(d_mat)).cuda()
    mds = MDS.MDS(mds_params)

    new_x = mds.algorithm(d_mat_t, shape.weights, x0, phi_t)
    # shape.mesh.vertices = new_x.cpu().numpy()
    # shape.mesh.show()
    #
    # plt.plot(mds.stress_list)
    # plt.show()
if __name__ == '__main__':
    main()

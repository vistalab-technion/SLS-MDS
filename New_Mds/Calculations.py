import numpy as np
import torch
from torch.autograd import Variable
from scipy import sparse


class Calculations:
    gpu_a = 0
    gpu_w = 0
    gpu_l = 0

    def __init__(self):
        pass

    # l satisfying the strong triangle inequality (equ 4)
    def is_valid(self, mat, mesh_faces):
        i, j, k = 0, 1, 2
        t_mesh_faces = torch.from_numpy(mesh_faces).cuda(self.gpu_l)

        for f in t_mesh_faces:
            if mat[f[i]][f[j]].data[0] < 0 or mat[f[j]][f[k]].data[0] < 0 or mat[f[i]][f[k]].data[0] < 0 \
                    or ((mat[f[i]][f[j]].data[0] + mat[f[j]][f[k]].data[0] - mat[f[i]][f[k]].data[0]) <= 0
                        or (mat[f[j]][f[k]].data[0] + mat[f[k]][f[i]].data[0] - mat[f[i]][f[j]].data[
                            0]) <= 0
                        or (mat[f[k]][f[i]].data[0] + mat[f[i]][f[j]].data[0] - mat[f[j]][f[k]].data[
                            0]) <= 0):
                print("False")
                return False

        # print("True")
        return True

    # returns the invert of a diagonal mat
    def compute_diag_mat_inv(self, mat):
        mat_a_np = mat.data.cpu().numpy()

        if np.linalg.det(mat_a_np) == 0:
            mat_i = np.eye(len(mat_a_np)) * 1e-8
            mat_a_np = np.add(mat_a_np, mat_i)

        mat_a_t = Variable(torch.from_numpy(mat_a_np).type(torch.FloatTensor)).cuda(self.gpu_a)
        mat_a_diag = torch.diag(mat_a_t)
        mat_diag_inv = 1 / mat_a_diag

        return torch.diag(mat_diag_inv)

    # returns the pseudo invert of mat
    def compute_mat_pinv(self, mat, num_of_vertices):  # time =  9.1305.
        mat_i = torch.eye(num_of_vertices) * 1e-8
        inverse_w = torch.add(mat_i, mat.data.type(torch.FloatTensor))
        w_inv = torch.inverse(inverse_w).type(torch.FloatTensor).cuda(self.gpu_w)
        return w_inv

    def dist_mat(self, mesh, size):
        print('compute mat_l')
        mat = torch.zeros((size, size))
        t_mesh_edges = torch.from_numpy(mesh.edges)
        t_mesh_vertices = torch.from_numpy(mesh.vertices)

        for e in t_mesh_edges:
            first = t_mesh_vertices[e[0]].type(torch.FloatTensor)
            sec = t_mesh_vertices[e[1]].type(torch.FloatTensor)
            mat[e[0]][e[1]] = np.sqrt((first[0]-sec[0])**2.0 + (first[1]-sec[1])**2.0 + (first[2]-sec[2])**2.0)
            mat[e[1]][e[0]] = mat[e[0]][e[1]]

        return Variable(mat.type(torch.FloatTensor).cuda(self.gpu_l), requires_grad=True)

    def mass_mat(self, l, mesh, size):
        print('start compute mat a')
        i, j, k = 0, 1, 2
        local_area_elements = Variable(torch.from_numpy(
            np.zeros(size)).type(torch.FloatTensor).cuda(self.gpu_a))
        t_mesh_faces = torch.from_numpy(mesh.faces).cuda(self.gpu_a)
        mat_l_a = l.cuda(self.gpu_a)

        for f in t_mesh_faces:
            v_i, v_j, v_k = f[i], f[j], f[k]

            l_ik = mat_l_a[v_i][v_k]
            l_kj = mat_l_a[v_k][v_j]
            l_ij = mat_l_a[v_i][v_j]

            s = (l_ik + l_kj + l_ij) / 2

            # a_ijk is the area of triangle ijk
            a_ijk = torch.sqrt(s * (s - l_ik) * (s - l_kj) * (s - l_ij))

            local_area_elements[v_i] = local_area_elements[v_i] + a_ijk
            local_area_elements[v_j] = local_area_elements[v_j] + a_ijk
            local_area_elements[v_k] = local_area_elements[v_k] + a_ijk

        torch.div(local_area_elements, 3.0)

        mass_mat = torch.diag(local_area_elements)

        return mass_mat

    # computes matrix W according to equ (7) in the SFO article
    def stiffness_mat(self, l, mesh, size):
        print('start compute mat w')

        stiffness_mat = Variable(torch.from_numpy(np.zeros((size, size))).type(
            torch.FloatTensor).cuda(self.gpu_w))
        mat_l_w = Variable(l.data.cuda(self.gpu_w))
        v_k, v_h = 0, 0
        i = 0
        t_mesh_face_adjacency = torch.from_numpy(mesh.face_adjacency).cuda(self.gpu_w)
        t_mesh_area_faces = torch.from_numpy(mesh.area_faces).cuda(self.gpu_w)
        t_mesh_face_adjacency_edges = torch.from_numpy(mesh.face_adjacency_edges).cuda(self.gpu_w)

        for f_adj in t_mesh_face_adjacency:
            [a_ijk, a_ijh] = t_mesh_area_faces[f_adj]
            [v_i, v_j] = t_mesh_face_adjacency_edges[i]

            for v_0, v_1 in zip(mesh.faces[f_adj[0]], mesh.faces[f_adj[1]]):
                if not v_0 == v_i and not v_0 == v_j:
                    v_k = v_0
                if not v_1 == v_i and not v_1 == v_j:
                    v_h = v_1

            l_ik = mat_l_w[v_i, v_k]
            l_jk = mat_l_w[v_k, v_j]
            l_ij = mat_l_w[v_i, v_j]
            l_ih = mat_l_w[v_i, v_h]
            l_jh = mat_l_w[v_j, v_h]

            tmp = ((-(l_ij ** 2.0) + l_jk ** 2.0 + l_ik ** 2.0) / (8.0 * a_ijk)) \
                + ((-(l_ij ** 2.0) + l_jh ** 2.0 + l_ih ** 2.0) / (8.0 * a_ijh))
            stiffness_mat[v_i, v_j] = tmp
            stiffness_mat[v_j, v_i] = tmp
            i += 1

        return stiffness_mat

    def laplacian(self, mesh, size):
        nv = np.shape(mesh.vertices)[0]
        nf = np.shape(mesh.faces)[0]
        V = mesh.vertices
        F = mesh.faces

        l1 = np.sqrt(np.sum(np.square(V[F[:, 1], :] - V[F[:, 2], :]), 1))
        l2 = np.sqrt(np.sum(np.square(V[F[:, 2], :] - V[F[:, 0], :]), 1))
        l3 = np.sqrt(np.sum(np.square(V[F[:, 0], :] - V[F[:, 1], :]), 1))

        i1 = F[:, 0]
        i2 = F[:, 1]
        i3 = F[:, 2]

        s = (l1 + l2 + l3) * 0.5
        # triangle - wise area
        fA = np.sqrt(np.multiply(np.multiply(np.multiply(s, s - l1), s - l2), s - l3))
        # cotangent weight
        cot12 = (l1 ** 2 + l2 ** 2 - l3 ** 2) / fA / 4.0
        cot23 = (l2 ** 2 + l3 ** 2 - l1 ** 2) / fA / 4.0
        cot31 = (l1 ** 2 + l3 ** 2 - l2 ** 2) / fA / 4.0

        # cot12 = np.expand_dims(cot12,1); cot23 = np.expand_dims(cot23,1); cot31 = np.expand_dims(cot31,1);

        diag1 = -cot12 - cot31
        diag2 = -cot12 - cot23
        diag3 = -cot31 - cot23

        i = np.concatenate((i1, i2, i2, i3, i3, i1, i1, i2, i3), 0)
        j = np.concatenate((i2, i1, i3, i2, i1, i3, i1, i2, i3), 0)

        # values corresponding to pairs form(i, j)

        v = np.concatenate((cot12, cot12, cot23, cot23, cot31, cot31, diag1, diag2, diag3), 0)

        stiffness = sparse.coo_matrix((v, (i, j)), (nv, nv))

        tri2ver = sparse.coo_matrix((np.ones(nf), (F[:, 0], np.arange(0, nf))), (nv, nf))
        tri2ver += sparse.coo_matrix((np.ones(nf), (F[:, 1], np.arange(0, nf))), (nv, nf))
        tri2ver += sparse.coo_matrix((np.ones(nf), (F[:, 2], np.arange(0, nf))), (nv, nf))

        tri2ver[tri2ver.nonzero()[0], tri2ver.nonzero()[1]] = 1

        mass = sparse.spdiags(tri2ver * fA / 3, 0, nv, nv)

        return mass.tocsc(), stiffness.tocsc()


# '-- Pseudo-inverse \
# -- x is supposed to be MxN matrix, where M samples(trials) and each sample(trial) is N dim
# -- returns Generalized inverse (also called the Moore-Penrose pseudo-inverse)
# -- defined for all real or complex matrices
# function pinv(x)
#    local u,s,v = torch.svd(x,'A')
#    local idx = torch.sum(torch.gt(s,0))
#    local stm = s:pow(-1):narrow(1,1,idx)
#    local n = stm:size()[1]
#    local ss=torch.expand(torch.reshape(stm,n,1),n,n) -- for elementwise mult
#    local vv = v:narrow(1,1,idx)
#    local uu = u:narrow(1,1,idx)
#    local pin = torch.mm(vv,torch.cmul(uu:t(),ss))
#    return pin
# end'
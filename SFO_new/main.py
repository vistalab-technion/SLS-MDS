import random

from Shape import Shape
import numpy as np

from algo import SFO


def main():
    print("start main\n")

    # import mesh
    print('Hi')
    shape = Shape("input/sphere.off")
    shape_a = Shape("input/sphere.off")
    shape_b = Shape("input/sphere_bump.off")
    # tmp = np.array([2, 3, 0, 1])
    # tmp2 = np.array([[2, 3], [0, 1], [1, 1], [6, 7]])
    # print(tmp2, "\n\n",  tmp[2:4], "\n")
    # print(tmp2[tmp[2:4], :])
    sfo = SFO.SFO()
    sfo.algorithm(shape, shape_a, shape_b)
    # shape.mesh.show()





    # TODO: plot stress

if __name__ == '__main__':
    main()

Subspace least-squares multidimensional scaling
https://arxiv.org/abs/1709.03484

© Amit Boyarski, Adi Weinberger, 2020

The code solves a stress based MDS problem where the input is a triangular mesh. It the SMACOF algorithm with a subspace parametrization based on the eigedecomposition of the Laplace-Beltrami operator. This parametrization allows solving huge MDS problems in a fraction of the time compared to the standard SMACOF algorithm.

Currently only the Numpy version is working. pyTorch version is under construction.

In order to run the project, you should put a 'shape_name.off' file, and a pre-computed matrix of pairwise (geodeisc) distances 'D_shape_name.mat' in the 'input' folder, and update the parameters in the main.py file.  Then run 'main.py' and it will generate the embedding.

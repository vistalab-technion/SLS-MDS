class MdsParams:

    def __init__(self, shape, _args):
        self.shape = shape
        self.max_iter = _args.max_iter
        self.a_tol = _args.a_tol
        self.r_tol = _args.r_tol
        self.c = _args.c
        self.plot_flag = _args.plot_flag
        self.compute_full_stress_flag = _args.compute_full_stress_flag
        self.compute_full_embedding_flag = self.compute_full_stress_flag
        self.display_every = _args.display_every
        self.max_size_for_pinv = _args.max_size_for_pinv
        self.samples_array = []
        self.weights = []
        self.set_p_q(_args.p, _args.q)

    def set_p_q(self, p=None, q=None):
        self.p = p
        if p is None:
            self.p = [self.shape.size]
        if q is None:
            # self.q = np.multiply(self.c, p)
            self.q = self.c * self.p
        else:
            self.q = q

    def set_weights(self, weights):
        self.weights = weights

    def set_optim_param(self, max_iter, a_tol, r_tol):
        self.max_iter = max_iter
        self.a_tol = a_tol
        self.r_tol = r_tol

    def set_samples(self, samples_array):
        self.samples_array = samples_array.copy()

    def set_shape(self, shape):
        self.shape = shape

    def set_compute_full_stress_flag(self, compute_full_stress_flag):
        self.compute_full_stress_flag = compute_full_stress_flag

    def set_compute_full_embedding_flag(self, compute_full_embedding_flag):
        self.compute_full_embedding_flag = compute_full_embedding_flag

    def set_plot_flag(self, plot_flag):
        self.plot_flag = plot_flag






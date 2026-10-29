"""
Algorithm 36 -- MGDA, Multiple-Gradient Descent Algorithm (Gradient-based, 5E; Desideri 2012, Sener & Koltun 2018)
Lecture steps per iteration
  1. Gradients g_i = grad f_i(x) of every objective / task (w.r.t. the SHARED variables).
  2. Min-norm element of their convex hull:  alpha* = argmin_{alpha in simplex} || sum_i alpha_i g_i ||^2
     (closed form for two tasks, Frank-Wolfe otherwise);  d = sum_i alpha_i g_i.
  3. If ||d|| ~ 0 the point is Pareto-critical (stop).  Otherwise  -d is a common descent direction:
     x <- x - eta d  decreases ALL objectives simultaneously.
Part (a): population of starting points on P13a -> points of the Pareto set (segment a-b).
Part (b): train the shared-body network with the MGDA weights (see algos._multitask.train).
"""
import numpy as np
from moo_base import MOAlgorithm
from algos._multitask import min_norm_weights, train, SharedNet, make_data


class MGDA(MOAlgorithm):
    number, name, family = 36, "MGDA", "Gradient-based (5E)"

    def __init__(self, problem, pop_size=20, seed=0, lr=0.1, verbose=False, **params):
        assert "jac" in problem.data, "MGDA needs a problem with analytic gradients (P13a)"
        super().__init__(problem, pop_size, seed, verbose, **params)
        self.lr = lr

    def initialize(self):
        super().initialize()
        self.X[0] = self.problem.data["start"]                  # sheet's starting point (0.5, 5)
        self.X, self.F, self.CV = self.evaluate(self.X)
        self.nfe -= 0

    def direction(self, x):
        G = self.problem.data["jac"](x)
        a = min_norm_weights(G)
        return a, a @ G

    def iterate(self):
        Xn = self.X.copy()
        for i, x in enumerate(self.X):
            a, d = self.direction(x)
            if np.linalg.norm(d) > 1e-8:
                Xn[i] = x - self.lr * d
        self.X, self.F, self.CV = self.evaluate(Xn)
        self.log("alpha of start point", self.direction(self.X[0])[0])

    def run_network(self, steps=800, lr=0.05, seed=0, reg_scale=5.0):
        X, yc, yr = make_data(seed=seed, reg_scale=reg_scale)
        net = SharedNet(X, yc, yr, seed=seed)
        return train(net, "mgda", steps, lr)


def unit_test():
    """Key rules: orthogonal unit gradients -> alpha=(1/2,1/2), d=(1/2,1/2); parallel gradients -> alpha picks the
    smaller one; opposite gradients -> d=0 (Pareto-critical); the step decreases BOTH objectives."""
    a = min_norm_weights(np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert np.allclose(a, [0.5, 0.5])
    a = min_norm_weights(np.array([[1.0, 0.0], [2.0, 0.0]]))
    assert np.allclose(a, [1.0, 0.0])
    a = min_norm_weights(np.array([[1.0, 1.0], [-1.0, -1.0]]))
    assert np.allclose(a @ np.array([[1.0, 1.0], [-1.0, -1.0]]), 0)
    a3 = min_norm_weights(np.eye(3))
    assert np.allclose(a3, 1 / 3, atol=1e-3), "Frank-Wolfe for 3 tasks"
    from moo_problems import get_problem
    p = get_problem("P13a")
    alg = MGDA(p, pop_size=5, seed=0)
    alg.budget = 10 ** 9
    alg.initialize()
    f0 = alg.F.copy()
    alg.iterate()
    assert np.all(alg.F <= f0 + 1e-9), "common descent direction: no objective may increase"
    r = MGDA(p, pop_size=10, seed=0).run(3000)
    x = r["X"]
    assert np.abs(x[:, 0] - x[:, 1]).max() < 1e-3, "converges to the Pareto set x1 = x2"
    return True

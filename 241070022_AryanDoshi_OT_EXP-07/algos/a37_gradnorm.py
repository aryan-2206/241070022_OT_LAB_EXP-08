"""
Algorithm 37 -- GradNorm (Gradient-based, 5E; Chen, Badrinarayanan, Lee, Rabinovich 2018)
Lecture steps per iteration (task weights w_i, sum w_i = T)
  1. G_i = || grad_W (w_i L_i) ||  = w_i ||g_i||   (gradient norm w.r.t. the shared weights);  G_avg = mean G_i.
  2. Relative inverse training rate  r_i = (L_i/L_i(0)) / mean_j (L_j/L_j(0))  (slower tasks have larger r_i).
  3. Target norm  G_avg r_i^alpha ;   GradNorm loss  sum_i | G_i - G_avg r_i^alpha |  (target treated as constant).
  4. w_i <- w_i - eta_w d(loss)/d w_i  = w_i - eta_w sign(G_i - target_i) ||g_i||, clip > 0, renormalise to sum T.
  5. Update the network with  sum_i w_i g_i.
Part (a): population of starting points on P13a (adaptive weights); Part (b): shared network.
"""
import numpy as np
from moo_base import MOAlgorithm
from algos._multitask import train, SharedNet, make_data


def gradnorm_weight_step(w, gnorm, L, L0, alpha=1.5, lr_w=0.02):
    T = len(w)
    Gi = w * gnorm
    r = (L / L0) / np.mean(L / L0)
    target = Gi.mean() * r ** alpha
    w = np.clip(w - lr_w * np.sign(Gi - target) * gnorm, 1e-3, None)
    return w * T / w.sum()


class GradNorm(MOAlgorithm):
    number, name, family = 37, "GradNorm", "Gradient-based (5E)"

    def __init__(self, problem, pop_size=20, seed=0, lr=0.05, alpha=1.5, lr_w=0.02, verbose=False, **params):
        assert "jac" in problem.data, "GradNorm needs a problem with analytic gradients (P13a)"
        super().__init__(problem, pop_size, seed, verbose, **params)
        self.lr, self.alpha, self.lr_w = lr, alpha, lr_w

    def initialize(self):
        super().initialize()
        self.X[0] = self.problem.data["start"]
        self.X, self.F, self.CV = self.evaluate(self.X)
        self.L0 = self.F.copy()
        self.w = np.ones((self.N, 2))

    def iterate(self):
        Xn = self.X.copy()
        for i, x in enumerate(self.X):
            G = self.problem.data["jac"](x)
            gn = np.linalg.norm(G, axis=1)
            self.w[i] = gradnorm_weight_step(self.w[i], gn, self.F[i], np.maximum(self.L0[i], 1e-9), self.alpha, self.lr_w)
            Xn[i] = x - self.lr * (self.w[i] @ G) / 2
        self.X, self.F, self.CV = self.evaluate(Xn)
        self.log("task weights of start point", self.w[0])

    def run_network(self, steps=800, lr=0.05, seed=0, reg_scale=5.0):
        X, yc, yr = make_data(seed=seed, reg_scale=reg_scale)
        net = SharedNet(X, yc, yr, seed=seed)
        return train(net, "gradnorm", steps, lr, alpha_gn=self.alpha, lr_w=self.lr_w)


def unit_test():
    """Key rule: the task with the LARGER weighted gradient norm gets a smaller weight, the other a larger one,
    and the weights keep sum = T."""
    w = np.array([1.0, 1.0])
    w2 = gradnorm_weight_step(w, np.array([4.0, 1.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0]))
    assert w2[0] < 1.0 < w2[1] and abs(w2.sum() - 2.0) < 1e-12
    # a task that trains slowly (larger r) gets a larger target -> its weight is increased
    w3 = gradnorm_weight_step(w, np.array([1.0, 1.0]), np.array([1.0, 2.0]), np.array([1.0, 1.0]))
    assert w3[1] > w3[0]
    from moo_problems import get_problem
    r = GradNorm(get_problem("P13a"), pop_size=8, seed=0).run(2000)
    assert len(r["F"]) >= 1
    return True

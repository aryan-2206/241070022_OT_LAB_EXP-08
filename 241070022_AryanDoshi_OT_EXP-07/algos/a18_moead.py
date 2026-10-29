"""
Algorithm 18 -- MOEA/D (Elitist, decomposition, Part 3; Zhang & Li 2007)
Lecture steps per generation
  1. N weight vectors lambda_i (Das-Dennis); each defines a Tchebycheff sub-problem
        g(x | lambda_i, z*) = max_j lambda_ij |f_j(x) - z*_j|   (objectives normalised).
  2. Neighbourhood B(i) = the T closest weight vectors; sub-problem i keeps ONE solution x_i.
  3. Reproduction: two parents from B(i) (probability delta, else the whole population), crossover + mutation.
  4. Update z*, then replace up to n_r solutions of B(i) whose Tchebycheff value is worse than the child's.
  Cooperative solving: neighbours exchange good solutions.  Constraints: feasibility first, then g.
"""
import numpy as np
from moo_base import MOAlgorithm, vary, tchebycheff
from moo_utils import das_dennis


def weight_vectors(M, N):
    if M == 2:
        t = np.linspace(0, 1, N)
        return np.column_stack([t, 1 - t])
    p = 1
    while len(das_dennis(M, p + 1)) <= N:
        p += 1
    return das_dennis(M, p)


class MOEAD(MOAlgorithm):
    number, name, family = 18, "MOEA/D", "Elitist, decomposition (Part 3)"

    def __init__(self, problem, pop_size=100, seed=0, T=20, delta=0.9, nr=2, verbose=False, **params):
        W = weight_vectors(problem.n_obj, pop_size)
        super().__init__(problem, len(W), seed, verbose, **params)
        self.W, self.T, self.delta, self.nr = W, min(T, len(W)), delta, nr
        d = np.sqrt(((W[:, None, :] - W[None, :, :]) ** 2).sum(axis=2))
        self.B = np.argsort(d, axis=1)[:, :self.T]

    def initialize(self):
        super().initialize()
        self.z = self.F.min(axis=0)

    def make_offspring(self):
        pa = np.empty(self.N, int)
        pb = np.empty(self.N, int)
        for i in range(self.N):
            pool = self.B[i] if self.rng.random() < self.delta else np.arange(self.N)
            pa[i], pb[i] = self.rng.choice(pool, 2, replace=len(pool) < 2)
        P = np.empty((2 * self.N,) + self.X.shape[1:], dtype=self.X.dtype)
        P[0::2], P[1::2] = self.X[pa], self.X[pb]
        return vary(self.problem, P, self.rng)[0::2]

    def update(self, Xo, Fo, CVo):
        self.z = np.minimum(self.z, Fo.min(axis=0))
        zn = self.F.max(axis=0)
        for i in self.rng.permutation(self.N):
            pool = self.B[i] if self.rng.random() < self.delta else np.arange(self.N)
            pool = self.rng.permutation(pool)
            g_new = tchebycheff(Fo[i], self.W[pool], self.z, zn)
            g_old = tchebycheff(self.F[pool], self.W[pool], self.z, zn)
            better = (CVo[i] < self.CV[pool]) | ((CVo[i] <= 0) & (self.CV[pool] <= 0) & (g_new < g_old))
            rep = pool[better][:self.nr]
            self.X[rep], self.F[rep], self.CV[rep] = Xo[i], Fo[i], CVo[i]

    def iterate(self):
        Xo, Fo, CVo = self.evaluate(self.make_offspring())
        self.update(Xo, Fo, CVo)
        self.log("ideal point z*", self.z)


def unit_test():
    """Key rule: a child replaces a neighbour only if its Tchebycheff value is smaller (and never when worse)."""
    w = np.array([0.5, 0.5])
    z = np.array([0.0, 0.0])
    assert abs(tchebycheff(np.array([1.0, 3.0]), w, z) - 1.5) < 1e-12        # max(0.5*1, 0.5*3)
    from moo_problems import get_problem
    a = MOEAD(get_problem("P1"), pop_size=10, T=3, nr=1, seed=0)
    a.budget = 10 ** 9
    a.initialize()
    before = a.F.copy()
    # a child much worse than everyone must not replace anything
    Xo = np.full((10,) + a.X.shape[1:], 10.0)
    Fo = np.full((10, 2), 1e6)
    a.update(Xo, Fo, np.zeros(10))
    assert np.allclose(a.F, before)
    # a perfect child (ideal point) replaces nr solutions
    Fo = np.zeros((10, 2))
    a.update(Xo, Fo, np.zeros(10))
    assert (a.F == 0).any(axis=1).sum() >= 10 or np.sum(np.all(a.F == 0, axis=1)) >= 5
    return True

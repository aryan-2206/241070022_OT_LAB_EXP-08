"""
Algorithm 29 -- MOBA, Multi-Objective Bat Algorithm (Swarm, Part 4; Yang 2011 extended)
Lecture steps per iteration
  1. Every bat has frequency f_i = f_min + (f_max - f_min) beta, velocity v_i, loudness A_i, pulse rate r_i.
  2. Guide x* = archive member chosen on the grid;  v_i <- v_i + (x_i - x*) f_i ;  x_i <- x_i + v_i.
  3. With probability (1 - r_i) a local random walk around the guide:  x_new = x* + eps A_mean (ub-lb) 0.1.
  4. Acceptance: new dominates old, or incomparable with probability A_i; on acceptance
        A_i <- alpha A_i,  r_i <- r_0 (1 - exp(-gamma t)).
  5. The archive keeps all non-dominated solutions.
"""
import numpy as np
from moo_base import MOAlgorithm, Archive, grid_roulette, row_dominates


class MOBA(MOAlgorithm):
    number, name, family = 29, "MOBA", "Swarm (Part 4)"

    def __init__(self, problem, pop_size=100, seed=0, fmin=0.0, fmax=2.0, alpha=0.9, gamma=0.9, r0=0.5,
                 verbose=False, **params):
        super().__init__(problem, pop_size, seed, verbose, **params)
        self.fmin, self.fmax, self.alpha, self.gamma, self.r0 = fmin, fmax, alpha, gamma, r0
        self.archive = Archive(pop_size, mode="grid")

    def initialize(self):
        super().initialize()
        N = self.N
        self.V = np.zeros(self.X.shape)
        self.A = np.ones(N)
        self.r = np.full(N, self.r0)
        self.t = 0
        self.archive.add(self.X, self.F, self.CV, self.rng)

    def move(self, X, V, guide, f):
        V = V + (X - guide) * f[:, None]
        return X + V, V

    def iterate(self):
        N, lb, ub = self.N, self.problem.lb, self.problem.ub
        self.t += 1
        guide = self.archive.X[grid_roulette(self.archive.F, self.rng, N)].astype(float)
        f = self.fmin + (self.fmax - self.fmin) * self.rng.random(N)
        Xn, Vn = self.move(self.X.astype(float), self.V, guide, f)
        walk = self.rng.random(N) > self.r
        Xw = guide + 0.1 * self.rng.normal(size=Xn.shape) * self.A.mean() * (ub - lb)
        Xn = np.where(walk[:, None], Xw, Xn)
        Xn = np.clip(Xn, lb, ub)
        Xe, Fe, CVe = self.evaluate(Xn)
        better = row_dominates(Fe, CVe, self.F, self.CV)
        inc = ~better & ~row_dominates(self.F, self.CV, Fe, CVe) & (self.rng.random(N) < self.A)
        take = better | inc
        self.X[take], self.F[take], self.CV[take], self.V[take] = Xe[take], Fe[take], CVe[take], Vn[take]
        self.A[take] *= self.alpha
        self.r[take] = self.r0 * (1 - np.exp(-self.gamma * self.t))
        self.archive.add(Xe, Fe, CVe, self.rng)
        self.log("mean loudness / pulse rate", [self.A.mean(), self.r.mean()])


def unit_test():
    """Key rule: v' = v + (x - x*) f, x' = x + v'. With f=1 and v=0 the bat is thrown to 2x - x*;
    with f=0 it stays."""
    a = MOBA.__new__(MOBA)
    X = np.array([[1.0, 1.0]]); G = np.array([[0.0, 0.0]])
    Xn, V = a.move(X, np.zeros((1, 2)), G, np.array([1.0]))
    assert np.allclose(Xn, [[2.0, 2.0]])
    Xn, V = a.move(X, np.zeros((1, 2)), G, np.array([0.0]))
    assert np.allclose(Xn, X)
    return True

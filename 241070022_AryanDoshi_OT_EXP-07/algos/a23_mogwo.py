"""
Algorithm 23 -- MOGWO, Multi-Objective Grey Wolf Optimizer (Swarm, Part 4; Mirjalili et al. 2016)
Lecture steps per iteration
  1. Archive of non-dominated solutions (grid, bounded).
  2. Leaders alpha, beta, delta = three different archive members chosen by roulette on the grid
     (less crowded cells favoured).
  3. For every wolf:  X_k = L_k - A_k |C_k L_k - X|  (k = alpha, beta, delta),  A = 2a r1 - a, C = 2 r2,
     a decreases linearly from 2 to 0,   X <- (X_1 + X_2 + X_3)/3  (encircling and hunting).
  4. Evaluate all wolves, add to the archive, remove members of the most crowded cell if it is full.
Binary problems (P9): positions are real numbers in [0,1] and are thresholded at 0.5.
"""
import numpy as np
from moo_base import MOAlgorithm, Archive, grid_leader, progress


def gwo_move(X, leaders, a, rng):
    """leaders: array (3, n) alpha, beta, delta. X: (N, n)."""
    Xs = []
    for L in leaders:
        A = 2 * a * rng.random(X.shape) - a
        C = 2 * rng.random(X.shape)
        Xs.append(L - A * np.abs(C * L - X))
    return sum(Xs) / 3.0


class MOGWO(MOAlgorithm):
    number, name, family = 23, "MOGWO", "Swarm (Part 4)"

    def __init__(self, problem, pop_size=100, seed=0, ndiv=10, verbose=False, **params):
        super().__init__(problem, pop_size, seed, verbose, **params)
        self.ndiv = ndiv
        self.archive = Archive(pop_size, mode="grid", ndiv=ndiv)

    def initialize(self):
        super().initialize()
        self.archive.add(self.X, self.F, self.CV, self.rng)

    def iterate(self):
        a = 2.0 * (1 - progress(self))
        idx = grid_leader(self.archive.F, self.rng, self.ndiv, power=2.0, k=3) if len(self.archive) >= 3 else \
            list(self.rng.integers(0, len(self.archive), 3))
        leaders = self.archive.X[idx]
        leaderF = self.archive.F[idx]
        Xn = gwo_move(self.X.astype(float), leaders, a, self.rng)
        Xn = np.clip(Xn, self.problem.lb, self.problem.ub)
        self.X, self.F, self.CV = self.evaluate(Xn)
        self.archive.add(self.X, self.F, self.CV, self.rng)
        self.log("a (2 -> 0)", a)
        self.log("leader objective vectors", leaderF)


def unit_test():
    """Key rule: when |A|<1 wolves step towards the leaders; at a=0 every wolf sits exactly on the leaders' mean."""
    rng = np.random.default_rng(0)
    X = rng.random((5, 4))
    leaders = np.array([[0.1] * 4, [0.2] * 4, [0.3] * 4])
    Xn = gwo_move(X, leaders, 0.0, rng)
    assert np.allclose(Xn, 0.2), "a=0 -> A=0 -> X_k = L_k -> mean of the leaders"
    Xn = gwo_move(X, leaders, 2.0, rng)
    assert Xn.shape == X.shape
    return True

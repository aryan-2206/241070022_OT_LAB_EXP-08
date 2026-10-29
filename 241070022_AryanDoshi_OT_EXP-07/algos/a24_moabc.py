"""
Algorithm 24 -- MOABC, Multi-Objective Artificial Bee Colony (Swarm, Part 4)
Lecture steps per cycle
  1. Employed bees: every food source x_i is perturbed in one random dimension j:
        v_ij = x_ij + phi (x_ij - x_kj),  phi ~ U(-1,1), k != i random;  GREEDY choice by dominance
        (v replaces x_i if v dominates x_i; if neither dominates the other, with probability 1/2).
  2. Onlooker bees: choose sources with probability proportional to a Pareto fitness 1/(1+rank), then repeat step 1.
  3. Scout bees: a source that has not improved for `limit` trials is replaced by a random solution.
  4. An external archive stores all non-dominated solutions found.
"""
import numpy as np
from moo_base import MOAlgorithm, Archive, row_dominates
from moo_utils import fast_non_dominated_sort


def abc_neighbour(X, i_idx, k_idx, j_idx, phi):
    V = X[i_idx].astype(float).copy()
    rows = np.arange(len(i_idx))
    V[rows, j_idx] = X[i_idx, j_idx] + phi * (X[i_idx, j_idx] - X[k_idx, j_idx])
    return V


class MOABC(MOAlgorithm):
    number, name, family = 24, "MOABC", "Swarm (Part 4)"

    def __init__(self, problem, pop_size=100, seed=0, limit=None, verbose=False, **params):
        super().__init__(problem, max(4, pop_size // 2), seed, verbose, **params)
        self.limit = limit or max(20, self.N)
        self.archive = Archive(pop_size, mode="crowding")

    def initialize(self):
        super().initialize()
        self.trial = np.zeros(self.N, int)
        self.archive.add(self.X, self.F, self.CV, self.rng)

    def search(self, src):
        n = self.problem.n_var
        k = (src + 1 + self.rng.integers(0, self.N - 1, len(src))) % self.N
        j = self.rng.integers(0, n, len(src))
        if self.problem.vtype == "perm":
            V = self.X[src].copy()
            for r in range(len(src)):
                a, b = self.rng.choice(n, 2, replace=False)
                V[r, a], V[r, b] = V[r, b], V[r, a]
        else:
            V = abc_neighbour(self.X, src, k, j, self.rng.uniform(-1, 1, len(src)))
        Xv, Fv, CVv = self.evaluate(V)
        better = row_dominates(Fv, CVv, self.F[src], self.CV[src])
        tie = ~better & ~row_dominates(self.F[src], self.CV[src], Fv, CVv) & (self.rng.random(len(src)) < 0.5)
        take = better | tie
        s = src[take]
        self.X[s], self.F[s], self.CV[s] = Xv[take], Fv[take], CVv[take]
        for t, ok in zip(src, take):                      # trial counters (one update per distinct source)
            self.trial[t] = 0 if ok else self.trial[t] + 1
        self.archive.add(Xv, Fv, CVv, self.rng)

    def iterate(self):
        self.search(np.arange(self.N))                                        # employed bees
        _, rank = fast_non_dominated_sort(self.F, self.CV)
        p = 1.0 / (1.0 + rank)
        src = self.rng.choice(self.N, self.N, p=p / p.sum())                  # onlookers
        self.search(src)
        for i in np.where(self.trial > self.limit)[0]:                        # scouts
            x = self.problem.random(1, self.rng)
            Xs, Fs, CVs = self.evaluate(x)
            self.X[i], self.F[i], self.CV[i], self.trial[i] = Xs[0], Fs[0], CVs[0], 0
            self.archive.add(Xs, Fs, CVs, self.rng)
        self.log("max trial counter", self.trial.max())


def unit_test():
    """Key rule: v_ij = x_ij + phi (x_ij - x_kj) changes exactly one dimension; phi=0 leaves the source unchanged."""
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    V = abc_neighbour(X, np.array([0]), np.array([1]), np.array([2]), np.array([0.5]))
    assert np.allclose(V, [[1.0, 2.0, 3.0 + 0.5 * (3.0 - 6.0)]])
    V0 = abc_neighbour(X, np.array([0]), np.array([1]), np.array([1]), np.array([0.0]))
    assert np.allclose(V0, X[:1])
    return True

"""
Algorithm 27 -- MODE, Multi-Objective Differential Evolution (Evolutionary, Part 4)
Lecture steps per generation
  1. For every target x_i: mutant v = x_r1 + F (x_r2 - x_r3) (DE/rand/1), binomial crossover with rate CR -> trial u.
  2. Selection: the trial REPLACES the target only if it dominates it (constrained dominance).
     A trial dominated by the target, or non-dominated w.r.t. it, is discarded.
  3. All feasible non-dominated points ever evaluated are kept in an external archive (the reported set).
Real / integer variables only.
"""
import numpy as np
from moo_base import MOAlgorithm, Archive, row_dominates
from algos.a19_moead_de import de_rand_1


class MODE(MOAlgorithm):
    number, name, family = 27, "MODE", "Evolutionary (Part 4)"

    def __init__(self, problem, pop_size=100, seed=0, F=0.5, CR=0.9, verbose=False, **params):
        super().__init__(problem, pop_size, seed, verbose, **params)
        self.Fm, self.CR = F, CR
        self.archive = Archive(pop_size, mode="crowding")

    def initialize(self):
        super().initialize()
        self.archive.add(self.X, self.F, self.CV, self.rng)

    def iterate(self):
        N = self.N
        r = np.array([self.rng.choice(N, 3, replace=False) for _ in range(N)]).T
        U = de_rand_1(self.X.astype(float), r[0], r[1], r[2], self.Fm, self.CR, self.problem.lb, self.problem.ub, self.rng)
        # binomial crossover with the TARGET vector
        cross = self.rng.random(U.shape) < self.CR
        cross[np.arange(N), self.rng.integers(0, U.shape[1], N)] = True
        trial = np.where(cross, U, self.X)
        Xt, Ft, CVt = self.evaluate(trial)
        win = row_dominates(Ft, CVt, self.F, self.CV)
        self.X[win], self.F[win], self.CV[win] = Xt[win], Ft[win], CVt[win]
        self.archive.add(Xt, Ft, CVt, self.rng)
        self.log("replacements", int(win.sum()))


def unit_test():
    """Key rule: the trial only replaces the target when it dominates it."""
    Fa = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    Ft = np.array([[0.5, 0.5], [2.0, 2.0], [0.5, 2.0]])           # dominates, dominated, incomparable
    z = np.zeros(3)
    assert list(row_dominates(Ft, z, Fa, z)) == [True, False, False]
    return True

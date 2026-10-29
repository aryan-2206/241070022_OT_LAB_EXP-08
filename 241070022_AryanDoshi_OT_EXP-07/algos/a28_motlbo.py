"""
Algorithm 28 -- MO-TLBO, Multi-Objective Teaching-Learning-Based Optimisation (Human-inspired, Part 4)
Lecture steps per generation
  1. TEACHER PHASE: teacher T = random member of the non-dominated archive, class mean M;
        x_new = x + r (T - T_F M),   T_F = teaching factor that SHRINKS from 2 to 1 during the run.
     Greedy dominance acceptance (new dominates old, or incomparable with probability 1/2).
  2. LEARNER PHASE: learner i meets random learner j:  if j is better (dominates) move towards j,
        x_new = x_i + r (x_j - x_i), otherwise away from j,  x_new = x_i + r (x_i - x_j); same acceptance.
  3. Non-dominated archive collects results.
"""
import numpy as np
from moo_base import MOAlgorithm, Archive, row_dominates, progress


def teacher_phase(X, teacher, TF, r, lb, ub):
    M = X.mean(axis=0)
    return np.clip(X + r * (teacher - TF * M), lb, ub)


class MOTLBO(MOAlgorithm):
    number, name, family = 28, "MO-TLBO", "Human-inspired (Part 4)"

    def __init__(self, problem, pop_size=100, seed=0, verbose=False, **params):
        super().__init__(problem, max(4, pop_size // 2), seed, verbose, **params)
        self.archive = Archive(pop_size, mode="crowding")

    def initialize(self):
        super().initialize()
        self.archive.add(self.X, self.F, self.CV, self.rng)

    def accept(self, Xn):
        Xn, Fn, CVn = self.evaluate(Xn)
        better = row_dominates(Fn, CVn, self.F, self.CV)
        tie = ~better & ~row_dominates(self.F, self.CV, Fn, CVn) & (self.rng.random(len(Xn)) < 0.5)
        take = better | tie
        self.X[take], self.F[take], self.CV[take] = Xn[take], Fn[take], CVn[take]
        self.archive.add(Xn, Fn, CVn, self.rng)

    def iterate(self):
        N, lb, ub = self.N, self.problem.lb, self.problem.ub
        TF = 2.0 - progress(self)                                   # shrinking teaching factor 2 -> 1
        teacher = self.archive.X[self.rng.integers(0, len(self.archive), N)].astype(float)
        r = self.rng.random((N, 1))
        self.accept(teacher_phase(self.X.astype(float), teacher, TF, r, lb, ub))
        j = (np.arange(N) + 1 + self.rng.integers(0, N - 1, N)) % N
        towards = row_dominates(self.F[j], self.CV[j], self.F, self.CV)
        away = row_dominates(self.F, self.CV, self.F[j], self.CV[j])
        sign = np.where(towards, 1.0, np.where(away, -1.0, np.where(self.rng.random(N) < 0.5, 1.0, -1.0)))[:, None]
        Xn = np.clip(self.X + self.rng.random((N, 1)) * sign * (self.X[j] - self.X), lb, ub)
        self.accept(Xn)
        self.log("teaching factor", TF)


def unit_test():
    """Key rule: x_new = x + r (T - TF M). If the class mean M equals the teacher and TF=1 nothing moves;
    with r=1 and TF=1 a learner jumps by (T - M)."""
    X = np.array([[0.0, 0.0], [2.0, 2.0]])
    M = X.mean(axis=0)
    T = M.copy()
    assert np.allclose(teacher_phase(X, T, 1.0, 1.0, -9, 9), X)
    T2 = np.array([3.0, 3.0])
    out = teacher_phase(X, T2, 1.0, 1.0, -9, 9)
    assert np.allclose(out, X + (T2 - M))
    return True

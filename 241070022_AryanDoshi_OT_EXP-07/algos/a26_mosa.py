"""
Algorithm 26 -- MOSA, Multi-Objective Simulated Annealing, archive based (Physics-inspired, Part 4; AMOSA-style)
Lecture steps
  1. Current solution x_c, archive of non-dominated solutions, temperature T (geometric cooling over the budget).
  2. Neighbour move x_n = N(x_c) (Gaussian / swap / bit-flip).
  3. Domination amount  dd(a,b) = prod_{i: f_i(a)!=f_i(b)} |f_i(a)-f_i(b)|/R_i  (objectives scaled by R).
  4. Boltzmann acceptance: if x_n is dominated (by x_c or by archive members) accept with probability
        exp(-avg dd / T);  if x_n dominates x_c or is non-dominated w.r.t. the archive, accept.
  5. Accepted non-dominated solutions enter the archive (dominated members are removed, size is bounded).
"""
import numpy as np
from moo_base import MOAlgorithm, Archive, neighbour, progress, penalised


def dom_amount(fa, fb, R):
    d = np.abs(fa - fb) / R
    d = d[d > 0]
    return float(np.prod(d)) if len(d) else 0.0


class MOSA(MOAlgorithm):
    number, name, family = 26, "MOSA", "Physics-inspired (Part 4)"

    def __init__(self, problem, pop_size=100, seed=0, T0=0.01, Tmin=1e-5, verbose=False, **params):
        super().__init__(problem, pop_size, seed, verbose, **params)
        self.T0, self.Tmin = T0, Tmin
        self.archive = Archive(pop_size, mode="crowding")

    def initialize(self):
        n0 = 20
        X = self.problem.random(n0, self.rng)
        self.X, self.F, self.CV = self.evaluate(X)
        self.scale = np.maximum(np.ptp(self.F, axis=0), 1e-12)
        Fp = penalised(self.F, self.CV, self.scale)
        self.archive.add(self.X, Fp, self.CV * 0, self.rng)      # archive works on penalised values
        self.archive.F = self.archive.F.copy()
        i = int(np.argmin(Fp.sum(axis=1)))
        self.xc, self.fc = self.X[i].copy(), Fp[i].copy()
        self.raw = {}

    def iterate(self):
        T = self.T0 * (self.Tmin / self.T0) ** progress(self)
        xn = neighbour(self.problem, self.xc, self.rng)
        Xn, Fn, CVn = self.evaluate(xn)
        fn = penalised(Fn, CVn, self.scale)[0]
        AF = self.archive.F
        R = np.maximum(np.ptp(np.vstack([AF, fn[None]]), axis=0), 1e-12)
        dom = lambda a, B: (B <= a).all(axis=1) & (B < a).any(axis=1)
        dominated_by = np.vstack([AF[dom(fn, AF)], self.fc[None] if (self.fc <= fn).all() and (self.fc < fn).any() else np.zeros((0, len(fn)))])
        new_dom_cur = (fn <= self.fc).all() and (fn < self.fc).any()
        accept = False
        if len(dominated_by) > 0:                                   # new point is dominated -> Boltzmann
            avg = np.mean([dom_amount(b, fn, R) for b in dominated_by])
            accept = self.rng.random() < np.exp(-avg / T)
        else:                                                       # non-dominated w.r.t. archive and current
            accept = True
        if accept:
            self.xc, self.fc = Xn[0].copy(), fn.copy()
        if len(dominated_by) == 0 or new_dom_cur:
            self.archive.add(Xn, fn[None], np.zeros(1), self.rng)
        self.log("T", T)

    def solution_set(self):
        # the BOUNDED archive, reported with its true (unpenalised) objective values
        X = self.archive.X
        F, CV = self.problem.evaluate(X)
        return X, F, CV


def unit_test():
    """Key rules: domination amount formula; at very low T a dominated move is (almost) never accepted,
    at very high T it is almost always accepted."""
    R = np.array([1.0, 1.0])
    assert abs(dom_amount(np.array([0.0, 0.0]), np.array([0.5, 0.2]), R) - 0.1) < 1e-12
    assert dom_amount(np.array([0.0, 0.0]), np.array([0.5, 0.0]), R) == 0.5      # equal coordinate ignored
    rng = np.random.default_rng(0)
    dd = 0.1
    low = np.mean([rng.random() < np.exp(-dd / 1e-3) for _ in range(2000)])
    high = np.mean([rng.random() < np.exp(-dd / 1e3) for _ in range(2000)])
    assert low < 0.01 and high > 0.99
    return True

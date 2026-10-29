"""
Algorithm 22 -- MOPSO, Multi-Objective Particle Swarm (Swarm, Part 4; Coello, Pulido, Lechuga 2004)
Lecture steps per iteration
  1. Every particle has position x, velocity v and personal best p (updated by dominance).
  2. The GLOBAL guide is chosen from an external archive of non-dominated solutions: grid cells are picked by
     roulette with weight 1/count (sparse regions first), then a random member of the cell.
  3. v <- w v + c1 r1 (p - x) + c2 r2 (g - x);  x <- x + v  (bounce at the bounds).
  4. Turbulence (mutation) with a probability that shrinks during the run.
  5. Evaluate; update personal bests (dominance, tie -> random); add to the archive (bounded, grid).
"""
import numpy as np
from moo_base import MOAlgorithm, Archive, grid_roulette, poly_mut, row_dominates, progress


class MOPSO(MOAlgorithm):
    number, name, family = 22, "MOPSO", "Swarm (Part 4)"

    def __init__(self, problem, pop_size=100, seed=0, w=0.4, c1=1.5, c2=1.5, ndiv=10, verbose=False, **params):
        super().__init__(problem, pop_size, seed, verbose, **params)
        self.w, self.c1, self.c2, self.ndiv = w, c1, c2, ndiv
        self.archive = Archive(pop_size, mode="grid", ndiv=ndiv)

    def initialize(self):
        super().initialize()
        span = self.problem.ub - self.problem.lb
        self.V = np.zeros_like(self.X, dtype=float) if self.problem.vtype == "perm" else \
            self.rng.uniform(-0.1, 0.1, self.X.shape) * span
        self.PX, self.PF, self.PCV = self.X.copy(), self.F.copy(), self.CV.copy()
        self.archive.add(self.X, self.F, self.CV, self.rng)

    def move(self, X, V, PX, G):
        r1, r2 = self.rng.random(X.shape), self.rng.random(X.shape)
        V = self.w * V + self.c1 * r1 * (PX - X) + self.c2 * r2 * (G - X)
        Xn = X + V
        lb, ub = self.problem.lb, self.problem.ub
        out = (Xn < lb) | (Xn > ub)
        V = np.where(out, -V, V)
        return np.clip(Xn, lb, ub), V

    def iterate(self):
        lead = grid_roulette(self.archive.F, self.rng, self.N, self.ndiv)
        G = self.archive.X[lead]
        Xn, self.V = self.move(self.X, self.V, self.PX, G)
        pmut = (1 - progress(self)) ** 2 * 0.5                          # turbulence shrinks with time
        mut = self.rng.random(self.N) < pmut
        if mut.any():
            Xn[mut] = poly_mut(Xn[mut], self.problem.lb, self.problem.ub, self.rng, pm=1.0 / self.problem.n_var + 0.1)
        self.X, self.F, self.CV = self.evaluate(Xn)
        better = row_dominates(self.F, self.CV, self.PF, self.PCV)
        tie = ~better & ~row_dominates(self.PF, self.PCV, self.F, self.CV) & (self.rng.random(self.N) < 0.5)
        upd = better | tie
        self.PX[upd], self.PF[upd], self.PCV[upd] = self.X[upd], self.F[upd], self.CV[upd]
        self.archive.add(self.X, self.F, self.CV, self.rng)
        self.log("archive size", len(self.archive))


def unit_test():
    """Key rule: v' = w v + c1 r1 (p-x) + c2 r2 (g-x). With w=0, c1=0, c2=1, r2=1 the particle moves onto g."""
    from moo_problems import get_problem
    a = MOPSO(get_problem("P3"), pop_size=10, w=0.0, c1=0.0, c2=1.0, seed=0)
    a.rng = type("R", (), {"random": staticmethod(lambda s: np.ones(s))})()
    X = np.full((1, 10), 0.2); G = np.full((1, 10), 0.8)
    Xn, V = a.move(X, np.zeros((1, 10)), X, G)
    assert np.allclose(Xn, G) and np.allclose(V, 0.6)
    # velocity inertia: v' = w v
    a.w, a.c2 = 0.5, 0.0
    Xn, V = a.move(X, np.full((1, 10), 0.1), X, G)
    assert np.allclose(V, 0.05)
    return True

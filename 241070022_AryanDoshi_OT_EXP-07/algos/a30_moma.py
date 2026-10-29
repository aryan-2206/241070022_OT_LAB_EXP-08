"""
Algorithm 30 -- MOMA, Multi-Objective Memetic Algorithm (Hybrid, Part 4)
Global search = NSGA-II generation; local search = hill climbing on a few non-dominated offspring.
Lecture steps per generation
  1. NSGA-II reproduction: crowded tournament, crossover, mutation -> offspring Q.
  2. Local search (Lamarckian): choose n_ls non-dominated members of Q; for ls_steps rounds make a small
     neighbour move (Gaussian / swap / bit flip) and keep it if it DOMINATES the current member.
  3. Survival: R = P u Q', fast non-dominated sort, crowding truncation.
"""
import numpy as np
from algos.a14_nsga2 import NSGA2
from moo_base import vary, tournament_by_rank, nsga2_select, neighbour, row_dominates
from moo_utils import non_dominated_mask


class MOMA(NSGA2):
    number, name, family = 30, "MOMA", "Hybrid (Part 4)"
    use_cv = True

    def __init__(self, problem, pop_size=100, seed=0, n_ls=5, ls_steps=3, verbose=False, **params):
        super().__init__(problem, pop_size, seed, verbose, **params)
        self.n_ls, self.ls_steps = n_ls, ls_steps

    def local_search(self, X, F, CV):
        nd = np.where(non_dominated_mask(F, CV))[0]
        pick = self.rng.choice(nd, min(self.n_ls, len(nd)), replace=False)
        improved = 0
        for _ in range(self.ls_steps):
            cand = np.array([neighbour(self.problem, X[i], self.rng) for i in pick])
            Xc, Fc, CVc = self.evaluate(cand)
            ok = row_dominates(Fc, CVc, F[pick], CV[pick])
            X[pick[ok]], F[pick[ok]], CV[pick[ok]] = Xc[ok], Fc[ok], CVc[ok]
            improved += int(ok.sum())
        return improved

    def iterate(self):
        par = tournament_by_rank(self.rng, self.N, self.rank, self.crowd)
        off = vary(self.problem, self.X[par], self.rng)
        Xo, Fo, CVo = self.evaluate(off)
        imp = self.local_search(Xo, Fo, CVo)
        X, F, CV = np.vstack([self.X, Xo]), np.vstack([self.F, Fo]), np.concatenate([self.CV, CVo])
        idx, self.rank, self.crowd = nsga2_select(F, CV, self.N)
        self.X, self.F, self.CV = X[idx], F[idx], CV[idx]
        self.log("local-search improvements", imp)


def unit_test():
    """Key rule: local search keeps a neighbour only if it DOMINATES the current member (never worsens)."""
    from moo_problems import get_problem
    a = MOMA(get_problem("P3"), pop_size=20, seed=0)
    a.budget = 10 ** 9
    a.initialize()
    X, F, CV = a.X.copy(), a.F.copy(), a.CV.copy()
    F0 = F.copy()
    a.local_search(X, F, CV)
    for i in range(len(F)):
        assert np.all(F[i] <= F0[i]), "a member can only be replaced by a dominating point"
    return True

"""
Algorithm 35 -- R-NSGA-II, Reference-point based preference NSGA-II (Preference-based, 5E; Deb, Sundar 2006)
Lecture steps per generation
  1. Decision maker (DM) gives reference point(s) r (in objective space).
  2. NSGA-II reproduction and non-dominated sorting of R = P u Q.
  3. Inside the last front crowding distance is replaced by the normalised Euclidean DISTANCE TO r
     (smaller distance = better, i.e. closer to the DM's wish).
  4. epsilon-clearing: walk through the front in order of distance, keep a solution and push every other
     solution closer than epsilon (normalised) to it to the back of the order -> keeps spread inside the region of interest.
  5. Tournament: rank, then preference distance.
"""
import numpy as np
from moo_base import MOAlgorithm, vary, tournament_by_rank, penalised
from moo_utils import fast_non_dominated_sort


def pref_order(Fn, ref_n, eps):
    """Order of a front for survival: closest to ref first with epsilon-clearing. Fn, ref_n normalised."""
    d = np.sqrt(((Fn[:, None, :] - ref_n[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    order = list(np.argsort(d, kind="stable"))
    first, later = [], []
    remaining = order[:]
    while remaining:
        i = remaining.pop(0)
        first.append(i)
        close = [j for j in remaining if np.linalg.norm(Fn[i] - Fn[j]) < eps]
        later += close
        remaining = [j for j in remaining if j not in close]
    return np.array(first + later), d


class RNSGA2(MOAlgorithm):
    number, name, family = 35, "R-NSGA-II", "Preference-based (5E)"

    def __init__(self, problem, pop_size=100, seed=0, ref_points=None, eps=0.002, verbose=False, **params):
        super().__init__(problem, pop_size, seed, verbose, **params)
        self.ref = None if ref_points is None else np.atleast_2d(np.asarray(ref_points, float))
        self.eps = eps

    def survive(self, X, F, CV):
        Fe = penalised(F, CV, self.scale)
        lo, hi = Fe.min(axis=0), Fe.max(axis=0)
        span = np.maximum(hi - lo, 1e-12)
        Fn, ref_n = (Fe - lo) / span, (self.ref - lo) / span
        fronts, rank = fast_non_dominated_sort(Fe)
        chosen, pdist = [], np.zeros(len(F))
        for fr in fronts:
            order, d = pref_order(Fn[fr], ref_n, self.eps)
            pdist[fr] = d
            if len(chosen) + len(fr) <= self.N:
                chosen += list(fr)
            else:
                chosen += list(fr[order[:self.N - len(chosen)]])
                break
        idx = np.array(chosen)
        return idx, rank[idx], pdist[idx]

    def initialize(self):
        super().initialize()
        if self.ref is None:
            self.ref = np.atleast_2d(self.F.min(axis=0))
        _, self.rank, self.pd = self.survive(self.X, self.F, self.CV)

    def iterate(self):
        par = tournament_by_rank(self.rng, self.N, self.rank, -self.pd)
        off = vary(self.problem, self.X[par], self.rng)
        Xo, Fo, CVo = self.evaluate(off)
        X, F, CV = np.vstack([self.X, Xo]), np.vstack([self.F, Fo]), np.concatenate([self.CV, CVo])
        idx, self.rank, self.pd = self.survive(X, F, CV)
        self.X, self.F, self.CV = X[idx], F[idx], CV[idx]
        self.log("min distance to reference", self.pd.min())


def unit_test():
    """Key rules: the front member closest to the reference point is ranked first; epsilon-clearing pushes
    near-duplicates of an already chosen point behind the others."""
    Fn = np.array([[0.0, 1.0], [0.1, 0.9], [0.101, 0.899], [1.0, 0.0]])
    ref = np.array([[0.1, 0.9]])
    order, d = pref_order(Fn, ref, eps=0.05)
    assert order[0] == 1, "point 1 is exactly at the reference"
    assert list(order).index(2) > list(order).index(3), "point 2 is within eps of point 1 -> cleared (pushed back)"
    return True

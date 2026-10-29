"""
Algorithm 14 -- NSGA-II (Elitist, Part 3; Deb, Pratap, Agarwal, Meyarivan 2002)
Lecture steps per generation
  1. Binary tournament on the crowded-comparison operator (lower rank, then larger crowding distance).
  2. Crossover + mutation -> offspring Q (|Q| = N).
  3. Merge R = P u Q, fast non-dominated sort into fronts.
  4. Fill the next population front by front; truncate the last front by crowding distance (elitism).
Constraints: static penalty (the dedicated Deb-rules version is algorithm 21).
"""
import numpy as np
from moo_base import MOAlgorithm, vary, tournament_by_rank, nsga2_select, penalised
from moo_utils import crowding_distance, fast_non_dominated_sort


class NSGA2(MOAlgorithm):
    number, name, family = 14, "NSGA-II", "Elitist (Part 3)"
    use_cv = False

    def _eff(self, F, CV):
        return F if self.use_cv else penalised(F, CV, self.scale)

    def _cv(self, CV):
        return CV if self.use_cv else None

    def initialize(self):
        super().initialize()
        _, self.rank, self.crowd = nsga2_select(self._eff(self.F, self.CV), self._cv(self.CV), self.N)

    def iterate(self):
        par = tournament_by_rank(self.rng, self.N, self.rank, self.crowd)
        off = vary(self.problem, self.X[par], self.rng)
        Xo, Fo, CVo = self.evaluate(off)
        X, F, CV = np.vstack([self.X, Xo]), np.vstack([self.F, Fo]), np.concatenate([self.CV, CVo])
        idx, self.rank, self.crowd = nsga2_select(self._eff(F, CV), self._cv(CV), self.N)
        self.log("fronts sizes", np.bincount(self.rank))
        self.X, self.F, self.CV = X[idx], F[idx], CV[idx]


def unit_test():
    """Key rule: when the last front must be truncated the boundary points (infinite crowding) survive,
    and non-dominated individuals are never lost (elitism)."""
    F = np.array([[0, 4], [1, 3], [1.1, 2.9], [4, 0], [5, 5], [6, 6]], float)    # front 1 = first 4 points
    idx, rank, crowd = nsga2_select(F, None, 3)
    assert set(idx) == {0, 3, 1} or set(idx) == {0, 3, 2}, "boundary points must survive"
    idx, rank, crowd = nsga2_select(F, None, 4)
    assert set(idx) == {0, 1, 2, 3}, "whole first front survives when N=4"
    from moo_problems import get_problem
    a = NSGA2(get_problem("P3"), pop_size=40, seed=1)
    a.run(4000, record=True)
    ideal = np.array([f.min(axis=0) for _, f in a.history if len(f)])
    assert np.all(np.diff(ideal, axis=0) <= 1e-12), "elitism: the best value of every objective never gets worse"
    return True

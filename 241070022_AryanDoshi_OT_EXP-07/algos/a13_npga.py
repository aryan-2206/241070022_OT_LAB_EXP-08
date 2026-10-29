"""
Algorithm 13 -- NPGA, Niched-Pareto GA of Horn, Nafpliotis & Goldberg (Non-elitist, Part 2; 1994)
Lecture steps for every selected parent
  1. Draw two candidate individuals and a comparison set of t_dom random individuals.
  2. A candidate is DOMINATED if some member of the comparison set dominates it.
  3. If exactly one candidate is non-dominated it wins (Pareto-domination tournament).
  4. Tie (both dominated or both not): the candidate in the less crowded niche (smaller niche count) wins.
Then crossover, mutation, generational replacement.
Constraints: constrained dominance in the tournament.
"""
import numpy as np
from moo_base import MOAlgorithm, vary, sharing_counts
from moo_utils import dominance_matrix


def npga_compare(D, niche, a, b, comp):
    """Winner of candidates a, b against the comparison set `comp` (D[i,j] = i dominates j)."""
    a_dom, b_dom = D[comp, a].any(), D[comp, b].any()
    if a_dom != b_dom:
        return b if a_dom else a          # the non-dominated candidate wins
    return a if niche[a] <= niche[b] else b   # tie -> smaller niche count


def npga_tournament(D, niche, rng, n, t_dom=10):
    """Returns the n winners of n independent NPGA tournaments."""
    N = len(niche)
    win = np.empty(n, int)
    for s in range(n):
        a, b = rng.integers(0, N, 2)
        comp = rng.integers(0, N, t_dom)
        win[s] = npga_compare(D, niche, a, b, comp)
    return win


class NPGA(MOAlgorithm):
    number, name, family = 13, "NPGA", "Non-elitist (Part 2)"

    def iterate(self):
        D = dominance_matrix(self.F, self.CV)
        niche = sharing_counts(self.F, self.params.get("sigma", 0.05))
        par = npga_tournament(D, niche, self.rng, self.N, self.params.get("t_dom", 10))
        self.log("comparison-set size", self.params.get("t_dom", 10))
        off = vary(self.problem, self.X[par], self.rng)
        self.X, self.F, self.CV = self.evaluate(off)


def unit_test():
    """Key rule: a non-dominated candidate beats a dominated one; ties are broken by the smaller niche count."""
    F = np.array([[0, 0], [5, 5], [1, 4], [4, 1]], float)
    D = dominance_matrix(F)
    niche = np.array([1.0, 1.0, 3.0, 1.0])
    comp = np.array([0, 1, 2, 3])                       # comparison set = whole population
    assert npga_compare(D, niche, 0, 1, comp) == 0 and npga_compare(D, niche, 1, 0, comp) == 0
    # candidates 2 and 3 are both dominated by point 0 -> tie -> niche count decides (3 has the smaller one)
    assert npga_compare(D, niche, 2, 3, comp) == 3 and npga_compare(D, niche, 3, 2, comp) == 3
    # empty comparison dominance: both non-dominated -> tie -> niche count
    D0 = np.zeros((4, 4), bool)
    assert npga_compare(D0, niche, 2, 0, np.array([1])) == 0
    return True

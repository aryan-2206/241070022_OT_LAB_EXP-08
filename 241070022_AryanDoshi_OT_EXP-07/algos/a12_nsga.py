"""
Algorithm 12 -- NSGA, Non-dominated Sorting GA of Srinivas & Deb (Non-elitist, Part 2; 1994)
Lecture steps per generation
  1. Sort the population into non-dominated fronts F1, F2, ...
  2. Front 1 gets a large dummy fitness (= N); sharing INSIDE the front divides it by the niche count.
  3. The next front gets dummy fitness slightly below the minimum SHARED fitness of the previous front,
     so that no member of a lower front can beat a member of a higher front.
  4. Binary tournament on shared fitness, crossover, mutation, generational replacement.
Constraints: static penalty on the objectives.
"""
import numpy as np
from moo_base import MOAlgorithm, vary, tournament_by_fitness, sharing_counts, penalised
from moo_utils import fast_non_dominated_sort


def nsga_fitness(F, sigma=0.1, eps=1e-3):
    N = len(F)
    fronts, rank = fast_non_dominated_sort(F)
    fit = np.zeros(N)
    dummy = float(N)
    for fr in fronts:
        m_i = sharing_counts(F[fr], sigma)
        fit[fr] = dummy / m_i
        dummy = fit[fr].min() - eps
    return rank, fit


class NSGA(MOAlgorithm):
    number, name, family = 12, "NSGA", "Non-elitist (Part 2)"

    def iterate(self):
        Fp = penalised(self.F, self.CV, self.scale)
        rank, fit = nsga_fitness(Fp, self.params.get("sigma", 0.1))
        self.log("front rank", rank[:8])
        self.log("shared dummy fitness", fit[:8])
        par = tournament_by_fitness(self.rng, self.N, fit)
        off = vary(self.problem, self.X[par], self.rng)
        self.X, self.F, self.CV = self.evaluate(off)


def unit_test():
    """Key rule: every member of front k+1 has strictly smaller (shared) fitness than every member of front k."""
    rng = np.random.default_rng(0)
    F = rng.random((40, 2))
    rank, fit = nsga_fitness(F)
    for k in range(rank.max()):
        assert fit[rank == k].min() > fit[rank == k + 1].max(), "dummy fitness ordering violated"
    # crowded points share fitness
    F2 = np.array([[0, 1], [0.01, 0.99], [1, 0]], float)
    _, f2 = nsga_fitness(F2, sigma=0.2)
    assert f2[2] > f2[0] and f2[2] > f2[1]
    return True

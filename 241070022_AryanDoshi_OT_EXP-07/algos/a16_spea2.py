"""
Algorithm 16 -- SPEA2 (Elitist, Part 3; Zitzler, Laumanns, Thiele 2001)
Lecture steps per generation (population P, archive A, sizes N and N_a)
  1. Strength S(i) = number of solutions in P u A that i dominates.
  2. Raw fitness R(i) = sum of S(j) over all j that dominate i (R = 0  <=>  non-dominated).
  3. Density D(i) = 1/(sigma_i^k + 2), sigma_i^k = distance to the k-th nearest neighbour, k = sqrt(|P|+|A|).
  4. Fitness F(i) = R(i) + D(i)  (F < 1 for non-dominated).
  5. Environmental selection: archive = all F < 1; if too small fill with the best dominated, if too large
     truncate repeatedly the individual with the smallest distance to its nearest neighbour.
  6. Binary tournament on F in the archive -> crossover/mutation -> new population.
"""
import numpy as np
from moo_base import MOAlgorithm, vary, tournament_by_fitness
from moo_utils import dominance_matrix, normalize


def spea2_fitness(F, CV=None):
    n = len(F)
    D = dominance_matrix(F, CV)
    S = D.sum(axis=1)
    R = (D * S[:, None]).sum(axis=0)
    Fn = normalize(F)
    dist = np.sqrt(((Fn[:, None, :] - Fn[None, :, :]) ** 2).sum(axis=2))
    k = max(1, int(np.sqrt(n)))
    sk = np.sort(dist, axis=1)[:, min(k, n - 1)]
    dens = 1.0 / (sk + 2.0)
    return S, R, dens, R + dens


def truncate(F, n_keep):
    """SPEA2 truncation: remove the point with the lexicographically smallest sorted-distance vector."""
    Fn = normalize(F)
    dist = np.sqrt(((Fn[:, None, :] - Fn[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    alive = np.arange(len(F))
    while len(alive) > n_keep:
        sub = np.sort(dist[np.ix_(alive, alive)], axis=1)
        order = np.lexsort(sub.T[::-1])
        alive = np.delete(alive, order[0])
    return alive


class SPEA2(MOAlgorithm):
    number, name, family = 16, "SPEA2", "Elitist (Part 3)"

    def __init__(self, problem, pop_size=100, seed=0, archive_size=None, verbose=False, **params):
        super().__init__(problem, pop_size, seed, verbose, **params)
        self.Na = archive_size or pop_size

    def initialize(self):
        super().initialize()
        self.AX, self.AF, self.ACV = self.X[:0], self.F[:0], self.CV[:0]
        self.environmental_selection()

    def environmental_selection(self):
        X, F, CV = np.vstack([self.AX, self.X]), np.vstack([self.AF, self.F]), np.concatenate([self.ACV, self.CV])
        S, R, dens, fit = spea2_fitness(F, CV)
        nd = np.where(fit < 1)[0]
        if len(nd) < self.Na:
            rest = np.where(fit >= 1)[0]
            nd = np.concatenate([nd, rest[np.argsort(fit[rest])][:self.Na - len(nd)]])
        elif len(nd) > self.Na:
            nd = nd[truncate(F[nd], self.Na)]
        self.AX, self.AF, self.ACV, self.Afit = X[nd], F[nd], CV[nd], fit[nd]

    def iterate(self):
        par = tournament_by_fitness(self.rng, self.N, -self.Afit)
        off = vary(self.problem, self.AX[par], self.rng)
        self.X, self.F, self.CV = self.evaluate(off)
        self.environmental_selection()
        self.log("archive fitness (first 8)", self.Afit[:8])

    def solution_set(self):
        return self.AX, self.AF, self.ACV


def unit_test():
    """Hand example: F = (1,3),(2,2),(3,1),(3,3): point 3 is dominated by the other three."""
    F = np.array([[1, 3], [2, 2], [3, 1], [3, 3]], float)
    S, R, dens, fit = spea2_fitness(F)
    assert list(S) == [1, 1, 1, 0], "strengths: points 0,1,2 dominate (3,3) only"
    assert list(R) == [0, 0, 0, 3], "raw fitness of the dominated point = 1+1+1"
    assert np.all(fit[:3] < 1) and fit[3] > 1, "F<1 <=> non-dominated"
    keep = truncate(np.array([[0, 1], [0.01, 0.99], [1, 0]], float), 2)
    assert 2 in keep, "truncation removes one of the two close points, keeps the far one"
    return True

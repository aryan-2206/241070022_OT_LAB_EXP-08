"""
Algorithm 11 -- MOGA, Multi-Objective GA of Fonseca & Fleming (Non-elitist, Part 2; 1993)
Lecture steps per generation
  1. Rank:  r_i = 1 + (number of individuals that dominate i).
  2. Raw fitness: sort by rank, assign fitness N ... 1 by linear interpolation, average inside equal ranks.
  3. Fitness sharing in objective space: niche count m_i = sum_j sh(d_ij);  F'_i = F_i / m_i,
     rescaled so that the mean fitness of every rank group is preserved.
  4. Binary tournament on shared fitness, crossover, mutation, generational replacement.
Constraints: constrained dominance inside the rank computation.
"""
import numpy as np
from moo_base import MOAlgorithm, vary, tournament_by_fitness, sharing_counts
from moo_utils import dominance_matrix


def moga_fitness(F, CV, sigma=0.05, alpha=1.0):
    N = len(F)
    D = dominance_matrix(F, CV)
    rank = 1 + D.sum(axis=0)                           # 1 + number of dominators
    order = np.argsort(rank, kind="stable")
    raw_sorted = np.linspace(N, 1, N)
    raw = np.empty(N)
    raw[order] = raw_sorted
    for r in np.unique(rank):                          # average raw fitness inside a rank group
        m = rank == r
        raw[m] = raw[m].mean()
    m_i = sharing_counts(F, sigma, alpha, groups=rank)
    shared = raw / m_i
    for r in np.unique(rank):                          # keep group mean (Fonseca-Fleming scaling)
        m = rank == r
        shared[m] *= raw[m].mean() / shared[m].mean()
    return rank, raw, m_i, shared


class MOGA(MOAlgorithm):
    number, name, family = 11, "MOGA", "Non-elitist (Part 2)"

    def iterate(self):
        sigma = self.params.get("sigma", 0.05)
        rank, raw, m_i, shared = moga_fitness(self.F, self.CV, sigma)
        self.log("rank", rank[:8])
        self.log("shared fitness", shared[:8])
        par = tournament_by_fitness(self.rng, self.N, shared)
        off = vary(self.problem, self.X[par], self.rng)
        self.X, self.F, self.CV = self.evaluate(off)


def unit_test():
    """Key rule: rank = 1 + #dominators; sharing lowers the fitness of crowded individuals inside a rank."""
    F = np.array([[1, 1], [2, 2], [3, 3], [0, 5], [0.01, 4.99], [5, 0]], float)
    rank, raw, m_i, shared = moga_fitness(F, np.zeros(6), sigma=0.2)
    assert list(rank) == [1, 2, 3, 1, 1, 1] or list(rank) == [1, 2, 3, 1, 1, 1]
    assert raw[0] == raw[3] and raw[0] > raw[1] > raw[2]
    # points 3 and 4 are neighbours (same rank) -> they share niche and get lower fitness than point 5
    assert m_i[3] > 1 and m_i[4] > 1 and shared[5] > shared[3]
    return True

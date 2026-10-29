"""
Algorithm 10 -- VEGA, Vector Evaluated Genetic Algorithm (Non-elitist, Part 2; Schaffer 1985)
Lecture steps per generation
  1. Split the mating pool into M sub-populations of equal size, one per objective.
  2. Sub-population k is filled by selection (here binary tournament) on objective k ONLY.
  3. Shuffle the M sub-populations together.
  4. Apply crossover (SBX / OX) and mutation to the shuffled pool -> next generation (no elitism).
Constraints: static penalty on the objectives.
"""
import numpy as np
from moo_base import MOAlgorithm, vary, penalised


class VEGA(MOAlgorithm):
    number, name, family = 10, "VEGA", "Non-elitist (Part 2)"

    def select(self):
        M, N = self.problem.n_obj, self.N
        Fp = penalised(self.F, self.CV, self.scale)
        sizes = [N // M + (1 if k < N % M else 0) for k in range(M)]
        groups = []
        for k in range(M):
            a, b = self.rng.integers(0, N, sizes[k]), self.rng.integers(0, N, sizes[k])
            groups.append(np.where(Fp[b, k] < Fp[a, k], b, a))
        return groups

    def iterate(self):
        groups = self.select()
        pool = np.concatenate(groups)
        self.rng.shuffle(pool)
        self.log("sub-population sizes", [len(g) for g in groups])
        off = vary(self.problem, self.X[pool], self.rng)
        self.X, self.F, self.CV = self.evaluate(off)           # generational replacement


def unit_test():
    """Key rule: sub-population k is selected on objective k only -> its members are better in f_k than average."""
    from moo_problems import get_problem
    p = get_problem("P1")
    a = VEGA(p, pop_size=60, seed=0)
    a.budget = 10 ** 9
    a.initialize()
    groups = a.select()
    for k, g in enumerate(groups):
        assert a.F[g, k].mean() < a.F[:, k].mean(), "group %d must prefer objective %d" % (k, k)
    return True

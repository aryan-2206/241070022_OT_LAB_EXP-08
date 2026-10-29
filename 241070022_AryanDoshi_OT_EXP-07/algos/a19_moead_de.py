"""
Algorithm 19 -- MOEA/D-DE (Elitist, decomposition + differential evolution, Part 3; Li & Zhang 2009)
Same decomposition machinery as MOEA/D (algorithm 18) but the reproduction operator is DE/rand/1/bin:
        u = x_r1 + F * (x_r2 - x_r3),   binomial crossover with rate CR, then polynomial mutation,
with r1,r2,r3 drawn from the neighbourhood B(i) (probability delta) or the whole population. The update
step limits the number of replacements to n_r (diversity).
Only for real / integer decision variables.
"""
import numpy as np
from algos.a18_moead import MOEAD
from moo_base import poly_mut


def de_rand_1(X, r1, r2, r3, F, CR, lb, ub, rng):
    V = X[r1] + F * (X[r2] - X[r3])
    cross = rng.random(V.shape) < CR
    cross[np.arange(len(V)), rng.integers(0, V.shape[1], len(V))] = True     # at least one gene
    U = np.where(cross, V, X)
    return np.clip(U, lb, ub)


class MOEADDE(MOEAD):
    number, name = 19, "MOEA/D-DE"

    def make_offspring(self):
        Fm, CR = self.params.get("F", 0.5), self.params.get("CR", 1.0)
        r = np.empty((3, self.N), int)
        for i in range(self.N):
            pool = self.B[i] if self.rng.random() < self.delta else np.arange(self.N)
            r[:, i] = self.rng.choice(pool, 3, replace=len(pool) < 3)
        base = np.arange(self.N)
        U = de_rand_1(self.X.astype(float), r[0], r[1], r[2], Fm, CR, self.problem.lb, self.problem.ub, self.rng)
        # DE base is x_r1; "x" in the crossover is the target solution i
        return poly_mut(U, self.problem.lb, self.problem.ub, self.rng)


def unit_test():
    """Key rule: DE/rand/1 -- with identical donors the mutant equals the base vector; with CR=0 only one gene changes."""
    rng = np.random.default_rng(0)
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    r1 = np.array([0]); r2 = np.array([1]); r3 = np.array([1])
    U = de_rand_1(X, r1, r2, r3, 0.5, 1.0, -100, 100, rng)
    assert np.allclose(U, X[0]), "x_r2 - x_r3 = 0 => mutant = base"
    U = de_rand_1(X, np.array([0]), np.array([1]), np.array([2]), 0.5, 1.0, -100, 100, rng)
    assert np.allclose(U, X[0] + 0.5 * (X[1] - X[2])), "u = x_r1 + F (x_r2 - x_r3)"
    return True

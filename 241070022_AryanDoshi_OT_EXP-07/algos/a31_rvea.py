"""
Algorithm 31 -- RVEA, Reference Vector guided EA (Many-objective, 5E; Cheng, Jin, Olhofer, Sendhoff 2016)
Lecture steps per generation t (of t_max)
  1. Unit reference vectors V (Das-Dennis, normalised to length 1).
  2. Random mating, crossover, mutation -> offspring; merge with the parents.
  3. Translate objectives by the ideal point z*;  angle theta_ij between solution i and vector v_j.
  4. Assign every solution to the vector with the smallest angle.
  5. Angle-Penalised Distance  APD = (1 + M (t/t_max)^a  theta / gamma_v) ||f'||,
     gamma_v = smallest angle between v and any other reference vector.  Keep the solution with min APD per vector.
  6. Reference-vector adaptation every f_r * t_max generations: V <- V0 * (z_max - z_min) (renormalised).
"""
import numpy as np
from moo_base import MOAlgorithm, vary, progress
from moo_utils import das_dennis
from algos.a15_nsga3 import choose_divisions


def apd_select(Fp, V, gamma, M, frac, alpha=2.0):
    """Fp: translated objectives. Returns indices of the selected solutions (one per non-empty vector)."""
    norm = np.maximum(np.linalg.norm(Fp, axis=1), 1e-12)
    cos = np.clip((Fp / norm[:, None]) @ V.T, -1, 1)
    theta = np.arccos(cos)
    assign = theta.argmin(axis=1)
    sel = []
    for j in np.unique(assign):
        mem = np.where(assign == j)[0]
        apd = (1 + M * (frac ** alpha) * theta[mem, j] / gamma[j]) * norm[mem]
        sel.append(mem[np.argmin(apd)])
    return np.array(sel, int), assign


class RVEA(MOAlgorithm):
    number, name, family = 31, "RVEA", "Many-objective (5E)"

    def __init__(self, problem, pop_size=100, seed=0, alpha=2.0, fr=0.1, verbose=False, **params):
        M = problem.n_obj
        p = pop_size - 1 if M == 2 else choose_divisions(M, pop_size)
        W = das_dennis(M, p)
        V0 = W / np.linalg.norm(W, axis=1, keepdims=True)
        super().__init__(problem, len(W), seed, verbose, **params)
        self.V0, self.V, self.alpha, self.fr = V0, V0.copy(), alpha, fr
        self.last_adapt = 0.0

    def gammas(self, V):
        c = np.clip(V @ V.T, -1, 1)
        np.fill_diagonal(c, -1)
        return np.maximum(np.arccos(c.max(axis=1)), 1e-6)

    def initialize(self):
        super().initialize()
        self.z = self.F.min(axis=0)

    def iterate(self):
        M = self.problem.n_obj
        idx = self.rng.integers(0, len(self.X), self.N)
        off = vary(self.problem, self.X[idx], self.rng)
        Xo, Fo, CVo = self.evaluate(off)
        X, F, CV = np.vstack([self.X, Xo]), np.vstack([self.F, Fo]), np.concatenate([self.CV, CVo])
        if (CV <= 0).any():
            keep = CV <= 0
            X, F, CV = X[keep], F[keep], CV[keep]
        self.z = np.minimum(self.z, F.min(axis=0))
        Fp = F - self.z
        frac = progress(self)
        sel, assign = apd_select(Fp, self.V, self.gammas(self.V), M, frac, self.alpha)
        self.X, self.F, self.CV = X[sel], F[sel], CV[sel]
        if frac - self.last_adapt >= self.fr:                         # reference vector adaptation
            self.last_adapt = frac
            span = np.maximum(self.F.max(axis=0) - self.z, 1e-12)
            V = self.V0 * span
            self.V = V / np.linalg.norm(V, axis=1, keepdims=True)
        self.log("population size / vectors used", [len(self.X), len(np.unique(assign))])


def unit_test():
    """Key rule: APD = (1 + M (t/tmax)^a theta/gamma) ||f'||.  Two vectors on the axes, M=2, gamma=pi/2."""
    V = np.array([[1.0, 0.0], [0.0, 1.0]])
    Fp = np.array([[2.0, 1.0], [3.0, 0.0], [0.1, 4.0]])
    gamma = np.array([np.pi / 2, np.pi / 2])
    sel, assign = apd_select(Fp, V, gamma, 2, 0.0)          # frac=0 -> pure distance
    assert list(assign) == [0, 0, 1]
    assert set(sel) == {0, 2}, "at t=0 the smaller norm wins inside vector 0"
    # late in the run the angle penalty dominates: (3,0) is perfectly aligned (theta=0), (2,1) is not
    sel2, _ = apd_select(Fp, V, gamma, 2, 1.0)
    assert 1 in sel2, "late: the better aligned solution (3,0) wins"
    return True

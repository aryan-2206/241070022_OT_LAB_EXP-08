"""
Algorithm 34 -- HypE, Hypervolume Estimation algorithm (Indicator-based, 5E; Bader & Zitzler 2011)
Lecture steps per generation
  1. Monte-Carlo sampling: S uniform sample points in the box [min F, ref] estimate hypervolume shares.
  2. For a sample dominated by n individuals each of them receives alpha_n = (1/n) prod_{i=1}^{n-1} (k-i)/(|P|-i)
     (k = number of individuals still to be removed; alpha_n = 0 for n > k)  ->  fitness = sum of shares x volume/S.
  3. Mating: binary tournament on fitness; crossover + mutation.
  4. Environmental selection on R = P u Q: repeatedly remove the individual with the smallest fitness and
     recompute the shares until |R| = N (k = |R| - N at the start).
Constraints: static penalty.
"""
import numpy as np
from moo_base import MOAlgorithm, vary, tournament_by_fitness, penalised


def hype_alpha(k, n):
    """alpha_j for j = 0..n (index 0 unused)."""
    a = np.zeros(n + 1)
    for j in range(1, n + 1):
        if j - 1 >= k:
            break
        prod = 1.0
        for i in range(1, j):
            prod *= (k - i) / (n - i)
        a[j] = prod / j
    return a


def hype_fitness_from_dom(dom, k, vol):
    n = dom.shape[1]
    a = hype_alpha(k, n)
    cnt = dom.sum(axis=1)
    return (dom * a[cnt][:, None]).sum(axis=0) * vol / dom.shape[0]


class HypE(MOAlgorithm):
    number, name, family = 34, "HypE", "Indicator-based (5E)"

    def __init__(self, problem, pop_size=100, seed=0, n_samples=1000, verbose=False, **params):
        super().__init__(problem, pop_size, seed, verbose, **params)
        self.S = n_samples

    def samples_dom(self, F):
        ref = F.max(axis=0) + 0.1 * np.maximum(np.ptp(F, axis=0), 1e-9)
        lo = F.min(axis=0)
        S = lo + self.rng.random((self.S, F.shape[1])) * (ref - lo)
        dom = (F[None, :, :] <= S[:, None, :]).all(axis=2)
        return dom, float(np.prod(ref - lo))

    def initialize(self):
        super().initialize()
        dom, vol = self.samples_dom(penalised(self.F, self.CV, self.scale))
        self.fit = hype_fitness_from_dom(dom, len(self.F), vol)

    def iterate(self):
        par = tournament_by_fitness(self.rng, self.N, self.fit)
        off = vary(self.problem, self.X[par], self.rng)
        Xo, Fo, CVo = self.evaluate(off)
        X, F, CV = np.vstack([self.X, Xo]), np.vstack([self.F, Fo]), np.concatenate([self.CV, CVo])
        dom, vol = self.samples_dom(penalised(F, CV, self.scale))
        alive = np.arange(len(F))
        k = len(F) - self.N
        while len(alive) > self.N:
            fit = hype_fitness_from_dom(dom[:, alive], k, vol)
            alive = np.delete(alive, np.argmin(fit))
        self.fit = hype_fitness_from_dom(dom[:, alive], self.N, vol)
        self.X, self.F, self.CV = X[alive], F[alive], CV[alive]
        self.log("fitness range", [self.fit.min(), self.fit.max()])


def unit_test():
    """Hand example: n=2, k=2: a sample dominated by both individuals gives each alpha_2 = 1/2 * (2-1)/(2-1) = 1/2;
    a sample dominated by one gives it alpha_1 = 1. (The shares of one sample add up to 1 when k = n.)"""
    a = hype_alpha(2, 2)
    assert abs(a[1] - 1.0) < 1e-12 and abs(a[2] - 0.5) < 1e-12
    dom = np.array([[True, True], [True, False], [False, False]])
    fit = hype_fitness_from_dom(dom, 2, 3.0)           # volume 3, 3 samples -> each sample is worth 1
    assert np.allclose(fit, [1.5, 0.5])
    a1 = hype_alpha(1, 3)
    assert a1[1] == 1.0 and a1[2] == 0.0 and a1[3] == 0.0, "k=1: only exclusively dominated samples count"
    return True

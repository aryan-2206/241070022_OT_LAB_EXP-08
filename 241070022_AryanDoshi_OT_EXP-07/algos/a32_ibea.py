"""
Algorithm 32 -- IBEA, Indicator-Based EA with the additive epsilon indicator (Many-objective, 5E; Zitzler & Kunzli 2004)
Lecture steps per generation
  1. Normalise objectives to [0,1].  Indicator  I_eps+(a,b) = min eps such that f(a) - eps <= f(b) = max_k (f_k(a)-f_k(b)).
  2. Fitness  F(x) = sum_{y != x} -exp(-I_eps+(y,x) / (c kappa)),  c = max |I|  (larger is better).
  3. Mating: binary tournament on F, crossover, mutation.
  4. Environmental selection on R = P u Q: repeatedly delete the worst individual and update the fitness of the
     rest:  F(x) += exp(-I_eps+(worst,x)/(c kappa))  until |R| = N.   No explicit diversity mechanism.
Constraints: static penalty.
"""
import numpy as np
from moo_base import MOAlgorithm, vary, tournament_by_fitness, penalised
from moo_utils import normalize


def ibea_matrix(Fn):
    return (Fn[:, None, :] - Fn[None, :, :]).max(axis=2)          # I[i,j] = I_eps+(i,j)


def ibea_fitness(Fn, kappa=0.05):
    I = ibea_matrix(Fn)
    c = max(np.abs(I).max(), 1e-12)
    E = -np.exp(-I / (c * kappa))
    np.fill_diagonal(E, 0.0)
    return E.sum(axis=0), I, c        # fitness_j = sum_i  -exp(-I[i,j]/(c kappa))


class IBEA(MOAlgorithm):
    number, name, family = 32, "IBEA", "Indicator-based (5E)"

    def __init__(self, problem, pop_size=100, seed=0, kappa=0.05, verbose=False, **params):
        super().__init__(problem, pop_size, seed, verbose, **params)
        self.kappa = kappa

    def fitness(self, F, CV):
        Fn = normalize(penalised(F, CV, self.scale))
        return ibea_fitness(Fn, self.kappa)

    def initialize(self):
        super().initialize()
        self.fit = self.fitness(self.F, self.CV)[0]

    def iterate(self):
        par = tournament_by_fitness(self.rng, self.N, self.fit)
        off = vary(self.problem, self.X[par], self.rng)
        Xo, Fo, CVo = self.evaluate(off)
        X, F, CV = np.vstack([self.X, Xo]), np.vstack([self.F, Fo]), np.concatenate([self.CV, CVo])
        fit, I, c = self.fitness(F, CV)
        alive = np.ones(len(F), bool)
        while alive.sum() > self.N:
            worst = np.where(alive)[0][np.argmin(fit[alive])]
            alive[worst] = False
            fit = fit + np.exp(-I[worst, :] / (c * self.kappa))          # remove the worst one's contribution
        idx = np.where(alive)[0]
        self.X, self.F, self.CV, self.fit = X[idx], F[idx], CV[idx], fit[idx]
        self.log("min / max fitness", [self.fit.min(), self.fit.max()])


def unit_test():
    """Hand example: I_eps+(a,b) = max_k (a_k - b_k); a dominating point has the highest fitness."""
    Fn = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    I = ibea_matrix(Fn)
    assert I[0, 1] == -1.0 and I[1, 0] == 1.0 and I[2, 0] == 0.5
    fit, _, _ = ibea_fitness(Fn, 0.05)
    assert fit.argmax() == 0 and fit.argmin() == 1, "dominating point best, dominated point worst"
    return True

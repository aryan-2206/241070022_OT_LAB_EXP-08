"""
Algorithm 33 -- SMS-EMOA (Indicator-based, steady state, 5E; Beume, Naujoks, Emmerich 2007)
Lecture steps per iteration (mu + 1 evolution)
  1. Choose two random parents, crossover + mutation -> ONE offspring.
  2. Add it to the population (size mu+1) and sort into non-dominated fronts.
  3. In the WORST front remove the individual with the smallest exclusive hypervolume contribution
     (2-D exact, >=3-D Monte-Carlo estimate); infeasible solutions are removed by largest violation first.
"""
import numpy as np
from moo_base import MOAlgorithm, vary
from moo_utils import fast_non_dominated_sort


def hv_contributions(F, ref, rng=None, n_samples=4000):
    """Exclusive hypervolume contribution of every row of F (minimisation)."""
    n, M = F.shape
    if n == 1:
        return np.array([np.prod(ref - F[0])])
    if M == 2:
        o = np.argsort(F[:, 0], kind="stable")
        Fs = F[o]
        contrib = np.empty(n)
        for k in range(n):
            right = Fs[k + 1, 0] if k + 1 < n else ref[0]
            up = Fs[k - 1, 1] if k > 0 else ref[1]
            contrib[k] = max(right - Fs[k, 0], 0) * max(up - Fs[k, 1], 0)
        out = np.empty(n)
        out[o] = contrib
        return out
    rng = rng or np.random.default_rng(0)
    lo = F.min(axis=0)
    S = lo + rng.random((n_samples, M)) * (ref - lo)
    dom = (F[None, :, :] <= S[:, None, :]).all(axis=2)
    only = dom.sum(axis=1) == 1
    return (dom & only[:, None]).sum(axis=0) / n_samples * np.prod(ref - lo)


class SMSEMOA(MOAlgorithm):
    number, name, family = 33, "SMS-EMOA", "Indicator-based (5E)"

    def iterate(self):
        i, j = self.rng.integers(0, self.N, 2)
        off = vary(self.problem, self.X[[i, j]], self.rng)[:1]
        Xo, Fo, CVo = self.evaluate(off)
        X, F, CV = np.vstack([self.X, Xo]), np.vstack([self.F, Fo]), np.concatenate([self.CV, CVo])
        fronts, rank = fast_non_dominated_sort(F, CV)
        last = fronts[-1]
        if (CV[last] > 0).any():
            worst = last[np.argmax(CV[last])]
        elif len(last) == 1:
            worst = last[0]
        else:
            ref = F.max(axis=0) + 1e-6 + 0.1 * np.ptp(F, axis=0)
            c = hv_contributions(F[last], ref, self.rng)
            worst = last[np.argmin(c)]
        keep = np.arange(len(F)) != worst
        self.X, self.F, self.CV = X[keep], F[keep], CV[keep]


def unit_test():
    """Hand example: front (1,3),(2,2),(3,1), ref (4,4): contributions are 1, 1, 1 (boundary included)."""
    F = np.array([[1, 3], [2, 2], [3, 1]], float)
    c = hv_contributions(F, np.array([4.0, 4.0]))
    assert np.allclose(c, [1, 1, 1])
    F2 = np.array([[1, 3], [1.9, 2.9], [3, 1]], float)       # middle point nearly redundant
    c2 = hv_contributions(F2, np.array([4.0, 4.0]))
    assert c2.argmin() == 1, "the least contributing point is the one removed"
    return True

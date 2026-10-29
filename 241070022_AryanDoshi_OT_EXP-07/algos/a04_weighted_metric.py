"""
Algorithm 4 -- Weighted Metric method, p-norm and Tchebycheff (Classical, Part 1)
Lecture steps
  1. Ideal point z* and nadir from the payoff table; normalise.
  2. For a weight vector w and exponent p solve   min ( sum_i w_i |fhat_i|^p )^(1/p).
     p = 1 is the weighted sum, p = 2 Euclidean distance to the ideal point, p = inf Tchebycheff:
     min t  s.t.  w_i * fhat_i <= t   (epigraph form).
  3. Sweep w for every p and compare the shapes of the found sets.
"""
import numpy as np
from algos._classical import ClassicalMethod


class WeightedMetric(ClassicalMethod):
    number, name = 4, "Weight Metric (p-norm, Tchebycheff)"

    def __init__(self, problem, pop_size=100, seed=0, n_sub=11, ps=(1, 2, np.inf), verbose=False, **params):
        super().__init__(problem, pop_size, seed, n_sub=n_sub, verbose=verbose, **params)
        self.ps = list(ps)

    def setup(self):
        W = self.weight_set()
        self.tasks = [(p, w) for p in self.ps for w in W]

    def n_tasks(self):
        return len(self.tasks)

    def solve_task(self, i):
        p, w = self.tasks[i]
        n = self.n
        if np.isinf(p):
            bnds = self.bounds() + [(0, None)]
            g_own = self.con_ineq()

            def ineq(z):
                c = z[n] - w * self.nf(self.F_(z[:n]))
                return c if g_own is None else np.concatenate([c, g_own(z)])
            res = self.best_of(lambda z: float(z[n]), bnds, ineq, k=3, extra=1)
        else:
            def obj(x):
                fh = np.maximum(self.nf(self.F_(x)), 0)
                return float(((w * fh ** p).sum()) ** (1.0 / p))
            res = self.best_of(obj, self.bounds(), self.con_ineq(), k=3)
        if res is not None:
            self.add_solution(res.x[:n], tag=(p, tuple(w)))


def unit_test():
    """Key rule: Tchebycheff (p=inf) reaches the interior of the non-convex ZDT2 front, p=1 does not."""
    from moo_problems import get_problem
    p3 = get_problem("P3")
    a_inf = WeightedMetric(p3, n_sub=9, ps=(np.inf,), seed=0)
    r_inf = a_inf.run(60000)
    f1 = r_inf["F"][:, 0]
    assert np.any((f1 > 0.2) & (f1 < 0.8)), "Tchebycheff must reach interior of a non-convex front"
    a_1 = WeightedMetric(p3, n_sub=9, ps=(1,), seed=0)
    r_1 = a_1.run(60000)
    assert np.all((r_1["F"][:, 0] < 0.02) | (r_1["F"][:, 0] > 0.98)), "p=1 is the weighted sum"
    return True

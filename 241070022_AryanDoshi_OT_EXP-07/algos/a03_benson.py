"""
Algorithm 3 -- Benson's Method (Classical, Part 1)
Lecture steps
  1. Choose a (feasible) reference point  f0 = f(x0).
  2. Solve   max  sum_i lambda_i * l_i   s.t.  f_i(x) + l_i = f0_i,  l_i >= 0,  x feasible.
     l_i is the improvement ("distance past the reference point") in objective i.
  3. The optimum is a Pareto-optimal point that dominates f0 (if f0 is not Pareto-optimal itself).
  4. Use several reference points x0 to cover the front.
"""
import numpy as np
from algos._classical import ClassicalMethod


class Benson(ClassicalMethod):
    number, name = 3, "Benson's Method"

    def setup(self):
        p = self.problem
        self.refs = []
        tries = 0
        while len(self.refs) < self.n_sub and tries < 5000:
            x0 = p.lb + self.rng.random(p.n_var) * (p.ub - p.lb)
            tries += 1
            if self.feasible(x0, 0.0):
                self.refs.append(x0)

    def n_tasks(self):
        return len(self.refs)

    def solve_task(self, i):
        n, M = self.n, self.problem.n_obj
        x0 = self.refs[i]
        f0 = self.problem.f1(x0)
        lam = np.ones(M)
        scale = np.where(self.nadir - self.ideal > 1e-9, self.nadir - self.ideal, 1.0)
        bnds = self.bounds() + [(0, None)] * M
        g_own = self.con_ineq()

        def obj(z):
            return -float((lam * z[n:] / scale).sum())

        def eq(z):
            return (self.F_(z[:n]) + z[n:] - f0) / scale
        z0 = np.concatenate([x0, np.zeros(M)])
        res = self.solve(obj, z0, bnds, g_own, eq, maxiter=200)
        if res is not None and np.all(np.isfinite(res.x)) and self.feasible(res.x[:n], 1e-5):
            f = self.problem.f1(res.x[:n])
            if np.all(f <= f0 + 1e-6):
                self.add_solution(res.x[:n], tag=tuple(f0))
                self.f0s = getattr(self, "f0s", []) + [f0]


def unit_test():
    """Key rule: the point found dominates (or equals) its reference point and lies on the Pareto set."""
    from moo_problems import get_problem
    from moo_utils import dominates
    p1 = get_problem("P1")
    alg = Benson(p1, n_sub=8, seed=1)
    r = alg.run(8000)
    assert len(alg.f0s) >= 5
    for f0, x in zip(alg.f0s, alg.sols):
        f = p1.f1(x)
        assert np.all(f <= f0 + 1e-6)
        assert 3 - 1e-3 <= x[0] <= 7 + 1e-3, "result must be Pareto-optimal (3<=x<=7)"
    return True

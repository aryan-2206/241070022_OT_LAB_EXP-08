"""
Algorithm 2 -- epsilon-Constraint (Classical, Part 1)
Lecture steps
  1. Payoff table -> ideal and nadir.
  2. Keep one objective f_k as THE objective, turn the others into constraints f_j(x) <= eps_j.
  3. Sweep eps_j between ideal_j and nadir_j; for each eps solve  min f_k  s.t. f_j <= eps_j (SLSQP).
  4. Each solution is (weakly) Pareto-optimal and the method also reaches NON-convex parts.
"""
import numpy as np
from algos._classical import ClassicalMethod
from moo_utils import das_dennis


class EpsilonConstraint(ClassicalMethod):
    number, name = 2, "epsilon-Constraint"

    def setup(self):
        M = self.problem.n_obj
        self.k = 0
        others = [j for j in range(M) if j != self.k]
        if M == 2:
            t = np.linspace(0, 1, self.n_sub)[:, None]
        else:
            p = 1
            while len(das_dennis(M, p)) < self.n_sub:
                p += 1
            t = das_dennis(M, p)[:, 1:]
        self.others = others
        self.EPS = self.ideal[others] + t * (self.nadir[others] - self.ideal[others])

    def n_tasks(self):
        return len(self.EPS)

    def solve_task(self, i):
        eps = self.EPS[i]
        n, k, others = self.n, self.k, self.others
        g_own = self.con_ineq()

        def ineq(x):
            c = eps - self.F_(x)[others]
            return c if g_own is None else np.concatenate([c, g_own(x)])
        res = self.best_of(lambda x: float(self.F_(x)[k]), self.bounds(), ineq, k=4,
                           check=lambda z: np.all(self.problem.f1(z[:n])[others] <= eps + 1e-5))
        if res is not None:
            self.add_solution(res.x, tag=tuple(eps))
            self.eps_used = getattr(self, "eps_used", []) + [eps]


def unit_test():
    """Key rule: every returned point satisfies f_j <= eps_j and ZDT2's concave front is reached."""
    from moo_problems import get_problem
    p3 = get_problem("P3")
    alg = EpsilonConstraint(p3, n_sub=9, seed=0)
    r = alg.run(30000)
    assert len(r["F"]) >= 6, "epsilon-constraint should find many points on ZDT2"
    f1 = np.sort(r["F"][:, 0])
    assert np.any((f1 > 0.2) & (f1 < 0.8)), "interior points of the NON-convex front must be found"
    assert np.allclose(r["F"][:, 1], 1 - r["F"][:, 0] ** 2, atol=2e-2)
    return True

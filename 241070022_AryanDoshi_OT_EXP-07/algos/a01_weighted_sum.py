"""
Algorithm 1 -- Weighted Sum (Classical, Part 1)
Lecture steps
  1. Normalise the objectives with the payoff table (ideal z*, nadir z^nad).
  2. Pick a weight vector w (w_i >= 0, sum w = 1).
  3. Solve the single-objective problem  min  sum_i w_i * fhat_i(x)  (scipy SLSQP, multi-start).
  4. Vary w (sweep) and collect the solutions; keep the non-dominated ones.
Known limit: only solutions on the CONVEX part of the front can be found.
"""
import numpy as np
from algos._classical import ClassicalMethod


class WeightedSum(ClassicalMethod):
    number, name = 1, "Weighted Sum"

    def setup(self):
        self.W = self.weight_set()

    def n_tasks(self):
        return len(self.W)

    def solve_task(self, i):
        w = self.W[i]
        res = self.best_of(lambda x: float(w @ self.nf(self.F_(x))), self.bounds(), self.con_ineq(), k=3)
        if res is not None:
            self.add_solution(res.x, tag=tuple(w))


def unit_test():
    """Key rule: weighted sum finds every point of a CONVEX front but only end points of a non-convex one."""
    from moo_problems import get_problem
    p1 = get_problem("P1")                       # convex front
    r = WeightedSum(p1, n_sub=11, seed=0).run(5000)
    assert len(r["F"]) >= 8, "should find a spread of points on a convex front"
    assert np.allclose(r["F"][:, 1], (4 - np.sqrt(r["F"][:, 0])) ** 2, atol=1e-3), "points must lie on the true front"
    p3 = get_problem("P3")                       # ZDT2 concave front
    r = WeightedSum(p3, n_sub=11, seed=0).run(20000)
    f1 = np.sort(r["F"][:, 0])
    assert np.all((f1 < 0.02) | (f1 > 0.98)), "on a concave front only the two end points can be reached"
    return True

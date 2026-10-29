"""
Algorithm 7 -- Lexicographic method (Classical, Part 1 and Bonus 5E)
Lecture steps
  1. DM ranks the objectives by priority (order).
  2. Minimise the first-priority objective alone; call the optimum f*_1.
  3. Add the constraint  f_1(x) <= f*_1 + tol_1  ("tolerance"), then minimise the second priority, etc.
  4. The last solution is returned. A larger tolerance lets lower priorities improve (trade-off).
"""
import numpy as np
from algos._classical import ClassicalMethod


class Lexicographic(ClassicalMethod):
    number, name = 7, "Lexicographic"
    family = "Classical (Part 1, 5E)"

    def __init__(self, problem, pop_size=100, seed=0, order=None, tols=None, verbose=False, **params):
        super().__init__(problem, pop_size, seed, verbose=verbose, **params)
        self.order = list(range(problem.n_obj)) if order is None else list(order)
        self.tols = np.linspace(0, 0.5, 11) if tols is None else np.asarray(tols, float)   # relative to the range

    def n_tasks(self):
        return len(self.tols)

    def solve_task(self, i):
        tol = self.tols[i]
        n = self.n
        span = self.nadir - self.ideal
        limits = []                                   # (objective index, upper limit) added so far
        x_last = None
        g_own = self.con_ineq()
        for lev, k in enumerate(self.order):
            def ineq(x, limits=list(limits)):
                c = [lim - self.F_(x)[j] for j, lim in limits]
                if g_own is not None:
                    c = list(g_own(x)) + c
                return np.array(c) if len(c) else np.array([1.0])
            res = self.best_of(lambda x, k=k: float(self.F_(x)[k]), self.bounds(), ineq, k=4,
                               check=lambda z, limits=list(limits): all(self.problem.f1(z[:n])[j] <= lim + 1e-5 for j, lim in limits))
            if res is None:
                return
            x_last = res.x[:n]
            fk = self.problem.f1(x_last)[k]
            limits.append((k, fk + tol * span[k] if lev < len(self.order) - 1 else fk))
        self.add_solution(x_last, tag=tol)


def unit_test():
    """Key rule: lower priorities may never worsen a higher priority beyond its tolerance (tol=0 -> equal optimum)."""
    from moo_problems import get_problem
    p = get_problem("P11")                                   # profit first, units second
    alg = Lexicographic(p, order=[0, 1], tols=[0.0, 0.1], seed=0)
    alg.run(20000)
    F = p.original(alg.F)
    assert abs(F[0, 0] - 2400) < 1e-2 and abs(F[0, 1] - 500) < 1e-1, "tol=0: profit 2400 first, then max units = 500"
    assert F[1, 0] >= 2400 - 0.1 * 800 - 1e-2, "profit stays within tolerance of its optimum"
    return True

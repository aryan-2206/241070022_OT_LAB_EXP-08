"""
Algorithm 6 -- Goal Programming (Classical, Part 1 and Bonus 5E)
Lecture steps
  1. DM sets a target t_i for every objective (in minimisation form: wish f_i <= t_i).
  2. Introduce deviation variables d_i^- , d_i^+ >= 0 with   f_i(x) - t_i - d_i^+ + d_i^- = 0.
  3. Only the undesirable deviation d_i^+ (excess over the target) is penalised.
  4. Solve  min sum_i w_i d_i^+   (weights: unit, or normalised by the goal scale).
"""
import numpy as np
from algos._classical import ClassicalMethod


class GoalProgramming(ClassicalMethod):
    number, name = 6, "Goal Programming"
    family = "Classical (Part 1, 5E)"

    def __init__(self, problem, pop_size=100, seed=0, goals=None, weight_sets=None, verbose=False, **params):
        super().__init__(problem, pop_size, seed, verbose=verbose, **params)
        self.goals = None if goals is None else np.asarray(goals, float)
        self.weight_sets = weight_sets            # dict name -> weight vector, or None for a sweep

    def setup(self):
        M = self.problem.n_obj
        if self.goals is None:
            self.goals = self.ideal.copy()           # unattainable goal -> compromise solutions
        if self.weight_sets is None:
            span = np.maximum(self.nadir - self.ideal, 1e-9)
            self.weight_sets = {"w=%s" % np.round(w, 2): w / span for w in self.weight_set()}   # normalised deviations
        self.names = list(self.weight_sets)
        self.scale = np.where(np.abs(self.goals) > 1e-9, np.abs(self.goals), 1.0)

    def n_tasks(self):
        return len(self.names)

    def solve_task(self, i):
        name = self.names[i]
        w = self.weight_sets[name]
        w = np.asarray(w, float) if not isinstance(w, str) else (np.ones(self.problem.n_obj) if w == "unit" else 1.0 / self.scale)
        n, M = self.n, self.problem.n_obj
        bnds = self.bounds() + [(0, None)] * (2 * M)
        t = self.goals
        g_own = self.con_ineq()

        def obj(z):
            return float((w * z[n + M:]).sum())             # penalise d+ only

        def eq(z):
            return self.F_(z[:n]) - t - z[n + M:] + z[n:n + M]
        best = None
        for x0 in self.starts(3):
            f0 = self.problem.f1(x0)
            dm0, dp0 = np.maximum(t - f0, 0), np.maximum(f0 - t, 0)
            res = self.solve(obj, np.concatenate([x0, dm0, dp0]), bnds, g_own, eq, maxiter=200)
            if res is None or not np.all(np.isfinite(res.x)) or not self.feasible(res.x[:n], 1e-5):
                continue
            if np.abs(eq(res.x)).max() > 1e-4:
                continue
            if best is None or res.fun < best.fun:
                best = res
        if best is not None:
            self.add_solution(best.x[:n], tag=name)
            self.dev = getattr(self, "dev", []) + [(name, best.x[n:n + M].copy(), best.x[n + M:].copy())]


def unit_test():
    """Key rule: a goal that is attainable gives zero deviation; P11's 600-unit goal leaves a shortfall >= 100."""
    from moo_problems import get_problem
    import numpy as np
    p = get_problem("P11")
    # goals in minimisation form: profit >= 2400 -> -profit <= -2400 ; units >= 600 -> -units <= -600
    alg = GoalProgramming(p, goals=[-2400, -600], weight_sets={"unit": "unit", "normalised": "norm"}, seed=0)
    alg.run(20000)
    for name, dm, dp in alg.dev:
        assert dp[0] < 1e-3, "profit goal 2400 is attainable"
        assert dp[1] >= 100 - 1e-2, "units goal 600 cannot be met: shortfall >= 100"
    # attainable goal => zero deviations
    alg2 = GoalProgramming(p, goals=[-2000, -400], weight_sets={"unit": "unit"}, seed=0)
    alg2.run(20000)
    assert np.abs(alg2.dev[0][2]).max() < 1e-3
    return True

"""
Algorithm 5 -- Value Function method (Classical, Part 1)
Lecture steps
  1. The decision maker (DM) supplies a value (utility) function V(f) that is monotone in every objective.
  2. Normalise the objectives, convert to "goodness" g_i = 1 - fhat_i in [0,1].
  3. Maximise a multiplicative (Cobb-Douglas) value V(g) = prod g_i^w_i  (log form for stability); the
     exponents w_i are the DM's preference input (profiles such as conservative / balanced / aggressive).
  4. The optimum of V is a single, Pareto-optimal compromise solution per DM profile.
"""
import numpy as np
from algos._classical import ClassicalMethod


class ValueFunction(ClassicalMethod):
    number, name = 5, "Value Function"

    def __init__(self, problem, pop_size=100, seed=0, profiles=None, verbose=False, **params):
        super().__init__(problem, pop_size, seed, verbose=verbose, **params)
        M = problem.n_obj
        if profiles is None:
            profiles = {"balanced": np.ones(M) / M}
            if M == 2:
                profiles = {"objective-1 focused": np.array([0.8, 0.2]), "balanced": np.array([0.5, 0.5]),
                            "objective-2 focused": np.array([0.2, 0.8])}
        self.profiles = {k: np.asarray(v, float) / np.sum(v) for k, v in profiles.items()}
        self.names = list(self.profiles)

    def n_tasks(self):
        return len(self.names)

    def solve_task(self, i):
        w = self.profiles[self.names[i]]

        def obj(x):
            g = np.clip(1.0 - self.nf(self.F_(x)), 1e-6, None)
            return -float((w * np.log(g)).sum())
        res = self.best_of(obj, self.bounds(), self.con_ineq(), k=4)
        if res is not None:
            self.add_solution(res.x, tag=self.names[i])


def unit_test():
    """Key rule: putting more weight on objective 1 gives a solution with a better (smaller) f1."""
    from moo_problems import get_problem
    p1 = get_problem("P1")
    alg = ValueFunction(p1, seed=0)
    r = alg.run(5000)
    F = alg.F
    assert F[0, 0] < F[1, 0] < F[2, 0], "f1 must increase as the weight on f1 decreases"
    assert abs(F[1, 0] - F[1, 1]) < 0.05, "balanced preference -> symmetric compromise (4,4)"
    return True

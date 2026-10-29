"""
_classical.py -- shared machinery for the scalarisation / classical methods (algorithms 1-7).
Only the single-objective sub-problem is solved with scipy.optimize.minimize (SLSQP), which the
assignment allows ("single-objective or LP sub-problem may use scipy.optimize"). Everything else
(weights, payoff table, sweeping, bookkeeping, Pareto filtering) is our own code.
"""
import numpy as np
from scipy.optimize import minimize
from moo_base import MOAlgorithm, BudgetExhausted
from moo_utils import das_dennis


class ClassicalMethod(MOAlgorithm):
    family = "Classical (Part 1)"

    def __init__(self, problem, pop_size=100, seed=0, n_sub=21, verbose=False, **params):
        super().__init__(problem, pop_size, seed, verbose, **params)
        self.n_sub = int(n_sub)
        self.sub_i = 0
        self.sols = []              # decision vectors found
        self.tags = []              # label of the sub-problem that produced each solution
        self.ideal = self.nadir = None

    # -- evaluation counting (every call of the objective vector is one evaluation) ---------------
    def F_(self, x):
        self.nfe += 1
        if self.nfe > self.budget:
            raise BudgetExhausted()
        return self.problem.f1(x)

    def G_(self, x):
        return self.problem.g1(x)

    @property
    def n(self):
        return self.problem.n_var

    def bounds(self):
        return list(zip(self.problem.lb, self.problem.ub))

    def starts(self, k=3):
        p = self.problem
        pts = [0.5 * (p.lb + p.ub)]
        for _ in range(k - 1):
            pts.append(p.lb + self.rng.random(p.n_var) * (p.ub - p.lb))
        return pts

    def solve(self, fun, x0, bounds, ineq=None, eq=None, maxiter=150):
        cons = []
        if ineq is not None:
            cons.append({"type": "ineq", "fun": ineq})
        if eq is not None:
            cons.append({"type": "eq", "fun": eq})
        try:
            res = minimize(fun, x0, method="SLSQP", bounds=bounds, constraints=cons,
                           options={"maxiter": maxiter, "ftol": 1e-10})
        except BudgetExhausted:
            raise
        except Exception:
            return None
        return res

    def feasible(self, x, tol=1e-6):
        g = self.G_(x)
        return len(g) == 0 or np.all(g <= tol)

    def con_ineq(self):
        """x -> -g(x) >= 0 for the problem's own constraints (None if unconstrained)."""
        if self.problem.gfun is None:
            return None
        n = self.n
        return lambda z: -self.G_(z[:n])

    def best_of(self, fun, bounds, ineq=None, eq=None, k=3, extra=0, maxiter=150, check=None):
        """Multi-start wrapper; returns the best feasible result."""
        best = None
        for x0 in self.starts(k):
            z0 = np.concatenate([x0, np.zeros(extra)]) if extra else x0
            res = self.solve(fun, z0, bounds, ineq, eq, maxiter)
            if res is None or not np.all(np.isfinite(res.x)):
                continue
            x = res.x[:self.n]
            if not self.feasible(x, 1e-5):
                continue
            if check is not None and not check(res.x):
                continue
            if best is None or res.fun < best.fun:
                best = res
        return best

    # -- payoff table: ideal and nadir estimate ---------------------------------------------------
    def payoff(self):
        """Payoff table. Stage 1: minimise f_k alone. Stage 2 (lexicographic clean-up): among the minimisers of
        f_k minimise the sum of the other objectives, so that the table row is not arbitrary."""
        M = self.problem.n_obj
        rows, ideal = [], np.zeros(M)
        g_own = self.con_ineq()
        for k in range(M):
            res = self.best_of(lambda x, k=k: self.F_(x)[k], self.bounds(), g_own, k=3)
            x = res.x[:self.n] if res is not None else self.starts(1)[0]
            fk = self.problem.f1(x)[k]
            ideal[k] = fk
            others = [j for j in range(M) if j != k]

            def ineq(z, k=k, fk=fk):
                c = np.array([fk + 1e-6 * (1 + abs(fk)) - self.F_(z)[k]])
                return c if g_own is None else np.concatenate([c, g_own(z)])
            res2 = self.best_of(lambda z, o=others: float(self.F_(z)[o].sum()), self.bounds(), ineq, k=2,
                                check=lambda z, k=k, fk=fk: self.problem.f1(z[:self.n])[k] <= fk + 1e-4 * (1 + abs(fk)))
            if res2 is not None:
                x = res2.x[:self.n]
            rows.append(self.problem.f1(x))
        rows = np.array(rows)
        nadir = rows.max(axis=0)
        span = nadir - ideal
        nadir = np.where(span > 1e-9, nadir, ideal + 1.0)
        self.ideal, self.nadir = ideal, nadir
        return ideal, nadir

    def nf(self, f):
        return (f - self.ideal) / (self.nadir - self.ideal)

    def weight_set(self):
        M = self.problem.n_obj
        if M == 2:
            t = np.linspace(0, 1, self.n_sub)
            return np.column_stack([t, 1 - t])
        p = 1
        while len(das_dennis(M, p)) < self.n_sub:
            p += 1
        return das_dennis(M, p)

    def add_solution(self, x, tag=None):
        self.sols.append(np.asarray(x, float)[:self.n])
        self.tags.append(tag)

    def finish(self):
        """Evaluate the collected decision vectors (repaired) and store them."""
        if not self.sols:
            return
        X = self.problem.repair(np.array(self.sols))
        F, CV = self.problem.evaluate(X)
        self.X, self.F, self.CV = X, F, CV

    # -- driver -----------------------------------------------------------------------------------
    def initialize(self):
        self.payoff()
        self.setup()

    def setup(self):
        pass

    def n_tasks(self):
        raise NotImplementedError

    def iterate(self):
        if self.sub_i >= self.n_tasks():
            self.finish()
            raise BudgetExhausted()
        self.solve_task(self.sub_i)
        self.sub_i += 1
        self.finish()

    def result(self):
        if self.X is None:
            self.finish()
            if self.X is None:
                return dict(X=np.zeros((0, self.n)), F=np.zeros((0, self.problem.n_obj)),
                            CV=np.zeros(0), feasible=False, nfe=self.nfe)
        return super().result()

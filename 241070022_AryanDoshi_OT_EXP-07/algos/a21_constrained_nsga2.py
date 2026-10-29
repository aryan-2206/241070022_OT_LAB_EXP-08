"""
Algorithm 21 -- Constrained NSGA-II with Deb's feasibility rules (Constrained, Part 4; Deb 2000/2002)
NSGA-II (algorithm 14) where the dominance relation is replaced by CONSTRAINED DOMINANCE:
  Rule 1  a feasible solution is better than an infeasible one;
  Rule 2  of two infeasible solutions the one with the smaller total violation CV is better;
  Rule 3  of two feasible solutions the usual Pareto dominance / crowded comparison decides.
Rules are used in the binary tournament, in the non-dominated sorting of R = P u Q (constrained
dominance) and therefore in the survival selection.  No penalty parameter is needed.
"""
import numpy as np
from algos.a14_nsga2 import NSGA2
from moo_base import vary
from moo_utils import constrained_dominates


def deb_tournament(F, CV, crowd, rng, n):
    """Binary tournament with Deb's three rules; ties (neither better) resolved by crowding distance."""
    N = len(F)
    out = np.empty(n, int)
    for s in range(n):
        a = rng.integers(0, N)
        b = (a + 1 + rng.integers(0, N - 1)) % N          # two DIFFERENT candidates
        if constrained_dominates(F[a], CV[a], F[b], CV[b]):
            out[s] = a
        elif constrained_dominates(F[b], CV[b], F[a], CV[a]):
            out[s] = b
        else:
            out[s] = a if crowd[a] >= crowd[b] else b
    return out


class ConstrainedNSGA2(NSGA2):
    number, name, family = 21, "Constrained NSGA-II", "Constrained (Part 4)"
    use_cv = True

    def iterate(self):
        from moo_base import nsga2_select
        par = deb_tournament(self.F, self.CV, self.crowd, self.rng, self.N)
        off = vary(self.problem, self.X[par], self.rng)
        Xo, Fo, CVo = self.evaluate(off)
        X, F, CV = np.vstack([self.X, Xo]), np.vstack([self.F, Fo]), np.concatenate([self.CV, CVo])
        idx, self.rank, self.crowd = nsga2_select(F, CV, self.N)
        self.log("feasible fraction", (CV[idx] <= 0).mean())
        self.X, self.F, self.CV = X[idx], F[idx], CV[idx]


def unit_test():
    """Key rules on a hand example with 3 points: feasible beats infeasible, smaller CV beats larger CV."""
    F = np.array([[5.0, 5.0], [0.0, 0.0], [0.1, 0.1]])
    CV = np.array([0.0, 3.0, 1.0])
    crowd = np.ones(3)
    rng = np.random.default_rng(0)
    # whenever points 0 (feasible) and 1 (infeasible) meet, 0 must win; points 2 (CV=1) beats 1 (CV=3)
    wins = deb_tournament(F, CV, crowd, rng, 3000)
    assert not np.any(wins == 1), "the most violated point can never win a tournament"
    assert constrained_dominates(F[0], 0.0, F[1], 3.0)            # rule 1
    assert constrained_dominates(F[2], 1.0, F[1], 3.0)            # rule 2
    assert not constrained_dominates(F[1], 3.0, F[0], 0.0)
    from moo_problems import get_problem
    p = get_problem("P6")
    r = ConstrainedNSGA2(p, pop_size=60, seed=0).run(6000)
    assert r["feasible"] and np.all(r["CV"] <= 0) and len(r["F"]) > 20
    return True

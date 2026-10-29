"""
Algorithm 9 -- VIKOR (Compromise ranking / MCDM, Part 1 and Bonus 5E)
Lecture steps (ranks a GIVEN set of alternatives)
  1. Best f*_j and worst f^-_j value of every criterion (according to its sense).
  2. Group utility  S_i = sum_j w_j (f*_j - f_ij)/(f*_j - f^-_j)   (weighted sum of regrets),
     individual regret R_i = max_j w_j (f*_j - f_ij)/(f*_j - f^-_j).
  3. Q_i = v (S_i - S*)/(S^- - S*) + (1-v)(R_i - R*)/(R^- - R*),  v = weight of "majority" strategy.
  4. Rank by increasing Q.  Compromise check: C1 acceptable advantage Q(2nd)-Q(1st) >= 1/(J-1),
     C2 acceptable stability: the best Q is also best in S or R.
"""
import numpy as np


class VIKOR:
    number, name, family = 9, "VIKOR", "Ranking / MCDM (Part 1, 5E)"

    def __init__(self, weights, sense, v=0.5):
        self.w = np.asarray(weights, float) / np.sum(weights)
        self.sense = np.asarray(sense)
        self.v = v

    def score(self, A, verbose=False):
        A = np.asarray(A, float)
        best = np.where(self.sense > 0, A.max(axis=0), A.min(axis=0))
        worst = np.where(self.sense > 0, A.min(axis=0), A.max(axis=0))
        span = np.where(np.abs(best - worst) > 1e-12, best - worst, 1.0)
        D = self.w * (best - A) / span
        S, R = D.sum(axis=1), D.max(axis=1)
        def nrm(x):
            r = x.max() - x.min()
            return (x - x.min()) / (r if r > 1e-12 else 1.0)
        Q = self.v * nrm(S) + (1 - self.v) * nrm(R)
        if verbose:
            print("S:", S.round(4), "R:", R.round(4), "Q:", Q.round(4))
        return Q, S, R

    def rank(self, A):
        Q, S, R = self.score(A)
        order = np.argsort(Q, kind="stable")
        J = len(Q)
        c1 = (Q[order[1]] - Q[order[0]]) >= 1.0 / (J - 1) if J > 1 else True
        c2 = (order[0] == np.argmin(S)) or (order[0] == np.argmin(R))
        return order, Q, dict(C1_advantage=bool(c1), C2_stability=bool(c2))

    def run(self, A):
        return self.rank(A)


def unit_test():
    """Key rule: v=1 ranks by group utility S, v=0 by individual regret R, in between a compromise."""
    A = np.array([[10.0, 0.0], [0.0, 10.0], [6.0, 6.0]])      # two extremes and a balanced alternative
    o1, Q1, _ = VIKOR([0.5, 0.5], [+1, +1], v=1.0).rank(A)
    S = VIKOR([0.5, 0.5], [+1, +1]).score(A)[1]
    assert np.argmin(S) == 2 and o1[0] == 2
    o0, Q0, _ = VIKOR([0.5, 0.5], [+1, +1], v=0.0).rank(A)
    assert o0[0] == 2, "balanced alternative has the smallest maximal regret"
    # change weights strongly: 0.9 on criterion 1 -> alternative 0 becomes best by S
    S2 = VIKOR([0.9, 0.1], [+1, +1]).score(A)[1]
    assert np.argmin(S2) == 0
    return True

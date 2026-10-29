"""
Algorithm 8 -- TOPSIS (Ranking / MCDM, Part 1 and Bonus 5E)
Lecture steps (ranks a GIVEN set of alternatives, it never creates new ones)
  1. Decision matrix A (alternatives x criteria), weights w, sense (+1 benefit, -1 cost).
  2. Vector-normalise each column:  r_ij = a_ij / sqrt(sum_i a_ij^2);  weight it: v_ij = w_j r_ij.
  3. Ideal solution v+_j = max (benefit) / min (cost) of column j; anti-ideal v-_j the opposite.
  4. Distances S+_i , S-_i to the ideal and the anti-ideal (Euclidean).
  5. Closeness  C_i = S-_i / (S+_i + S-_i); rank by decreasing C_i.
"""
import numpy as np


class TOPSIS:
    number, name, family = 8, "TOPSIS", "Ranking / MCDM (Part 1, 5E)"

    def __init__(self, weights, sense):
        self.w = np.asarray(weights, float) / np.sum(weights)
        self.sense = np.asarray(sense)

    def score(self, A, verbose=False):
        A = np.asarray(A, float)
        R = A / np.sqrt((A ** 2).sum(axis=0))
        V = R * self.w
        best = np.where(self.sense > 0, V.max(axis=0), V.min(axis=0))
        worst = np.where(self.sense > 0, V.min(axis=0), V.max(axis=0))
        Sp = np.sqrt(((V - best) ** 2).sum(axis=1))
        Sm = np.sqrt(((V - worst) ** 2).sum(axis=1))
        C = Sm / (Sp + Sm)
        if verbose:
            print("normalised+weighted V:\n", np.round(V, 4), "\nS+:", Sp.round(4), "\nS-:", Sm.round(4), "\nC:", C.round(4))
        return C

    def rank(self, A):
        """Returns the order (best first) and the closeness scores."""
        C = self.score(A)
        return np.argsort(-C, kind="stable"), C

    def run(self, A):
        return self.rank(A)


def unit_test():
    """Key rule on a hand example: an alternative equal to the ideal gets C = 1, the anti-ideal gets C = 0,
    and the ranking follows C."""
    A = np.array([[10.0, 10.0], [0.0, 0.0], [5.0, 5.0]])
    t = TOPSIS([0.5, 0.5], [+1, +1])
    order, C = t.rank(A)
    assert abs(C[0] - 1) < 1e-12 and abs(C[1]) < 1e-12 and abs(C[2] - 0.5) < 1e-12
    assert list(order) == [0, 2, 1]
    # cost criterion flips preference
    t2 = TOPSIS([0.5, 0.5], [-1, +1])
    order2, _ = t2.rank(np.array([[1.0, 5.0], [2.0, 5.0]]))
    assert list(order2) == [0, 1]
    return True

"""Track B: Discrete Optimization - QUBO and Ising Formulations.

Implements:
1. Max-Cut formulation as Quadratic Unconstrained Binary Optimization (QUBO)
2. Binary-to-spin conversion: s_i = 1 - 2*x_i (where x_i in {0, 1} and s_i in {-1, +1})
3. Penalty-based constrained selection QUBO: min x^T Q x + lambda * (sum x_i - K)^2
4. Classical baseline solver (Brute-force and Simulated Annealing) for Track C.
"""

import numpy as np


class MaxCutQUBO:
    """
    Max-Cut on graph G=(V, E) with edge weights w_{ij}.
    Binary variable x_i in {0, 1} indicates partition membership.
    Cut value:
        Cut(x) = sum_{(i,j) in E} w_{ij} * (x_i + x_j - 2 * x_i * x_j)
    To pose as a QUBO minimization: min -Cut(x) = min x^T Q x
    """
    def __init__(self, num_nodes, edges):
        """
        num_nodes: integer n
        edges: list of tuples (u, v, weight)
        """
        self.n = num_nodes
        self.edges = edges
        self.Q = np.zeros((num_nodes, num_nodes))
        self._build_qubo_matrix()

    def _build_qubo_matrix(self):
        # min sum_{(u,v)} w_{uv} * (2 x_u x_v - x_u - x_v)
        for u, v, w in self.edges:
            self.Q[u, u] -= w
            self.Q[v, v] -= w
            self.Q[u, v] += w
            self.Q[v, u] += w  # Symmetric representation

    def evaluate_cut(self, x):
        """Calculates total cut weight for binary string x."""
        cut = 0.0
        for u, v, w in self.edges:
            if x[u] != x[v]:
                cut += w
        return cut

    def to_ising(self):
        """
        Converts QUBO to Ising Hamiltonian:
        x_i = (1 - s_i) / 2 where s_i in {-1, +1}
        H(s) = sum_{i < j} J_{ij} s_i s_j + sum_i h_i s_i + offset
        For Max-Cut: Cut = 0.5 * sum_{(u,v)} w_{uv} (1 - s_u s_v)
        So minimizing energy H(s) = 0.5 * sum_{(u,v)} w_{uv} s_u s_v maximizes cut!
        """
        J = np.zeros((self.n, self.n))
        h = np.zeros(self.n)
        for u, v, w in self.edges:
            if u < v:
                J[u, v] = 0.5 * w
            else:
                J[v, u] = 0.5 * w
        offset = 0.0
        return J, h, offset

    def solve_classical_exact(self):
        """Exhaustive brute force over all 2^n configurations (classical baseline)."""
        best_cut = -1.0
        best_x = None
        for i in range(1 << self.n):
            x = np.array([(i >> bit) & 1 for bit in range(self.n)])
            cut = self.evaluate_cut(x)
            if cut > best_cut:
                best_cut = cut
                best_x = x.copy()
        return {
            "method": "Classical Exact (Brute Force)",
            "max_cut": float(best_cut),
            "best_partition": best_x.tolist()
        }

    def solve_simulated_annealing(self, steps=1000, t_init=10.0, t_final=0.01, seed=42):
        """Simulated Annealing heuristic on QUBO landscape."""
        np.random.seed(seed)
        current_x = np.random.randint(0, 2, size=self.n)
        current_cut = self.evaluate_cut(current_x)

        best_x = current_x.copy()
        best_cut = current_cut

        t = t_init
        alpha = (t_final / t_init) ** (1.0 / steps)

        for _ in range(steps):
            # Propose flipping a random bit
            flip_idx = np.random.randint(0, self.n)
            proposed_x = current_x.copy()
            proposed_x[flip_idx] = 1 - proposed_x[flip_idx]
            proposed_cut = self.evaluate_cut(proposed_x)

            delta_cut = proposed_cut - current_cut
            # We want to MAXIMIZE cut, so delta_cut > 0 is an improvement
            if delta_cut > 0 or np.random.rand() < np.exp(delta_cut / max(t, 1e-8)):
                current_x = proposed_x
                current_cut = proposed_cut
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_x = current_x.copy()
            t *= alpha

        return {
            "method": "Classical Simulated Annealing",
            "max_cut": float(best_cut),
            "best_partition": best_x.tolist()
        }


class ConstrainedFeatureSelectionQUBO:
    """
    Feature selection formulated as QUBO with quadratic penalty for cardinality constraint:
    min x^T S x - f^T x + lambda * (sum x_i - K)^2
    where S is feature redundancy, f is feature relevance, and K is target count.
    """
    def __init__(self, relevance_vector, redundancy_matrix, target_k=3, penalty_lambda=10.0):
        self.n = len(relevance_vector)
        self.f = relevance_vector
        self.S = redundancy_matrix
        self.k = target_k
        self.lam = penalty_lambda
        self.Q = np.zeros((self.n, self.n))
        self._build_qubo()

    def _build_qubo(self):
        # Objective: x^T S x - sum f_i x_i + lam * (x^T 1 1^T x - 2 K 1^T x + K^2)
        # Note: x_i^2 = x_i for binary variables
        for i in range(self.n):
            for j in range(self.n):
                self.Q[i, j] += self.S[i, j] + self.lam * 1.0
            self.Q[i, i] += -self.f[i] - 2.0 * self.lam * self.k

    def evaluate(self, x):
        """Computes cost including penalty."""
        cost = float(x.T @ self.S @ x - self.f @ x + self.lam * (np.sum(x) - self.k) ** 2)
        penalty_violation = abs(np.sum(x) - self.k)
        return cost, penalty_violation

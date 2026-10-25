"""Track B: Discrete Optimization - Combinatorial Solvers.

Implements:
1. Greedy Heuristic (value/weight ratio)
2. Dynamic Programming (exact O(N*W))
3. Branch-and-Bound (exact tree search with fractional LP relaxation pruning)
4. Integer Linear Programming (ILP) Solver via SciPy MILP
"""

import numpy as np
from scipy.optimize import milp, LinearConstraint


class GreedyKnapsackSolver:
    """Greedy approximation for 0/1 Knapsack based on efficiency ratio v_i / w_i."""
    def solve(self, values, weights, capacity):
        n = len(values)
        ratios = [(values[i] / weights[i], i) for i in range(n)]
        ratios.sort(key=lambda x: x[0], reverse=True)

        total_value = 0.0
        total_weight = 0.0
        selected = np.zeros(n, dtype=int)

        for _, idx in ratios:
            if total_weight + weights[idx] <= capacity:
                selected[idx] = 1
                total_value += values[idx]
                total_weight += weights[idx]

        return {
            "method": "Greedy",
            "total_value": float(total_value),
            "total_weight": float(total_weight),
            "selected": selected.tolist()
        }


class DynamicProgrammingKnapsackSolver:
    """Exact pseudo-polynomial Dynamic Programming solver for 0/1 Knapsack."""
    def solve(self, values, weights, capacity):
        n = len(values)
        cap = int(capacity)
        # dp[i][w] = max value using subset of first i items with weight limit w
        dp = np.zeros((n + 1, cap + 1), dtype=float)

        for i in range(1, n + 1):
            v = values[i - 1]
            w = int(weights[i - 1])
            for current_w in range(cap + 1):
                if w <= current_w:
                    dp[i, current_w] = max(dp[i - 1, current_w], dp[i - 1, current_w - w] + v)
                else:
                    dp[i, current_w] = dp[i - 1, current_w]

        # Backtrack to reconstruct chosen items
        selected = np.zeros(n, dtype=int)
        w_rem = cap
        for i in range(n, 0, -1):
            if dp[i, w_rem] != dp[i - 1, w_rem]:
                selected[i - 1] = 1
                w_rem -= int(weights[i - 1])

        total_val = float(dp[n, cap])
        total_w = float(np.sum(weights * selected))
        return {
            "method": "Dynamic Programming",
            "total_value": total_val,
            "total_weight": total_w,
            "selected": selected.tolist()
        }


class BranchAndBoundKnapsackSolver:
    """Exact Branch-and-Bound solver using fractional knapsack relaxation as upper bound."""
    def __init__(self):
        self.best_value = 0.0
        self.best_selected = None
        self.nodes_explored = 0

    def _fractional_bound(self, level, current_weight, current_val, values, weights, capacity, sorted_indices):
        n = len(values)
        if current_weight >= capacity:
            return 0.0

        bound = current_val
        w = current_weight

        for i in range(level, n):
            idx = sorted_indices[i]
            if w + weights[idx] <= capacity:
                w += weights[idx]
                bound += values[idx]
            else:
                # Fractional inclusion of item
                fraction = (capacity - w) / weights[idx]
                bound += values[idx] * fraction
                break
        return bound

    def solve(self, values, weights, capacity):
        n = len(values)
        # Sort items by descending ratio
        ratios = [(values[i] / weights[i], i) for i in range(n)]
        ratios.sort(key=lambda x: x[0], reverse=True)
        sorted_indices = [idx for _, idx in ratios]

        self.best_value = 0.0
        self.best_selected = np.zeros(n, dtype=int)
        self.nodes_explored = 0

        current_choice = np.zeros(n, dtype=int)

        def search(level, current_weight, current_val):
            self.nodes_explored += 1
            if current_weight <= capacity and current_val > self.best_value:
                self.best_value = current_val
                self.best_selected = current_choice.copy()

            if level >= n or current_weight >= capacity:
                return

            # Compute upper bound
            upper_bound = self._fractional_bound(level, current_weight, current_val, values, weights, capacity, sorted_indices)
            if upper_bound <= self.best_value:
                # Prune branch
                return

            idx = sorted_indices[level]

            # Branch 1: Include item idx if capacity allows
            if current_weight + weights[idx] <= capacity:
                current_choice[idx] = 1
                search(level + 1, current_weight + weights[idx], current_val + values[idx])
                current_choice[idx] = 0

            # Branch 2: Exclude item idx
            search(level + 1, current_weight, current_val)

        search(0, 0.0, 0.0)

        return {
            "method": "Branch-and-Bound",
            "total_value": float(self.best_value),
            "total_weight": float(np.sum(weights * self.best_selected)),
            "selected": self.best_selected.tolist(),
            "nodes_explored": self.nodes_explored
        }


class ILPKnapsackSolver:
    """Exact Integer Linear Programming (ILP) solver using SciPy MILP."""
    def solve(self, values, weights, capacity):
        n = len(values)
        # SciPy MILP minimizes c^T x, so c = -values
        c = -np.array(values, dtype=float)
        # Constraint: weights^T x <= capacity
        A = np.array([weights], dtype=float)
        constraints = LinearConstraint(A, -np.inf, capacity)
        # Integrality: 1 for binary integer variables x_i in {0, 1}
        integrality = np.ones(n, dtype=int)
        bounds = [(0, 1) for _ in range(n)]
        from scipy.optimize import Bounds
        lower_b = np.zeros(n)
        upper_b = np.ones(n)

        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=Bounds(lower_b, upper_b))

        selected = np.round(res.x).astype(int)
        total_val = float(-res.fun)
        total_w = float(np.sum(weights * selected))

        return {
            "method": "ILP (SciPy)",
            "total_value": total_val,
            "total_weight": total_w,
            "selected": selected.tolist(),
            "status": res.message
        }

"""Track H: DBMS Optimization - Index Selection via 0/1 Knapsack ILP.

Formulates:
Maximize total query workload execution latency reduction:
    max sum_{i=1}^m Delta T_i * x_i
subject to:
    sum_{i=1}^m Size_i * x_i <= Storage_Budget_B
    x_i in {0, 1}
"""

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


class IndexSelector:
    def __init__(self, candidate_indices, storage_budget_kb=130):
        """
        candidate_indices: list of dicts with:
        {'name': str, 'table': str, 'column': str, 'size_kb': float, 'benefit_score': float, 'sql': str}
        """
        self.candidates = candidate_indices
        self.budget = storage_budget_kb

    def solve_greedy(self):
        """Greedy selection ordering by benefit / size ratio."""
        ratios = [(c["benefit_score"] / c["size_kb"], i) for i, c in enumerate(self.candidates)]
        ratios.sort(key=lambda x: x[0], reverse=True)

        selected = []
        total_size = 0.0
        total_benefit = 0.0

        for _, idx in ratios:
            cand = self.candidates[idx]
            if total_size + cand["size_kb"] <= self.budget:
                selected.append(cand)
                total_size += cand["size_kb"]
                total_benefit += cand["benefit_score"]

        return {
            "method": "Greedy Index Selection",
            "selected_indices": selected,
            "total_size_kb": total_size,
            "total_benefit": total_benefit
        }

    def solve_ilp(self):
        """Exact 0/1 Knapsack ILP solver via SciPy MILP."""
        n = len(self.candidates)
        # Minimize -benefit
        c = -np.array([cand["benefit_score"] for cand in self.candidates], dtype=float)
        sizes = np.array([cand["size_kb"] for cand in self.candidates], dtype=float)

        A = np.array([sizes])
        constraints = LinearConstraint(A, -np.inf, self.budget)
        integrality = np.ones(n, dtype=int)
        bounds = Bounds(np.zeros(n), np.ones(n))

        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        sol = np.round(res.x).astype(int)

        selected = [self.candidates[i] for i in range(n) if sol[i] == 1]
        total_size = sum(c["size_kb"] for c in selected)
        total_benefit = sum(c["benefit_score"] for c in selected)

        return {
            "method": "ILP Optimal (Knapsack)",
            "selected_indices": selected,
            "total_size_kb": total_size,
            "total_benefit": total_benefit
        }

    def apply_indices_to_db(self, conn, selected_indices):
        """Executes CREATE INDEX DDL on the SQLite connection."""
        cur = conn.cursor()
        for idx in selected_indices:
            cur.execute(idx["sql"])
        conn.commit()

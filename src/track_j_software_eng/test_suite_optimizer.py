"""Track J: Software-Engineering Optimization - Test-Suite Minimization & Prioritization.

Formulates:
Given a regression test suite T = {t_1, ..., t_m} with execution runtimes dur_i,
and code branches / requirements B = {b_1, ..., b_k} with coverage matrix C_{i, j}:
Maximize cumulative code branch coverage:
    max sum_{j=1}^k y_j
subject to:
    sum_{i=1}^m dur_i * x_i <= Time_Budget_T
    y_j <= sum_{i: C_{ij} == 1} x_i    for all branches j in {1..k}
    x_i in {0, 1},  y_j in {0, 1}
"""

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


class TestSuiteOptimizer:
    def __init__(self, num_tests=40, num_branches=100, seed=42):
        np.random.seed(seed)
        self.m = num_tests
        self.k = num_branches

        # Test execution durations in milliseconds (5ms to 120ms)
        self.durations = np.random.uniform(5.0, 120.0, size=num_tests)

        # Binary coverage matrix: C[i, j] = 1 if test i covers branch j
        self.coverage_matrix = np.zeros((num_tests, num_branches), dtype=int)
        for i in range(num_tests):
            n_covered = np.random.randint(5, 25)
            branches = np.random.choice(num_branches, size=n_covered, replace=False)
            self.coverage_matrix[i, branches] = 1

    def solve_naive(self, time_budget_ms):
        """Naive baseline: runs tests sequentially in random order until budget is exhausted."""
        total_time = 0.0
        selected = []
        covered_branches = set()

        for i in range(self.m):
            if total_time + self.durations[i] <= time_budget_ms:
                selected.append(i)
                total_time += self.durations[i]
                for j in range(self.k):
                    if self.coverage_matrix[i, j] == 1:
                        covered_branches.add(j)

        return {
            "method": "Naive Sequential Selection",
            "selected_tests": selected,
            "total_time_ms": total_time,
            "covered_branches": len(covered_branches),
            "coverage_pct": (len(covered_branches) / self.k) * 100.0
        }

    def solve_greedy_additional_coverage(self, time_budget_ms):
        """Greedy heuristic: repeatedly selects the test maximizing new uncovered branches per ms."""
        remaining_tests = set(range(self.m))
        selected = []
        total_time = 0.0
        covered_branches = set()

        while remaining_tests:
            best_test = None
            best_ratio = -1.0
            best_new_covered = 0

            for i in remaining_tests:
                if total_time + self.durations[i] <= time_budget_ms:
                    test_branches = set(np.where(self.coverage_matrix[i] == 1)[0])
                    new_branches = test_branches - covered_branches
                    gain = len(new_branches)
                    ratio = gain / max(self.durations[i], 1e-6)

                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_test = i
                        best_new_covered = gain

            if best_test is not None and best_new_covered > 0:
                selected.append(best_test)
                total_time += self.durations[best_test]
                covered_branches.update(np.where(self.coverage_matrix[best_test] == 1)[0])
                remaining_tests.remove(best_test)
            else:
                break

        return {
            "method": "Greedy Additional Coverage",
            "selected_tests": selected,
            "total_time_ms": total_time,
            "covered_branches": len(covered_branches),
            "coverage_pct": (len(covered_branches) / self.k) * 100.0
        }

    def solve_ilp(self, time_budget_ms):
        """
        Exact Integer Linear Programming (ILP) formulation:
        Decision variables:
        x_i in {0, 1} for i=0..m-1 (tests)
        y_j in {0, 1} for j=0..k-1 (branches covered)
        Total variables: m + k

        Maximize sum_j y_j  <=>  Minimize -sum_j y_j
        subject to:
            sum_i dur_i * x_i <= Budget
            y_j - sum_{i: C_ij=1} x_i <= 0 for all j
        """
        m, k = self.m, self.k
        total_vars = m + k

        # Objective: minimize -sum y_j
        c = np.zeros(total_vars)
        c[m:] = -1.0  # Coefficient -1 for each y_j

        # Constraint 1: Time budget: sum_i dur_i * x_i <= Budget
        A_budget = np.zeros(total_vars)
        A_budget[:m] = self.durations

        # Constraint 2: Coverage implication: y_j - sum_{i: C_ij=1} x_i <= 0
        A_coverage = []
        for j in range(k):
            row = np.zeros(total_vars)
            row[m + j] = 1.0  # y_j
            for i in range(m):
                if self.coverage_matrix[i, j] == 1:
                    row[i] = -1.0
            A_coverage.append(row)

        A_all = np.vstack([A_budget, np.array(A_coverage)])
        lb_all = -np.inf * np.ones(len(A_all))
        ub_all = np.concatenate([[time_budget_ms], np.zeros(k)])

        constraints = LinearConstraint(A_all, lb_all, ub_all)
        integrality = np.ones(total_vars, dtype=int)
        bounds = Bounds(np.zeros(total_vars), np.ones(total_vars))

        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        sol = np.round(res.x).astype(int)

        selected_tests = [i for i in range(m) if sol[i] == 1]
        covered_branches = [j for j in range(k) if sol[m + j] == 1]
        total_time = float(np.sum(self.durations[selected_tests]))

        return {
            "method": "ILP Optimal Formulation",
            "selected_tests": selected_tests,
            "total_time_ms": total_time,
            "covered_branches": len(covered_branches),
            "coverage_pct": (len(covered_branches) / k) * 100.0
        }

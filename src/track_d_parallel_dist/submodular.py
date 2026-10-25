"""Track D: Submodular Optimization.

Implements:
1. Maximum Coverage / Facility Location submodular objective function.
2. Diminishing returns property verification test:
   f(A U {x}) - f(A) >= f(B U {x}) - f(B) for all A subseteq B.
3. Greedy selection algorithm.
4. Approximation guarantee verification: f(S_greedy) >= (1 - 1/e) * f(S_opt).
"""

import numpy as np


class MaximumCoverageSubmodular:
    """
    Given a ground set of items U, and a family of subsets S_1, ..., S_m,
    choose a collection of at most k subsets to maximize |Union_{i in Selected} S_i|.
    This is a classic monotone submodular function with proven (1 - 1/e) guarantee.
    """
    def __init__(self, ground_set_size, num_candidate_sets, seed=42):
        np.random.seed(seed)
        self.U_size = ground_set_size
        self.m = num_candidate_sets
        self.sets = []
        for _ in range(num_candidate_sets):
            # Each candidate set covers a random subset of ground elements
            k_items = np.random.randint(5, max(6, ground_set_size // 3))
            covered_items = set(np.random.choice(ground_set_size, size=k_items, replace=False))
            self.sets.append(covered_items)

    def evaluate(self, selected_indices):
        """Computes f(S) = |Union_{i in selected} S_i|."""
        if not selected_indices:
            return 0
        covered = set()
        for idx in selected_indices:
            covered.update(self.sets[idx])
        return len(covered)

    def marginal_gain(self, current_selection, candidate_idx):
        """Computes Delta(x | S) = f(S U {x}) - f(S)."""
        if candidate_idx in current_selection:
            return 0
        current_covered = set()
        for idx in current_selection:
            current_covered.update(self.sets[idx])
        new_items = self.sets[candidate_idx] - current_covered
        return len(new_items)

    def verify_diminishing_returns(self, num_checks=50, seed=42):
        """
        Numerically verifies the defining diminishing returns condition:
        For random pairs A subseteq B and element e not in B:
        f(A U {e}) - f(A) >= f(B U {e}) - f(B).
        """
        np.random.seed(seed)
        violations = 0
        margins_A = []
        margins_B = []

        for _ in range(num_checks):
            # Sample subset B of size 3 to 6
            b_size = np.random.randint(3, 7)
            b_indices = set(np.random.choice(self.m, size=b_size, replace=False))

            # Sample subset A subseteq B
            a_size = np.random.randint(1, b_size)
            a_indices = set(np.random.choice(list(b_indices), size=a_size, replace=False))

            # Element e not in B
            remaining = list(set(range(self.m)) - b_indices)
            if not remaining:
                continue
            e = np.random.choice(remaining)

            gain_A = self.marginal_gain(a_indices, e)
            gain_B = self.marginal_gain(b_indices, e)

            margins_A.append(gain_A)
            margins_B.append(gain_B)

            if gain_A < gain_B:
                violations += 1

        return {
            "num_checks": num_checks,
            "violations": violations,
            "is_submodular": violations == 0,
            "avg_margin_A": float(np.mean(margins_A)),
            "avg_margin_B": float(np.mean(margins_B))
        }

    def greedy_select(self, budget_k):
        """Greedy selection: repeatedly choose element with maximum marginal gain."""
        selected = []
        for _ in range(budget_k):
            best_e = None
            best_gain = -1
            for i in range(self.m):
                if i not in selected:
                    gain = self.marginal_gain(selected, i)
                    if gain > best_gain:
                        best_gain = gain
                        best_e = i
            if best_e is not None and best_gain > 0:
                selected.append(best_e)
            else:
                break
        return selected, self.evaluate(selected)

    def exact_optimal_select(self, budget_k):
        """Exact brute-force search over all C(m, k) subsets (for small m)."""
        import itertools
        best_val = -1
        best_subset = None
        for comb in itertools.combinations(range(self.m), budget_k):
            val = self.evaluate(comb)
            if val > best_val:
                best_val = val
                best_subset = list(comb)
        return best_subset, best_val

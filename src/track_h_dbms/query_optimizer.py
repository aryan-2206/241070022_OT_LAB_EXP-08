"""Track H: DBMS Optimization - System-R Dynamic Programming Join-Order Optimizer.

Implements:
1. Selinger Cost Model for query cardinality and disk I/O estimation.
2. System-R Dynamic Programming Join Enumeration over table subsets.
3. Greedy Join Order Selection baseline.
4. Generates rewritten SQL join trees.
"""

import itertools


class JoinOptimizer:
    def __init__(self, table_cardinalities, distinct_values, join_graph):
        """
        table_cardinalities: dict {table_name: row_count}
        distinct_values: dict {(table, col): distinct_value_count}
        join_graph: dict {(table1, table2): (col1, col2)}
        """
        self.cards = table_cardinalities
        self.v_vals = distinct_values
        self.joins = join_graph

    def estimate_cardinality(self, table1, table2):
        """Estimates join size |R bowtie S| using standard uniform independence assumption."""
        key = (table1, table2) if (table1, table2) in self.joins else (table2, table1)
        if key not in self.joins:
            # Cartesian cross-product penalty
            return self.cards[table1] * self.cards[table2] * 0.1

        col1, col2 = self.joins[key]
        v1 = self.v_vals.get((table1, col1), 100)
        v2 = self.v_vals.get((table2, col2), 100)
        return (self.cards[table1] * self.cards[table2]) / max(v1, v2)

    def solve_greedy(self, tables):
        """Greedy join order selection: iteratively joins the relation with minimum intermediate size."""
        remaining = list(tables)
        # Start with smallest table
        current = min(remaining, key=lambda t: self.cards[t])
        remaining.remove(current)
        plan = [current]
        total_cost = self.cards[current]

        while remaining:
            best_t = None
            best_cost = float("inf")
            for cand in remaining:
                cand_cost = self.estimate_cardinality(plan[-1], cand)
                if cand_cost < best_cost:
                    best_cost = cand_cost
                    best_t = cand

            plan.append(best_t)
            remaining.remove(best_t)
            total_cost += best_cost

        return {
            "method": "Greedy Join Ordering",
            "order": plan,
            "estimated_cost": float(total_cost)
        }

    def solve_system_r_dp(self, tables):
        """
        System-R Dynamic Programming over subsets S subseteq {R_1, ..., R_m}:
        DP[S] = min_{R in S} { DP[S \\ {R}] + Cost( (S \\ {R}) bowtie R ) }
        """
        # dp[frozenset(S)] = (best_cost, best_order, estimated_cardinality)
        dp = {}

        # Base case: single tables
        for t in tables:
            dp[frozenset([t])] = (self.cards[t], [t], self.cards[t])

        # Subsets of size 2 to m
        for size in range(2, len(tables) + 1):
            for subset in itertools.combinations(tables, size):
                subset_set = frozenset(subset)
                best_cost = float("inf")
                best_order = None
                best_card = float("inf")

                for t in subset:
                    prev_subset = subset_set - frozenset([t])
                    prev_cost, prev_order, prev_card = dp[prev_subset]

                    # Cost to join prev_order with table t
                    join_card = self.estimate_cardinality(prev_order[-1], t)
                    cost = prev_cost + prev_card + join_card

                    if cost < best_cost:
                        best_cost = cost
                        best_order = prev_order + [t]
                        best_card = join_card

                dp[subset_set] = (best_cost, best_order, best_card)

        all_tables = frozenset(tables)
        final_cost, final_order, _ = dp[all_tables]

        return {
            "method": "System-R Dynamic Programming",
            "order": final_order,
            "estimated_cost": float(final_cost)
        }

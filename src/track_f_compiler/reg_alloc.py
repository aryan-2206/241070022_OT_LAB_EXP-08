"""Track F: Compiler Optimization - Register Allocation via Graph Coloring and ILP.

Formulates Register Allocation as an optimization problem:
Given a set of virtual variables V and K available physical registers,
assign each variable to a register such that no two simultaneously live
(interfering) variables share the same register, minimizing spill code.
"""

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


class RegisterAllocator:
    def __init__(self, num_registers=3):
        self.k = num_registers

    def compute_liveness_and_interference(self, ir):
        """
        Computes variable live ranges and constructs undirected interference graph:
        An edge exists between (u, v) if u and v are simultaneously live.
        """
        # Step 1: Collect all variable names
        variables = set()
        for instr in ir:
            if instr.dest:
                variables.add(instr.dest)
            for arg in [instr.arg1, instr.arg2]:
                if isinstance(arg, str) and not arg.isdigit():
                    variables.add(arg)
        var_list = sorted(list(variables))
        var_to_idx = {v: i for i, v in enumerate(var_list)}
        n_vars = len(var_list)

        # Step 2: Backward scan to find live intervals [start_line, end_line]
        intervals = {v: [float("inf"), -1] for v in var_list}
        for line_no, instr in enumerate(ir):
            if instr.dest:
                intervals[instr.dest][0] = min(intervals[instr.dest][0], line_no)
                intervals[instr.dest][1] = max(intervals[instr.dest][1], line_no)
            for arg in [instr.arg1, instr.arg2]:
                if isinstance(arg, str) and not arg.isdigit():
                    intervals[arg][0] = min(intervals[arg][0], line_no)
                    intervals[arg][1] = max(intervals[arg][1], line_no)

        # Step 3: Interference graph
        adj_matrix = np.zeros((n_vars, n_vars), dtype=int)
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                u, v = var_list[i], var_list[j]
                # Two variables interfere if their live intervals overlap
                start_u, end_u = intervals[u]
                start_v, end_v = intervals[v]
                if max(start_u, start_v) <= min(end_u, end_v):
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1

        return var_list, adj_matrix, intervals

    def solve_chaitin_heuristic(self, var_list, adj_matrix):
        """Chaitin's graph coloring heuristic."""
        n = len(var_list)
        if n == 0:
            return {"method": "Chaitin Heuristic", "assignment": {}, "spills": [], "spill_count": 0}
        degrees = np.sum(adj_matrix, axis=1)
        removed = set()
        stack = []

        # Simplify phase
        for _ in range(n):
            candidate = None
            for i in range(n):
                if i not in removed and degrees[i] < self.k:
                    candidate = i
                    break
            if candidate is None:
                # Potential spill: choose node with highest degree
                active = [i for i in range(n) if i not in removed]
                candidate = max(active, key=lambda i: degrees[i])

            stack.append(candidate)
            removed.add(candidate)
            for j in range(n):
                if adj_matrix[candidate, j] and j not in removed:
                    degrees[j] -= 1

        # Select / Color phase
        color_assignment = {}
        spills = []

        for node_idx in reversed(stack):
            neighbor_colors = set()
            for j in range(n):
                if adj_matrix[node_idx, j] and j in color_assignment:
                    neighbor_colors.add(color_assignment[j])

            # Pick smallest available color in {0, ..., k-1}
            chosen_color = None
            for c in range(self.k):
                if c not in neighbor_colors:
                    chosen_color = c
                    break

            if chosen_color is not None:
                color_assignment[node_idx] = chosen_color
            else:
                spills.append(var_list[node_idx])

        return {
            "method": "Chaitin Heuristic",
            "assignment": {var_list[i]: f"R{c}" for i, c in color_assignment.items()},
            "spills": spills,
            "spill_count": len(spills)
        }

    def solve_ilp(self, var_list, adj_matrix):
        """
        Integer Linear Programming (ILP) formulation for optimal register allocation:
        Minimize total spilled variables: min sum_v s_v
        subject to:
            sum_{r=1}^K x_{v, r} + s_v = 1  for all v
            x_{u, r} + x_{v, r} <= 1         for all (u, v) in E, for all r in {1..K}
        """
        n = len(var_list)
        k = self.k
        if n == 0:
            return {"method": "ILP Optimal (SciPy)", "assignment": {}, "spills": [], "spill_count": 0}

        # Decision variables: x_{v, r} for v in 0..n-1, r in 0..k-1 (n*k vars)
        # plus s_v for v in 0..n-1 (n vars) -> Total vars: n*(k + 1)
        total_vars = n * (k + 1)

        # Objective c: 0 for x_{v,r}, 1 for s_v (minimize spills)
        c = np.zeros(total_vars)
        for v in range(n):
            c[n * k + v] = 1.0  # Spill penalty

        # Constraint 1: sum_r x_{v, r} + s_v == 1 for every variable v
        A_eq = []
        for v in range(n):
            row = np.zeros(total_vars)
            for r in range(k):
                row[v * k + r] = 1.0
            row[n * k + v] = 1.0
            A_eq.append(row)
        A_eq = np.array(A_eq)
        b_eq = np.ones(n)

        # Constraint 2: x_{u, r} + x_{v, r} <= 1 for each interfering edge and register r
        A_ub = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i, j] == 1:
                    for r in range(k):
                        row = np.zeros(total_vars)
                        row[i * k + r] = 1.0
                        row[j * k + r] = 1.0
                        A_ub.append(row)

        if A_ub:
            A_ub = np.array(A_ub)
            b_ub = np.ones(len(A_ub))
            A_all = np.vstack([A_eq, A_ub])
            lb_all = np.concatenate([b_eq, -np.inf * np.ones(len(A_ub))])
            ub_all = np.concatenate([b_eq, b_ub])
        else:
            A_all = A_eq
            lb_all = b_eq
            ub_all = b_eq

        constraints = LinearConstraint(A_all, lb_all, ub_all)
        integrality = np.ones(total_vars, dtype=int)
        bounds = Bounds(np.zeros(total_vars), np.ones(total_vars))

        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)

        sol = np.round(res.x).astype(int)
        assignment = {}
        spills = []

        for v in range(n):
            spilled = sol[n * k + v]
            if spilled == 1:
                spills.append(var_list[v])
            else:
                for r in range(k):
                    if sol[v * k + r] == 1:
                        assignment[var_list[v]] = f"R{r}"
                        break

        return {
            "method": "ILP Optimal (SciPy)",
            "assignment": assignment,
            "spills": spills,
            "spill_count": len(spills)
        }

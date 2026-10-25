"""Track E: EcoRoute Solvers.

Implements:
1. Baseline: Greedy Nearest-Neighbor Dispatcher
2. Metaheuristic: Simulated Annealing Router with 2-opt and intra/inter-route swap
3. Exact Solver: Branch-and-Bound for small benchmark instances (N <= 8)
"""

import copy
import itertools
import numpy as np


class GreedyNearestNeighborSolver:
    """Greedy dispatch baseline: sequentially routes to the nearest unvisited feasible stop."""
    def solve(self, network):
        unvisited = set(range(1, network.num_nodes))
        routes = []

        for v in range(network.num_vehicles):
            route = [0]
            current_node = 0
            current_load = 0.0

            while unvisited:
                # Find nearest unvisited node that does not violate vehicle capacity
                best_next = None
                best_dist = float("inf")

                for candidate in unvisited:
                    cand_demand = network.demands[candidate]
                    if current_load + cand_demand <= network.capacity:
                        d = network.dist_matrix[current_node, candidate]
                        if d < best_dist:
                            best_dist = d
                            best_next = candidate

                if best_next is not None:
                    route.append(best_next)
                    current_load += network.demands[best_next]
                    current_node = best_next
                    unvisited.remove(best_next)
                else:
                    # Vehicle is full or cannot reach more stops
                    break

            route.append(0)  # Return to depot
            routes.append(route)

        # If any unvisited nodes remain, greedily append to the vehicle with minimum additional distance
        for missing in list(unvisited):
            best_v = 0
            best_cost = float("inf")
            for idx, r in enumerate(routes):
                candidate_route = r[:-1] + [missing, 0]
                met = network.evaluate_route(candidate_route)
                if met["dist"] < best_cost:
                    best_cost = met["dist"]
                    best_v = idx
            routes[best_v] = routes[best_v][:-1] + [missing, 0]

        eval_res = network.evaluate_fleet_solution(routes)
        return {
            "method": "Greedy Nearest-Neighbor (Baseline)",
            "routes": routes,
            "metrics": eval_res
        }


class SimulatedAnnealingRouter:
    """Simulated Annealing metaheuristic for carbon-aware vehicle routing."""
    def __init__(self, t_init=50.0, t_final=0.05, steps=1200, seed=42):
        self.t_init = t_init
        self.t_final = t_final
        self.steps = steps
        self.seed = seed

    def _apply_neighbor_operator(self, routes):
        """Generates a neighbor solution using 2-opt, relocate, or inter-route swap."""
        new_routes = [r.copy() for r in routes]
        op = np.random.choice(["2opt", "relocate", "swap"])

        # Pick random active vehicle
        v_indices = [i for i, r in enumerate(new_routes) if len(r) > 2]
        if not v_indices:
            return new_routes

        v1 = np.random.choice(v_indices)
        r1 = new_routes[v1]

        if op == "2opt" and len(r1) >= 5:
            # 2-opt reversal of customer subsegment
            i, j = sorted(np.random.choice(range(1, len(r1) - 1), size=2, replace=False))
            r1[i:j+1] = reversed(r1[i:j+1])
            new_routes[v1] = r1

        elif op == "relocate" and len(new_routes) > 1 and len(r1) > 3:
            # Move a customer stop from vehicle v1 to vehicle v2
            v2 = np.random.choice([i for i in range(len(new_routes)) if i != v1])
            r2 = new_routes[v2]
            stop_idx = np.random.randint(1, len(r1) - 1)
            stop = r1.pop(stop_idx)
            # Insert at best or random position in r2
            insert_pos = np.random.randint(1, len(r2))
            r2.insert(insert_pos, stop)
            new_routes[v1] = r1
            new_routes[v2] = r2

        elif op == "swap" and len(new_routes) > 1:
            # Swap one customer stop between two vehicles
            v2 = np.random.choice([i for i in range(len(new_routes)) if i != v1])
            r2 = new_routes[v2]
            if len(r1) > 3 and len(r2) > 3:
                idx1 = np.random.randint(1, len(r1) - 1)
                idx2 = np.random.randint(1, len(r2) - 1)
                r1[idx1], r2[idx2] = r2[idx2], r1[idx1]
                new_routes[v1] = r1
                new_routes[v2] = r2

        return new_routes

    def solve(self, network, initial_routes=None):
        np.random.seed(self.seed)

        if initial_routes is None:
            # Start from greedy solution
            init_res = GreedyNearestNeighborSolver().solve(network)
            current_routes = init_res["routes"]
        else:
            current_routes = copy.deepcopy(initial_routes)

        current_eval = network.evaluate_fleet_solution(current_routes)
        current_cost = current_eval["cost"]

        best_routes = copy.deepcopy(current_routes)
        best_cost = current_cost
        best_eval = current_eval

        cost_history = [current_cost]
        t = self.t_init
        cooling_rate = (self.t_final / self.t_init) ** (1.0 / self.steps)

        for step in range(self.steps):
            proposed_routes = self._apply_neighbor_operator(current_routes)
            prop_eval = network.evaluate_fleet_solution(proposed_routes)
            prop_cost = prop_eval["cost"]

            delta_cost = prop_cost - current_cost

            # Minimize cost: accept if improved or by Metropolis probability
            if delta_cost < 0 or np.random.rand() < np.exp(-delta_cost / max(t, 1e-6)):
                current_routes = proposed_routes
                current_cost = prop_cost
                current_eval = prop_eval

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_routes = copy.deepcopy(current_routes)
                    best_eval = prop_eval

            cost_history.append(best_cost)
            t *= cooling_rate

        return {
            "method": "Simulated Annealing (EcoRoute)",
            "routes": best_routes,
            "metrics": best_eval,
            "cost_history": cost_history
        }


class ExactSmallInstanceSolver:
    """Exact exhaustive solver for small verification benchmarks (N <= 8)."""
    def solve(self, network):
        assert network.num_nodes <= 8, "Exact solver only practical for N <= 8"
        nodes = list(range(1, network.num_nodes))

        best_cost = float("inf")
        best_routes = None
        best_metrics = None

        # Partition nodes across vehicles and permute visit orders
        for perm in itertools.permutations(nodes):
            # For 2 vehicles, partition perm at every possible split point
            n_perm = len(perm)
            for split in range(1, n_perm):
                r1 = [0] + list(perm[:split]) + [0]
                r2 = [0] + list(perm[split:]) + [0]
                routes = [r1, r2]
                res = network.evaluate_fleet_solution(routes)
                if res["cost"] < best_cost:
                    best_cost = res["cost"]
                    best_routes = routes
                    best_metrics = res

        return {
            "method": "Exact Exhaustive Search",
            "routes": best_routes,
            "metrics": best_metrics
        }

"""Track E: Main Real-Process Case Study - EcoRoute.
Campus Logistics & Material Collection Optimization.

Formulates:
- 25-Node Campus Logistics Network with coordinates, package demands, time windows, and vehicle specs.
- Objective function: Carbon emissions (load-dependent) + transit distance + time-window penalties.
- Independent feasibility checker.
"""

import numpy as np


class CampusNetwork:
    """Represents the campus map, customer stops, and electric delivery fleet."""
    def __init__(self, num_nodes=20, num_vehicles=3, vehicle_capacity=100.0, seed=42):
        np.random.seed(seed)
        self.num_nodes = num_nodes
        self.num_vehicles = num_vehicles
        self.capacity = vehicle_capacity
        self.battery_range_km = 45.0  # Max km on single charge

        # Node 0 is the Central Logistics Depot at (5.0, 5.0)
        self.coords = np.zeros((num_nodes, 2))
        self.coords[0] = [5.0, 5.0]
        # Nodes 1..N are academic buildings, research labs, hostels across a 10 km x 10 km campus
        self.coords[1:] = np.random.uniform(0.5, 9.5, size=(num_nodes - 1, 2))

        # Demands (kg of lab supplies / parcel deliveries), depot has 0 demand
        self.demands = np.zeros(num_nodes)
        self.demands[1:] = np.random.uniform(5.0, 25.0, size=num_nodes - 1)

        # Time windows [Start, End] in minutes from 08:00 AM (0 to 360 mins)
        self.time_windows = np.zeros((num_nodes, 2))
        for i in range(1, num_nodes):
            earliest = np.random.uniform(10.0, 180.0)
            window_width = np.random.uniform(40.0, 90.0)
            self.time_windows[i] = [earliest, earliest + window_width]

        # Pairwise Euclidean distance matrix (km)
        self.dist_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                self.dist_matrix[i, j] = np.linalg.norm(self.coords[i] - self.coords[j])

        # Carbon emissions model: base vehicle emission + load penalty (kg CO2 per km)
        self.base_emission_per_km = 0.08  # Electric grid equivalent
        self.payload_emission_factor = 0.0012  # per kg payload

    def evaluate_route(self, route):
        """
        Evaluates a single vehicle route starting and ending at depot 0.
        Route: list of node indices, e.g. [0, 4, 2, 7, 0]
        Returns:
            dist: total distance in km
            carbon: total carbon emitted in kg CO2
            lateness: total time window lateness in minutes
            load_violations: amount exceeding capacity
            is_feasible: boolean
        """
        if len(route) <= 2:
            return {"dist": 0.0, "carbon": 0.0, "lateness": 0.0, "load_violation": 0.0, "is_feasible": True}

        total_dist = 0.0
        total_carbon = 0.0
        current_time = 0.0
        current_load = sum(self.demands[node] for node in route[1:-1])
        load_violation = max(0.0, current_load - self.capacity)

        avg_speed_kmh = 25.0  # Campus speed limit 25 km/h -> 2.4 mins per km

        for idx in range(len(route) - 1):
            u = route[idx]
            v = route[idx + 1]
            d = self.dist_matrix[u, v]
            total_dist += d

            # Carbon accounting with diminishing payload as items are dropped off
            emission_rate = self.base_emission_per_km + self.payload_emission_factor * current_load
            total_carbon += emission_rate * d

            # Time progression
            travel_mins = (d / avg_speed_kmh) * 60.0
            current_time += travel_mins

            # Deliver demand at stop v
            if v != 0:
                earliest, latest = self.time_windows[v]
                # Wait if arrived early
                if current_time < earliest:
                    current_time = earliest
                current_load -= self.demands[v]

        # Calculate lateness
        total_lateness = 0.0
        sim_time = 0.0
        for idx in range(len(route) - 1):
            u = route[idx]
            v = route[idx + 1]
            d = self.dist_matrix[u, v]
            sim_time += (d / avg_speed_kmh) * 60.0
            if v != 0:
                earliest, latest = self.time_windows[v]
                if sim_time < earliest:
                    sim_time = earliest
                if sim_time > latest:
                    total_lateness += (sim_time - latest)

        battery_violation = max(0.0, total_dist - self.battery_range_km)
        is_feasible = (load_violation == 0.0) and (battery_violation == 0.0)

        return {
            "dist": total_dist,
            "carbon": total_carbon,
            "lateness": total_lateness,
            "load_violation": load_violation,
            "battery_violation": battery_violation,
            "is_feasible": is_feasible
        }

    def evaluate_fleet_solution(self, routes, lateness_penalty=0.5, violation_penalty=50.0):
        """
        Evaluates a complete fleet routing solution.
        routes: list of routes, one per vehicle.
        """
        total_dist = 0.0
        total_carbon = 0.0
        total_lateness = 0.0
        total_penalty = 0.0
        all_feasible = True

        visited = set()
        for r in routes:
            metrics = self.evaluate_route(r)
            total_dist += metrics["dist"]
            total_carbon += metrics["carbon"]
            total_lateness += metrics["lateness"]
            if not metrics["is_feasible"]:
                all_feasible = False
                total_penalty += violation_penalty * (metrics["load_violation"] + metrics["battery_violation"])
            for node in r[1:-1]:
                visited.add(node)

        # Check coverage of all required customer nodes
        missing = set(range(1, self.num_nodes)) - visited
        if missing:
            all_feasible = False
            total_penalty += len(missing) * violation_penalty * 2.0

        # Multi-objective weighted scalar cost:
        cost = total_carbon + 0.1 * total_dist + lateness_penalty * total_lateness + total_penalty

        return {
            "cost": float(cost),
            "total_dist": float(total_dist),
            "total_carbon": float(total_carbon),
            "total_lateness": float(total_lateness),
            "all_feasible": all_feasible,
            "num_visited": len(visited),
            "missing_stops": len(missing)
        }

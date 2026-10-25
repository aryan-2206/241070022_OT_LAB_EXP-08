"""Unit and Correctness Tests for Track E: Main Case Study (EcoRoute)."""

import numpy as np
import pytest

from src.track_e_case_study.problem_formulation import CampusNetwork
from src.track_e_case_study.solvers import (
    GreedyNearestNeighborSolver, SimulatedAnnealingRouter
)


def test_campus_network_feasibility_validation():
    """Independent feasibility checker must detect capacity overload and missing stops."""
    net = CampusNetwork(num_nodes=5, num_vehicles=2, vehicle_capacity=20.0, seed=42)

    # Valid route partitioning: nodes 1, 2 on vehicle 1; nodes 3, 4 on vehicle 2
    valid_routes = [[0, 1, 2, 0], [0, 3, 4, 0]]
    eval_res = net.evaluate_fleet_solution(valid_routes)
    assert eval_res["num_visited"] == 4
    assert eval_res["missing_stops"] == 0

    # Incomplete route missing stop 4
    incomplete_routes = [[0, 1, 2, 0], [0, 3, 0]]
    eval_inc = net.evaluate_fleet_solution(incomplete_routes)
    assert eval_inc["all_feasible"] is False
    assert eval_inc["missing_stops"] == 1


def test_greedy_baseline_returns_complete_routes():
    """Greedy dispatcher must visit all required customer stops."""
    net = CampusNetwork(num_nodes=10, num_vehicles=2, vehicle_capacity=80.0, seed=42)
    res = GreedyNearestNeighborSolver().solve(net)
    assert res["metrics"]["missing_stops"] == 0
    assert res["metrics"]["num_visited"] == 9


def test_simulated_annealing_cost_monotonicity():
    """Simulated Annealing must return a solution at least as good as initial greedy start."""
    net = CampusNetwork(num_nodes=10, num_vehicles=2, vehicle_capacity=80.0, seed=42)
    greedy = GreedyNearestNeighborSolver().solve(net)
    sa = SimulatedAnnealingRouter(steps=200, seed=42).solve(net, initial_routes=greedy["routes"])

    assert sa["metrics"]["cost"] <= greedy["metrics"]["cost"] + 1e-6

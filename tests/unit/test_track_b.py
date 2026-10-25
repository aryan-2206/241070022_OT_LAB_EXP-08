"""Unit and Correctness Tests for Track B: Discrete Optimization."""

import numpy as np
import pytest

from src.track_b_discrete.solvers import (
    GreedyKnapsackSolver, DynamicProgrammingKnapsackSolver,
    BranchAndBoundKnapsackSolver, ILPKnapsackSolver
)
from src.track_b_discrete.qubo_formulation import MaxCutQUBO, ConstrainedFeatureSelectionQUBO


def test_knapsack_exact_solvers_consistency():
    """Branch-and-Bound, Dynamic Programming, and ILP must return the identical global optimum."""
    values = np.array([10, 40, 30, 50])
    weights = np.array([5, 4, 6, 3])
    capacity = 10

    dp = DynamicProgrammingKnapsackSolver().solve(values, weights, capacity)
    bnb = BranchAndBoundKnapsackSolver().solve(values, weights, capacity)
    ilp = ILPKnapsackSolver().solve(values, weights, capacity)

    assert dp["total_value"] == bnb["total_value"], "DP and BnB values do not match"
    assert dp["total_value"] == ilp["total_value"], "DP and ILP values do not match"
    assert dp["total_weight"] <= capacity
    assert bnb["total_weight"] <= capacity
    assert ilp["total_weight"] <= capacity


def test_greedy_feasibility_and_approximation():
    """Greedy heuristic must always produce a feasible solution."""
    values = np.array([60, 100, 120])
    weights = np.array([10, 20, 30])
    capacity = 50

    greedy = GreedyKnapsackSolver().solve(values, weights, capacity)
    assert greedy["total_weight"] <= capacity
    assert greedy["total_value"] > 0


def test_maxcut_qubo_and_ising_mapping():
    """Verifies that Max-Cut QUBO formulation matches graph cut evaluation."""
    edges = [(0, 1, 1.0), (1, 2, 2.0), (0, 2, 1.5)]
    mc = MaxCutQUBO(num_nodes=3, edges=edges)

    # Test partition x = [0, 1, 0] -> cuts edge (0,1) weight 1.0 and (1,2) weight 2.0 -> cut = 3.0
    x = np.array([0, 1, 0])
    assert mc.evaluate_cut(x) == 3.0

    # Verify Ising matrix properties
    J, h, _ = mc.to_ising()
    assert J.shape == (3, 3)
    assert np.allclose(h, 0.0)  # Max-Cut has no linear field terms


def test_constrained_feature_selection_penalty():
    """Penalty term must penalize solutions violating cardinality constraint K."""
    relevance = np.array([0.9, 0.8, 0.2, 0.1])
    redundancy = np.zeros((4, 4))
    target_k = 2
    fs = ConstrainedFeatureSelectionQUBO(relevance, redundancy, target_k=target_k, penalty_lambda=50.0)

    # Feasible solution: 2 features selected
    x_feas = np.array([1, 1, 0, 0])
    cost_feas, viol_feas = fs.evaluate(x_feas)
    assert viol_feas == 0

    # Infeasible solution: 3 features selected (penalty applies)
    x_infeas = np.array([1, 1, 1, 0])
    cost_infeas, viol_infeas = fs.evaluate(x_infeas)
    assert viol_infeas == 1
    assert cost_infeas > cost_feas, "Penalty failed to penalize infeasible solution"

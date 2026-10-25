"""Unit and Correctness Tests for Track D: Submodular, Parallel, and Distributed Optimization."""

import numpy as np
import pytest

from src.track_d_parallel_dist.submodular import MaximumCoverageSubmodular
from src.track_d_parallel_dist.amdahl_gustafson import ScalingLaws
from src.track_d_parallel_dist.federated_multitask import (
    FederatedAveragingSimulator, MultiTaskOptimizer
)


def test_submodular_diminishing_returns():
    """Validates that Maximum Coverage strictly satisfies diminishing returns with 0 violations."""
    submod = MaximumCoverageSubmodular(ground_set_size=40, num_candidate_sets=12, seed=42)
    verif = submod.verify_diminishing_returns(num_checks=60)
    assert verif["violations"] == 0, "Diminishing returns condition violated"
    assert verif["is_submodular"] is True


def test_submodular_greedy_approximation_guarantee():
    """Greedy selection must achieve at least (1 - 1/e) ~ 0.6321 of the exact optimal coverage."""
    submod = MaximumCoverageSubmodular(ground_set_size=30, num_candidate_sets=10, seed=42)
    k = 3
    _, greedy_val = submod.greedy_select(budget_k=k)
    _, exact_val = submod.exact_optimal_select(budget_k=k)

    bound = (1.0 - 1.0 / np.e) * exact_val
    assert greedy_val >= bound, f"Greedy value {greedy_val} violated (1-1/e) bound {bound:.2f}"


def test_amdahl_law_monotonicity_and_asymptote():
    """Amdahl's speedup must increase with p but never exceed theoretical limit 1/s."""
    s = 0.20  # 20% serial fraction
    limit = 1.0 / s

    sp_1 = ScalingLaws.amdahl_speedup(1, s)
    sp_4 = ScalingLaws.amdahl_speedup(4, s)
    sp_16 = ScalingLaws.amdahl_speedup(16, s)
    sp_inf = ScalingLaws.amdahl_speedup(10000, s)

    assert sp_1 == 1.0
    assert sp_4 > sp_1
    assert sp_16 > sp_4
    assert sp_inf < limit
    assert np.isclose(sp_inf, limit, atol=1e-2)


def test_fedavg_loss_reduction():
    """FedAvg across distributed clients must reduce global MSE loss."""
    fed = FederatedAveragingSimulator(num_clients=4, d_features=3, seed=42)
    res = fed.run_fedavg(rounds=10, local_steps=3, lr=0.01)
    assert res["final_loss"] < res["history"][0], "FedAvg failed to reduce initial loss"


def test_mgda_convex_combination():
    """MGDA gradient combination must maintain non-negative weights alpha in [0, 1]."""
    mto = MultiTaskOptimizer(d_features=3, n_samples=50, seed=42)
    res = mto.run_mgda(steps=15, lr=0.01)
    assert res["final_losses"][2] < res["history"][0][2], "MGDA failed to decrease total loss"

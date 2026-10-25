"""Unit and Correctness Tests for Track J: Software-Engineering Optimization."""

import pytest

from src.track_j_software_eng.test_suite_optimizer import TestSuiteOptimizer
from src.track_j_software_eng.runner import count_codebase_lines


def test_codebase_scale_requirement():
    """Validates that the project codebase satisfies the >= 1,000 LOC requirement."""
    total_loc, _ = count_codebase_lines(root_dir="src")
    assert total_loc >= 1000, f"Codebase scale requirement failed: {total_loc} < 1000 LOC"


def test_time_budget_constraint_satisfaction():
    """Selected test subset must not exceed the allowed CI execution time budget."""
    opt = TestSuiteOptimizer(num_tests=30, num_branches=50, seed=42)
    budget = 400.0

    naive_res = opt.solve_naive(time_budget_ms=budget)
    greedy_res = opt.solve_greedy_additional_coverage(time_budget_ms=budget)
    ilp_res = opt.solve_ilp(time_budget_ms=budget)

    assert naive_res["total_time_ms"] <= budget
    assert greedy_res["total_time_ms"] <= budget
    assert ilp_res["total_time_ms"] <= budget


def test_optimization_advantage_over_naive():
    """Greedy additional coverage and ILP must achieve higher branch coverage than naive sequential selection."""
    opt = TestSuiteOptimizer(num_tests=40, num_branches=80, seed=42)
    budget = 350.0

    naive_res = opt.solve_naive(time_budget_ms=budget)
    greedy_res = opt.solve_greedy_additional_coverage(time_budget_ms=budget)
    ilp_res = opt.solve_ilp(time_budget_ms=budget)

    assert greedy_res["coverage_pct"] >= naive_res["coverage_pct"]
    assert ilp_res["coverage_pct"] >= greedy_res["coverage_pct"]

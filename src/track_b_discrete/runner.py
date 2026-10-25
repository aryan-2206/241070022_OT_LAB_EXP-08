"""Track B: Discrete Optimization - Benchmark Runner & Experiment Suite.

Executes:
1. Combinatorial Knapsack Benchmark: Greedy vs DP vs Branch-and-Bound vs ILP.
2. Max-Cut QUBO & Ising formulation with classical exact & simulated annealing baselines.
3. Constrained Feature Selection with penalty terms.
4. Generates comparison tables and plots.
"""

import os
import sys
import time
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import matplotlib.pyplot as plt

from src.track_b_discrete.solvers import (
    GreedyKnapsackSolver, DynamicProgrammingKnapsackSolver,
    BranchAndBoundKnapsackSolver, ILPKnapsackSolver
)
from src.track_b_discrete.qubo_formulation import MaxCutQUBO, ConstrainedFeatureSelectionQUBO


def run_knapsack_experiment():
    """Compares Greedy, DP, Branch-and-Bound, and ILP on a benchmark knapsack instance."""
    # Test instance with 12 items
    np.random.seed(42)
    values = np.array([24, 18, 12, 10, 40, 30, 22, 15, 28, 35, 14, 20])
    weights = np.array([10, 8, 6, 5, 18, 12, 11, 7, 14, 16, 8, 10])
    capacity = 45

    solvers = {
        "Greedy": GreedyKnapsackSolver(),
        "Dynamic Programming": DynamicProgrammingKnapsackSolver(),
        "Branch-and-Bound": BranchAndBoundKnapsackSolver(),
        "ILP (SciPy)": ILPKnapsackSolver()
    }

    results = []
    for name, solver in solvers.items():
        start_t = time.perf_counter()
        res = solver.solve(values, weights, capacity)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        results.append({
            "Problem": "0/1 Knapsack (N=12, C=45)",
            "Solver": name,
            "Total_Value": res["total_value"],
            "Total_Weight": res["total_weight"],
            "Capacity": capacity,
            "Elapsed_ms": elapsed_ms,
            "Optimality_Gap_%": 0.0  # Will compute against DP/ILP optimum
        })

    # Set exact optimal value from DP
    opt_val = max(r["Total_Value"] for r in results if r["Solver"] in ["Dynamic Programming", "ILP (SciPy)"])
    for r in results:
        r["Optimality_Gap_%"] = ((opt_val - r["Total_Value"]) / opt_val) * 100.0

    return results


def run_qubo_maxcut_experiment():
    """Formulates and solves a 5-node Max-Cut graph classically as the baseline for Track C."""
    # 5-node graph with 7 weighted edges
    nodes = 5
    edges = [
        (0, 1, 1.0),
        (0, 2, 2.0),
        (1, 2, 1.0),
        (1, 3, 3.0),
        (2, 3, 2.0),
        (2, 4, 1.0),
        (3, 4, 2.0)
    ]
    maxcut = MaxCutQUBO(num_nodes=nodes, edges=edges)

    t0 = time.perf_counter()
    exact_res = maxcut.solve_classical_exact()
    exact_time_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    sa_res = maxcut.solve_simulated_annealing(steps=500, seed=42)
    sa_time_ms = (time.perf_counter() - t0) * 1000.0

    qubo_results = [
        {
            "Problem": "Max-Cut 5-Node Graph",
            "Solver": exact_res["method"],
            "Total_Value": exact_res["max_cut"],
            "Total_Weight": float(np.nan),
            "Capacity": float(np.nan),
            "Elapsed_ms": exact_time_ms,
            "Optimality_Gap_%": 0.0
        },
        {
            "Problem": "Max-Cut 5-Node Graph",
            "Solver": sa_res["method"],
            "Total_Value": sa_res["max_cut"],
            "Total_Weight": float(np.nan),
            "Capacity": float(np.nan),
            "Elapsed_ms": sa_time_ms,
            "Optimality_Gap_%": ((exact_res["max_cut"] - sa_res["max_cut"]) / exact_res["max_cut"]) * 100.0
        }
    ]
    return qubo_results, exact_res


def run_all_track_b_experiments(output_dir="results"):
    print("=" * 60)
    print("Running Track B: Discrete Optimization Experiments...")
    print("=" * 60)

    knapsack_res = run_knapsack_experiment()
    qubo_res, exact_maxcut = run_qubo_maxcut_experiment()

    all_res = knapsack_res + qubo_res
    df = pd.DataFrame(all_res)

    csv_path = os.path.join(output_dir, "raw", "track_b_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results table to {csv_path}")

    # Plot comparison of Knapsack solvers
    knapsack_df = df[df["Problem"].str.contains("Knapsack")].copy()
    plt.figure(figsize=(10, 4.5))

    plt.subplot(1, 2, 1)
    bars = plt.bar(knapsack_df["Solver"], knapsack_df["Total_Value"], color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"])
    plt.title("Knapsack Objective Value Comparison")
    plt.ylabel("Total Value Achieved")
    plt.ylim(0, max(knapsack_df["Total_Value"]) * 1.2)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.1f}", ha="center", va="bottom")
    plt.xticks(rotation=15)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.subplot(1, 2, 2)
    bars2 = plt.bar(knapsack_df["Solver"], knapsack_df["Elapsed_ms"], color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"])
    plt.title("Solver Execution Time (ms)")
    plt.ylabel("Elapsed Time (ms)")
    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f}ms", ha="center", va="bottom")
    plt.xticks(rotation=15)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "figures", "track_b_comparison.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved benchmark figure to {fig_path}")

    print("\nTrack B Results Summary:")
    print(df.to_string(index=False))
    print(f"\nClassical Max-Cut Ground Truth Baseline for Track C: Value = {exact_maxcut['max_cut']}, Partition = {exact_maxcut['best_partition']}")
    return df, exact_maxcut


if __name__ == "__main__":
    run_all_track_b_experiments()

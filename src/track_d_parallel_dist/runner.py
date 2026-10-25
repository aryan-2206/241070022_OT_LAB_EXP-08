"""Track D: Submodular, Parallel, and Distributed Optimization - Runner Suite.

Executes:
1. Submodular Coverage: Diminishing returns verification, Greedy vs Exact, (1 - 1/e) bound.
2. Parallel Processing: Data, Pipeline, and Dynamic Work Stealing across workers.
3. Amdahl's & Gustafson's Law fitting.
4. Federated Learning: FedAvg vs Centralized SGD.
5. Multi-task optimization: Equal Weighting vs MGDA vs GradNorm.
6. Saves CSVs and figures.
"""

import os
import sys
import time
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import matplotlib.pyplot as plt

from src.track_d_parallel_dist.submodular import MaximumCoverageSubmodular
from src.track_d_parallel_dist.parallel_exec import ParallelBenchmarkSuite
from src.track_d_parallel_dist.amdahl_gustafson import ScalingLaws
from src.track_d_parallel_dist.federated_multitask import (
    FederatedAveragingSimulator, MultiTaskOptimizer
)


def run_all_track_d_experiments(output_dir="results"):
    print("=" * 60)
    print("Running Track D: Submodular, Parallel & Distributed Optimization...")
    print("=" * 60)

    results_records = []

    # 1. Submodular Coverage Experiment
    submod = MaximumCoverageSubmodular(ground_set_size=50, num_candidate_sets=16, seed=42)
    diminishing_res = submod.verify_diminishing_returns(num_checks=40)
    k_budget = 4
    greedy_sets, greedy_val = submod.greedy_select(budget_k=k_budget)
    exact_sets, exact_val = submod.exact_optimal_select(budget_k=k_budget)
    approx_ratio = greedy_val / exact_val
    theoretical_guarantee = 1.0 - 1.0 / np.e  # ~0.6321

    results_records.append({
        "Category": "Submodular Coverage",
        "Experiment": f"Greedy Selection (k={k_budget})",
        "Metric": "Covered Elements",
        "Baseline_Value": exact_val,
        "Optimized_Value": greedy_val,
        "Ratio_or_Speedup": approx_ratio,
        "Notes": f"Guarantee: >= {theoretical_guarantee:.3f}, Violations: {diminishing_res['violations']}"
    })

    # 2. Parallel Computing Benchmark (Skewed task workload)
    par_suite = ParallelBenchmarkSuite(n_tasks=80, skew="skewed", seed=42)
    t_seq, _ = par_suite.run_sequential()

    worker_counts = [1, 2, 4]
    speedups_static = []
    speedups_stealing = []

    for p in worker_counts:
        t_stat, _ = par_suite.run_data_parallel_static(num_workers=p)
        t_steal, _ = par_suite.run_dynamic_work_stealing(num_workers=p)

        sp_stat = t_seq / max(t_stat, 1e-6)
        sp_steal = t_seq / max(t_steal, 1e-6)
        speedups_static.append(sp_stat)
        speedups_stealing.append(sp_steal)

        results_records.append({
            "Category": "Parallel Computing",
            "Experiment": f"Static Chunking (p={p})",
            "Metric": "Elapsed Time (s)",
            "Baseline_Value": t_seq,
            "Optimized_Value": t_stat,
            "Ratio_or_Speedup": sp_stat,
            "Notes": f"Parallel Efficiency: {sp_stat / p:.2f}"
        })
        results_records.append({
            "Category": "Parallel Computing",
            "Experiment": f"Work Stealing (p={p})",
            "Metric": "Elapsed Time (s)",
            "Baseline_Value": t_seq,
            "Optimized_Value": t_steal,
            "Ratio_or_Speedup": sp_steal,
            "Notes": f"Parallel Efficiency: {sp_steal / p:.2f}"
        })

    # 3. Pipeline Parallelism
    t_pipe, _ = par_suite.run_pipeline_parallel(num_stages=3)
    results_records.append({
        "Category": "Parallel Computing",
        "Experiment": "Pipeline Parallelism (3 stages)",
        "Metric": "Elapsed Time (s)",
        "Baseline_Value": t_seq,
        "Optimized_Value": t_pipe,
        "Ratio_or_Speedup": t_seq / max(t_pipe, 1e-6),
        "Notes": "Staged queue concurrency"
    })

    # 4. Federated Learning: FedAvg vs Centralized SGD
    fed_sim = FederatedAveragingSimulator(num_clients=5, d_features=4, seed=42)
    cent_res = fed_sim.run_centralized(rounds=30, lr=0.01)
    fed_e1 = fed_sim.run_fedavg(rounds=15, local_steps=1, lr=0.01)
    fed_e5 = fed_sim.run_fedavg(rounds=15, local_steps=5, lr=0.01)

    results_records.append({
        "Category": "Distributed Learning",
        "Experiment": "FedAvg (E=1 local step)",
        "Metric": "Global MSE Loss",
        "Baseline_Value": cent_res["final_loss"],
        "Optimized_Value": fed_e1["final_loss"],
        "Ratio_or_Speedup": cent_res["final_loss"] / max(fed_e1["final_loss"], 1e-6),
        "Notes": "15 communication rounds"
    })
    results_records.append({
        "Category": "Distributed Learning",
        "Experiment": "FedAvg (E=5 local steps)",
        "Metric": "Global MSE Loss",
        "Baseline_Value": cent_res["final_loss"],
        "Optimized_Value": fed_e5["final_loss"],
        "Ratio_or_Speedup": cent_res["final_loss"] / max(fed_e5["final_loss"], 1e-6),
        "Notes": "15 communication rounds, lower network traffic"
    })

    # 5. Multi-Task Optimization: Equal Weighting vs MGDA vs GradNorm
    mto = MultiTaskOptimizer(d_features=4, n_samples=120, seed=42)
    eq_res = mto.run_equal_weighting(steps=60, lr=0.005)
    mgda_res = mto.run_mgda(steps=60, lr=0.005)
    gn_res = mto.run_gradnorm(steps=60, lr=0.005)

    results_records.append({
        "Category": "Multi-Task Optimization",
        "Experiment": "MGDA (Pareto Stationarity)",
        "Metric": "Total Combined Loss (L1 + L2)",
        "Baseline_Value": eq_res["final_losses"][2],
        "Optimized_Value": mgda_res["final_losses"][2],
        "Ratio_or_Speedup": eq_res["final_losses"][2] / max(mgda_res["final_losses"][2], 1e-6),
        "Notes": f"L1: {mgda_res['final_losses'][0]:.3f}, L2: {mgda_res['final_losses'][1]:.3f}"
    })
    results_records.append({
        "Category": "Multi-Task Optimization",
        "Experiment": "GradNorm",
        "Metric": "Total Combined Loss (L1 + L2)",
        "Baseline_Value": eq_res["final_losses"][2],
        "Optimized_Value": gn_res["final_losses"][2],
        "Ratio_or_Speedup": eq_res["final_losses"][2] / max(gn_res["final_losses"][2], 1e-6),
        "Notes": f"L1: {gn_res['final_losses'][0]:.3f}, L2: {gn_res['final_losses'][1]:.3f}"
    })

    # Save to CSV
    df = pd.DataFrame(results_records)
    csv_path = os.path.join(output_dir, "raw", "track_d_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results table to {csv_path}")

    # Plot Submodular Guarantee, Speedup vs Amdahl's Law, and FedAvg/Multi-Task
    plt.figure(figsize=(15, 4.5))

    # Subplot 1: Submodular Guarantee
    plt.subplot(1, 3, 1)
    bars = plt.bar(["Theoretical Bound (1-1/e)", "Greedy Observed", "Exact Optimum"],
                   [theoretical_guarantee * exact_val, greedy_val, exact_val],
                   color=["gray", "#3274a1", "#e1812c"])
    plt.title("Submodular Max Coverage Guarantee")
    plt.ylabel("Ground Elements Covered")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}", ha="center", va="bottom")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Subplot 2: Parallel Speedup vs Amdahl's Law
    plt.subplot(1, 3, 2)
    p_fine = np.linspace(1, 4, 50)
    amdahl_curve = [ScalingLaws.amdahl_speedup(p, serial_fraction=0.15) for p in p_fine]
    plt.plot(p_fine, amdahl_curve, "--", label="Amdahl's Law (s=15%)", color="gray")
    plt.plot(worker_counts, speedups_static, "o-", label="Static Chunking", color="#c44e52")
    plt.plot(worker_counts, speedups_stealing, "s-", label="Work Stealing", color="#55a868")
    plt.plot(worker_counts, worker_counts, ":", label="Ideal Linear Speedup", color="black", alpha=0.5)
    plt.title("Parallel Speedup & Work Stealing")
    plt.xlabel("Worker Count (p)")
    plt.ylabel(r"Speedup $S_p = T_1 / T_p$")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 3: Multi-Task Loss Trajectories
    plt.subplot(1, 3, 3)
    eq_tot = [x[2] for x in eq_res["history"]]
    mgda_tot = [x[2] for x in mgda_res["history"]]
    gn_tot = [x[2] for x in gn_res["history"]]
    plt.plot(eq_tot, label="Equal Weighting", color="#c44e52")
    plt.plot(mgda_tot, label="MGDA (Pareto)", color="#3274a1")
    plt.plot(gn_tot, label="GradNorm", color="#55a868")
    plt.title("Multi-Task Convergence (L1 + L2)")
    plt.xlabel("Step")
    plt.ylabel("Combined Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "figures", "track_d_submodular_parallel.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved figure to {fig_path}")

    print("\nTrack D Execution Summary:")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    run_all_track_d_experiments()

"""Track G: xv6 Operating System Optimization - Benchmark Runner & Suite.

Executes:
1. Workload 1: CPU-Bound processes under Round-Robin vs Priority/Stride.
2. Workload 2: I/O-Bound interactive processes under Round-Robin vs Priority/Stride.
3. Workload 3: Mixed Contention (Priority 20 interactive vs Priority 5 batch).
4. Generates performance plots and CSV records.
"""

import os
import sys
import copy
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import matplotlib.pyplot as plt

from src.track_g_xv6_os.xv6_kernel_sim import SimProc, XV6KernelSimulator


def get_workload(workload_type):
    """Factory creating processes for the 3 compulsory benchmark workloads."""
    if workload_type == "cpu_bound":
        return [
            SimProc(1, "matrix_mul_1", burst_ticks=40, priority=10),
            SimProc(2, "matrix_mul_2", burst_ticks=35, priority=10),
            SimProc(3, "fft_compute", burst_ticks=50, priority=10),
            SimProc(4, "prime_sieve", burst_ticks=30, priority=10),
        ]
    elif workload_type == "io_bound":
        return [
            SimProc(1, "shell_ui", burst_ticks=25, priority=15, io_frequency=3),
            SimProc(2, "editor", burst_ticks=30, priority=15, io_frequency=4),
            SimProc(3, "net_listener", burst_ticks=20, priority=12, io_frequency=2),
            SimProc(4, "log_writer", burst_ticks=35, priority=10, io_frequency=5),
        ]
    elif workload_type == "mixed_contention":
        return [
            # High priority interactive task
            SimProc(1, "interactive_gui", burst_ticks=20, priority=20, io_frequency=3),
            # Low priority batch compute tasks
            SimProc(2, "batch_render_1", burst_ticks=60, priority=5),
            SimProc(3, "batch_render_2", burst_ticks=60, priority=5),
            SimProc(4, "batch_compress", burst_ticks=50, priority=5),
        ]


def run_all_track_g_experiments(output_dir="results"):
    print("=" * 60)
    print("Running Track G: xv6 Operating System Optimization Experiments...")
    print("=" * 60)

    workloads = ["cpu_bound", "io_bound", "mixed_contention"]
    records = []

    for wl in workloads:
        # Run Round-Robin baseline
        procs_rr = get_workload(wl)
        sim_rr = XV6KernelSimulator(scheduler_policy="round_robin")
        res_rr = sim_rr.run_workload(procs_rr)

        # Run Priority & Stride with Aging
        procs_stride = get_workload(wl)
        sim_stride = XV6KernelSimulator(scheduler_policy="stride_priority")
        res_stride = sim_stride.run_workload(procs_stride)

        # For mixed workload, check response time of interactive_gui process
        interactive_tat_rr = next((p.completion_tick for p in res_rr["processes"] if "interactive" in p.name), res_rr["avg_turnaround_ticks"])
        interactive_tat_stride = next((p.completion_tick for p in res_stride["processes"] if "interactive" in p.name), res_stride["avg_turnaround_ticks"])

        records.append({
            "Workload": wl,
            "Policy": "Round-Robin (Default xv6)",
            "Avg_Turnaround_Ticks": res_rr["avg_turnaround_ticks"],
            "Avg_Wait_Ticks": res_rr["avg_wait_ticks"],
            "Avg_Response_Ticks": res_rr["avg_response_ticks"],
            "Context_Switches": res_rr["context_switches"],
            "Fairness_Index": res_rr["fairness_index"],
            "Interactive_TAT": interactive_tat_rr
        })
        records.append({
            "Workload": wl,
            "Policy": "Stride & Priority + Aging (Optimized)",
            "Avg_Turnaround_Ticks": res_stride["avg_turnaround_ticks"],
            "Avg_Wait_Ticks": res_stride["avg_wait_ticks"],
            "Avg_Response_Ticks": res_stride["avg_response_ticks"],
            "Context_Switches": res_stride["context_switches"],
            "Fairness_Index": res_stride["fairness_index"],
            "Interactive_TAT": interactive_tat_stride
        })

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "raw", "track_g_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results table to {csv_path}")

    # Plot Comparison Across Workloads
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    wl_names = ["CPU-Bound", "I/O-Bound", "Mixed Contention"]
    rr_resp = [r["Avg_Response_Ticks"] for r in records if "Round-Robin" in r["Policy"]]
    stride_resp = [r["Avg_Response_Ticks"] for r in records if "Stride" in r["Policy"]]

    x = np.arange(len(wl_names))
    width = 0.35
    plt.bar(x - width/2, rr_resp, width, label="Round-Robin (Baseline)", color="#c44e52")
    plt.bar(x + width/2, stride_resp, width, label="Priority & Stride (Optimized)", color="#55a868")
    plt.xticks(x, wl_names)
    plt.ylabel("Average Response Time (Ticks)")
    plt.title("Process Response Time Across Workloads")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.subplot(1, 2, 2)
    # Mixed contention interactive turnaround time
    mixed_rr = [r["Interactive_TAT"] for r in records if r["Workload"] == "mixed_contention" and "Round-Robin" in r["Policy"]][0]
    mixed_stride = [r["Interactive_TAT"] for r in records if r["Workload"] == "mixed_contention" and "Stride" in r["Policy"]][0]

    bars = plt.bar(["Default Round-Robin", "Priority Stride"], [mixed_rr, mixed_stride], color=["#c44e52", "#55a868"])
    plt.title("Interactive Task Completion Time (Mixed Contention)")
    plt.ylabel("Completion Tick (Lower is Better)")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval} ticks", ha="center", va="bottom")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "figures", "track_g_xv6_benchmarks.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved xv6 benchmark figure to {fig_path}")

    print("\nTrack G Execution Summary:")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    run_all_track_g_experiments()

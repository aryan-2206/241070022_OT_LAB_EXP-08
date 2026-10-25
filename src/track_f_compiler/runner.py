"""Track F: Compiler Optimization - Benchmark Runner & Experiment Suite.

Executes:
1. Compiles benchmark programs to 3-Address Code (TAC).
2. Runs 4 optimization passes (Constant Folding, DCE, CSE, Strength Reduction).
3. Verifies output equivalence on ToyVM.
4. Solves Register Allocation as graph coloring and ILP.
5. Saves CSV and plots.
"""

import os
import sys
import time
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import matplotlib.pyplot as plt

from src.track_f_compiler.compiler import ToyCompiler, ToyVM
from src.track_f_compiler.passes import OptimizationPipeline
from src.track_f_compiler.reg_alloc import RegisterAllocator


def run_all_track_f_experiments(output_dir="results"):
    print("=" * 60)
    print("Running Track F: Compiler Optimization Experiments...")
    print("=" * 60)

    # Benchmark program exhibiting constant folding, dead code, CSE, and strength reduction
    source_program = """
    # Dynamic variables x and y provided at runtime
    # Redundant common subexpression: x + y
    t1 = x + y
    # Strength reduction: x * 8 -> x << 3
    t2 = x * 8
    # Re-computation of identical expression
    t3 = x + y
    # Constant folding: 12 * 4 -> 48
    c1 = 12 * 4
    # Dead code: unused heavy calculation
    unused_heavy = t2 * 999
    # Strength reduction: y / 4 -> y >> 2
    t4 = y / 4
    # Re-use of common subexpression with offset
    t5 = t1 + c1
    # Final result combining live registers
    res = t2 + t3 + t4 + t5
    return res
    """

    compiler = ToyCompiler()
    vm = ToyVM()
    pipeline = OptimizationPipeline()

    initial_env = {"x": 10, "y": 8}

    # 1. Generate unoptimized IR
    t0 = time.perf_counter()
    unopt_ir = compiler.parse_and_generate_ir(source_program)
    compile_time_unopt_ms = (time.perf_counter() - t0) * 1000.0

    # Execute unoptimized IR on VM
    unopt_res, unopt_cycles, _ = vm.run(unopt_ir, initial_env=initial_env)

    # 2. Run optimization pipeline
    t0 = time.perf_counter()
    opt_ir, _ = pipeline.optimize(unopt_ir)
    compile_time_opt_ms = (time.perf_counter() - t0) * 1000.0

    # Execute optimized IR on VM
    opt_res, opt_cycles, _ = vm.run(opt_ir, initial_env=initial_env)

    # Verification: identical outputs
    assert unopt_res == opt_res, f"Correctness violated! Unoptimized={unopt_res}, Optimized={opt_res}"

    unopt_inst_count = len(unopt_ir)
    opt_inst_count = len(opt_ir)
    inst_reduction_pct = ((unopt_inst_count - opt_inst_count) / unopt_inst_count) * 100.0
    cycle_reduction_pct = ((unopt_cycles - opt_cycles) / unopt_cycles) * 100.0

    # 3. Register Allocation via Graph Coloring and ILP
    allocator = RegisterAllocator(num_registers=3)
    var_list, adj_mat, _ = allocator.compute_liveness_and_interference(opt_ir)

    chaitin_res = allocator.solve_chaitin_heuristic(var_list, adj_mat)
    ilp_res = allocator.solve_ilp(var_list, adj_mat)

    records = [
        {
            "Stage": "Unoptimized IR",
            "Instruction_Count": unopt_inst_count,
            "VM_Cycles": unopt_cycles,
            "Output_Value": unopt_res,
            "Compile_Time_ms": compile_time_unopt_ms,
            "Reduction_%": 0.0
        },
        {
            "Stage": "Optimized IR (4 Passes)",
            "Instruction_Count": opt_inst_count,
            "VM_Cycles": opt_cycles,
            "Output_Value": opt_res,
            "Compile_Time_ms": compile_time_opt_ms,
            "Reduction_%": inst_reduction_pct
        }
    ]

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "raw", "track_f_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results table to {csv_path}")

    # Plot instruction count and VM cycles before and after optimization
    plt.figure(figsize=(10, 4.5))

    plt.subplot(1, 2, 1)
    bars1 = plt.bar(["Unoptimized IR", "Optimized IR"], [unopt_inst_count, opt_inst_count],
                    color=["#c44e52", "#55a868"])
    plt.title("IR Instruction Count Reduction")
    plt.ylabel("Instruction Count")
    for bar in bars1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval}", ha="center", va="bottom")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.subplot(1, 2, 2)
    bars2 = plt.bar(["Unoptimized IR", "Optimized IR"], [unopt_cycles, opt_cycles],
                    color=["#c44e52", "#55a868"])
    plt.title("Simulated VM Clock Cycles")
    plt.ylabel("Clock Cycles")
    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval} cycles", ha="center", va="bottom")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "figures", "track_f_compiler.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved compiler optimization figure to {fig_path}")

    print("\nTrack F Execution Summary:")
    print(df.to_string(index=False))
    print(f"\nCorrectness Check: Unoptimized Output = {unopt_res}, Optimized Output = {opt_res} (100% Match!)")
    print(f"Instruction Reduction: {inst_reduction_pct:.1f}%, VM Cycle Reduction: {cycle_reduction_pct:.1f}%")
    print(f"Register Allocation (K=3): Chaitin Spills = {chaitin_res['spill_count']}, ILP Spills = {ilp_res['spill_count']}")
    return df


if __name__ == "__main__":
    run_all_track_f_experiments()

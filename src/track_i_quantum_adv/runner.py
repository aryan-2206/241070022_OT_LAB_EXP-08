"""Track I: Quantum Algorithms & QML Advantage - Benchmark Runner & Suite.

Executes:
1. Suite of 7 quantum algorithms (DJ, BV, Grover, QFT, QPE, Shor, Quantum Annealing).
2. Grover's Algorithmic Advantage vs Classical Linear Search scaling (Queries & Wall-Clock).
3. QML Quantum Kernel vs Classical RBF Classifier comparison.
4. Classical simulation overhead analysis ($2^n$ memory barrier).
5. Generates figures and CSV results.
"""

import os
import sys
import time
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import matplotlib.pyplot as plt

from src.track_i_quantum_adv.quantum_algorithms import QuantumAlgorithmSuite
from src.track_i_quantum_adv.qml_comparison import (
    QuantumKernelClassifier, ClassicalRBFClassifier
)


def run_all_track_i_experiments(output_dir="results"):
    print("=" * 60)
    print("Running Track I: Quantum Algorithms & QML Advantage Experiments...")
    print("=" * 60)

    # 1. Run Quantum Algorithm Suite
    suite = QuantumAlgorithmSuite()
    dj_res = suite.run_deutsch_jozsa(n_qubits=3, is_balanced=True)
    bv_res = suite.run_bernstein_vazirani(hidden_bitstring="1011")
    grover_res = suite.run_grovers_search(n_qubits=4, target_index=11)
    qft_res = suite.run_quantum_fourier_transform(n_qubits=3)
    qpe_res = suite.run_quantum_phase_estimation(true_phase=0.375, n_eval_qubits=3)
    shor_res = suite.run_shors_order_finding(a=7, N=15)
    qa_res = suite.run_quantum_annealing_simulation(steps=50)

    algo_records = [
        {"Algorithm": "Deutsch-Jozsa", "Problem": "Constant vs Balanced Oracle",
         "Quantum_Queries": dj_res["quantum_queries"], "Classical_Queries": dj_res["classical_worst_case_queries"],
         "Success": dj_res["detected"] == "balanced"},
        {"Algorithm": "Bernstein-Vazirani", "Problem": "Hidden Bitstring '1011'",
         "Quantum_Queries": bv_res["quantum_queries"], "Classical_Queries": bv_res["classical_queries"],
         "Success": bv_res["success"]},
        {"Algorithm": "Grover's Search", "Problem": "Target Search in N=16",
         "Quantum_Queries": grover_res["quantum_queries"], "Classical_Queries": grover_res["classical_avg_queries"],
         "Success": grover_res["target_probability"] > 0.90},
        {"Algorithm": "QFT", "Problem": "Basis Transformation",
         "Quantum_Queries": 1, "Classical_Queries": 3 * 8,  # FFT O(N log N)
         "Success": qft_res["is_uniform_spectrum"]},
        {"Algorithm": "QPE", "Problem": "Phase 0.375 Recovery",
         "Quantum_Queries": 3, "Classical_Queries": 8,
         "Success": np.isclose(qpe_res["estimated_phase"], qpe_res["true_phase"], atol=1e-3)},
        {"Algorithm": "Shor's Factoring", "Problem": "Factorize N=15",
         "Quantum_Queries": shor_res["period_r"], "Classical_Queries": 15,
         "Success": shor_res["discovered_factors"] == [3, 5]}
    ]

    # 2. Grover Search Scaling vs Classical Linear Search
    search_qubits = [3, 4, 5, 6, 7]
    grover_scaling = []
    for n in search_qubits:
        N = 1 << n
        target = N - 3
        # Measure quantum simulation time
        t0 = time.perf_counter()
        g_run = suite.run_grovers_search(n_qubits=n, target_index=target)
        q_time_ms = (time.perf_counter() - t0) * 1000.0

        # Measure classical linear search time
        items = list(range(N))
        t0 = time.perf_counter()
        c_queries = 0
        for item in items:
            c_queries += 1
            if item == target:
                break
        c_time_ms = (time.perf_counter() - t0) * 1000.0

        grover_scaling.append({
            "Qubits": n,
            "Search_Space_N": N,
            "Quantum_Queries": g_run["quantum_queries"],
            "Classical_Queries": c_queries,
            "Query_Advantage_Ratio": c_queries / max(g_run["quantum_queries"], 1),
            "Quantum_Sim_Time_ms": q_time_ms,
            "Classical_Time_ms": c_time_ms
        })

    # 3. QML vs Classical Comparison
    np.random.seed(42)
    N_samples = 60
    # 2-class concentric circles
    r_in = np.random.uniform(0.1, 0.7, size=N_samples // 2)
    theta_in = np.random.uniform(0, 2*np.pi, size=N_samples // 2)
    X_in = np.column_stack([r_in * np.cos(theta_in), r_in * np.sin(theta_in)])
    y_in = -np.ones(N_samples // 2)

    r_out = np.random.uniform(1.1, 1.8, size=N_samples // 2)
    theta_out = np.random.uniform(0, 2*np.pi, size=N_samples // 2)
    X_out = np.column_stack([r_out * np.cos(theta_out), r_out * np.sin(theta_out)])
    y_out = np.ones(N_samples // 2)

    X_all = np.vstack([X_in, X_out])
    y_all = np.concatenate([y_in, y_out])

    # Shuffle
    perm = np.random.permutation(N_samples)
    X_all = X_all[perm]
    y_all = y_all[perm]

    split = 40
    X_train, X_test = X_all[:split], X_all[split:]
    y_train, y_test = y_all[:split], y_all[split:]

    # Train Quantum Kernel Classifier
    q_clf = QuantumKernelClassifier(n_qubits=2)
    q_train_ms = q_clf.fit(X_train, y_train)
    q_preds = q_clf.predict(X_test)
    q_acc = float(np.mean(q_preds == y_test))

    # Train Classical RBF Classifier
    c_clf = ClassicalRBFClassifier(gamma=1.5)
    c_train_ms = c_clf.fit(X_train, y_train)
    c_preds = c_clf.predict(X_test)
    c_acc = float(np.mean(c_preds == y_test))

    qml_records = [
        {
            "Model": "Quantum Kernel Classifier (VQC Feature Map)",
            "Accuracy_%": q_acc * 100.0,
            "Training_Time_ms": q_train_ms,
            "Circuit_Evaluations": q_clf.circuit_evaluations,
            "Data_Loading_Overhead": "O(2^d) unitary synthesis",
            "Notes": "Simulated pure statevector overlap"
        },
        {
            "Model": "Classical RBF SVM",
            "Accuracy_%": c_acc * 100.0,
            "Training_Time_ms": c_train_ms,
            "Circuit_Evaluations": 0,
            "Data_Loading_Overhead": "None (direct memory vector)",
            "Notes": "Closed-form dual kernel ridge solution"
        }
    ]

    # Combine into CSV
    df_scaling = pd.DataFrame(grover_scaling)
    csv_path = os.path.join(output_dir, "raw", "track_i_results.csv")
    df_scaling.to_csv(csv_path, index=False)
    print(f"Saved results table to {csv_path}")

    # Plot Grover Query Scaling, Wall-Clock Simulation Inversion, and QML Comparison
    plt.figure(figsize=(15, 4.5))

    # Subplot 1: Grover Query Complexity (Quantum Advantage)
    plt.subplot(1, 3, 1)
    N_vals = [s["Search_Space_N"] for s in grover_scaling]
    q_queries = [s["Quantum_Queries"] for s in grover_scaling]
    c_queries = [s["Classical_Queries"] for s in grover_scaling]
    plt.plot(N_vals, c_queries, "o-", label="Classical Linear Search O(N)", color="#c44e52")
    plt.plot(N_vals, q_queries, "s-", label=r"Quantum Grover $O(\sqrt{N})$", color="#55a868")
    plt.title("Algorithmic Query Advantage")
    plt.xlabel("Search Space Size (N)")
    plt.ylabel("Required Queries / Oracles")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Classical Simulation Wall-Clock Inversion (Honest Negative Result)
    plt.subplot(1, 3, 2)
    q_times = [s["Quantum_Sim_Time_ms"] for s in grover_scaling]
    c_times = [s["Classical_Time_ms"] for s in grover_scaling]
    plt.plot(N_vals, q_times, "s-", label="Quantum Simulation (CPU)", color="#3274a1")
    plt.plot(N_vals, c_times, "o-", label="Classical Direct Search", color="#e1812c")
    plt.yscale("log")
    plt.title("Classical Simulation Wall-Clock Overhead")
    plt.xlabel("Search Space Size (N)")
    plt.ylabel("Time in ms (log scale)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)

    # Subplot 3: QML vs Classical Accuracy & Latency
    plt.subplot(1, 3, 3)
    bars = plt.bar(["Classical RBF SVM", "Quantum Kernel"], [c_acc * 100.0, q_acc * 100.0],
                   color=["#55a868", "#8172b2"])
    plt.title("QML vs Classical Accuracy")
    plt.ylabel("Test Accuracy (%)")
    plt.ylim(0, 110)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "figures", "track_i_quantum.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved quantum advantage figure to {fig_path}")

    print("\nTrack I Execution Summary:")
    print("Algorithm Verification:")
    for a in algo_records:
        print(f"  [{'PASS' if a['Success'] else 'FAIL'}] {a['Algorithm']}: {a['Problem']} (Q-Queries: {a['Quantum_Queries']} vs C-Queries: {a['Classical_Queries']})")
    print("\nGrover vs Classical Query Scaling:")
    print(df_scaling.to_string(index=False))
    print("\nQML vs Classical Model Comparison:")
    for m in qml_records:
        print(f"  {m['Model']}: Acc={m['Accuracy_%']:.1f}%, Train Time={m['Training_Time_ms']:.2f}ms, Evals={m['Circuit_Evaluations']}")
    return df_scaling


if __name__ == "__main__":
    run_all_track_i_experiments()

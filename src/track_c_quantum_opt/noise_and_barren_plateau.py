"""Track C: Quantum Optimization - Noise, Finite Shots, and Barren Plateaus.

Demonstrates:
1. Finite measurement shot noise error scaling (1 / sqrt(N_shots))
2. Depolarizing hardware noise impact on QAOA performance
3. Barren Plateau phenomenon: Exponential decay of gradient variance Var[d<H>/dtheta] ~ 2^{-n}
"""

import numpy as np
from src.track_c_quantum_opt.simulator import QuantumCircuit
from src.track_c_quantum_opt.gradients import QuantumGradients


def evaluate_finite_shot_scaling(circuit_fn, params, diag_H, shot_counts=(100, 500, 1000, 5000, 10000), trials=20):
    """Measures variance and standard error of expectation value vs number of shots."""
    qc = circuit_fn(params)
    exact_exp = qc.expectation_diagonal_hamiltonian(diag_H)

    results = []
    for shots in shot_counts:
        errors = []
        for t in range(trials):
            counts = qc.sample_shots(n_shots=shots, seed=42 + t)
            shot_probs = counts / shots
            shot_exp = float(np.sum(shot_probs * diag_H))
            errors.append(abs(shot_exp - exact_exp))

        mean_error = float(np.mean(errors))
        results.append({
            "shots": shots,
            "mean_abs_error": mean_error,
            "theoretical_scaling": 1.0 / np.sqrt(shots)
        })
    return results, exact_exp


def evaluate_depolarizing_noise_impact(qaoa_instance, params, noise_levels=(0.0, 0.05, 0.1, 0.2, 0.3)):
    """Measures degradation of QAOA expectation under depolarizing hardware noise."""
    results = []
    for p_noise in noise_levels:
        qc = qaoa_instance.construct_circuit(params)
        qc.apply_depolarizing_noise(p_noise)
        noisy_cut = qc.expectation_diagonal_hamiltonian(qaoa_instance.diag_cut)
        results.append({
            "depolarizing_noise": p_noise,
            "expected_cut": float(noisy_cut)
        })
    return results


def study_barren_plateaus(qubit_range=(2, 3, 4, 5, 6, 7), num_samples=30):
    """
    Demonstrates Barren Plateaus in random parameterized quantum circuits.
    As qubit count n increases, Var[d<H>/dtheta_1] decays exponentially as O(2^{-n}).
    """
    results = []
    # Hamiltonian: Observable Z on qubit 0: H = Z_0 (x) I ...
    for n in qubit_range:
        dim = 1 << n
        diag_H = np.zeros(dim)
        for i in range(dim):
            # Bit 0 value: 0 -> +1, 1 -> -1
            diag_H[i] = 1.0 if ((i >> 0) & 1) == 0 else -1.0

        grads = []
        for s in range(num_samples):
            np.random.seed(100 * n + s)
            # Circuit depth d = 3
            depth = 3
            params = np.random.uniform(0, 2*np.pi, size=n * depth)

            def eval_cost(p):
                qc = QuantumCircuit(n)
                idx = 0
                for d in range(depth):
                    for q in range(n):
                        qc.ry(p[idx], q)
                        idx += 1
                    # Entangling layer
                    for q in range(n - 1):
                        qc.cnot(q, q + 1)
                return qc.expectation_diagonal_hamiltonian(diag_H)

            # Compute gradient with respect to the first parameter theta_0
            g_0 = QuantumGradients.parameter_shift(eval_cost, params)[0]
            grads.append(g_0)

        variance = float(np.var(grads))
        results.append({
            "num_qubits": n,
            "gradient_variance": variance,
            "log2_variance": float(np.log2(max(variance, 1e-15)))
        })

    return results

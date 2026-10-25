"""Unit and Correctness Tests for Track I: Quantum Algorithms and QML Advantage."""

import numpy as np
import pytest

from src.track_i_quantum_adv.quantum_algorithms import QuantumAlgorithmSuite
from src.track_i_quantum_adv.qml_comparison import QuantumKernelClassifier


def test_deutsch_jozsa_exact_oracle_detection():
    """Deutsch-Jozsa must identify balanced and constant oracles in exactly 1 query."""
    suite = QuantumAlgorithmSuite()
    res_balanced = suite.run_deutsch_jozsa(n_qubits=3, is_balanced=True)
    res_constant = suite.run_deutsch_jozsa(n_qubits=3, is_balanced=False)

    assert res_balanced["detected"] == "balanced"
    assert res_balanced["quantum_queries"] == 1
    assert res_constant["detected"] == "constant"
    assert res_constant["quantum_queries"] == 1


def test_bernstein_vazirani_hidden_string_recovery():
    """Bernstein-Vazirani must extract the secret bitstring in exactly 1 query."""
    suite = QuantumAlgorithmSuite()
    secret = "1101"
    res = suite.run_bernstein_vazirani(hidden_bitstring=secret)

    assert res["recovered_string"] == secret
    assert res["quantum_queries"] == 1
    assert res["success"] is True


def test_grover_search_high_probability():
    """Grover's search on N=16 must amplify target state probability above 90%."""
    suite = QuantumAlgorithmSuite()
    res = suite.run_grovers_search(n_qubits=4, target_index=7)

    assert res["best_measured_index"] == 7
    assert res["target_probability"] > 0.90
    assert res["quantum_queries"] < res["classical_avg_queries"]


def test_quantum_kernel_matrix_properties():
    """Quantum Gram matrix must be symmetric with unit diagonal (K_{ii} == 1.0)."""
    q_clf = QuantumKernelClassifier(n_qubits=2)
    X = np.array([[0.5, 1.2], [-0.3, 0.8], [1.1, -0.9]])
    K = q_clf.compute_kernel_matrix(X, X)

    # Unit diagonal
    assert np.allclose(np.diag(K), 1.0, atol=1e-8)
    # Symmetry
    assert np.allclose(K, K.T, atol=1e-8)
    # Valid probability overlaps in [0, 1]
    assert np.all(K >= 0.0) and np.all(K <= 1.0)


def test_shors_order_finding_factorization():
    """Shor's period finding must factorize 15 into 3 and 5."""
    suite = QuantumAlgorithmSuite()
    res = suite.run_shors_order_finding(a=7, N=15)
    assert res["discovered_factors"] == [3, 5]
    assert res["period_r"] == 4

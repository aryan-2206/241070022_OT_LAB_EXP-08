"""Unit and Correctness Tests for Track C: Quantum Optimization."""

import numpy as np
import pytest

from src.track_c_quantum_opt.simulator import QuantumCircuit
from src.track_c_quantum_opt.gradients import QuantumGradients
from src.track_c_quantum_opt.qaoa import MaxCutQAOA


def test_quantum_simulator_unitarity_and_normalization():
    """Statevector must maintain strict L2 normalization (sum |c_i|^2 == 1.0)."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.ry(0.6, 1)
    qc.cnot(0, 2)
    qc.rz(1.2, 2)
    qc.rzz(0.8, 1, 2)

    probs = qc.get_probabilities()
    assert np.isclose(np.sum(probs), 1.0, atol=1e-12)
    assert np.all(probs >= 0.0)


def test_parameter_shift_rule_correctness():
    """Parameter-shift rule must produce exact analytic gradients matching finite differences."""
    n = 2
    diag_H = np.array([1.0, -1.0, -1.0, 1.0])  # Z_0 Z_1 observable

    def circuit_exp(params):
        qc = QuantumCircuit(n)
        qc.ry(params[0], 0)
        qc.rx(params[1], 1)
        qc.cnot(0, 1)
        return qc.expectation_diagonal_hamiltonian(diag_H)

    params = np.array([0.7, -1.1])
    ps_grad = QuantumGradients.parameter_shift(circuit_exp, params)

    # Numerical finite differences
    eps = 1e-5
    num_grad = np.zeros(2)
    for i in range(2):
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[i] += eps
        p_minus[i] -= eps
        num_grad[i] = (circuit_exp(p_plus) - circuit_exp(p_minus)) / (2.0 * eps)

    assert np.allclose(ps_grad, num_grad, atol=1e-4), "Parameter shift gradient diverges from numerical gradient"


def test_qaoa_cut_expectation():
    """QAOA cost evaluation on a known 2-node graph."""
    edges = [(0, 1, 2.0)]
    qaoa = MaxCutQAOA(num_nodes=2, edges=edges, p_layers=1)

    # Initial state is |++>, expected cut is 0.5 * 2.0 = 1.0
    init_params = np.array([0.0, 0.0])
    cost = -qaoa.evaluate_cost(init_params)
    assert np.isclose(cost, 1.0, atol=1e-6)

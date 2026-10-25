"""Track C & I: High-Fidelity Pure-NumPy Statevector Quantum Simulator.

Supports:
- n-qubit statevectors |psi> in C^{2^n}
- Universal quantum gates: H, X, Y, Z, S, T, Rx, Ry, Rz, CNOT, CZ, Rzz
- Expectation values <psi| H |psi> for diagonal & Pauli Hamiltonians
- Finite-shot measurement sampling
- Depolarizing noise simulation
"""

import numpy as np


class QuantumCircuit:
    def __init__(self, num_qubits):
        self.n = num_qubits
        self.dim = 1 << num_qubits
        self.state = np.zeros(self.dim, dtype=complex)
        self.state[0] = 1.0  # Initialize in |0...0>
        self.gates = []

    def reset(self):
        """Resets circuit to |0...0>."""
        self.state = np.zeros(self.dim, dtype=complex)
        self.state[0] = 1.0
        self.gates = []

    def _apply_1q_matrix(self, matrix_2x2, target):
        """Applies a 2x2 unitary gate to target qubit."""
        state = self.state.reshape([2] * self.n)
        # Move target axis to the front
        state = np.swapaxes(state, 0, target)
        # Apply matrix product: (2, 2) x (2, rest) -> (2, rest)
        orig_shape = state.shape
        state = state.reshape((2, -1))
        state = matrix_2x2 @ state
        state = state.reshape(orig_shape)
        # Swap back
        state = np.swapaxes(state, 0, target)
        self.state = state.reshape(self.dim)

    def _apply_2q_matrix(self, matrix_4x4, control, target):
        """Applies a 4x4 two-qubit unitary gate."""
        state = self.state.reshape([2] * self.n)
        # Move control to axis 0, target to axis 1
        state = np.swapaxes(state, 0, control)
        # If target was at 0, it moved to control
        target_axis = control if target == 0 else target
        state = np.swapaxes(state, 1, target_axis)

        orig_shape = state.shape
        state = state.reshape((4, -1))
        state = matrix_4x4 @ state
        state = state.reshape(orig_shape)

        # Swap back in reverse order
        state = np.swapaxes(state, 1, target_axis)
        state = np.swapaxes(state, 0, control)
        self.state = state.reshape(self.dim)

    # 1-Qubit Gates
    def h(self, target):
        """Hadamard gate."""
        mat = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
        self._apply_1q_matrix(mat, target)

    def x(self, target):
        """Pauli-X gate (NOT)."""
        mat = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        self._apply_1q_matrix(mat, target)

    def y(self, target):
        """Pauli-Y gate."""
        mat = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
        self._apply_1q_matrix(mat, target)

    def z(self, target):
        """Pauli-Z gate."""
        mat = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        self._apply_1q_matrix(mat, target)

    def rx(self, theta, target):
        """Rx(theta) = exp(-i * theta/2 * X) = cos(theta/2) I - i sin(theta/2) X."""
        c = np.cos(theta / 2.0)
        s = np.sin(theta / 2.0)
        mat = np.array([[c, -1.0j * s], [-1.0j * s, c]], dtype=complex)
        self._apply_1q_matrix(mat, target)

    def ry(self, theta, target):
        """Ry(theta) = exp(-i * theta/2 * Y) = cos(theta/2) I - i sin(theta/2) Y."""
        c = np.cos(theta / 2.0)
        s = np.sin(theta / 2.0)
        mat = np.array([[c, -s], [s, c]], dtype=complex)
        self._apply_1q_matrix(mat, target)

    def rz(self, theta, target):
        """Rz(theta) = exp(-i * theta/2 * Z) = diag(exp(-i theta/2), exp(i theta/2))."""
        mat = np.array([[np.exp(-1.0j * theta / 2.0), 0.0],
                        [0.0, np.exp(1.0j * theta / 2.0)]], dtype=complex)
        self._apply_1q_matrix(mat, target)

    # 2-Qubit Gates
    def cnot(self, control, target):
        """Controlled-NOT gate."""
        mat = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
        self._apply_2q_matrix(mat, control, target)

    def cz(self, control, target):
        """Controlled-Z gate."""
        mat = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=complex)
        self._apply_2q_matrix(mat, control, target)

    def rzz(self, theta, q1, q2):
        """Rzz(theta) = exp(-i * theta/2 * Z_1 (x) Z_2).
        Equivalent to CNOT(q1, q2) -> Rz(theta, q2) -> CNOT(q1, q2).
        """
        self.cnot(q1, q2)
        self.rz(theta, q2)
        self.cnot(q1, q2)

    # Measurements & Observables
    def get_probabilities(self):
        """Returns classical probability distribution over all computational basis states."""
        return np.abs(self.state) ** 2

    def sample_shots(self, n_shots=1000, seed=42):
        """Simulates finite measurement shots using multinomial sampling."""
        np.random.seed(seed)
        probs = self.get_probabilities()
        probs = np.clip(probs, 0.0, 1.0)
        probs /= np.sum(probs)
        counts = np.random.multinomial(n_shots, probs)
        return counts

    def expectation_diagonal_hamiltonian(self, diag_H):
        """Analytic expectation <psi| H |psi> for a diagonal Hamiltonian."""
        probs = self.get_probabilities()
        return float(np.sum(probs * diag_H))

    def apply_depolarizing_noise(self, noise_prob):
        """Applies depolarizing channel: with probability noise_prob, replaces state with maximally mixed state."""
        if noise_prob <= 0.0:
            return
        probs = self.get_probabilities()
        mixed_probs = (1.0 - noise_prob) * probs + noise_prob / self.dim
        # Reconstruct decohered state amplitudes with randomized phase
        phases = np.exp(1.0j * np.random.uniform(0, 2*np.pi, size=self.dim))
        self.state = np.sqrt(mixed_probs) * phases

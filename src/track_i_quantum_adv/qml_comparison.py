"""Track I: Quantum Machine Learning vs Classical Computing Comparison.

Implements:
1. Quantum Kernel Estimator via parameterized feature mapping:
   K_Q(x, x') = |<phi(x)|phi(x')>|^2
2. Classical RBF Support Vector Machine baseline.
3. Compares accuracy, training time, circuit evaluations, parameter count,
   and documents the classical simulation overhead and data-loading bottleneck.
"""

import time
import numpy as np
from src.track_c_quantum_opt.simulator import QuantumCircuit


class QuantumKernelClassifier:
    """Quantum Kernel Machine using statevector overlap feature mapping."""
    def __init__(self, n_qubits=2):
        self.n = n_qubits
        self.X_train = None
        self.y_train = None
        self.alpha = None
        self.circuit_evaluations = 0

    def _feature_map(self, x):
        """Encodes classical vector x in R^2 into quantum state |phi(x)>."""
        qc = QuantumCircuit(self.n)
        for i in range(self.n):
            qc.h(i)
            qc.ry(x[i], i)
        # Entangling phase
        if self.n > 1:
            qc.cnot(0, 1)
            qc.rz(x[0] * x[1], 1)
            qc.cnot(0, 1)
        return qc.state.copy()

    def compute_kernel_matrix(self, X1, X2):
        """Computes Gram matrix K_{ij} = |<phi(x_1_i) | phi(x_2_j)>|^2."""
        N1 = len(X1)
        N2 = len(X2)
        K = np.zeros((N1, N2))

        # Precompute statevectors
        states1 = [self._feature_map(x) for x in X1]
        states2 = [self._feature_map(x) for x in X2]
        self.circuit_evaluations += N1 + N2

        for i in range(N1):
            for j in range(N2):
                overlap = np.vdot(states1[i], states2[j])
                K[i, j] = float(np.abs(overlap) ** 2)

        return K

    def fit(self, X, y, reg_lambda=0.01):
        """Trains dual kernel ridge regression weights: alpha = (K + lambda I)^{-1} y."""
        self.X_train = X
        self.y_train = y
        self.circuit_evaluations = 0

        t0 = time.perf_counter()
        K = self.compute_kernel_matrix(X, X)
        K_reg = K + reg_lambda * np.eye(len(X))
        self.alpha = np.linalg.solve(K_reg, y)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return elapsed_ms

    def predict(self, X_test):
        """Predicts labels: sign(sum_i alpha_i K(x_i, x_test))."""
        K_test = self.compute_kernel_matrix(self.X_train, X_test)
        scores = K_test.T @ self.alpha
        return np.sign(scores)


class ClassicalRBFClassifier:
    """Classical RBF Kernel Ridge Classifier baseline."""
    def __init__(self, gamma=1.0):
        self.gamma = gamma
        self.X_train = None
        self.alpha = None

    def _rbf_kernel(self, X1, X2):
        from scipy.spatial.distance import cdist
        dists = cdist(X1, X2, metric="sqeuclidean")
        return np.exp(-self.gamma * dists)

    def fit(self, X, y, reg_lambda=0.01):
        self.X_train = X
        t0 = time.perf_counter()
        K = self._rbf_kernel(X, X)
        K_reg = K + reg_lambda * np.eye(len(X))
        self.alpha = np.linalg.solve(K_reg, y)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return elapsed_ms

    def predict(self, X_test):
        K_test = self._rbf_kernel(self.X_train, X_test)
        scores = K_test.T @ self.alpha
        return np.sign(scores)

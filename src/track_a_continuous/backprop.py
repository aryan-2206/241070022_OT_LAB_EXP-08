"""Track A: Continuous Optimization - Explicit Backpropagation.

Demonstrates backpropagation as the analytical gradient computation mechanism
behind a Multi-Layer Perceptron (MLP), explicitly distinct from the parameter
update optimizer.
"""

import numpy as np


class TwoLayerMLP:
    """Two-layer neural network with explicit forward pass, analytical backpropagation,
    and parameter unpacking.
    Architecture: Input (d) -> Linear -> ReLU -> Linear -> Sigmoid -> Binary Cross-Entropy.
    """
    def __init__(self, in_features, hidden_features, out_features=1, seed=42):
        np.random.seed(seed)
        # He initialization
        self.W1 = np.random.randn(in_features, hidden_features) * np.sqrt(2.0 / in_features)
        self.b1 = np.zeros((1, hidden_features))
        self.W2 = np.random.randn(hidden_features, out_features) * np.sqrt(2.0 / hidden_features)
        self.b2 = np.zeros((1, out_features))

        # Cached activations for backprop
        self.X_cache = None
        self.z1_cache = None
        self.a1_cache = None
        self.z2_cache = None
        self.y_hat_cache = None

    def forward(self, X):
        """Forward pass computing activations."""
        self.X_cache = X
        self.z1_cache = X @ self.W1 + self.b1
        self.a1_cache = np.maximum(0.0, self.z1_cache)  # ReLU
        self.z2_cache = self.a1_cache @ self.W2 + self.b2
        # Sigmoid activation
        self.y_hat_cache = 1.0 / (1.0 + np.exp(-np.clip(self.z2_cache, -15.0, 15.0)))
        return self.y_hat_cache

    def compute_loss(self, y_true, y_pred, eps=1e-12):
        """Binary cross-entropy loss."""
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        loss = -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
        return float(loss)

    def backward(self, y_true):
        """Analytical Backpropagation computing exact gradients:
        dL/dW2, dL/db2, dL/dW1, dL/db1.
        """
        N = y_true.shape[0]
        # Output layer gradient: dL/dz2 = (y_hat - y) / N
        dz2 = (self.y_hat_cache - y_true) / N
        dW2 = self.a1_cache.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Hidden layer gradient through ReLU:
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1_cache > 0.0)
        dW1 = self.X_cache.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def get_params_vector(self):
        """Flattens all parameters into a 1D vector."""
        return np.concatenate([self.W1.ravel(), self.b1.ravel(), self.W2.ravel(), self.b2.ravel()])

    def set_params_vector(self, p_vec):
        """Reconstructs layer matrices from a 1D parameter vector."""
        idx = 0
        w1_size = self.W1.size
        self.W1 = p_vec[idx:idx + w1_size].reshape(self.W1.shape)
        idx += w1_size

        b1_size = self.b1.size
        self.b1 = p_vec[idx:idx + b1_size].reshape(self.b1.shape)
        idx += b1_size

        w2_size = self.W2.size
        self.W2 = p_vec[idx:idx + w2_size].reshape(self.W2.shape)
        idx += w2_size

        b2_size = self.b2.size
        self.b2 = p_vec[idx:idx + b2_size].reshape(self.b2.shape)

    def grads_to_vector(self, grads_dict):
        """Flattens gradient dictionary into a 1D vector."""
        return np.concatenate([
            grads_dict["W1"].ravel(),
            grads_dict["b1"].ravel(),
            grads_dict["W2"].ravel(),
            grads_dict["b2"].ravel()
        ])

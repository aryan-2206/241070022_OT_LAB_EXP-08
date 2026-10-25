"""Track D: Distributed and Multi-Task Optimization.

Implements:
1. Federated Averaging (FedAvg): Distributed client training with varying local steps E and rounds R.
2. Multi-Task Optimization:
   - Naive Equal Weighting
   - MGDA (Multiple Gradient Descent Algorithm for Pareto Stationarity)
   - GradNorm (Dynamic Gradient Normalization)
"""

import numpy as np


class FederatedAveragingSimulator:
    """
    Simulates Federated Learning with non-IID clients:
    In each round:
    1. Server broadcasts global weights w_t to active clients.
    2. Each client k runs E local SGD epochs on its local partition D_k.
    3. Server aggregates updates: w_{t+1} = sum_k (n_k / N) * w_k^{t, E}.
    """
    def __init__(self, num_clients=5, d_features=4, seed=42):
        np.random.seed(seed)
        self.num_clients = num_clients
        self.d = d_features
        # True underlying parameter vector
        self.w_true = np.random.randn(d_features)

        # Generate non-IID datasets per client (varying feature shifts)
        self.client_data = []
        for c in range(num_clients):
            n_samples = 40
            mean_shift = np.random.randn(d_features) * 1.5
            X_c = np.random.randn(n_samples, d_features) + mean_shift
            y_c = X_c @ self.w_true + np.random.randn(n_samples) * 0.1
            self.client_data.append((X_c, y_c))

    def evaluate_global_loss(self, w):
        """Evaluates global MSE across all client datasets."""
        total_loss = 0.0
        total_samples = 0
        for X_c, y_c in self.client_data:
            preds = X_c @ w
            total_loss += np.sum((preds - y_c) ** 2)
            total_samples += len(y_c)
        return float(total_loss / total_samples)

    def run_centralized(self, rounds=20, lr=0.01):
        """Centralized pooled dataset training baseline."""
        all_X = np.vstack([X for X, y in self.client_data])
        all_y = np.concatenate([y for X, y in self.client_data])

        w = np.zeros(self.d)
        loss_history = [self.evaluate_global_loss(w)]

        for _ in range(rounds):
            grad = (2.0 / len(all_y)) * (all_X.T @ (all_X @ w - all_y))
            w -= lr * grad
            loss_history.append(self.evaluate_global_loss(w))

        return {"method": "Centralized SGD", "final_loss": loss_history[-1], "history": loss_history, "w": w}

    def run_fedavg(self, rounds=10, local_steps=5, lr=0.01):
        """FedAvg execution across distributed clients."""
        w_global = np.zeros(self.d)
        loss_history = [self.evaluate_global_loss(w_global)]
        total_n = sum(len(y) for _, y in self.client_data)

        for _ in range(rounds):
            client_weights = []
            for X_c, y_c in self.client_data:
                w_local = w_global.copy()
                n_c = len(y_c)
                for _ in range(local_steps):
                    grad = (2.0 / n_c) * (X_c.T @ (X_c @ w_local - y_c))
                    w_local -= lr * grad
                client_weights.append((n_c, w_local))

            # Server aggregation: weighted average
            w_global = np.zeros(self.d)
            for n_c, w_l in client_weights:
                w_global += (n_c / total_n) * w_l

            loss_history.append(self.evaluate_global_loss(w_global))

        return {
            "method": f"FedAvg (E={local_steps})",
            "rounds": rounds,
            "local_steps": local_steps,
            "final_loss": loss_history[-1],
            "history": loss_history,
            "w": w_global
        }


class MultiTaskOptimizer:
    """
    Two-task learning problem with shared representation:
    Task 1: y1 = X @ w_shared + b1
    Task 2: y2 = (X @ w_shared)^2 + b2  (non-linear conflict)
    """
    def __init__(self, d_features=4, n_samples=100, seed=42):
        np.random.seed(seed)
        self.d = d_features
        self.X = np.random.randn(n_samples, d_features)
        self.w_true = np.random.randn(d_features)
        self.y1 = self.X @ self.w_true + np.random.randn(n_samples) * 0.1
        self.y2 = (self.X @ self.w_true) ** 2 + np.random.randn(n_samples) * 0.1

    def compute_task_losses_and_grads(self, w):
        """Computes loss and analytical gradient for both tasks w.r.t shared parameters w."""
        pred1 = self.X @ w
        loss1 = np.mean((pred1 - self.y1) ** 2)
        grad1 = (2.0 / len(self.y1)) * (self.X.T @ (pred1 - self.y1))

        pred2 = (self.X @ w) ** 2
        loss2 = np.mean((pred2 - self.y2) ** 2)
        # Chain rule: d/dw = (4/N) * X^T [ ((Xw)^2 - y2) * (Xw) ]
        grad2 = (4.0 / len(self.y2)) * (self.X.T @ ((pred2 - self.y2) * pred1))

        return loss1, grad1, loss2, grad2

    def run_equal_weighting(self, steps=50, lr=0.01, w_init=None):
        """Naive equal sum of task losses: L = L1 + L2."""
        w = np.ones(self.d) * 0.5 if w_init is None else w_init.copy()
        history = []
        for _ in range(steps):
            l1, g1, l2, g2 = self.compute_task_losses_and_grads(w)
            history.append((l1, l2, l1 + l2))
            g_total = g1 + g2
            w -= lr * g_total
        return {"method": "Equal Weighting", "history": history, "final_losses": history[-1]}

    def run_mgda(self, steps=50, lr=0.01, w_init=None):
        """
        Multiple Gradient Descent Algorithm (MGDA) for 2 tasks:
        Finds minimum-norm vector in convex hull of gradients:
        min_{alpha in [0, 1]} ||alpha * g1 + (1 - alpha) * g2||^2
        Analytical 1D solution for 2 tasks:
        alpha = ((g2 - g1)^T g2) / ||g1 - g2||^2 clipped to [0, 1].
        """
        w = np.ones(self.d) * 0.5 if w_init is None else w_init.copy()
        history = []
        for _ in range(steps):
            l1, g1, l2, g2 = self.compute_task_losses_and_grads(w)
            history.append((l1, l2, l1 + l2))

            # Solve min ||alpha g1 + (1-alpha) g2||^2
            diff = g1 - g2
            denom = float(diff.T @ diff)
            if denom < 1e-12:
                alpha = 0.5
            else:
                alpha = float(((g2 - g1).T @ g2) / denom)
                alpha = np.clip(alpha, 0.0, 1.0)

            g_pareto = alpha * g1 + (1.0 - alpha) * g2
            w -= lr * g_pareto
        return {"method": "MGDA (Pareto Stationarity)", "history": history, "final_losses": history[-1]}

    def run_gradnorm(self, steps=50, lr=0.01, alpha_gn=0.5, w_init=None):
        """
        GradNorm: Dynamically balances gradient norms based on relative inverse training rates.
        """
        w = np.ones(self.d) * 0.5 if w_init is None else w_init.copy()
        weights = np.array([1.0, 1.0])
        history = []

        # Initial loss reference
        l1_init, _, l2_init, _ = self.compute_task_losses_and_grads(w)

        for _ in range(steps):
            l1, g1, l2, g2 = self.compute_task_losses_and_grads(w)
            history.append((l1, l2, l1 + l2))

            # Relative training rates
            r1 = (l1 / max(l1_init, 1e-6))
            r2 = (l2 / max(l2_init, 1e-6))
            r_mean = (r1 + r2) / 2.0

            # Target gradient norms
            norm1 = np.linalg.norm(g1)
            norm2 = np.linalg.norm(g2)
            mean_norm = (norm1 + norm2) / 2.0

            target1 = mean_norm * ((r1 / max(r_mean, 1e-6)) ** alpha_gn)
            target2 = mean_norm * ((r2 / max(r_mean, 1e-6)) ** alpha_gn)

            # Update weights to drive norm towards target
            w1_adapt = float(np.clip(target1 / max(norm1, 1e-6), 0.2, 5.0))
            w2_adapt = float(np.clip(target2 / max(norm2, 1e-6), 0.2, 5.0))

            # Normalize weights to sum to 2.0
            sum_w = w1_adapt + w2_adapt
            weights = np.array([w1_adapt, w2_adapt]) * (2.0 / sum_w)

            g_balanced = weights[0] * g1 + weights[1] * g2
            w -= lr * g_balanced

        return {"method": "GradNorm", "history": history, "final_losses": history[-1]}

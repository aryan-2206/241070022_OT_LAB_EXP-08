"""Track A: Continuous Optimization - Second-Order & Curvature Solvers.

Implements:
- Newton's Method (with exact Hessian inversion and regularization)
- Gauss-Newton Method (for non-linear least squares problems)
"""

import numpy as np


class NewtonOptimizer:
    """Newton-Raphson Optimization: theta_{t+1} = theta_t - [H(theta_t) + lambda I]^{-1} grad(theta_t)."""
    def __init__(self, damping=1e-4):
        self.damping = damping

    def step(self, params, grad, hessian):
        p = len(params)
        # Regularized Hessian to guarantee positive definiteness
        H_reg = hessian + self.damping * np.eye(p)
        delta = np.linalg.solve(H_reg, grad)
        return params - delta


class GaussNewtonOptimizer:
    """Gauss-Newton Optimization for Non-linear Least Squares:
    Minimizes f(theta) = 0.5 * ||r(theta)||^2
    Update: theta_{t+1} = theta_t - (J^T J + damping * I)^{-1} J^T r
    """
    def __init__(self, damping=1e-4):
        self.damping = damping

    def step(self, params, residuals, jacobian):
        """
        params: 1D array of shape (p,)
        residuals: 1D array of shape (N,)
        jacobian: 2D array of shape (N, p) where J_{ij} = dr_i / dtheta_j
        """
        JT_J = jacobian.T @ jacobian
        p = JT_J.shape[0]
        JT_J_reg = JT_J + self.damping * np.eye(p)
        JT_r = jacobian.T @ residuals
        delta = np.linalg.solve(JT_J_reg, JT_r)
        return params - delta

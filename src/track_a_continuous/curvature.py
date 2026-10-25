"""Track A: Continuous Optimization - Curvature Approximations.

Implements:
1. Empirical Fisher Information Matrix
2. Diagonal Hessian Approximation
3. Pearlmutter Hessian-Vector Products (HVP)
"""

import numpy as np


def compute_empirical_fisher(grad_samples):
    """
    Computes Empirical Fisher Information Matrix from individual sample gradients:
    F = 1/N * sum_{i=1}^N g_i * g_i^T
    grad_samples: 2D array of shape (N, p)
    """
    N, p = grad_samples.shape
    F = np.zeros((p, p))
    for i in range(N):
        g = grad_samples[i:i+1].T
        F += g @ g.T
    return F / N


def compute_diagonal_hessian(loss_fn, theta, eps=1e-5):
    """
    Approximates diagonal elements of the Hessian matrix via second-order central difference:
    H_{ii} = [f(theta + eps * e_i) - 2 * f(theta) + f(theta - eps * e_i)] / eps^2
    """
    p = len(theta)
    diag_H = np.zeros(p)
    f0 = loss_fn(theta)
    for i in range(p):
        e_i = np.zeros(p)
        e_i[i] = eps
        f_plus = loss_fn(theta + e_i)
        f_minus = loss_fn(theta - e_i)
        diag_H[i] = (f_plus - 2.0 * f0 + f_minus) / (eps ** 2)
    return diag_H


def hessian_vector_product(grad_fn, theta, v, r=1e-5):
    """
    Pearlmutter's Hessian-Vector Product (HVP) without forming the explicit p x p matrix:
    H * v = [grad(theta + r * v) - grad(theta - r * v)] / (2 * r)
    O(p) memory complexity instead of O(p^2).
    """
    grad_plus = grad_fn(theta + r * v)
    grad_minus = grad_fn(theta - r * v)
    return (grad_plus - grad_minus) / (2.0 * r)

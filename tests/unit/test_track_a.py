"""Unit and Correctness Tests for Track A: Continuous Optimization."""

import numpy as np
import pytest

from src.track_a_continuous.optimizers import (
    GradientDescent, MomentumOptimizer, AdaGrad, RMSProp, Adam
)
from src.track_a_continuous.second_order import NewtonOptimizer, GaussNewtonOptimizer
from src.track_a_continuous.backprop import TwoLayerMLP
from src.track_a_continuous.landscapes import QuadraticBowl, Rosenbrock, SaddlePointLandscape
from src.track_a_continuous.curvature import compute_diagonal_hessian, hessian_vector_product


def test_first_order_optimizer_steps():
    """Verifies that first-order optimizer step arithmetic matches mathematical formulation."""
    params = np.array([1.0, -2.0])
    grads = np.array([0.5, -1.0])

    # GD: theta - lr * g
    gd = GradientDescent(lr=0.1)
    new_params = gd.step(params.copy(), grads)
    expected = params - 0.1 * grads
    assert np.allclose(new_params, expected)

    # Momentum first step: v = 0.1 * grads, params -= v
    mom = MomentumOptimizer(lr=0.1, beta=0.9)
    new_params = mom.step(params.copy(), grads)
    assert np.allclose(new_params, params - 0.1 * grads)


def test_newton_quadratic_convergence():
    """Newton's method on a strictly convex quadratic bowl must reach the exact global optimum in 1 step."""
    quad = QuadraticBowl(cond_number=5.0)
    init_x = np.array([4.2, -7.8])
    newton = NewtonOptimizer(damping=0.0)

    g = quad.grad(init_x)
    H = quad.hessian(init_x)
    x_opt = newton.step(init_x, g, H)

    assert np.allclose(x_opt, [0.0, 0.0], atol=1e-8)
    assert quad(x_opt) < 1e-12


def test_backpropagation_finite_difference_check():
    """Compares analytical backpropagation gradients against numerical finite differences."""
    np.random.seed(42)
    mlp = TwoLayerMLP(in_features=2, hidden_features=4, out_features=1, seed=42)
    X = np.array([[0.5, -0.3], [1.2, 0.8]])
    y = np.array([[1.0], [0.0]])

    mlp.forward(X)
    analytical_grads = mlp.backward(y)
    p_vec = mlp.get_params_vector()

    # Numerical gradient check via central differences
    eps = 1e-5
    num_grad_vec = np.zeros_like(p_vec)
    for i in range(len(p_vec)):
        p_plus = p_vec.copy()
        p_plus[i] += eps
        mlp.set_params_vector(p_plus)
        y_plus = mlp.forward(X)
        l_plus = mlp.compute_loss(y, y_plus)

        p_minus = p_vec.copy()
        p_minus[i] -= eps
        mlp.set_params_vector(p_minus)
        y_minus = mlp.forward(X)
        l_minus = mlp.compute_loss(y, y_minus)

        num_grad_vec[i] = (l_plus - l_minus) / (2.0 * eps)

    # Reset original parameters
    mlp.set_params_vector(p_vec)
    ana_grad_vec = mlp.grads_to_vector(analytical_grads)

    max_rel_error = np.max(np.abs(ana_grad_vec - num_grad_vec) / (np.abs(ana_grad_vec) + np.abs(num_grad_vec) + 1e-8))
    assert max_rel_error < 1e-4, f"Backprop gradient check failed with max rel error {max_rel_error}"


def test_pearlemutter_hvp():
    """Validates Pearlmutter Hessian-Vector Product against exact Hessian product."""
    rosen = Rosenbrock()
    x = np.array([1.2, 1.5])
    v = np.array([0.7, -0.4])

    exact_hvp = rosen.hessian(x) @ v
    approx_hvp = hessian_vector_product(rosen.grad, x, v, r=1e-5)

    assert np.allclose(approx_hvp, exact_hvp, atol=1e-4)

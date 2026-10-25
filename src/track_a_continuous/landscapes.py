"""Track A: Continuous Optimization - Optimization Landscapes.

Defines convex, ill-conditioned, and non-convex benchmark objective functions
with analytical gradients and Hessians.
"""

import numpy as np


class QuadraticBowl:
    """Strongly convex quadratic function: f(x) = 0.5 * x^T A x where A is positive definite."""
    def __init__(self, cond_number=10.0):
        self.A = np.diag([1.0, cond_number])

    def __call__(self, x):
        return 0.5 * float(x.T @ self.A @ x)

    def grad(self, x):
        return self.A @ x

    def hessian(self, x):
        return self.A


class Rosenbrock:
    """Ill-conditioned non-convex banana valley: f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2."""
    def __init__(self, a=1.0, b=100.0):
        self.a = a
        self.b = b

    def __call__(self, x):
        return float((self.a - x[0]) ** 2 + self.b * (x[1] - x[0] ** 2) ** 2)

    def grad(self, x):
        df_dx = -2.0 * (self.a - x[0]) - 4.0 * self.b * x[0] * (x[1] - x[0] ** 2)
        df_dy = 2.0 * self.b * (x[1] - x[0] ** 2)
        return np.array([df_dx, df_dy])

    def hessian(self, x):
        d2f_dx2 = 2.0 - 4.0 * self.b * x[1] + 12.0 * self.b * (x[0] ** 2)
        d2f_dxdy = -4.0 * self.b * x[0]
        d2f_dy2 = 2.0 * self.b
        return np.array([[d2f_dx2, d2f_dxdy], [d2f_dxdy, d2f_dy2]])


class SaddlePointLandscape:
    """Non-convex saddle point: f(x, y) = x^2 - y^2 + 0.1 * y^4 (stabilized).
    Saddle point at (0, 0) where gradient is zero, Hessian has mixed positive/negative eigenvalues.
    """
    def __call__(self, x):
        return float(x[0] ** 2 - x[1] ** 2 + 0.1 * (x[1] ** 4))

    def grad(self, x):
        return np.array([2.0 * x[0], -2.0 * x[1] + 0.4 * (x[1] ** 3)])

    def hessian(self, x):
        return np.array([[2.0, 0.0], [0.0, -2.0 + 1.2 * (x[1] ** 2)]])


class Rastrigin:
    """Highly multimodal non-convex landscape with many local minima:
    f(x) = 10 * d + sum_i (x_i^2 - 10 * cos(2 * pi * x_i))
    Global minimum at x* = [0, 0, ...] with f(x*) = 0.
    """
    def __init__(self, A=10.0):
        self.A = A

    def __call__(self, x):
        d = len(x)
        return float(self.A * d + np.sum(x ** 2 - self.A * np.cos(2.0 * np.pi * x)))

    def grad(self, x):
        return 2.0 * x + self.A * 2.0 * np.pi * np.sin(2.0 * np.pi * x)

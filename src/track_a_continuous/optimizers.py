"""Track A: Continuous Optimization - First-Order Optimizers.

Implements:
- Gradient Descent (GD)
- Stochastic Gradient Descent (SGD) and Mini-batch SGD
- Momentum
- Nesterov Accelerated Gradient (NAG)
- AdaGrad
- RMSProp
- Adam
"""

import numpy as np


class Optimizer:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        raise NotImplementedError


class GradientDescent(Optimizer):
    """Standard Vanilla Gradient Descent (GD)."""
    def __init__(self, lr=0.01):
        super().__init__(lr)

    def step(self, params, grads):
        return params - self.lr * grads


class MomentumOptimizer(Optimizer):
    """Classical Momentum optimizer: v = beta * v + lr * grads; params -= v."""
    def __init__(self, lr=0.01, beta=0.9):
        super().__init__(lr)
        self.beta = beta
        self.v = None

    def step(self, params, grads):
        if self.v is None:
            self.v = np.zeros_like(params)
        self.v = self.beta * self.v + self.lr * grads
        return params - self.v


class NesterovMomentumOptimizer(Optimizer):
    """Nesterov Accelerated Gradient (NAG)."""
    def __init__(self, lr=0.01, beta=0.9):
        super().__init__(lr)
        self.beta = beta
        self.v = None

    def step(self, params, grads):
        if self.v is None:
            self.v = np.zeros_like(params)
        v_prev = self.v.copy()
        self.v = self.beta * self.v + self.lr * grads
        # Lookahead approximation update: params - (beta * v_prev + (1 + beta) * lr * grads)
        return params - (self.beta * v_prev + self.lr * grads)


class AdaGrad(Optimizer):
    """AdaGrad optimizer: G_t = G_{t-1} + g_t^2; theta_t = theta_{t-1} - lr / sqrt(G_t + eps) * g_t."""
    def __init__(self, lr=0.05, eps=1e-8):
        super().__init__(lr)
        self.eps = eps
        self.G = None

    def step(self, params, grads):
        if self.G is None:
            self.G = np.zeros_like(params)
        self.G += grads ** 2
        return params - (self.lr / (np.sqrt(self.G) + self.eps)) * grads


class RMSProp(Optimizer):
    """RMSProp optimizer: v_t = beta * v_{t-1} + (1 - beta) * g_t^2."""
    def __init__(self, lr=0.01, beta=0.9, eps=1e-8):
        super().__init__(lr)
        self.beta = beta
        self.eps = eps
        self.v = None

    def step(self, params, grads):
        if self.v is None:
            self.v = np.zeros_like(params)
        self.v = self.beta * self.v + (1.0 - self.beta) * (grads ** 2)
        return params - (self.lr / (np.sqrt(self.v) + self.eps)) * grads


class Adam(Optimizer):
    """Adam: Adaptive Moment Estimation with bias corrections."""
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grads ** 2)
        # Bias-corrected moments
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        return params - (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat

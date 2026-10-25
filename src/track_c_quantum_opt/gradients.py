"""Track C: Quantum Optimization - Quantum Gradient Strategies.

Implements:
1. Exact Analytic Parameter-Shift Rule: d<H>/dtheta = (<H>_{theta + pi/2} - <H>_{theta - pi/2}) / 2
2. Simultaneous Perturbation Stochastic Approximation (SPSA): 2 circuit evaluations per step
3. Quantum Natural Gradient (QNG): Metric tensor preconditioned descent
"""

import numpy as np


class QuantumGradients:
    @staticmethod
    def parameter_shift(eval_fn, params):
        """
        Computes exact analytic gradient using the parameter-shift rule.
        Requires 2 * p circuit evaluations.
        """
        p = len(params)
        grads = np.zeros(p)
        shift = np.pi / 2.0

        for i in range(p):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += shift
            params_minus[i] -= shift

            val_plus = eval_fn(params_plus)
            val_minus = eval_fn(params_minus)
            grads[i] = (val_plus - val_minus) / 2.0

        return grads

    @staticmethod
    def spsa_gradient(eval_fn, params, c=0.1, seed=None):
        """
        SPSA: Simultaneous Perturbation Stochastic Approximation.
        Requires exactly 2 circuit evaluations regardless of dimension p.
        """
        if seed is not None:
            np.random.seed(seed)
        p = len(params)
        # Random perturbation vector from Rademacher distribution (Bernoulli +-1)
        delta = 2 * np.random.randint(0, 2, size=p) - 1

        params_plus = params + c * delta
        params_minus = params - c * delta

        val_plus = eval_fn(params_plus)
        val_minus = eval_fn(params_minus)

        grads = (val_plus - val_minus) / (2.0 * c * delta)
        return grads

    @staticmethod
    def quantum_natural_gradient(eval_fn, state_fn, params, lr=0.05, damping=1e-3):
        """
        Quantum Natural Gradient using the Fubini-Study metric tensor g_{ij}:
        g_{ij} = Re[ <d_i psi | d_j psi> - <d_i psi | psi><psi | d_j psi> ]
        Update: theta - lr * (g + damping * I)^{-1} grad
        """
        p = len(params)
        grad = QuantumGradients.parameter_shift(eval_fn, params)

        # Approximate block-diagonal Fubini-Study metric tensor
        shift = 1e-4
        psi_0 = state_fn(params)
        d_psi = np.zeros((p, len(psi_0)), dtype=complex)

        for i in range(p):
            p_shifted = params.copy()
            p_shifted[i] += shift
            psi_shifted = state_fn(p_shifted)
            d_psi[i] = (psi_shifted - psi_0) / shift

        # Compute Fubini-Study metric matrix:
        g = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                term1 = np.vdot(d_psi[i], d_psi[j])
                term2 = np.vdot(d_psi[i], psi_0) * np.vdot(psi_0, d_psi[j])
                g[i, j] = np.real(term1 - term2)

        g_reg = g + damping * np.eye(p)
        qng_step = np.linalg.solve(g_reg, grad)
        return qng_step, grad

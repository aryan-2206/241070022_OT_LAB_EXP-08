"""Track C: Quantum Optimization - Quantum Approximate Optimization Algorithm (QAOA).

Solves the Track B Max-Cut QUBO problem on the 5-node graph using QAOA
with p layers of alternating problem (cost) and mixer unitaries.
"""

import numpy as np
from src.track_c_quantum_opt.simulator import QuantumCircuit
from src.track_c_quantum_opt.gradients import QuantumGradients


class MaxCutQAOA:
    """QAOA solver for Max-Cut with p layers."""
    def __init__(self, num_nodes, edges, p_layers=1):
        self.n = num_nodes
        self.edges = edges
        self.p = p_layers
        self.diag_cut = self._build_diagonal_cost_hamiltonian()

    def _build_diagonal_cost_hamiltonian(self):
        """Constructs diagonal matrix of Max-Cut values for all 2^n computational basis states."""
        dim = 1 << self.n
        diag_cut = np.zeros(dim)
        for state_idx in range(dim):
            # Extract bits
            bits = [(state_idx >> bit) & 1 for bit in range(self.n)]
            cut = 0.0
            for u, v, w in self.edges:
                if bits[u] != bits[v]:
                    cut += w
            diag_cut[state_idx] = cut
        return diag_cut

    def construct_circuit(self, params):
        """
        params: 1D array of length 2*p: [gamma_1, ..., gamma_p, beta_1, ..., beta_p]
        """
        gammas = params[:self.p]
        betas = params[self.p:]

        qc = QuantumCircuit(self.n)
        # 1. Initialize in uniform superposition |+>^{\otimes n}
        for q in range(self.n):
            qc.h(q)

        # 2. Alternating layers
        for layer in range(self.p):
            gamma = gammas[layer]
            beta = betas[layer]

            # Cost unitary: exp(-i * gamma * C)
            # For each edge (u, v, w), exp(i * gamma * w/2 * Z_u Z_v)
            # We apply Rzz(-gamma * w, u, v)
            for u, v, w in self.edges:
                qc.rzz(-gamma * w, u, v)

            # Mixer unitary: exp(-i * beta * B) = prod_i Rx(2 * beta, i)
            for q in range(self.n):
                qc.rx(2.0 * beta, q)

        return qc

    def evaluate_cost(self, params):
        """Evaluates expected cut <psi| C |psi>. (We minimize -<C>)."""
        qc = self.construct_circuit(params)
        expected_cut = qc.expectation_diagonal_hamiltonian(self.diag_cut)
        return -expected_cut  # Negative because optimizers minimize

    def get_state(self, params):
        """Returns the full statevector for the circuit."""
        qc = self.construct_circuit(params)
        return qc.state.copy()

    def optimize(self, init_params=None, steps=40, lr=0.1, method="parameter_shift"):
        """Optimizes variational parameters gamma and beta using multi-start to avoid local minima."""
        best_cut_expected = -1.0
        best_params = None
        best_history = None

        seeds = [42, 7, 99] if init_params is None else [42]
        for s in seeds:
            if init_params is not None:
                params = init_params.copy()
            else:
                np.random.seed(s)
                # QAOA angles typically lie in gamma in [0, 2pi], beta in [0, pi]
                params = np.concatenate([
                    np.random.uniform(0.1, 1.2, size=self.p),
                    np.random.uniform(0.1, 0.8, size=self.p)
                ])

            history = [-self.evaluate_cost(params)]
            # Momentum buffer
            v = np.zeros_like(params)
            beta_m = 0.8

            for _ in range(steps):
                if method == "parameter_shift":
                    grads = QuantumGradients.parameter_shift(self.evaluate_cost, params)
                elif method == "spsa":
                    grads = QuantumGradients.spsa_gradient(self.evaluate_cost, params, c=0.05)
                elif method == "qng":
                    qng_step, _ = QuantumGradients.quantum_natural_gradient(
                        self.evaluate_cost, self.get_state, params, lr=lr
                    )
                    grads = qng_step / lr

                v = beta_m * v + lr * grads
                params -= v
                history.append(-self.evaluate_cost(params))

            final_cut = history[-1]
            if final_cut > best_cut_expected:
                best_cut_expected = final_cut
                best_params = params.copy()
                best_history = history

        qc_final = self.construct_circuit(best_params)
        probs = qc_final.get_probabilities()
        best_bitstring_idx = int(np.argmax(probs))
        best_bitstring = [(best_bitstring_idx >> b) & 1 for b in range(self.n)]

        return {
            "p_layers": self.p,
            "optimal_params": best_params.tolist(),
            "expected_cut": float(best_cut_expected),
            "max_prob_bitstring": best_bitstring,
            "max_prob": float(probs[best_bitstring_idx]),
            "history": best_history
        }

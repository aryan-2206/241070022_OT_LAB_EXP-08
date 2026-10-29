"""
Algorithm 20 -- MOEA/D-ACO (Elitist, decomposition + ant colony, Part 3; Ke, Zhang, Battiti 2013)
Defined for the bi-objective TSP (problem P8).  Lecture steps per generation
  1. N Tchebycheff sub-problems (as MOEA/D); every sub-problem i owns a pheromone matrix tau_i and
     one ant; weights lambda_i give the heuristic  eta_i(r,s) = (1/d_rs)^(lambda_i1) * (1/t_rs)^(lambda_i2).
  2. Ant i builds a tour node by node with probability  ~ tau_i(r,s)^alpha * eta_i(r,s)^beta.
  3. Evaluate the N tours; update the solutions of the neighbourhood B(i) (MOEA/D update, n_r replacements).
  4. Pheromone update of sub-problem i: evaporation (1-rho) and deposit of the solutions that currently
     belong to its neighbours B(i) (cooperation), amount 1/(1+g_te).
"""
import numpy as np
from algos.a18_moead import MOEAD


def construct_tour(tau, eta, alpha, beta, rng, n):
    """One ant builds a permutation of customers 0..n-1 (node 0 of the matrices is the depot)."""
    cur, unvisited, tour = 0, list(range(1, n + 1)), []
    while unvisited:
        u = np.array(unvisited)
        p = (tau[cur, u] ** alpha) * (eta[cur, u] ** beta)
        p = p / p.sum()
        nxt = int(rng.choice(u, p=p))
        tour.append(nxt - 1)
        unvisited.remove(nxt)
        cur = nxt
    return np.array(tour)


class MOEADACO(MOEAD):
    number, name, family = 20, "MOEA/D-ACO", "Elitist, decomposition (Part 3)"

    def __init__(self, problem, pop_size=50, seed=0, alpha=1.0, beta=2.0, rho=0.1, verbose=False, **params):
        assert problem.vtype == "perm" and "D" in problem.data, "MOEA/D-ACO needs the TSP problem P8"
        super().__init__(problem, pop_size, seed, T=params.pop("T", 10), verbose=verbose, **params)
        self.alpha, self.beta, self.rho = alpha, beta, rho
        D, Tm = problem.data["D"], problem.data["T"]
        iD = 1.0 / (D + np.eye(len(D)))
        iT = 1.0 / (Tm + np.eye(len(D)))
        self.eta = np.array([(iD ** w[0]) * (iT ** w[1]) for w in self.W])          # (N, n+1, n+1)
        self.tau = np.ones((self.N,) + D.shape)

    def make_offspring(self):
        n = self.problem.n_var
        return np.array([construct_tour(self.tau[i], self.eta[i], self.alpha, self.beta, self.rng, n)
                         for i in range(self.N)])

    def pheromone_update(self):
        zn = self.F.max(axis=0)
        for i in range(self.N):
            self.tau[i] *= (1 - self.rho)
            nb = self.B[i]
            g = tchebycheff_val(self.F[nb], self.W[i], self.z, zn)
            tours = np.hstack([np.zeros((len(nb), 1), int), self.X[nb] + 1, np.zeros((len(nb), 1), int)])
            a, b = tours[:, :-1], tours[:, 1:]
            dep = np.repeat((1.0 / (1.0 + g))[:, None], a.shape[1], axis=1)
            np.add.at(self.tau[i], (a, b), dep)
            np.add.at(self.tau[i], (b, a), dep)
            np.clip(self.tau[i], 1e-3, 1e3, out=self.tau[i])

    def iterate(self):
        Xo, Fo, CVo = self.evaluate(self.make_offspring())
        self.update(Xo, Fo, CVo)
        self.pheromone_update()
        self.log("mean pheromone of sub-problem 0", self.tau[0].mean())


def tchebycheff_val(F, w, z, zn):
    from moo_base import tchebycheff
    return tchebycheff(F, w, z, zn)


def unit_test():
    """Key rules: (a) an ant builds a valid permutation; (b) pheromone deposit raises the tour's edges."""
    from moo_problems import get_problem
    p = get_problem("P8")
    rng = np.random.default_rng(0)
    a = MOEADACO(p, pop_size=10, seed=0)
    t = construct_tour(a.tau[0], a.eta[0], 1.0, 2.0, rng, 15)
    assert sorted(t.tolist()) == list(range(15))
    a.budget = 10 ** 9
    a.initialize()
    before = a.tau[3].copy()
    a.pheromone_update()
    nb = a.B[3]
    tour = np.concatenate([[0], a.X[nb[0]] + 1, [0]])
    e = (tour[:-1], tour[1:])
    assert np.all(a.tau[3][e] > before[e] * (1 - a.rho) - 1e-12), "edges of neighbours' tours receive pheromone"
    return True

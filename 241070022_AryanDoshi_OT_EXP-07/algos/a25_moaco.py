"""
Algorithm 25 -- MOACO, Multi-Objective Ant Colony Optimisation with two pheromone matrices (Swarm, Part 4)
Defined for the bi-objective TSP (P8).  Lecture steps per iteration
  1. Two pheromone matrices tau^1 (distance), tau^2 (time) and two heuristics eta^1 = 1/d, eta^2 = 1/t.
  2. Every ant draws its own weight lambda ~ U(0,1) and moves from r to s with probability
        p(r,s)  ~  [ tau^1(r,s)^lambda tau^2(r,s)^(1-lambda) ]^alpha  [ eta^1(r,s)^lambda eta^2(r,s)^(1-lambda) ]^beta .
  3. Evaluate the tours, update the archive of non-dominated tours.
  4. Evaporation tau <- (1-rho) tau, then every archive member deposits  min f_k / f_k  on edge matrix k.
"""
import numpy as np
from moo_base import MOAlgorithm, Archive


def moaco_tour(tau1, tau2, eta1, eta2, lam, alpha, beta, rng, n):
    cur, unvisited, tour = 0, list(range(1, n + 1)), []
    while unvisited:
        u = np.array(unvisited)
        t = (tau1[cur, u] ** lam) * (tau2[cur, u] ** (1 - lam))
        h = (eta1[cur, u] ** lam) * (eta2[cur, u] ** (1 - lam))
        p = (t ** alpha) * (h ** beta)
        nxt = int(rng.choice(u, p=p / p.sum()))
        tour.append(nxt - 1)
        unvisited.remove(nxt)
        cur = nxt
    return np.array(tour)


class MOACO(MOAlgorithm):
    number, name, family = 25, "MOACO", "Swarm (Part 4)"

    def __init__(self, problem, pop_size=50, seed=0, alpha=1.0, beta=2.0, rho=0.1, verbose=False, **params):
        assert problem.vtype == "perm" and "D" in problem.data, "MOACO needs the TSP problem P8"
        super().__init__(problem, pop_size, seed, verbose, **params)
        D, T = problem.data["D"], problem.data["T"]
        self.eta1, self.eta2 = 1.0 / (D + np.eye(len(D))), 1.0 / (T + np.eye(len(D)))
        self.tau1, self.tau2 = np.ones(D.shape), np.ones(D.shape)
        self.alpha, self.beta, self.rho = alpha, beta, rho
        self.archive = Archive(100, mode="crowding")

    def initialize(self):
        self.X = self.F = self.CV = None

    def deposit(self):
        for tau, k in ((self.tau1, 0), (self.tau2, 1)):
            tau *= (1 - self.rho)
        A = self.archive
        if len(A) == 0:
            return
        fmin = A.F.min(axis=0)
        tours = np.hstack([np.zeros((len(A), 1), int), A.X + 1, np.zeros((len(A), 1), int)])
        a, b = tours[:, :-1], tours[:, 1:]
        for tau, k in ((self.tau1, 0), (self.tau2, 1)):
            dep = np.repeat((fmin[k] / A.F[:, k] / len(A))[:, None], a.shape[1], axis=1)
            np.add.at(tau, (a, b), dep)
            np.add.at(tau, (b, a), dep)
            np.clip(tau, 1e-3, 10.0, out=tau)

    def iterate(self):
        n = self.problem.n_var
        lam = self.rng.random(self.N)
        tours = np.array([moaco_tour(self.tau1, self.tau2, self.eta1, self.eta2, l, self.alpha, self.beta, self.rng, n)
                          for l in lam])
        self.X, self.F, self.CV = self.evaluate(tours)
        self.archive.add(self.X, self.F, self.CV, self.rng)
        self.deposit()
        self.log("archive size", len(self.archive))


def unit_test():
    """Key rules: ants build valid tours; deposit raises the pheromone on edges of archive tours only."""
    from moo_problems import get_problem
    p = get_problem("P8")
    a = MOACO(p, pop_size=20, seed=0)
    t = moaco_tour(a.tau1, a.tau2, a.eta1, a.eta2, 0.3, 1, 2, a.rng, 15)
    assert sorted(t.tolist()) == list(range(15))
    a.budget = 10 ** 9
    a.iterate()
    tour = np.concatenate([[0], a.archive.X[0] + 1, [0]])
    off_edge = a.tau1[0, 0]                                  # self loop never used
    assert a.tau1[tour[:-1], tour[1:]].mean() > 0.9 * 1.0 and off_edge <= 1.0
    before = a.tau1.copy()
    a.deposit()
    assert a.tau1[tour[:-1], tour[1:]].mean() > before[tour[:-1], tour[1:]].mean() * (1 - a.rho) - 1e-9
    return True

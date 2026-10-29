"""
Algorithm 17 -- PESA-II (Elitist, Part 3; Corne, Jerram, Knowles, Oates 2001)
Lecture steps per generation
  1. Internal population IP (small) and bounded external archive EP, objective space divided into a grid
     (hyper-boxes); each box knows how many archive members it contains (squeeze factor).
  2. REGION-BASED selection: choose a hyper-box (binary tournament, the box with fewer members wins),
     then pick a random archive member inside it.
  3. Crossover + mutation -> new IP; evaluate.
  4. Insert every non-dominated IP member into EP; if EP is too big remove random members of the most
     crowded hyper-box.
"""
import numpy as np
from moo_base import MOAlgorithm, Archive, vary, grid_cells


def region_select(F, rng, n, ndiv=10):
    cells = grid_cells(F, ndiv)
    uniq, inv, cnt = np.unique(cells, return_inverse=True, return_counts=True)
    out = []
    for _ in range(n):
        a, b = rng.integers(0, len(F), 2)
        box = inv[a] if cnt[inv[a]] <= cnt[inv[b]] else inv[b]       # tournament between two boxes
        out.append(int(rng.choice(np.where(inv == box)[0])))
    return np.array(out)


class PESA2(MOAlgorithm):
    number, name, family = 17, "PESA-II", "Elitist (Part 3)"

    def __init__(self, problem, pop_size=100, seed=0, ip_size=None, ndiv=10, verbose=False, **params):
        super().__init__(problem, pop_size, seed, verbose, **params)
        self.ip = ip_size or max(10, pop_size // 5)
        self.ndiv = ndiv
        self.archive = Archive(pop_size, mode="grid", ndiv=ndiv)

    def initialize(self):
        X = self.problem.random(self.ip, self.rng)
        self.X, self.F, self.CV = self.evaluate(X)
        self.archive.add(self.X, self.F, self.CV, self.rng)

    def iterate(self):
        par = region_select(self.archive.F, self.rng, self.ip, self.ndiv)
        off = vary(self.problem, self.archive.X[par], self.rng)
        self.X, self.F, self.CV = self.evaluate(off)
        self.archive.add(self.X, self.F, self.CV, self.rng)
        self.log("archive size", len(self.archive))


def unit_test():
    """Key rule: region-based selection prefers the sparse hyper-box (many points in one box, one in another)."""
    rng = np.random.default_rng(0)
    F = np.vstack([np.random.default_rng(1).random((30, 2)) * 0.05, [[1.0, 1.0]]])   # 30 points crowded, 1 isolated
    sel = region_select(F, rng, 2000, ndiv=4)
    frac = np.mean(sel == 30)
    assert frac > 1.6 / 31, "the lone point must be picked about twice as often as under uniform choice 1/31 (got %.3f)" % frac
    return True

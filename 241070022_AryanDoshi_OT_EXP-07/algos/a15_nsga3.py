"""
Algorithm 15 -- NSGA-III (Elitist, many-objective; Deb & Jain 2014)
Lecture steps per generation
  1. Structured reference points Z on the unit simplex (Das-Dennis lattice), |P| ~ |Z| (multiple of 4).
  2. Random mating -> crossover + mutation -> offspring Q; R = P u Q; non-dominated sort.
  3. Take whole fronts until |S| >= N; the last front F_l may be cut.
  4. Normalise S: translate by the ideal point, find extreme points with the achievement scalarising function,
     intercepts of the hyperplane through them, divide.
  5. Associate every member to the reference direction with the smallest perpendicular distance.
  6. Niche preservation: repeatedly pick the reference point with the fewest associated members already
     selected (random tie); from F_l take its closest member if its niche is empty, otherwise a random one.
"""
import numpy as np
from moo_base import MOAlgorithm, vary
from moo_utils import das_dennis, fast_non_dominated_sort


def choose_divisions(M, target):
    p = 1
    while len(das_dennis(M, p + 1)) <= target:
        p += 1
    return p


def nsga3_normalize(Fp):
    """Fp: objectives already translated by the ideal point. Returns intercepts (hyperplane/worst-point)."""
    n, M = Fp.shape
    idx = []
    for i in range(M):
        w = np.full(M, 1e-6)
        w[i] = 1.0
        idx.append(int(np.argmin((Fp / w).max(axis=1))))
    E = Fp[idx]
    try:
        a = np.linalg.solve(E, np.ones(M))
        icpt = 1.0 / a
        if np.any(~np.isfinite(icpt)) or np.any(icpt <= 1e-6):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        icpt = Fp.max(axis=0)
    return np.maximum(icpt, 1e-6)


def associate(Fn, W):
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    proj = Fn @ Wn.T
    d2 = (Fn ** 2).sum(axis=1, keepdims=True) - proj ** 2
    d = np.sqrt(np.maximum(d2, 0))
    j = d.argmin(axis=1)
    return j, d[np.arange(len(Fn)), j]


def niching(K, rho, assoc_last, dist_last, rng):
    """Pick K members of the last front (returns positions inside the last front)."""
    rho = rho.copy()
    avail = list(range(len(assoc_last)))
    chosen = []
    active = set(assoc_last.tolist())
    while len(chosen) < K and avail:
        cand = [j for j in active]
        mn = min(rho[j] for j in cand)
        js = [j for j in cand if rho[j] == mn]
        j = js[rng.integers(len(js))]
        mem = [m for m in avail if assoc_last[m] == j]
        if not mem:
            active.discard(j)
            continue
        m = min(mem, key=lambda t: dist_last[t]) if rho[j] == 0 else mem[rng.integers(len(mem))]
        chosen.append(m)
        avail.remove(m)
        rho[j] += 1
    return chosen


class NSGA3(MOAlgorithm):
    number, name, family = 15, "NSGA-III", "Elitist, many-objective (Part 3, 5E)"

    def __init__(self, problem, pop_size=100, seed=0, verbose=False, **params):
        M = problem.n_obj
        p = pop_size - 1 if M == 2 else choose_divisions(M, pop_size)
        W = das_dennis(M, p)
        super().__init__(problem, int(np.ceil(len(W) / 4) * 4), seed, verbose, **params)
        self.W = W

    def initialize(self):
        super().initialize()
        self.ideal = self.F.min(axis=0)

    def iterate(self):
        perm = self.rng.permutation(self.N)
        off = vary(self.problem, self.X[perm], self.rng)
        Xo, Fo, CVo = self.evaluate(off)
        X, F, CV = np.vstack([self.X, Xo]), np.vstack([self.F, Fo]), np.concatenate([self.CV, CVo])
        self.ideal = np.minimum(self.ideal, F.min(axis=0))
        fronts, rank = fast_non_dominated_sort(F, CV)
        S, k = [], 0
        while len(S) < self.N:
            S += list(fronts[k])
            k += 1
        S = np.array(S)
        if len(S) == self.N:
            sel = S
        else:
            Fl = fronts[k - 1]
            Sp = np.array([i for i in S if i not in set(Fl)], int)
            Fp = F[S] - self.ideal
            icpt = nsga3_normalize(Fp)
            Fn = (F - self.ideal) / icpt
            j_all, d_all = associate(Fn[S], self.W)
            pos = {int(s): t for t, s in enumerate(S)}
            rho = np.zeros(len(self.W), int)
            for s in Sp:
                rho[j_all[pos[int(s)]]] += 1
            a_last = np.array([j_all[pos[int(s)]] for s in Fl])
            d_last = np.array([d_all[pos[int(s)]] for s in Fl])
            pick = niching(self.N - len(Sp), rho, a_last, d_last, self.rng)
            sel = np.concatenate([Sp, Fl[pick]])
            self.log("intercepts", icpt)
            self.log("niche counts rho", rho[:12])
        self.X, self.F, self.CV = X[sel], F[sel], CV[sel]


def unit_test():
    """Key rules: (a) intercepts of the extreme-point hyperplane; (b) the niching step first fills EMPTY niches."""
    Fp = np.array([[2, 0.0, 0.0], [0.0, 3, 0.0], [0.0, 0.0, 4], [1, 1, 1]], float)
    assert np.allclose(nsga3_normalize(Fp), [2, 3, 4], atol=1e-3)
    W = np.array([[1, 0.0], [0.0, 1]])
    Fn = np.array([[1.0, 0.05], [0.9, 0.1], [0.05, 1.0]])
    j, d = associate(Fn, W)
    assert list(j) == [0, 0, 1]
    rng = np.random.default_rng(0)
    rho = np.array([5, 0])                                     # niche 1 is empty, niche 0 is crowded
    chosen = niching(1, rho, np.array([0, 0, 1]), np.array([0.1, 0.2, 0.3]), rng)
    assert chosen == [2], "the empty niche must be served first"
    return True

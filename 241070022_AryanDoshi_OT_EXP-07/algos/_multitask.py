"""
_multitask.py -- the shared-body network of problem P13(b), written with NumPy only (forward AND backward by hand).
    H = tanh(X W1 + b1)                    shared body
    task 1 (classification): p = sigmoid(H wc + bc),  L1 = mean binary cross-entropy
    task 2 (regression)    : y = H wr + br,          L2 = mean squared error
"""
import numpy as np


def make_data(n=400, d=5, seed=0, reg_scale=5.0):
    """Synthetic data. The regression target has a larger scale (reg_scale) so that task 2 has the larger gradients."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w1, w2 = rng.normal(size=d), rng.normal(size=d)
    y_cls = (X @ w1 + 0.5 * np.sin(2 * X[:, 0]) > 0).astype(float)
    y_reg = reg_scale * (np.tanh(X @ w2) + 0.1 * rng.normal(size=n))
    return X, y_cls, y_reg


class SharedNet:
    def __init__(self, X, yc, yr, hidden=16, seed=0):
        rng = np.random.default_rng(seed)
        self.X, self.yc, self.yr = X, yc, yr
        d = X.shape[1]
        self.W1 = rng.normal(0, 0.5, (d, hidden))
        self.b1 = np.zeros(hidden)
        self.wc, self.bc = rng.normal(0, 0.5, hidden), 0.0
        self.wr, self.br = rng.normal(0, 0.5, hidden), 0.0

    # forward / backward ------------------------------------------------------
    def forward(self):
        self.H = np.tanh(self.X @ self.W1 + self.b1)
        z = self.H @ self.wc + self.bc
        self.p = 1.0 / (1.0 + np.exp(-z))
        self.pred = self.H @ self.wr + self.br
        eps = 1e-12
        L1 = -np.mean(self.yc * np.log(self.p + eps) + (1 - self.yc) * np.log(1 - self.p + eps))
        L2 = np.mean((self.pred - self.yr) ** 2)
        return np.array([L1, L2])

    def backward(self):
        """Returns (shared_grads (2, P), head_grads [g_head1, g_head2]) for the two task losses."""
        n = len(self.X)
        dz = (self.p - self.yc) / n
        dpred = 2 * (self.pred - self.yr) / n
        out_shared, out_head = [], []
        for dout, w in ((dz, self.wc), (dpred, self.wr)):
            dH = np.outer(dout, w)
            dZ = dH * (1 - self.H ** 2)
            gW1, gb1 = self.X.T @ dZ, dZ.sum(axis=0)
            out_shared.append(np.concatenate([gW1.ravel(), gb1]))
            out_head.append(np.concatenate([self.H.T @ dout, [dout.sum()]]))
        return np.array(out_shared), out_head

    # parameter access --------------------------------------------------------
    def shared(self):
        return np.concatenate([self.W1.ravel(), self.b1])

    def set_shared(self, v):
        k = self.W1.size
        self.W1 = v[:k].reshape(self.W1.shape)
        self.b1 = v[k:]

    def step(self, shared_dir, head_dirs, lr):
        self.set_shared(self.shared() - lr * shared_dir)
        h = len(self.wc)
        self.wc, self.bc = self.wc - lr * head_dirs[0][:h], self.bc - lr * head_dirs[0][h]
        self.wr, self.br = self.wr - lr * head_dirs[1][:h], self.br - lr * head_dirs[1][h]

    def accuracy(self):
        return float(np.mean((self.p > 0.5) == (self.yc > 0.5)))


def min_norm_weights(G, iters=200):
    """alpha on the simplex minimising ||sum_i alpha_i g_i||^2 (G has one gradient per row).
    Two tasks: closed form. More: Frank-Wolfe."""
    T = len(G)
    if T == 2:
        d = G[0] - G[1]
        den = d @ d
        a = 0.5 if den < 1e-18 else float(np.clip(((G[1] - G[0]) @ G[1]) / den, 0, 1))
        return np.array([a, 1 - a])
    K = G @ G.T
    a = np.ones(T) / T
    for t in range(iters):
        grad = K @ a
        j = int(np.argmin(grad))
        e = np.zeros(T)
        e[j] = 1
        diff = e - a
        den = diff @ K @ diff
        step = 0.0 if den < 1e-18 else float(np.clip(-(a @ K @ diff) / den, 0, 1))
        if step < 1e-10:
            break
        a = a + step * diff
    return a


def train(model, method="fixed", steps=800, lr=0.05, alpha_gn=1.5, lr_w=0.02, seed=0):
    """method: 'fixed' (equal weights), 'mgda', 'gradnorm'. Returns history dict."""
    T = 2
    w = np.ones(T)
    L0 = None
    hist = dict(L=[], w=[], gnorm=[])
    for t in range(steps):
        L = model.forward()
        if L0 is None:
            L0 = L.copy()
        Gs, Gh = model.backward()
        gn = np.linalg.norm(Gs, axis=1)
        if method == "fixed":
            w = np.ones(T)
        elif method == "mgda":
            w = min_norm_weights(Gs) * T                  # scaled so that sum(w) = T
        elif method == "gradnorm":
            Gi = w * gn
            r = (L / L0) / np.mean(L / L0)
            target = Gi.mean() * r ** alpha_gn
            w = w - lr_w * np.sign(Gi - target) * gn
            w = np.clip(w, 1e-3, None)
            w = w * T / w.sum()
        shared_dir = (w[:, None] * Gs).sum(axis=0) / (T if method != "fixed" else 1)
        head_dirs = [w[i] * Gh[i] / (T if method != "fixed" else 1) for i in range(T)]
        if method == "fixed":
            shared_dir = Gs.sum(axis=0) / T
            head_dirs = [Gh[0], Gh[1]]
        model.step(shared_dir, head_dirs, lr)
        hist["L"].append(L.copy()); hist["w"].append(w.copy()); hist["gnorm"].append(gn.copy())
    hist["L"] = np.array(hist["L"]); hist["w"] = np.array(hist["w"]); hist["gnorm"] = np.array(hist["gnorm"])
    return hist

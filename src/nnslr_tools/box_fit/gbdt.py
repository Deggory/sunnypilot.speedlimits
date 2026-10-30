"""A small gradient-boosted tree regressor (squared loss) in numpy: the scorer of the colour fit.

Training bins every feature into ``bins`` quantiles and grows each tree depth first on a random ``subsample`` of the
rows; prediction walks all trees at once on the packed arrays (:func:`pack`), so the model is a handful of numpy
arrays in an ``.npz`` and needs nothing but numpy to run.
"""
from __future__ import annotations

import numpy as np

# The deepest tree prediction walks: a packed model of depth d needs d steps, this covers depth 11.
MAX_STEPS = 12


def _bin_edges(X, bins):
    return [np.unique(np.quantile(X[:, j], np.linspace(0, 1, bins + 1)[1:-1])) for j in range(X.shape[1])]


def _binned(X, edges):
    return np.stack([np.searchsorted(e, X[:, j], side='right') for j, e in enumerate(edges)], 1).astype(np.int16)


def _tree(B, g, depth, bins, min_leaf, lam):
    """One regression tree on the binned rows ``B`` and residuals ``g``: [(feature, bin, left, right, value)];
    a leaf has feature -1."""
    nodes = []

    def build(idx, d):
        me = len(nodes)
        nodes.append(None)
        G = g[idx].sum()
        n = len(idx)
        value = G / (n + lam)
        if d == depth or n < 2 * min_leaf:
            nodes[me] = (-1, 0, -1, -1, value)
            return me
        base = G * G / (n + lam)
        Bi = B[idx]
        gi = g[idx]
        nf = B.shape[1]
        # Gradient sums and counts per (feature, bin), cumulated: the gain of every split at once.
        flat = (Bi + (np.arange(nf) * (bins + 1))[None, :]).ravel()
        cg = np.bincount(flat, weights=np.repeat(gi, nf), minlength=nf * (bins + 1)).reshape(nf, bins + 1).cumsum(1)
        cn = np.bincount(flat, minlength=nf * (bins + 1)).reshape(nf, bins + 1).cumsum(1)
        ok = (cn >= min_leaf) & (n - cn >= min_leaf)
        gain = cg ** 2 / (cn + lam) + (G - cg) ** 2 / (n - cn + lam) - base
        gain[~ok] = -1
        j, t = np.unravel_index(int(np.argmax(gain)), gain.shape)
        if not gain[j, t] > 0:
            nodes[me] = (-1, 0, -1, -1, value)
            return me
        j, t = int(j), int(t)
        left = Bi[:, j] <= t
        nodes[me] = (j, t, build(idx[left], d + 1), build(idx[~left], d + 1), value)
        return me

    build(np.arange(len(g)), 0)
    return nodes


def _predict_tree(arr, X):
    node = np.zeros(len(X), int)
    for _ in range(MAX_STEPS + 4):
        f = arr[node, 0].astype(int)
        leaf = f < 0
        if leaf.all():
            break
        x = X[np.arange(len(X)), np.maximum(f, 0)]
        nxt = np.where(x < arr[node, 1], arr[node, 2], arr[node, 3]).astype(int)
        node = np.where(leaf, node, nxt)
    return arr[node, 4]


def train(X, y, *, n_trees=300, depth=6, lr=0.05, bins=32, min_leaf=20, lam=1.0, subsample=0.8, seed=0) -> dict:
    """Boosted trees on rows ``X`` and targets ``y``; the same arguments and data give the same model."""
    rng = np.random.default_rng(seed)
    edges = _bin_edges(X, bins)
    B = _binned(X, edges)
    base = float(y.mean())
    pred = np.full(len(y), base)
    trees = []
    for _ in range(n_trees):
        rows = rng.random(len(y)) < subsample
        nodes = _tree(B[rows], (y - pred)[rows], depth, bins, min_leaf, lam)
        # A split at bin t sends x < edges[t] to the left (searchsorted right: bin t holds edges[t-1] <= x < edges[t]).
        arr = np.array([(j, (edges[j][t] if t < len(edges[j]) else np.inf) if j >= 0 else 0, left, right, value)
                        for j, t, left, right, value in nodes], float)
        trees.append(arr)
        pred += lr * _predict_tree(arr, X)
    return {'base': base, 'lr': lr, 'trees': trees}


def pack(model: dict) -> dict:
    """The trees stacked into padded arrays (what the .npz holds and :func:`predict` reads)."""
    T = len(model['trees'])
    N = max(len(t) for t in model['trees'])
    A = np.zeros((T, N, 5))
    A[:, :, 0] = -1
    for i, t in enumerate(model['trees']):
        A[i, :len(t)] = t
    return {'base': model['base'], 'lr': model['lr'], 'feat': A[:, :, 0].astype(np.int64), 'thr': A[:, :, 1],
            'left': A[:, :, 2].astype(np.int64), 'right': A[:, :, 3].astype(np.int64), 'val': A[:, :, 4]}


def load(path) -> dict:
    with np.load(path) as z:
        return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}


def predict(pm: dict, X) -> np.ndarray:
    """The packed model's prediction for each row of ``X``."""
    T = pm['feat'].shape[0]
    node = np.zeros((len(X), T), np.int64)
    ti = np.arange(T)[None, :]
    for _ in range(MAX_STEPS):
        f = pm['feat'][ti, node]
        leaf = f < 0
        if leaf.all():
            break
        x = np.take_along_axis(X, np.maximum(f, 0), axis=1)
        nxt = np.where(x < pm['thr'][ti, node], pm['left'][ti, node], pm['right'][ti, node])
        node = np.where(leaf, node, nxt)
    return pm['base'] + pm['lr'] * pm['val'][ti, node].sum(1)

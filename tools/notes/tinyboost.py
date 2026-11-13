"""XGBoost's tree booster for one numeric feature, in plain Python.

This is the algorithm from Chen & Guestrin (KDD 2016), section 2, with the exact
greedy split search and the sparsity-aware default direction, written for reading
rather than speed. tools/notes/xgboost_runs.py checks it against the real library
(tree_method="exact"): same splits, same leaf values, same predictions.

Two details follow the library rather than the paper:
  * the gain it reports and compares with gamma has no factor 1/2;
  * each leaf value is already multiplied by the learning rate (eta).
Standard library only.
"""
from __future__ import annotations

import math


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def grad_hess(obj: str, y: list[float], f: list[float]) -> tuple[list[float], list[float]]:
    """First and second derivatives of the loss at the current scores f."""
    if obj == "squared":                      # loss (f - y)^2 / 2
        return [fi - yi for fi, yi in zip(f, y)], [1.0] * len(y)
    if obj == "logistic":                     # log loss, f is in log-odds
        p = [sigmoid(fi) for fi in f]
        return [pi - yi for pi, yi in zip(p, y)], [pi * (1 - pi) for pi in p]
    raise ValueError(obj)


def split_candidates(xs, g, h, idx, lam, mcw):
    """Every threshold between two different values of x, scored by
    gain = G_L^2/(H_L+lam) + G_R^2/(H_R+lam) - G^2/(H+lam).
    Rows whose x is None are missing: they are tried on the left and on the right."""
    present = sorted((i for i in idx if xs[i] is not None), key=lambda i: xs[i])
    miss = [i for i in idx if xs[i] is None]
    G, H = sum(g[i] for i in idx), sum(h[i] for i in idx)
    Gm, Hm = sum(g[i] for i in miss), sum(h[i] for i in miss)
    parent = G * G / (H + lam)
    out, GL, HL = [], 0.0, 0.0
    for k in range(len(present) - 1):
        GL += g[present[k]]
        HL += h[present[k]]
        a, b = xs[present[k]], xs[present[k + 1]]
        if a == b:
            continue
        for miss_left in ((True, False) if miss else (False,)):
            gl = GL + (Gm if miss_left else 0.0)
            hl = HL + (Hm if miss_left else 0.0)
            gr, hr = G - gl, H - hl
            out.append(dict(thr=(a + b) / 2, n_left=k + 1, miss_left=miss_left,
                            GL=gl, HL=hl, GR=gr, HR=hr,
                            gain=gl * gl / (hl + lam) + gr * gr / (hr + lam) - parent,
                            allowed=hl >= mcw and hr >= mcw))
    return out


def build_tree(xs, g, h, idx, depth, p):
    G, H = sum(g[i] for i in idx), sum(h[i] for i in idx)
    if depth < p["max_depth"]:
        ok = [c for c in split_candidates(xs, g, h, idx, p["lam"], p["mcw"])
              if c["allowed"] and c["gain"] > 1e-6]
        if ok:
            c = max(ok, key=lambda c: c["gain"])
            if c["gain"] >= p["gamma"]:
                left = [i for i in idx if (c["miss_left"] if xs[i] is None else xs[i] < c["thr"])]
                lset = set(left)
                right = [i for i in idx if i not in lset]
                return dict(thr=c["thr"], miss_left=c["miss_left"], gain=c["gain"], H=H,
                            left=build_tree(xs, g, h, left, depth + 1, p),
                            right=build_tree(xs, g, h, right, depth + 1, p))
    return dict(leaf=-G / (H + p["lam"]) * p["eta"], G=G, H=H, n=len(idx))


def tree_value(t: dict, x) -> float:
    while "leaf" not in t:
        go_left = t["miss_left"] if x is None else x < t["thr"]
        t = t["left"] if go_left else t["right"]
    return t["leaf"]


def fit(xs, y, obj="squared", rounds=10, eta=0.3, max_depth=6, lam=1.0, mcw=1.0, gamma=0.0,
        base=None):
    """Returns (starting score, trees). base defaults to the mean label, as in XGBoost 3."""
    if base is None:
        base = sum(y) / len(y)
    f0 = base if obj == "squared" else math.log(base / (1 - base))
    p = dict(eta=eta, max_depth=max_depth, lam=lam, mcw=mcw, gamma=gamma)
    f, trees = [f0] * len(y), []
    for _ in range(rounds):
        g, h = grad_hess(obj, y, f)
        t = build_tree(xs, g, h, list(range(len(y))), 0, p)
        trees.append(t)
        f = [fi + tree_value(t, x) for fi, x in zip(f, xs)]
    return f0, trees


def predict(model, x, n_trees=None) -> float:
    """Score after the first n_trees trees (all of them by default)."""
    f0, trees = model
    return f0 + sum(tree_value(t, x) for t in trees[:n_trees])

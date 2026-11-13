#!/usr/bin/env python3
"""Runs with the real XGBoost library behind the note content/notes/xgboost/.

    pip install xgboost scikit-learn pandas numpy
    python3 tools/notes/xgboost_runs.py --creditcard creditcard.csv \
        --heart path/to/heart-disease-prediction

1. Checks tools/notes/tinyboost.py against xgboost (tree_method="exact") on the note's toy
   data: same splits, gains, leaf values and predictions.
2. Credit-card fraud: the ULB dataset (284,807 card transactions over two days, 492 frauds;
   https://api.openml.org/d/42175, also at
   https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv).
   Split by time: the first 60% of transactions train, the next 20% validate, the last 20% test.
   Learning-rate curves, scale_pos_weight and max_delta_step, importance types and SHAP values.
3. Heart disease: the 80/20 split of https://github.com/HenryVarro666/heart-disease-prediction
   (processed_data.csv, random_state=4), re-run with XGBoost, plus 10 x 5-fold cross-validation.

Writes tools/notes/xgboost_runs.json, which xgboost_figs.py reads to draw the figures and
print the numbers. Takes a few minutes on two CPU cores. Results were produced with
xgboost 3.4.1, scikit-learn 1.9.1, numpy 2.5.3.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tinyboost as tb  # noqa: E402
from xgboost_figs import (WAVE, WAVE_TEST, TREND, WORKED_X, WORKED_Y,  # noqa: E402
                          MISSING_X, MISSING_Y)

OUT = HERE / "xgboost_runs.json"
warnings.filterwarnings("ignore", module="sklearn")  # deprecation notices for penalty="l1"
THREADS = 2


def r6(x):
    """Round floats for the JSON file (6 significant digits)."""
    if isinstance(x, float):
        return float(f"{x:.6g}")
    if isinstance(x, dict):
        return {k: r6(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r6(v) for v in x]
    if isinstance(x, np.generic):
        return r6(x.item())
    return x


# ---- 1. tinyboost against the library ---------------------------------------------------
def xgb_exact(xs, y, obj, rounds, base, **kw):
    X = np.array([np.nan if v is None else v for v in xs], dtype=np.float32).reshape(-1, 1)
    d = xgb.DMatrix(X, label=np.array(y, dtype=float), missing=np.nan)
    p = dict(objective=obj, tree_method="exact", base_score=base, nthread=THREADS)
    p.update(kw)
    return xgb.train(p, d, num_boost_round=rounds), d


def check_tinyboost() -> dict:
    out = {}
    # Regression on the wave: predictions on the training points and on a grid between them.
    (xs, ys) = WAVE
    base = sum(ys) / len(ys)
    for rounds in (1, 5, 30, 300):
        mine = tb.fit(xs, ys, "squared", rounds=rounds, eta=0.3, max_depth=2, lam=1, mcw=1)
        b, d = xgb_exact(xs, ys, "reg:squarederror", rounds, base, eta=0.3, max_depth=2,
                         reg_lambda=1, min_child_weight=1)
        grid = np.linspace(-0.4937, 10.4937, 517, dtype=np.float32)  # avoids split points
        theirs = b.predict(xgb.DMatrix(grid.reshape(-1, 1)))
        diff = max(abs(float(t) - tb.predict(mine, float(g))) for g, t in zip(grid, theirs))
        out[f"wave_{rounds}_max_abs_diff"] = diff
        assert diff < 1e-4, diff
    # The worked example: one stump on twelve transactions, log loss.
    base = sum(WORKED_Y) / len(WORKED_Y)
    stumps = {}
    for name, kw in (("default", {}), ("mcw0", dict(min_child_weight=0))):
        b, _ = xgb_exact(WORKED_X, WORKED_Y, "binary:logistic", 1, base, eta=0.3, max_depth=1,
                         reg_lambda=1, **({"min_child_weight": 1} | kw))
        j = json.loads(b.get_dump(dump_format="json", with_stats=True)[0])
        mine = tb.fit(WORKED_X, WORKED_Y, "logistic", rounds=1, eta=0.3, max_depth=1, lam=1,
                      mcw=kw.get("min_child_weight", 1), base=base)[1][0]
        assert abs(j["split_condition"] - mine["thr"]) < 1e-6
        assert abs(j["gain"] - mine["gain"]) < 1e-4
        assert abs(j["children"][0]["leaf"] - mine["left"]["leaf"]) < 1e-5
        assert abs(j["children"][1]["leaf"] - mine["right"]["leaf"]) < 1e-5
        stumps[name] = dict(threshold=j["split_condition"], gain=j["gain"],
                            leaves=[c["leaf"] for c in j["children"]],
                            covers=[c["cover"] for c in j["children"]])
    out["worked"] = stumps
    # gamma is compared with the gain as the library reports it (no factor 1/2).
    g = stumps["default"]["gain"]
    kept, _ = xgb_exact(WORKED_X, WORKED_Y, "binary:logistic", 1, base, eta=0.3, max_depth=1,
                        reg_lambda=1, gamma=g - 0.01)
    pruned, _ = xgb_exact(WORKED_X, WORKED_Y, "binary:logistic", 1, base, eta=0.3, max_depth=1,
                          reg_lambda=1, gamma=g + 0.01)
    half, _ = xgb_exact(WORKED_X, WORKED_Y, "binary:logistic", 1, base, eta=0.3, max_depth=1,
                        reg_lambda=1, gamma=g / 2 + 0.01)
    out["gamma_just_below_gain_keeps_split"] = "split_condition" in kept.get_dump(dump_format="json")[0]
    out["gamma_just_above_gain_prunes"] = "split_condition" not in pruned.get_dump(dump_format="json")[0]
    out["gamma_above_half_gain_keeps_split"] = "split_condition" in half.get_dump(dump_format="json")[0]
    assert out["gamma_just_below_gain_keeps_split"] and out["gamma_just_above_gain_prunes"]
    assert out["gamma_above_half_gain_keeps_split"]
    # Missing values: two more transactions with no value, both fraud.
    base = sum(MISSING_Y) / len(MISSING_Y)
    b, _ = xgb_exact(MISSING_X, MISSING_Y, "binary:logistic", 1, base, eta=0.3, max_depth=1,
                     reg_lambda=1, min_child_weight=1)
    j = json.loads(b.get_dump(dump_format="json", with_stats=True)[0])
    mine = tb.fit(MISSING_X, MISSING_Y, "logistic", rounds=1, eta=0.3, max_depth=1, lam=1,
                  mcw=1, base=base)[1][0]
    assert abs(j["split_condition"] - mine["thr"]) < 1e-6 and abs(j["gain"] - mine["gain"]) < 1e-4
    assert (j["missing"] == j["yes"]) == mine["miss_left"]
    out["missing"] = dict(threshold=j["split_condition"], gain=j["gain"],
                          missing_goes="left" if j["missing"] == j["yes"] else "right")
    # Extrapolation: the trend data.
    (xs, ys) = TREND
    base = sum(ys) / len(ys)
    mine = tb.fit(xs, ys, "squared", rounds=200, eta=0.1, max_depth=2, lam=1, mcw=1)
    b, _ = xgb_exact(xs, ys, "reg:squarederror", 200, base, eta=0.1, max_depth=2, reg_lambda=1,
                     min_child_weight=1)
    grid = np.linspace(-0.4937, 15.4937, 517, dtype=np.float32)
    theirs = b.predict(xgb.DMatrix(grid.reshape(-1, 1)))
    diff = max(abs(float(t) - tb.predict(mine, float(g))) for g, t in zip(grid, theirs))
    out["trend_max_abs_diff"] = diff
    assert diff < 1e-3, diff
    return out


# ---- 2. credit-card fraud ---------------------------------------------------------------
def load_creditcard(path: str):
    df = pd.read_csv(path)
    assert df["Time"].is_monotonic_increasing and len(df) == 284_807
    cols = list(df.columns[:-1])
    X = df[cols].to_numpy(np.float64)
    y = df["Class"].to_numpy(int)
    n = len(y)
    i1, i2 = int(n * 0.6), int(n * 0.8)
    parts = {k: (X[s], y[s]) for k, s in
             (("train", slice(0, i1)), ("valid", slice(i1, i2)), ("test", slice(i2, n)))}
    return cols, parts


def dm(parts, k, cols):
    X, y = parts[k]
    return xgb.DMatrix(X, label=y, feature_names=cols)


def leaves(j):
    return [j["leaf"]] if "leaf" in j else sum((leaves(c) for c in j["children"]), [])


def keep_rounds(n: int) -> list[int]:
    """Rounds (1-based) to store for a log-x plot: all of the first 60, then thinner."""
    ks = set(range(1, min(n, 60) + 1))
    k = 60
    while k < n:
        k = int(k * 1.04) + 1
        ks.add(min(k, n))
    ks.add(n)
    return sorted(ks)


def learning_rate_curves(cols, parts) -> dict:
    dtr, dva, dte = (dm(parts, k, cols) for k in ("train", "valid", "test"))
    yte = parts["test"][1]
    out = {}
    for eta, mds, rounds in ((0.3, 0, 600), (0.1, 0, 1000), (0.03, 0, 2000), (0.3, 1, 600)):
        ev: dict = {}
        p = dict(objective="binary:logistic", eta=eta, eval_metric=["logloss", "aucpr"],
                 seed=0, nthread=THREADS)
        if mds:
            p["max_delta_step"] = mds
        b = xgb.train(p, dtr, num_boost_round=rounds, evals=[(dtr, "train"), (dva, "valid")],
                      evals_result=ev, verbose_eval=False)
        vl = ev["valid"]["logloss"]
        best = int(np.argmin(vl)) + 1
        pte = b.predict(dte, iteration_range=(0, best))
        first = leaves(json.loads(b.get_dump(dump_format="json")[0]))
        ks = keep_rounds(rounds)
        out[f"eta{eta}_mds{mds}"] = dict(
            eta=eta, max_delta_step=mds, rounds=ks, valid_logloss=[vl[k - 1] for k in ks],
            best_round=best, best_valid_logloss=vl[best - 1],
            final_valid_logloss=vl[-1], first_train_logloss=ev["train"]["logloss"][0],
            first_tree_max_leaf=max(first), first_tree_min_leaf=min(first),
            test_ap_at_best=float(average_precision_score(yte, pte)),
            test_auc_at_best=float(roc_auc_score(yte, pte)))
    ytr = parts["train"][1]
    yva = parts["valid"][1]
    out["constant_valid_logloss"] = float(log_loss(yva, np.full(len(yva), ytr.mean())))
    out["constant_train_logloss"] = float(log_loss(ytr, np.full(len(ytr), ytr.mean())))
    return out


def fit_es(cols, parts, drop=(), **kw):
    keep = [c for c in cols if c not in drop]
    idx = [cols.index(c) for c in keep]
    mk = lambda k: xgb.DMatrix(parts[k][0][:, idx], label=parts[k][1], feature_names=keep)
    dtr, dva, dte = mk("train"), mk("valid"), mk("test")
    p = dict(objective="binary:logistic", eta=0.05, eval_metric=["aucpr", "logloss"], seed=0,
             nthread=THREADS)
    p.update(kw)
    b = xgb.train(p, dtr, num_boost_round=3000, evals=[(dva, "valid")],
                  early_stopping_rounds=300, verbose_eval=False)
    b = b[: b.best_iteration + 1]
    return b, dte


def pr_points(y, p):
    """(recall, precision) at each positive, walking down the ranking (ties broken by order)."""
    order = np.argsort(-p, kind="stable")
    ys = y[order]
    npos = int(y.sum())
    pts, hits = [], 0
    for i, v in enumerate(ys, 1):
        if v:
            hits += 1
            pts.append((hits / npos, hits / i))
    return pts


def at_threshold(y, p, t):
    f = p >= t
    return dict(flagged=int(f.sum()), tp=int((f & (y == 1)).sum()), fp=int((f & (y == 0)).sum()))


def imbalance(cols, parts) -> dict:
    ytr = parts["train"][1]
    yte = parts["test"][1]
    spw = float((ytr == 0).sum() / (ytr == 1).sum())
    out = dict(scale_pos_weight=spw, split={k: dict(rows=len(v[1]), frauds=int(v[1].sum()))
                                            for k, v in parts.items()},
               time_ranges={k: [float(v[0][0, 0]), float(v[0][-1, 0])] for k, v in parts.items()})
    preds = {}
    for name, kw in (("default", {}), ("spw", dict(scale_pos_weight=spw)),
                     ("mds1", dict(max_delta_step=1))):
        b, dte = fit_es(cols, parts, **kw)
        p = b.predict(dte)
        preds[name] = p
        bins = []
        for lo, hi in ((0, 0.01), (0.01, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 1.0001)):
            m = (p >= lo) & (p < hi)
            bins.append(dict(lo=lo, hi=hi, n=int(m.sum()), frauds=int(yte[m].sum()),
                             mean_p=float(p[m].mean()) if m.any() else None))
        out[name] = dict(trees=b.num_boosted_rounds(), ap=float(average_precision_score(yte, p)),
                         auc=float(roc_auc_score(yte, p)), logloss=float(log_loss(yte, p)),
                         mean_p=float(p.mean()), test_rate=float(yte.mean()),
                         median_p_fraud=float(np.median(p[yte == 1])),
                         median_p_legit=float(np.median(p[yte == 0])),
                         at_05=at_threshold(yte, p, 0.5), bins=bins, pr=pr_points(yte, p))
    # The default model cut at the same number of flags as scale_pos_weight's 0.5.
    k = out["spw"]["at_05"]["flagged"]
    t = float(np.sort(preds["default"])[::-1][k - 1])
    out["default_same_flags"] = dict(threshold=t, **at_threshold(yte, preds["default"], t))
    # Paired bootstrap of the AP difference, default minus scale_pos_weight.
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(2000):
        i = rng.integers(0, len(yte), len(yte))
        if yte[i].sum() == 0:
            continue
        diffs.append(average_precision_score(yte[i], preds["default"][i])
                     - average_precision_score(yte[i], preds["spw"][i]))
    out["ap_diff_default_minus_spw_95ci"] = [float(v) for v in np.percentile(diffs, [2.5, 97.5])]
    return out


def importance(cols, parts) -> dict:
    b, dte = fit_es(cols, parts)
    out = {}
    for kind in ("weight", "gain", "cover", "total_gain"):
        s = b.get_score(importance_type=kind)
        ranked = sorted(s, key=lambda k: -s[k])
        out[kind] = dict(top=[(k, s[k]) for k in ranked[:5]],
                         rank={c: ranked.index(c) + 1 for c in ("Time", "Amount", "V14", "V4")})
    sh = b.predict(dte, pred_contribs=True)
    mag = np.abs(sh[:, :-1]).mean(0)
    ranked = [cols[i] for i in np.argsort(-mag)]
    out["shap"] = dict(top=[(c, float(mag[cols.index(c)])) for c in ranked[:5]],
                       rank={c: ranked.index(c) + 1 for c in ("Time", "Amount", "V14", "V4")})
    margin = b.predict(dte, output_margin=True)
    out["shap_sum_max_err"] = float(np.abs(sh.sum(1) - margin).max())
    ti = cols.index("Time")
    dtr = xgb.DMatrix(parts["train"][0], feature_names=cols)
    shtr = b.predict(dtr, pred_contribs=True)
    out["time_shap"] = dict(test_mean=float(sh[:, ti].mean()), test_min=float(sh[:, ti].min()),
                            test_max=float(sh[:, ti].max()), train_mean=float(shtr[:, ti].mean()))
    dump = " ".join(b.get_dump())
    thr = [float(v.split("]")[0]) for v in dump.split("[Time<")[1:]]
    out["time_splits"] = dict(n=len(thr), max_threshold=max(thr))
    b2, dte2 = fit_es(cols, parts, drop=("Time",))
    yte = parts["test"][1]
    out["without_time"] = dict(ap=float(average_precision_score(yte, b2.predict(dte2))),
                               auc=float(roc_auc_score(yte, b2.predict(dte2))))
    out["with_time"] = dict(ap=float(average_precision_score(yte, b.predict(dte))),
                            auc=float(roc_auc_score(yte, b.predict(dte))))
    return out


# ---- 3. heart disease -------------------------------------------------------------------
def heart(repo: str) -> dict:
    df = pd.read_csv(Path(repo) / "processed_data.csv").iloc[:, 1:]
    x = MinMaxScaler().fit_transform(df.drop(["target"], axis=1))
    y = df["target"]
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=4)
    models = {
        "Logistic regression": LogisticRegression(max_iter=1000, random_state=1,
                                                  solver="liblinear", penalty="l1"),
        "Support vector machine": SVC(kernel="linear", max_iter=1000, C=10, probability=True,
                                      random_state=0),
        "Random forest": RandomForestClassifier(n_estimators=1000, random_state=1,
                                                max_leaf_nodes=20, min_samples_split=15),
        "XGBoost, defaults": xgb.XGBClassifier(random_state=0, n_jobs=THREADS),
        "XGBoost, small trees": xgb.XGBClassifier(n_estimators=300, learning_rate=0.05,
                                                  max_depth=2, subsample=0.8,
                                                  colsample_bytree=0.8, random_state=0,
                                                  n_jobs=THREADS),
    }
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
    out = dict(train=len(ytr), test=len(yte), test_class1=int(yte.sum()), models={})
    for name, m in models.items():
        m.fit(xtr, ytr)
        p = m.predict_proba(xte)[:, 1]
        s = cross_val_score(m, x, y, cv=cv, scoring="roc_auc")
        out["models"][name] = dict(test_accuracy=float(accuracy_score(yte, m.predict(xte))),
                                   test_auc=float(roc_auc_score(yte, p)),
                                   cv_auc_mean=float(s.mean()), cv_auc_sd=float(s.std()),
                                   cv_auc_min=float(s.min()), cv_auc_max=float(s.max()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--creditcard", required=True, help="path to creditcard.csv")
    ap.add_argument("--heart", required=True, help="path to a heart-disease-prediction checkout")
    a = ap.parse_args()
    t0 = time.time()
    res = dict(versions=dict(xgboost=xgb.__version__, sklearn=sklearn.__version__,
                             numpy=np.__version__))
    res["tinyboost_check"] = check_tinyboost()
    print("tinyboost matches xgboost", f"{time.time() - t0:.0f}s")
    cols, parts = load_creditcard(a.creditcard)
    res["lr_curves"] = learning_rate_curves(cols, parts)
    print("learning-rate curves", f"{time.time() - t0:.0f}s")
    res["imbalance"] = imbalance(cols, parts)
    print("imbalance", f"{time.time() - t0:.0f}s")
    res["importance"] = importance(cols, parts)
    print("importance", f"{time.time() - t0:.0f}s")
    res["heart"] = heart(a.heart)
    print("heart", f"{time.time() - t0:.0f}s")
    OUT.write_text(json.dumps(r6(res), indent=1, ensure_ascii=False) + "\n")
    print("wrote", OUT.relative_to(HERE.parents[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

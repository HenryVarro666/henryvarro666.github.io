#!/usr/bin/env python3
"""Figures and numbers for the note content/notes/xgboost/.

    python3 tools/notes/xgboost_figs.py

writes seven SVGs next to the note, in English (name.svg) and Chinese (name.zh.svg), and
prints every number the text quotes. Standard library only.

The toy figures (one tree, boosting rounds, the worked split, missing values, extrapolation)
are computed here with tinyboost.py, a from-scratch version of XGBoost's tree booster that
xgboost_runs.py checks against the real library. The credit-card figures read
xgboost_runs.json, written by xgboost_runs.py, which needs xgboost and the data.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "notes"))
sys.path.insert(0, str(ROOT / "tools" / "figs"))
import tinyboost as tb  # noqa: E402
from precision_vs_recall_figs import Canvas  # noqa: E402

OUT = ROOT / "content" / "notes" / "xgboost"
RUNS_FILE = ROOT / "tools" / "notes" / "xgboost_runs.json"
LANG = "en"
RUNS: dict = {}


def t(en: str, zh: str) -> str:
    return zh if LANG == "zh" else en


# ---- toy data ---------------------------------------------------------------------------
def wave_truth(x: float) -> float:
    return 2 * math.sin(0.8 * x) + 0.3 * x


def _wave(seed: int, n: int):
    r = random.Random(seed)
    xs = sorted(round(r.uniform(0, 10), 2) for _ in range(n))
    return xs, [round(wave_truth(x) + r.gauss(0, 0.35), 3) for x in xs]


def _trend(seed: int, n: int, lo: float, hi: float):
    r = random.Random(seed)
    xs = sorted(round(r.uniform(lo, hi), 2) for _ in range(n))
    return xs, [round(1.5 * x + 2 + r.gauss(0, 1.0), 3) for x in xs]


WAVE = _wave(3, 40)            # 40 training points
WAVE_TEST = _wave(4, 400)      # fresh points from the same curve, to measure error
TREND = _trend(5, 40, 0, 10)   # a straight-line trend, seen only for x in [0, 10]
TREND_FUTURE = _trend(6, 15, 10, 15)

# Twelve card transactions: x = purchases on the card in the past 24 hours, y = 1 for fraud.
WORKED_X = [float(v) for v in range(1, 13)]
WORKED_Y = [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1]
# The same twelve plus two transactions whose count is missing, both fraud.
MISSING_X = WORKED_X + [None, None]
MISSING_Y = WORKED_Y + [1, 1]

BOOST = dict(eta=0.3, max_depth=2, lam=1.0, mcw=1.0)   # the wave and its rounds
SHOWN_ROUNDS = (1, 5, 30, 300)


def rmse(model, data, n_trees=None) -> float:
    xs, ys = data
    return math.sqrt(sum((tb.predict(model, x, n_trees) - y) ** 2 for x, y in zip(xs, ys)) / len(ys))


def error_curve(model, data, n: int) -> list[float]:
    """RMSE after 1..n trees, computed incrementally."""
    xs, ys = data
    f = [model[0]] * len(xs)
    out = []
    for tree in model[1][:n]:
        f = [fi + tb.tree_value(tree, x) for fi, x in zip(f, xs)]
        out.append(math.sqrt(sum((fi - y) ** 2 for fi, y in zip(f, ys)) / len(ys)))
    return out


def single_tree():
    """A plain regression tree of depth 2: no shrinkage, no penalty, leaves = means."""
    return tb.fit(*WAVE, "squared", rounds=1, eta=1.0, max_depth=2, lam=0.0, mcw=1.0)


def wave_model():
    return tb.fit(*WAVE, "squared", rounds=max(SHOWN_ROUNDS), **BOOST)


def worked():
    base = sum(WORKED_Y) / len(WORKED_Y)
    f0 = math.log(base / (1 - base))
    g, h = tb.grad_hess("logistic", WORKED_Y, [f0] * len(WORKED_Y))
    cands = tb.split_candidates(WORKED_X, g, h, list(range(len(WORKED_X))), 1.0, 1.0)
    stump = tb.fit(WORKED_X, WORKED_Y, "logistic", rounds=1, eta=0.3, max_depth=1)[1][0]
    return dict(base=base, f0=f0, g=g, h=h, cands=cands, stump=stump)


def missing():
    base = sum(MISSING_Y) / len(MISSING_Y)
    f0 = math.log(base / (1 - base))
    g, h = tb.grad_hess("logistic", MISSING_Y, [f0] * len(MISSING_Y))
    cands = tb.split_candidates(MISSING_X, g, h, list(range(len(MISSING_X))), 1.0, 1.0)
    stump = tb.fit(MISSING_X, MISSING_Y, "logistic", rounds=1, eta=0.3, max_depth=1)[1][0]
    return dict(base=base, cands=cands, stump=stump)


def trend_models():
    trees = tb.fit(*TREND, "squared", rounds=200, eta=0.1, max_depth=2, lam=1.0, mcw=1.0)
    xs, ys = TREND
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / sum((a - mx) ** 2 for a in xs)
    return trees, (my - slope * mx, slope)


def f2(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}"


def sgn(x: float, d: int = 2) -> str:
    """Signed number with a real minus sign."""
    return f"{x:+.{d}f}".replace("-", "−")


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


# ---- 1. one tree ------------------------------------------------------------------------
def fig_one_tree() -> Canvas:
    model = single_tree()
    root = model[1][0]
    base = model[0]
    c = Canvas("one-tree", 720, 340, t(
        "A regression tree of depth two on forty points. Left, the tree: three yes-or-no questions about x "
        "and four leaves, each holding the mean of its rows. Right, the same tree as a function: a staircase "
        "with one step per leaf.",
        "一棵深度为 2 的回归树，拟合 40 个点。左边是树：三个关于 x 的是非题，四个叶子，每个叶子存它那些样本的均值。"
        "右边是同一棵树画成函数：一级台阶对应一个叶子。"))
    # left: the tree
    c.text(20, 28, t("the tree", "树"), 13.5, "f-t", weight=600)

    def node(cx, cy, label, w=104):
        c.rect(cx - w / 2, cy - 16, w, 32, "f-node", 6, 1.2)
        c.text(cx, cy + 4.5, label, 12.5, "f-t", "middle")

    def leaf(cx, cy, val, n):
        c.rect(cx - 31, cy - 22, 62, 44, "f-hl", 6, 1.4)
        c.text(cx, cy - 3, f2(val), 13, "f-t", "middle", 600)
        c.text(cx, cy + 14, t(f"{n} rows", f"{n} 个"), 11.5, "f-m", "middle")

    rx, ry = 160, 72
    lx, rxx, cy2 = 88, 232, 152
    leaves_x = (52, 124, 196, 268)
    ly = 252
    node(rx, ry, f"x < {root['thr']:g} ?", 112)
    node(lx, cy2, f"x < {root['left']['thr']:g} ?")
    node(rxx, cy2, f"x < {root['right']['thr']:g} ?")
    for (x1, y1, x2, y2, lab, side) in ((rx, ry + 16, lx, cy2 - 16, t("yes", "是"), -1),
                                         (rx, ry + 16, rxx, cy2 - 16, t("no", "否"), 1)):
        c.line(x1, y1, x2, y2, "f-ln", 1.4)
        c.text((x1 + x2) / 2 + side * 14, (y1 + y2) / 2, lab, 11.5, "f-m", "middle")
    kids = (root["left"]["left"], root["left"]["right"], root["right"]["left"], root["right"]["right"])
    parents = (lx, lx, rxx, rxx)
    for k, (cx, px, kid) in enumerate(zip(leaves_x, parents, kids)):
        c.line(px, cy2 + 16, cx, ly - 22, "f-ln", 1.4)
        leaf(cx, ly, base + kid["leaf"], kid["n"])
    c.text(20, 316, t("leaf value = mean of y in that leaf", "叶子的值 = 落进这个叶子的 y 的均值"), 12, "f-m")

    # right: the staircase
    x0, w, yt, yb = 372, 318, 44, 280
    ylo, yhi = -1.0, 5.0
    sx = lambda v: x0 + v / 10 * w
    sy = lambda v: yb - (v - ylo) / (yhi - ylo) * (yb - yt)
    c.text(x0 - 24, 28, t("the same tree as a function", "同一棵树，画成函数"), 13.5, "f-t", weight=600)
    for v in (0, 2, 4):
        c.line(x0, sy(v), x0 + w, sy(v), "f-lnrule", 1)
        c.text(x0 - 8, sy(v) + 4.5, f"{v:g}", 12, "f-m", "end")
    for v in (0, 2, 4, 6, 8, 10):
        c.line(sx(v), yb, sx(v), yb + 5, "f-lnrule", 1)
        c.text(sx(v), yb + 20, f"{v:g}", 12, "f-m", "middle")
    c.text(x0 + w / 2, yb + 40, "x", 12, "f-m", "middle")
    c.text(x0 - 8, sy(5) + 4.5, "y", 12, "f-m", "end")
    for th in (root["thr"], root["left"]["thr"], root["right"]["thr"]):
        c.line(sx(th), yt, sx(th), yb, "f-lnrule", 1, ' stroke-dasharray="3 3"')
    for x, y in zip(*WAVE):
        c.dot(sx(x), sy(y), 3.2, "f-dotm", ring=False)
    pts = []
    steps = 400
    for i in range(steps + 1):
        x = 10 * i / steps
        pts.append((sx(x), sy(tb.predict(model, x))))
    # draw as a staircase: vertical jumps at thresholds
    path = [pts[0]]
    for a, b in zip(pts, pts[1:]):
        if abs(a[1] - b[1]) > 1e-9:
            path.append((b[0], a[1]))
        path.append(b)
    c.path(path, "f-lnhl", 2.4)
    return c


# ---- 2. boosting rounds -----------------------------------------------------------------
def fig_rounds() -> Canvas:
    model = wave_model()
    n = max(SHOWN_ROUNDS)
    tr = error_curve(model, WAVE, n)
    te = error_curve(model, WAVE_TEST, n)
    best = min(range(n), key=lambda i: te[i]) + 1
    c = Canvas("boosting-rounds", 720, 486, t(
        f"Boosting on the same forty points: trees of depth two, learning rate {BOOST['eta']}. Top, the model after "
        f"1, 5, 30 and 300 trees, with the curve the points were drawn from dashed. Bottom, the error on the "
        f"training points keeps falling; the error on fresh points is lowest at {best} trees and then rises.",
        f"还是这 40 个点做 boosting：每棵树深度 2，学习率 {BOOST['eta']}。上排是 1、5、30、300 棵树之后的模型，虚线是生成数据的真实曲线。"
        f"下面的曲线：训练点上的误差一直下降；新数据上的误差在 {best} 棵树时最低，之后回升。"))
    pw, ph, py = 150, 112, 64
    ylo, yhi = -1.2, 5.2
    for k, r in enumerate(SHOWN_ROUNDS):
        x0 = 28 + k * 172
        sx = lambda v, x0=x0: x0 + v / 10 * pw
        sy = lambda v: py + ph - (v - ylo) / (yhi - ylo) * ph
        title = t(f"{r} tree" + ("s" if r > 1 else ""), f"{r} 棵树")
        c.text(x0, 26, title, 13.5, "f-t", weight=600)
        c.text(x0, 46, t(f"train {f2(tr[r - 1])} · new {f2(te[r - 1])}",
                         f"训练 {f2(tr[r - 1])} · 新数据 {f2(te[r - 1])}"), 11.5, "f-m")
        c.rect(x0, py, pw, ph, "f-node", 4, 1)
        c.path([(sx(i / 20), sy(wave_truth(i / 20))) for i in range(201)], "f-lnbar2", 1.2)
        c.raw(c.parts.pop().replace("/>", ' stroke-dasharray="4 3"/>'))
        for x, y in zip(*WAVE):
            c.dot(sx(x), sy(y), 2.2, "f-dotm", ring=False)
        pts = [(sx(i / 40), sy(tb.predict(model, i / 40, r))) for i in range(401)]
        c.path(pts, "f-lnhl", 2.0)
    c.text(28, 200, t("numbers above each panel: root-mean-square error (RMSE) on the 40 training points and on 400 new points",
                      "每格上方的数字：40 个训练点和 400 个新点上的均方根误差（RMSE）"), 11.5, "f-m")
    # bottom: error vs number of trees
    x0, w, yt, yb = 78, 470, 256, 426
    sx = lambda r: x0 + math.log10(r) / math.log10(n) * w
    elo, ehi = 0.0, 1.4
    sy = lambda v: yb - (v - elo) / (ehi - elo) * (yb - yt)
    c.text(x0, yt - 14, t("error against number of trees", "误差随树的数量变化"), 13.5, "f-t", weight=600)
    for v in (0, 0.5, 1.0):
        c.line(x0, sy(v), x0 + w, sy(v), "f-lnrule", 1)
        c.text(x0 - 8, sy(v) + 4.5, f"{v:g}", 12, "f-m", "end")
    for r in (1, 3, 10, 30, 100, 300):
        c.line(sx(r), yb, sx(r), yb + 5, "f-lnrule", 1)
        c.text(sx(r), yb + 20, str(r), 12, "f-m", "middle")
    c.text(x0 + w / 2, yb + 40, t("number of trees (log scale)", "树的数量（对数坐标）"), 12, "f-m", "middle")
    noise = 0.35
    c.line(x0, sy(noise), x0 + w, sy(noise), "f-lnrule", 1.5, ' stroke-dasharray="5 4"')
    c.path([(sx(i + 1), sy(v)) for i, v in enumerate(tr)], "f-lnbar2", 2.0)
    c.path([(sx(i + 1), sy(v)) for i, v in enumerate(te)], "f-lnhl", 2.4)
    c.dot(sx(best), sy(te[best - 1]), 5, "f-fillhl")
    c.text(sx(best), sy(te[best - 1]) - 12, t(f"lowest: {best} trees, {f2(te[best - 1])}",
                                              f"最低：{best} 棵树，{f2(te[best - 1])}"), 12, "f-t", "middle", 600)
    # direct labels at the right end
    lx = x0 + w + 10
    c.text(lx, sy(te[-1]) + 4, t("new points", "新数据"), 12.5, "f-h", weight=600)
    c.text(lx, sy(tr[-1]) + 4, t("training points", "训练点"), 12.5, "f-m", weight=600)
    c.text(lx, sy(noise) + 4, t("noise level", "噪声水平"), 12, "f-m")
    return c


# ---- 3. the worked split ----------------------------------------------------------------
def fig_split() -> Canvas:
    w = worked()
    cands = w["cands"]
    chosen = w["stump"]
    c = Canvas("split-gain", 720, 430, t(
        "Twelve card transactions sorted by the number of purchases on the card in the past 24 hours; four are fraud. "
        "Each bar is the gain of splitting between two neighbours. Grey bars are not allowed: one side would hold "
        "less than 1 in total hessian. The tallest allowed bar wins.",
        "12 笔刷卡交易，按这张卡过去 24 小时内的消费次数排序，其中 4 笔是盗刷。每根柱子是在相邻两笔之间切一刀的 gain。"
        "灰色的柱子不允许：切完有一边的 hessian 总和不到 1。允许的柱子里最高的那根胜出。"))
    X = lambda v: 76 + (v - 1) * 52
    yd = 78
    c.text(20, 28, t("purchases on the card in the past 24 hours", "这张卡过去 24 小时的消费次数"), 13.5, "f-t", weight=600)
    # legend
    c.dot(470, 23, 6, "f-alt", ring=False)
    c.text(482, 28, t("fraud", "盗刷"), 12.5, "f-t")
    c.raw(f'<circle cx="560" cy="23" r="6" class="f-hollow" stroke-width="2"/>')
    c.text(572, 28, t("legitimate", "正常"), 12.5, "f-t")
    c.line(X(1) - 20, yd, X(12) + 20, yd, "f-lnrule", 1)
    for v, yv in zip(WORKED_X, WORKED_Y):
        if yv:
            c.dot(X(v), yd, 9, "f-alt")
        else:
            c.raw(f'<circle cx="{X(v):.1f}" cy="{yd}" r="9" class="f-hollow" stroke-width="2"/>')
        c.text(X(v), yd + 28, f"{v:g}", 12, "f-m", "middle")
    c.text(20, 140, t("starting prediction p = 1/3 for every row:  fraud  g = −2/3,  legitimate  g = +1/3,  "
                      "all  h = 2/9",
                      "一开始每行的预测都是 p = 1/3：盗刷 g = −2/3，正常 g = +1/3，每行 h = 2/9"), 12.5, "f-t")
    # bars
    base_y, top_y = 372, 196
    gmax = 2.6
    sy = lambda g: base_y - g / gmax * (base_y - top_y)
    c.line(X(1) - 20, base_y, X(12) + 20, base_y, "f-lnrule", 1)
    c.text(20, 186, "gain", 12.5, "f-m")
    for v in (0, 1, 2):
        c.text(36, sy(v) + 4.5, f"{v}", 12, "f-m", "end")
        if v:
            c.line(46, sy(v), X(12) + 20, sy(v), "f-lnrule", 0.8, ' stroke-dasharray="2 4"')
    best_blocked = max((cc for cc in cands if not cc["allowed"]), key=lambda cc: cc["gain"])
    for cc in cands:
        x = X(cc["thr"])
        cls = "f-bar"
        if cc["allowed"]:
            cls = "f-fillhl" if abs(cc["thr"] - chosen["thr"]) < 1e-9 else "f-fillhl1"
        c.raw(f'<rect x="{x - 15:.1f}" y="{sy(cc["gain"]):.1f}" width="30" '
              f'height="{base_y - sy(cc["gain"]):.1f}" rx="3" class="{cls}"/>')
        c.text(x, base_y + 18, f"{cc['thr']:g}", 11.5, "f-m", "middle")
        c.text(x, sy(cc["gain"]) - 6, f2(cc["gain"]), 11.5, "f-t" if cc["allowed"] else "f-m", "middle")
    c.text(X(6.5), base_y + 38, t("threshold, halfway between two neighbouring values", "阈值：取相邻两个值的中点"), 11.5, "f-m", "middle")
    # annotations
    xa = X(chosen["thr"])
    c.text(xa, sy(chosen["gain"]) - 24, t("chosen", "选中"), 12.5, "f-h", "middle", 600)
    xb = X(best_blocked["thr"])
    c.text(xb + 22, sy(best_blocked["gain"]) - 30,
           t("highest, but the right side has 4 rows:", "最高，但右边只有 4 行："), 11.5, "f-m")
    c.text(xb + 22, sy(best_blocked["gain"]) - 14,
           t(f"hessian {f2(best_blocked['HR'])} < min_child_weight 1", f"hessian {f2(best_blocked['HR'])} < min_child_weight 1"),
           11.5, "f-m")
    return c


# ---- 4. missing values ------------------------------------------------------------------
def fig_missing() -> Canvas:
    m = missing()
    at = {cc["miss_left"]: cc for cc in m["cands"] if abs(cc["thr"] - 5.5) < 1e-9}
    best = m["stump"]
    c = Canvas("missing", 720, 310, t(
        f"The same twelve transactions plus two whose count is missing, both fraud. At the threshold 5.5, sending "
        f"the missing rows left gives a gain of {f2(at[True]['gain'])}; sending them right gives "
        f"{f2(at[False]['gain'])}. The direction is learned per split and stored in the tree.",
        f"同样 12 笔交易，再加两笔消费次数缺失的交易，都是盗刷。在阈值 5.5 处，把缺失的两行放左边，gain 是 "
        f"{f2(at[True]['gain'])}；放右边是 {f2(at[False]['gain'])}。每次切分都会学一个方向，存在树里。"))

    def rowdots(x, y, ys, per_row=7):
        for i, yv in enumerate(ys):
            cx, cy = x + (i % per_row) * 20, y + (i // per_row) * 20
            if yv is None:
                c.raw(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" class="f-altsoft" stroke-width="1.6" '
                      f'stroke-dasharray="2 2"/>')
                c.text(cx, cy + 4, "?", 10.5, "f-t", "middle", 600)
            elif yv:
                c.dot(cx, cy, 7, "f-alt", ring=False)
            else:
                c.raw(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" class="f-hollow" stroke-width="1.8"/>')

    left_rows = WORKED_Y[:5]
    right_rows = WORKED_Y[5:]
    for k, miss_left in enumerate((True, False)):
        x0 = 24 + k * 350
        cc = at[miss_left]
        c.text(x0, 30, t("missing → left" if miss_left else "missing → right",
                         "缺失值 → 左" if miss_left else "缺失值 → 右"), 13.5, "f-t", weight=600)
        c.rect(x0 + 92, 46, 136, 30, "f-node", 6, 1.2)
        c.text(x0 + 160, 66, "x < 5.5 ?", 12.5, "f-t", "middle")
        c.line(x0 + 130, 76, x0 + 74, 106, "f-ln", 1.3)
        c.line(x0 + 190, 76, x0 + 246, 106, "f-ln", 1.3)
        c.text(x0 + 88, 96, t("yes", "是"), 11.5, "f-m", "end")
        c.text(x0 + 232, 96, t("no", "否"), 11.5, "f-m")
        lrows = left_rows + ([None, None] if miss_left else [])
        rrows = right_rows + ([] if miss_left else [None, None])
        c.rect(x0, 108, 150, 64, "f-node", 6, 1)
        c.rect(x0 + 170, 108, 150, 64, "f-node", 6, 1)
        rowdots(x0 + 16, 126, lrows)
        rowdots(x0 + 186, 126, rrows)
        c.text(x0, 196, f"G_L = {sgn(cc['GL'])} · H_L = {cc['HL']:.2f}", 12, "f-m")
        c.text(x0 + 170, 196, f"G_R = {sgn(cc['GR'])} · H_R = {cc['HR']:.2f}", 12, "f-m")
        big = not miss_left
        c.text(x0 + 160, 232, f"gain {f2(cc['gain'])}", 16, "f-h" if big else "f-t", "middle", 600)
    c.text(24, 284, t(f"Best over all thresholds and both directions: x < {best['thr']:g}, missing rows go "
                      f"{'left' if best['miss_left'] else 'right'}, gain {f2(best['gain'])}.",
                      f"所有阈值、两个方向一起比，最好的是 x < {best['thr']:g}，缺失值走"
                      f"{'左' if best['miss_left'] else '右'}边，gain {f2(best['gain'])}。"), 12.5, "f-t")
    return c


# ---- 5. learning rate -------------------------------------------------------------------
def fig_learning_rate() -> Canvas:
    lc = RUNS["lr_curves"]
    curves = [("eta0.3_mds0", "f-lnhl", "0.3"), ("eta0.1_mds0", "f-lnhl2", "0.1"),
              ("eta0.03_mds0", "f-lnhl1", "0.03"), ("eta0.3_mds1", "f-lnalt", t("0.3, max_delta_step 1", "0.3，max_delta_step 1"))]
    const = lc["constant_valid_logloss"]
    c = Canvas("learning-rate", 720, 410, t(
        "Validation log loss on the credit-card fraud data as trees are added, for three learning rates and for the "
        "default rate with each leaf capped at 1. Dots mark the minimum of each curve. The dashed line predicts the "
        "training fraud rate for everyone.",
        "信用卡盗刷数据上，验证集 log loss 随树的数量变化：三种学习率，以及默认学习率加上叶子值上限 1。"
        "圆点是每条曲线的最低点。虚线是对所有交易都预测训练集的盗刷率。"))
    x0, w, yt, yb = 82, 420, 52, 340
    rmax = 2000
    lo, hi = math.log10(0.0018), math.log10(0.07)
    sx = lambda r: x0 + math.log10(r) / math.log10(rmax) * w
    sy = lambda v: yb - (math.log10(min(max(v, 10 ** lo), 10 ** hi)) - lo) / (hi - lo) * (yb - yt)
    c.text(x0 - 60, 28, t("validation log loss (log scale)", "验证集 log loss（对数坐标）"), 12, "f-m")
    for v in (0.002, 0.005, 0.01, 0.02, 0.05):
        c.line(x0, sy(v), x0 + w, sy(v), "f-lnrule", 1)
        c.text(x0 - 8, sy(v) + 4.5, f"{v:g}", 12, "f-m", "end")
    for r in (1, 10, 100, 1000):
        c.line(sx(r), yb, sx(r), yb + 5, "f-lnrule", 1)
        c.text(sx(r), yb + 20, f"{r:,}", 12, "f-m", "middle")
    c.text(x0 + w / 2, yb + 42, t("number of trees (log scale)", "树的数量（对数坐标）"), 12, "f-m", "middle")
    c.line(x0, sy(const), x0 + w, sy(const), "f-lnrule", 1.5, ' stroke-dasharray="5 4"')
    c.text(x0 + w - 4, sy(const) - 7, t(f"predict the training fraud rate: {const:.4f}",
                                         f"一律预测训练集盗刷率：{const:.4f}"), 11.5, "f-m", "end")
    for key, cls, _ in curves:
        d = lc[key]
        pts = [(sx(r), sy(v)) for r, v in zip(d["rounds"], d["valid_logloss"])]
        c.path(pts, cls, 2.2 if key != "eta0.3_mds1" else 2.0)
        if key == "eta0.3_mds1":
            c.raw(c.parts.pop().replace("/>", ' stroke-dasharray="6 4"/>'))
    for key, cls, _ in curves:
        d = lc[key]
        c.dot(sx(d["best_round"]), sy(d["best_valid_logloss"]), 4.5, "f-dotink")
    first = lc["eta0.3_mds0"]["valid_logloss"][0]
    c.text(sx(1) + 8, sy(0.07) + 14, t(f"0.3 starts at {first:.3f}", f"0.3 起点 {first:.3f}"), 11.5, "f-m")
    # legend
    lx, ly = 530, 70
    c.text(lx, ly - 22, t("learning rate (eta)", "学习率（eta）"), 12.5, "f-t", weight=600)
    for i, (key, cls, lab) in enumerate(curves):
        d = lc[key]
        y = ly + i * 46
        c.line(lx, y - 4, lx + 22, y - 4, cls, 2.4, ' stroke-dasharray="6 4"' if key == "eta0.3_mds1" else "")
        c.text(lx + 30, y, lab, 12.5, "f-t", weight=600)
        c.text(lx + 30, y + 17, t(f"best {d['best_valid_logloss']:.4f} at {d['best_round']} trees",
                                  f"最低 {d['best_valid_logloss']:.4f}，{d['best_round']} 棵树"), 11.5, "f-m")
    return c


# ---- 6. scale_pos_weight ----------------------------------------------------------------
def fig_imbalance() -> Canvas:
    im = RUNS["imbalance"]
    de, sp, same = im["default"], im["spw"], im["default_same_flags"]
    npos = len(de["pr"])
    c = Canvas("scale-pos-weight", 720, 400, t(
        f"Precision-recall curves on the last 20% of transactions ({npos} frauds), for XGBoost with default weights "
        f"and with scale_pos_weight = {im['scale_pos_weight']:.0f}. The curves almost coincide; what moves is where "
        f"the threshold 0.5 lands on them.",
        f"最后 20% 的交易（{npos} 笔盗刷）上的 PR 曲线：默认权重的 XGBoost，和 scale_pos_weight = "
        f"{im['scale_pos_weight']:.0f} 的 XGBoost。两条曲线几乎重合；变的是阈值 0.5 落在曲线上的位置。"))
    x0, w, yt, yb = 70, 380, 48, 330
    sx = lambda r: x0 + r * w
    plo = 0.0
    sy = lambda p: yb - (p - plo) / (1 - plo) * (yb - yt)
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        c.line(x0, sy(v), x0 + w, sy(v), "f-lnrule", 1)
        c.text(x0 - 8, sy(v) + 4.5, f"{v:g}", 12, "f-m", "end")
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        c.line(sx(v), yb, sx(v), yb + 5, "f-lnrule", 1)
        c.text(sx(v), yb + 20, f"{v:g}", 12, "f-m", "middle")
    c.text(x0 - 46, yt - 18, "precision", 12, "f-m")
    c.text(x0 + w / 2, yb + 42, "recall", 12, "f-m", "middle")
    for d, cls in ((de, "f-lnhl"), (sp, "f-lnalt")):
        pts = [(sx(r), sy(max(p, plo))) for r, p in d["pr"]]
        c.path(pts, cls, 2.2)
    pd_ = (de["at_05"]["tp"] / npos, de["at_05"]["tp"] / de["at_05"]["flagged"])
    ps_ = (sp["at_05"]["tp"] / npos, sp["at_05"]["tp"] / sp["at_05"]["flagged"])
    pq_ = (same["tp"] / npos, same["tp"] / same["flagged"])
    c.dot(sx(pd_[0]), sy(pd_[1]), 6, "f-fillhl")
    c.dot(sx(ps_[0]), sy(ps_[1]), 6, "f-alt")
    c.raw(f'<circle cx="{sx(pq_[0]):.1f}" cy="{sy(pq_[1]):.1f}" r="6" class="f-hollow" stroke-width="2.2"/>')
    c.line(sx(pd_[0]) - 7, sy(pd_[1]) - 4, sx(pd_[0]) - 70, sy(pd_[1]) - 22, "f-lnrule", 1)
    c.text(sx(pd_[0]) - 74, sy(pd_[1]) - 20, t("default, 0.5", "默认，0.5"), 12, "f-t", "end", 600)
    c.line(sx(ps_[0]) - 7, sy(ps_[1]) + 3, sx(ps_[0]) - 70, sy(ps_[1]) + 30, "f-lnrule", 1)
    c.text(sx(ps_[0]) - 74, sy(ps_[1]) + 34, t("scale_pos_weight, 0.5", "scale_pos_weight，0.5"), 12, "f-t", "end", 600)
    c.line(sx(pq_[0]) + 7, sy(pq_[1]) + 5, sx(pq_[0]) + 40, sy(pq_[1]) + 46, "f-lnrule", 1)
    c.text(sx(pq_[0]) + 36, sy(pq_[1]) + 62, t(f"default, {same['threshold']:.2f}", f"默认，{same['threshold']:.2f}"), 12, "f-t", "middle", 600)
    # right panel: the numbers
    rx = 490
    rows = (
        ("f-lnhl", t("default weights", "默认权重"), de,
         t(f"0.5 flags {de['at_05']['flagged']}: {de['at_05']['tp']} fraud", f"0.5 标出 {de['at_05']['flagged']} 笔：{de['at_05']['tp']} 笔盗刷")),
        ("f-lnalt", f"scale_pos_weight {im['scale_pos_weight']:.0f}", sp,
         t(f"0.5 flags {sp['at_05']['flagged']}: {sp['at_05']['tp']} fraud", f"0.5 标出 {sp['at_05']['flagged']} 笔：{sp['at_05']['tp']} 笔盗刷")),
    )
    y = 66
    for cls, name, d, line2 in rows:
        c.line(rx, y - 4, rx + 22, y - 4, cls, 2.6)
        c.text(rx + 30, y, name, 12.5, "f-t", weight=600)
        c.text(rx + 30, y + 18, f"AP {d['ap']:.2f} · ROC-AUC {d['auc']:.3f}", 12, "f-m")
        c.text(rx + 30, y + 36, line2, 12, "f-m")
        y += 74
    c.raw(f'<circle cx="{rx + 11}" cy="{y - 4}" r="6" class="f-hollow" stroke-width="2.2"/>')
    c.text(rx + 30, y, t("default weights,", "默认权重，"), 12.5, "f-t", weight=600)
    c.text(rx + 30, y + 18, t(f"threshold {same['threshold']:.2f}", f"阈值 {same['threshold']:.2f}"), 12, "f-m")
    c.text(rx + 30, y + 36, t(f"flags {same['flagged']}: {same['tp']} fraud", f"标出 {same['flagged']} 笔：{same['tp']} 笔盗刷"), 12, "f-m")
    lo, hi = im["ap_diff_default_minus_spw_95ci"]
    c.text(rx, y + 74, t("AP difference, 95% bootstrap interval:", "AP 之差的 95% bootstrap 区间："), 12, "f-m")
    c.text(rx, y + 92, f"{sgn(lo, 3)} … {sgn(hi, 3)}", 12, "f-t", weight=600)
    return c


# ---- 7. extrapolation -------------------------------------------------------------------
def fig_extrapolation() -> Canvas:
    trees, (a0, b1) = trend_models()
    c = Canvas("extrapolation", 720, 340, t(
        "A straight-line trend seen only for x between 0 and 10. Boosted trees fit the training points closely but "
        "predict a constant beyond the last one; a straight line, fitted to the same points, keeps the trend.",
        "一条直线趋势，训练时只见过 x 在 0 到 10 之间的点。boosting 的树把训练点拟合得很紧，但过了最后一个点就只会给常数；"
        "同样这些点拟合出的直线能把趋势延续下去。"))
    x0, w, yt, yb = 60, 470, 40, 286
    sx = lambda v: x0 + v / 15 * w
    ylo, yhi = 0, 27
    sy = lambda v: yb - (v - ylo) / (yhi - ylo) * (yb - yt)
    c.raw(f'<rect x="{sx(10):.1f}" y="{yt:.1f}" width="{sx(15) - sx(10):.1f}" height="{yb - yt:.1f}" '
          f'class="f-fillsoft"/>')
    c.text(sx(12.5), yt + 18, t("no training data here", "这里没有训练数据"), 12, "f-h", "middle", 600)
    for v in (0, 10, 20):
        c.line(x0, sy(v), x0 + w, sy(v), "f-lnrule", 1)
        c.text(x0 - 8, sy(v) + 4.5, f"{v}", 12, "f-m", "end")
    for v in (0, 5, 10, 15):
        c.line(sx(v), yb, sx(v), yb + 5, "f-lnrule", 1)
        c.text(sx(v), yb + 20, f"{v}", 12, "f-m", "middle")
    c.text(x0 + w / 2, yb + 40, "x", 12, "f-m", "middle")
    for x, y in zip(*TREND):
        c.dot(sx(x), sy(y), 3.2, "f-dotm", ring=False)
    for x, y in zip(*TREND_FUTURE):
        c.raw(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.6" class="f-hollow" stroke-width="1.6"/>')
    c.path([(sx(v / 20), sy(a0 + b1 * v / 20)) for v in (0, 300)], "f-lnalt", 2.0)
    c.raw(c.parts.pop().replace("/>", ' stroke-dasharray="6 4"/>'))
    pts = [(sx(i / 40), sy(tb.predict(trees, i / 40))) for i in range(601)]
    c.path(pts, "f-lnhl", 2.4)
    # labels at the right
    lx = x0 + w + 12
    yt15, yl15 = tb.predict(trees, 15), a0 + b1 * 15
    c.text(lx, sy(yl15) + 4, t(f"straight line: {yl15:.1f}", f"直线：{yl15:.1f}"), 12.5, "f-alt", weight=600)
    c.text(lx, sy(yt15) + 4, t(f"boosted trees: {yt15:.1f}", f"boosting 树：{yt15:.1f}"), 12.5, "f-h", weight=600)
    c.text(lx, sy(yl15) + 24, t(f"truth at 15: {1.5 * 15 + 2:.1f}", f"x=15 的真值：{1.5 * 15 + 2:.1f}"), 12, "f-m")
    return c


# ---- numbers ----------------------------------------------------------------------------
def numbers() -> str:
    L = []
    p = L.append
    m1 = single_tree()
    r = m1[1][0]
    p("== one tree (depth 2, plain): base %.3f" % m1[0])
    p(f"   root x<{r['thr']:.3f}; left x<{r['left']['thr']:.3f}; right x<{r['right']['thr']:.3f}")
    for k in ("left", "right"):
        for kk in ("left", "right"):
            lf = r[k][kk]
            p(f"   leaf {k}/{kk}: value {m1[0] + lf['leaf']:.3f}  rows {lf['n']}")
    model = wave_model()
    tr = error_curve(model, WAVE, 300)
    te = error_curve(model, WAVE_TEST, 300)
    best = min(range(300), key=lambda i: te[i]) + 1
    truth_rmse = math.sqrt(sum((wave_truth(x) - y) ** 2 for x, y in zip(*WAVE_TEST)) / len(WAVE_TEST[0]))
    p(f"== rounds: start RMSE train {rmse((model[0], []), WAVE):.3f} new {rmse((model[0], []), WAVE_TEST):.3f}")
    for k in SHOWN_ROUNDS:
        p(f"   {k:>3} trees: train {tr[k - 1]:.3f}  new {te[k - 1]:.3f}")
    p(f"   best on new points: {best} trees, {te[best - 1]:.3f}; the true curve itself scores {truth_rmse:.3f}")
    w = worked()
    p(f"== worked split: base p {w['base']:.4f}, f0 {w['f0']:.4f}; g fraud {min(w['g']):.4f} legit "
      f"{max(w['g']):.4f}; h {w['h'][0]:.4f}; G {sum(w['g']):.4f} H {sum(w['h']):.4f}")
    for cc in w["cands"]:
        p(f"   thr {cc['thr']:>4}: left {cc['n_left']:>2} rows  GL {cc['GL']:+.4f} HL {cc['HL']:.4f}  "
          f"GR {cc['GR']:+.4f} HR {cc['HR']:.4f}  gain {cc['gain']:.4f}  {'ok' if cc['allowed'] else 'BLOCKED'}")
    s = w["stump"]
    for side in ("left", "right"):
        lf = s[side]
        raw = -lf["G"] / (lf["H"] + 1)
        p(f"   {side}: rows {lf['n']} G {lf['G']:+.4f} H {lf['H']:.4f}  w* {raw:+.4f}  x eta {lf['leaf']:+.4f}  "
          f"p {w['base']:.4f} -> {sigmoid(w['f0'] + lf['leaf']):.4f}  (full step p {sigmoid(w['f0'] + raw):.4f})")
    p(f"   paper's form: half the gain {s['gain'] / 2:.4f}")
    mm = missing()
    p(f"== missing: base {mm['base']:.4f}")
    for cc in mm["cands"]:
        if cc["allowed"]:
            p(f"   thr {cc['thr']:>4} missing {'L' if cc['miss_left'] else 'R'}: gain {cc['gain']:.4f}")
    bs = mm["stump"]
    p(f"   chosen: x<{bs['thr']} missing {'left' if bs['miss_left'] else 'right'} gain {bs['gain']:.4f}")
    trees, (a0, b1) = trend_models()
    fut = TREND_FUTURE
    rm = lambda f, d: math.sqrt(sum((f(x) - y) ** 2 for x, y in zip(*d)) / len(d[0]))
    p(f"== extrapolation: line y = {a0:.3f} + {b1:.3f} x; last train x {max(TREND[0])}")
    for x in (10, 12, 15):
        p(f"   x={x}: trees {tb.predict(trees, x):.2f}  line {a0 + b1 * x:.2f}  truth {1.5 * x + 2:.2f}")
    p(f"   RMSE train: trees {rm(lambda x: tb.predict(trees, x), TREND):.2f} line {rm(lambda x: a0 + b1 * x, TREND):.2f}; "
      f"future: trees {rm(lambda x: tb.predict(trees, x), fut):.2f} line {rm(lambda x: a0 + b1 * x, fut):.2f}")
    if RUNS:
        ck = RUNS["tinyboost_check"]
        p(f"== tinyboost vs xgboost: {json.dumps(ck['worked'])}")
        p(f"   gamma checks: {ck['gamma_just_below_gain_keeps_split']} {ck['gamma_just_above_gain_prunes']} "
          f"{ck['gamma_above_half_gain_keeps_split']}; missing {ck['missing']}")
        lc = RUNS["lr_curves"]
        p(f"== learning rate: constant valid logloss {lc['constant_valid_logloss']:.5f} train {lc['constant_train_logloss']:.5f}")
        for k, d in lc.items():
            if isinstance(d, dict):
                p(f"   {k}: best {d['best_valid_logloss']:.5f} at {d['best_round']}, final {d['final_valid_logloss']:.5f}, "
                  f"first train logloss {d['first_train_logloss']:.4f}, first tree leaves {d['first_tree_min_leaf']:.2f}.."
                  f"{d['first_tree_max_leaf']:.2f}, test AP {d['test_ap_at_best']:.3f} AUC {d['test_auc_at_best']:.4f}")
        im = RUNS["imbalance"]
        p(f"== imbalance: spw {im['scale_pos_weight']:.2f}; split {im['split']}; time {im['time_ranges']}")
        for k in ("default", "spw", "mds1"):
            d = im[k]
            p(f"   {k}: trees {d['trees']} AP {d['ap']:.3f} AUC {d['auc']:.4f} logloss {d['logloss']:.5f} mean p {d['mean_p']:.5f} "
              f"(rate {d['test_rate']:.5f}) median p fraud {d['median_p_fraud']:.3f} legit {d['median_p_legit']:.2e} at 0.5 {d['at_05']}")
            p("      bins " + "; ".join(f"[{b['lo']},{b['hi']}) n={b['n']} fr={b['frauds']}" for b in d["bins"]))
        p(f"   default at spw's flag count: {im['default_same_flags']}; AP diff CI {im['ap_diff_default_minus_spw_95ci']}")
        imp = RUNS["importance"]
        p("== importance:")
        for k in ("weight", "gain", "cover", "total_gain", "shap"):
            p(f"   {k}: top {[(a, round(b, 2)) for a, b in imp[k]['top']]} ranks {imp[k]['rank']}")
        p(f"   shap sum err {imp['shap_sum_max_err']:.2e}; Time shap {imp['time_shap']}; Time splits {imp['time_splits']}")
        p(f"   with Time {imp['with_time']}; without {imp['without_time']}")
        h = RUNS["heart"]
        p(f"== heart: train {h['train']} test {h['test']} class1 in test {h['test_class1']}")
        for k, d in h["models"].items():
            p(f"   {k:24s} acc {d['test_accuracy']:.3f} ({round(d['test_accuracy'] * h['test'])}/{h['test']}) AUC {d['test_auc']:.3f} "
              f"| CV {d['cv_auc_mean']:.3f} ± {d['cv_auc_sd']:.3f} [{d['cv_auc_min']:.3f}, {d['cv_auc_max']:.3f}]")
    return "\n".join(L)


FIGS = (fig_one_tree, fig_rounds, fig_split, fig_missing, fig_learning_rate, fig_imbalance,
        fig_extrapolation)


def main() -> int:
    global LANG, RUNS
    if RUNS_FILE.exists():
        RUNS = json.loads(RUNS_FILE.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    warnings = []
    for lang in ("en", "zh"):
        LANG = lang
        for fn in FIGS:
            if fn in (fig_learning_rate, fig_imbalance) and not RUNS:
                print(f"skip {fn.__name__}: run xgboost_runs.py first")
                continue
            c = fn()
            name = c.name + (".zh" if lang == "zh" else "") + ".svg"
            (OUT / name).write_text(c.svg())
            warnings += c.warnings
    LANG = "en"
    print(numbers())
    for w in warnings:
        print("WARNING", w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

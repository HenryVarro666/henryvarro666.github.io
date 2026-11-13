#!/usr/bin/env python3
"""Figures and numbers for the note content/notes/roc-auc-vs-pr-auc/.

    python3 tools/notes/roc_auc_vs_pr_auc_figs.py

writes five SVGs next to the note, in English (name.svg) and Chinese (name.zh.svg),
and prints every number the text quotes. Standard library only; the drawing helpers
come from precision_vs_recall_figs.py, the script for the previous note.

Models are binormal: negative scores ~ N(0, 1), positive scores ~ N(mu, sigma^2).
Model A has equal variances; model B has a wider spread of positive scores, chosen
so both have the same ROC-AUC. The screening model is the one fitted to the BCSC
operating point in the previous note.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "notes"))
sys.path.insert(0, str(ROOT / "tools" / "figs"))
from precision_vs_recall_figs import Canvas, pct, MU as SCREEN_MU, PREV as SCREEN_PREV  # noqa: E402
from figlib import text_width  # noqa: E402

OUT = ROOT / "content" / "notes" / "roc-auc-vs-pr-auc"
ND = NormalDist()
LANG = "en"


def t(en: str, zh: str) -> str:
    return zh if LANG == "zh" else en


# ---- a toy ranking, small enough to count by hand ----------------------------------
# Highest score first. P = positive, N = negative.
TOY = "PNPPNNPNNNPNNNN"
NPOS, NNEG = TOY.count("P"), TOY.count("N")


def toy_auc() -> tuple[int, int]:
    """(correctly ordered pairs, all pairs)."""
    good, pos_seen = 0, 0
    for ch in TOY:
        if ch == "P":
            pos_seen += 1
        else:
            good += pos_seen
    return good, NPOS * NNEG


def toy_ap() -> list[tuple[int, float]]:
    """(rank, precision at that rank) for every positive."""
    out, hits = [], 0
    for i, ch in enumerate(TOY, 1):
        if ch == "P":
            hits += 1
            out.append((i, hits / i))
    return out


# ---- binormal models -----------------------------------------------------------------
AUC_AB = 0.90
SIG_B = 2.0
MU_A = ND.inv_cdf(AUC_AB) * math.sqrt(2)
MU_B = ND.inv_cdf(AUC_AB) * math.sqrt(1 + SIG_B ** 2)
MODELS = {"A": (MU_A, 1.0), "B": (MU_B, SIG_B), "S": (SCREEN_MU, 1.0)}


def auc(mu: float, sig: float) -> float:
    return ND.cdf(mu / math.sqrt(1 + sig * sig))


def tpr(th: float, mu: float, sig: float) -> float:
    return 1 - ND.cdf((th - mu) / sig)


def fpr(th: float) -> float:
    return 1 - ND.cdf(th)


def th_at_fpr(f: float) -> float:
    return ND.inv_cdf(1 - f)


def th_at_tpr(r: float, mu: float, sig: float) -> float:
    return mu + sig * ND.inv_cdf(1 - r)


def ap(mu: float, sig: float, prev: float, n: int = 6000) -> float:
    """Average precision: precision integrated over recall, i.e. averaged over the positives."""
    lo, hi = mu - 9 * sig, mu + 9 * sig
    h = (hi - lo) / n
    total = 0.0
    for i in range(n):
        th = lo + (i + 0.5) * h
        r, f = tpr(th, mu, sig), fpr(th)
        dens = math.exp(-((th - mu) / sig) ** 2 / 2) / (sig * math.sqrt(2 * math.pi))
        p = prev * r / (prev * r + (1 - prev) * f) if r > 0 else 1.0
        total += p * dens * h
    return total


def pauc(mu: float, sig: float, fmax: float, n: int = 4000) -> tuple[float, float]:
    """Partial area over FPR in [0, fmax], raw and McClish-standardized."""
    a = 0.0
    for i in range(n):
        f = (i + 0.5) * fmax / n
        a += tpr(th_at_fpr(f), mu, sig) * fmax / n
    amin, amax = fmax * fmax / 2, fmax
    return a, 0.5 * (1 + (a - amin) / (amax - amin))


def hanley_mcneil_se(a: float, npos: int, nneg: int) -> float:
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return math.sqrt((a * (1 - a) + (npos - 1) * (q1 - a * a) + (nneg - 1) * (q2 - a * a)) / (npos * nneg))


def fmt(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}"


# ---- 1. AUC as the share of correctly ordered pairs ----------------------------------
def fig_pairs() -> Canvas:
    good, total = toy_auc()
    c = Canvas("pairs", 720, 380, t(
        f"A ranked list of {len(TOY)} cases, {NPOS} positive and {NNEG} negative. On the right, a grid of all "
        f"{total} positive-negative pairs; a cell is filled when the positive is ranked above the negative. "
        f"{good} of {total} cells are blue, so the ROC-AUC is {good / total:.2f}. The filled region is the area "
        "under the ROC curve, drawn as a staircase over the grid.",
        f"一个排好序的列表，{len(TOY)} 个样本，{NPOS} 个阳性、{NNEG} 个阴性。右边是全部 {total} 个“阳性—阴性”配对，"
        f"阳性排在阴性前面就是蓝色。{total} 格里有 {good} 格是蓝色，所以 ROC-AUC 是 {good / total:.2f}。蓝色区域正好是 ROC "
        "曲线下的面积，阶梯就是 ROC 曲线。"))
    # ranked list
    x0, y0, rh, rw = 28, 52, 19, 82
    c.text(x0, 28, t("ranked by score", "按分数排序"), 13, "f-t", weight=600)
    for i, ch in enumerate(TOY):
        y = y0 + i * rh
        cls = "f-fillhl" if ch == "P" else "f-node"
        c.rect(x0 + 24, y, rw - 24, rh - 3, cls, rx=3, sw=0 if ch == "P" else 1)
        c.text(x0 + 16, y + rh - 7, str(i + 1), 11.5, "f-m", "end")
        lab = t("positive", "阳性") if ch == "P" else t("negative", "阴性")
        c.text(x0 + 24 + (rw - 24) / 2, y + rh - 7.5, lab, 11.5, "f-onhl" if ch == "P" else "f-m", "middle")
    c.text(x0 + 24, y0 + len(TOY) * rh + 14, t("↑ high score", "↑ 分数高"), 11.5, "f-m")

    # pair grid: columns = negatives in rank order, rows = positives in rank order (top-ranked at the bottom)
    gx, gy, cw, ch_ = 250, 74, 34, 34
    negs = [i for i, x in enumerate(TOY) if x == "N"]
    poss = [i for i, x in enumerate(TOY) if x == "P"]
    gw, gh = NNEG * cw, NPOS * ch_
    c.text(gx, 28, t(f"all {total} positive–negative pairs", f"全部 {total} 个阳性—阴性配对"), 13, "f-t", weight=600)
    c.text(gx, 46, t("blue: the positive is ranked above the negative; orange: the reverse",
                     "蓝色：阳性排在阴性前面；橙色：反过来"), 12, "f-m")
    for j, ni in enumerate(negs):
        for k, pi in enumerate(poss):
            x = gx + j * cw
            y = gy + gh - (k + 1) * ch_
            ok = pi < ni
            c.rect(x + 1, y + 1, cw - 2, ch_ - 2, "f-fillhl1" if ok else "f-altsoft", rx=3, sw=0 if ok else 1)
    # ROC staircase over the grid
    pts, px, py = [(gx, gy + gh)], gx, gy + gh
    for chh in TOY:
        if chh == "P":
            py -= ch_
        else:
            px += cw
        pts.append((px, py))
    c.path(pts, "f-lnhl", 2.5)
    c.text(gx + gw / 2, gy + gh + 22, t("negatives, in rank order → (false positive rate)", "阴性，按排名 →（假阳性率）"), 12, "f-m", "middle")
    c.text(gx - 10, gy + gh / 2 + 4, t("positives ↑", "阳性 ↑"), 12, "f-m", "end")
    c.text(gx + gw, gy + gh + 56, t(f"{good} of {total} pairs in the right order  →  ROC-AUC = {good / total:.2f}",
                                    f"{total} 对里有 {good} 对顺序正确  →  ROC-AUC = {good / total:.2f}"), 14, "f-t", "end", 600)
    c.text(gx + gw, gy + gh + 78, t("the staircase is the ROC curve; the blue cells are the area under it",
                                    "阶梯就是 ROC 曲线，蓝色格子就是它下面的面积"), 12, "f-m", "end")
    return c


# ---- 2. AP as the mean precision at each positive -------------------------------------
def fig_ap() -> Canvas:
    pts = toy_ap()
    apv = sum(p for _, p in pts) / len(pts)
    c = Canvas("average-precision", 720, 372, t(
        f"The same ranked list. At each positive, precision so far is written next to it: "
        + ", ".join(f"{p:.2f}" for _, p in pts) + f". Average precision is their mean, {apv:.2f}. "
        f"On the right, the precision-recall curve as a staircase; the shaded area under it equals the average "
        f"precision. A random ranking would sit at the prevalence, {NPOS}/{len(TOY)}.",
        f"同一个排序列表。每遇到一个阳性，旁边写上到此为止的 precision："
        + "、".join(f"{p:.2f}" for _, p in pts) + f"。average precision 就是它们的平均，{apv:.2f}。"
        f"右边是阶梯形的 PR 曲线，阴影面积等于 average precision。随机排序会落在阳性占比 {NPOS}/{len(TOY)}。"))
    x0, y0, rh, rw = 28, 52, 19, 82
    c.text(x0, 28, t("ranked by score", "按分数排序"), 13, "f-t", weight=600)
    hits = 0
    for i, ch in enumerate(TOY):
        y = y0 + i * rh
        cls = "f-fillhl" if ch == "P" else "f-node"
        c.rect(x0 + 24, y, rw - 24, rh - 3, cls, rx=3, sw=0 if ch == "P" else 1)
        c.text(x0 + 16, y + rh - 7, str(i + 1), 11.5, "f-m", "end")
        lab = t("positive", "阳性") if ch == "P" else t("negative", "阴性")
        c.text(x0 + 24 + (rw - 24) / 2, y + rh - 7.5, lab, 11.5, "f-onhl" if ch == "P" else "f-m", "middle")
        if ch == "P":
            hits += 1
            c.text(x0 + rw + 12, y + rh - 7, f"{hits}/{i + 1} = {hits / (i + 1):.2f}", 12.5, "f-t")
    c.text(x0 + rw + 12, 46, t("precision so far", "到此为止的 precision"), 12, "f-m")
    c.text(x0 + 24, y0 + len(TOY) * rh + 14, t("↑ high score", "↑ 分数高"), 11.5, "f-m")

    # PR staircase
    px0, pyb, size = 360, 300, 230
    sx = lambda r: px0 + r * size
    sy = lambda p: pyb - p * size
    for v in (0, 0.5, 1.0):
        c.line(px0, sy(v), px0 + size, sy(v), "f-lnrule", 1)
        c.text(px0 - 8, sy(v) + 4.5, f"{v:g}", 12, "f-m", "end")
        c.line(sx(v), pyb, sx(v), pyb + 5, "f-lnrule", 1)
        c.text(sx(v), pyb + 20, f"{v:g}", 12, "f-m", "middle")
    c.text(px0, 46, "precision", 12, "f-m")
    c.text(px0 + size / 2, pyb + 40, "recall", 12, "f-m", "middle")
    poly, prev_r = [(sx(0), pyb)], 0.0
    for k, (_, p) in enumerate(pts, 1):
        r = k / NPOS
        poly += [(sx(prev_r), sy(p)), (sx(r), sy(p))]
        prev_r = r
    poly.append((sx(1), pyb))
    c.path(poly, "f-fillhl1", close=True)
    c.path(poly[1:-1], "f-lnhl", 2)
    for k, (_, p) in enumerate(pts, 1):
        c.dot(sx(k / NPOS), sy(p), 4.5, "f-dotink")
    base = NPOS / len(TOY)
    c.line(px0, sy(base), px0 + size, sy(base), "f-lnalt", 1.5, ' stroke-dasharray="5 4"')
    c.text(px0 + size + 8, sy(base) + 4.5, t(f"random: {base:.2f}", f"随机：{base:.2f}"), 12, "f-t")
    c.text(px0 + size + 8, sy(apv) - 2, t(f"AP = {apv:.2f}", f"AP = {apv:.2f}"), 14, "f-t", weight=600)
    c.text(px0 + size + 8, sy(apv) + 16, t("mean of the five", "五个数的平均"), 12, "f-m")
    return c


# ---- 3. same ROC-AUC, two shapes ------------------------------------------------------
def fig_roc_ab() -> Canvas:
    (ma, sa), (mb, sb) = MODELS["A"], MODELS["B"]
    ta, tb = tpr(th_at_fpr(0.001), ma, sa), tpr(th_at_fpr(0.001), mb, sb)
    c = Canvas("same-auc", 720, 380, t(
        f"Two models with the same ROC-AUC, {AUC_AB:.2f}. Left, on ordinary axes the curves look alike. Right, with "
        f"the false positive rate on a log scale: at a false positive rate of 0.1% model B finds {pct(tb)} of "
        f"positives and model A {pct(ta)}; at high false positive rates A is ahead.",
        f"两个 ROC-AUC 相同（{AUC_AB:.2f}）的模型。左图用普通坐标，两条线差不多。右图把假阳性率换成对数坐标："
        f"假阳性率 0.1% 时，模型 B 能找到 {pct(tb)} 的阳性，模型 A 只有 {pct(ta)}；假阳性率高的地方 A 反超。"))
    yt, yb, size = 66, 296, 230

    def frame(x0, title, logx):
        c.text(x0, 26, title, 13.5, "f-t", weight=600)
        c.text(x0, 48, t("recall (true positive rate)", "recall（真阳性率）"), 12, "f-m")
        for v in (0, 0.5, 1.0):
            y = yb - v * size
            c.line(x0, y, x0 + size, y, "f-lnrule", 1)
            c.text(x0 - 8, y + 4.5, f"{v:g}", 12, "f-m", "end")
        ticks = [(1e-4, "0.01%"), (1e-3, "0.1%"), (1e-2, "1%"), (1e-1, "10%"), (1, "100%")] if logx else [(0, "0"), (0.5, "0.5"), (1, "1")]
        for v, lab in ticks:
            x = x0 + (math.log10(v) + 4) / 4 * size if logx else x0 + v * size
            c.line(x, yb, x, yb + 5, "f-lnrule", 1)
            c.text(x, yb + 20, lab, 12, "f-m", "middle")
        c.text(x0 + size / 2, yb + 42, t("false positive rate" + (" (log scale)" if logx else ""),
                                         "假阳性率" + ("（对数坐标）" if logx else "")), 12, "f-m", "middle")

    def curve(x0, mu, sig, logx, cls):
        pts = []
        for i in range(301):
            if logx:
                f = 10 ** (-4 + 4 * i / 300)
            else:
                f = i / 300
            f = min(max(f, 1e-9), 1 - 1e-12)
            r = tpr(th_at_fpr(f), mu, sig)
            x = x0 + ((math.log10(f) + 4) / 4 * size if logx else f * size)
            pts.append((x, yb - r * size))
        c.path(pts, cls, 2.2)

    for x0, logx, title in ((56, False, t("ordinary axes", "普通坐标")), (430, True, t("false positive rate on a log scale", "假阳性率取对数"))):
        frame(x0, title, logx)
        if not logx:
            c.line(x0, yb, x0 + size, yb - size, "f-lnrule", 1)
        curve(x0, ma, sa, logx, "f-lnhl")
        curve(x0, mb, sb, logx, "f-lnalt")
    # direct labels on the log panel
    x0 = 430
    fx = x0 + (math.log10(0.001) + 4) / 4 * size
    c.line(fx, yt - 6, fx, yb, "f-lnrule", 1, ' stroke-dasharray="3 3"')
    c.dot(fx, yb - tb * size, 4.5, "f-dotink")
    c.dot(fx, yb - ta * size, 4.5, "f-dotink")
    c.text(fx - 8, yb - tb * size - 8, f"B {pct(tb)}", 12.5, "f-t", "end", 600)
    c.text(fx + 8, yb - ta * size + 16, f"A {pct(ta)}", 12.5, "f-t", weight=600)
    # legend on the linear panel
    lx, ly = 56 + 120, yb - 64
    for i, (cls, lab) in enumerate((("f-lnhl", t(f"model A, AUC {auc(ma, sa):.2f}", f"模型 A，AUC {auc(ma, sa):.2f}")),
                                    ("f-lnalt", t(f"model B, AUC {auc(mb, sb):.2f}", f"模型 B，AUC {auc(mb, sb):.2f}")))):
        y = ly + i * 18
        c.line(lx, y - 4, lx + 18, y - 4, cls, 2.2)
        c.text(lx + 24, y, lab, 12.5, "f-t")
    return c


# ---- 4. the same two models in PR space -----------------------------------------------
PR_PREV = 0.005


def fig_pr_ab() -> Canvas:
    (ma, sa), (mb, sb) = MODELS["A"], MODELS["B"]
    apa, apb = ap(ma, sa, PR_PREV), ap(mb, sb, PR_PREV)
    c = Canvas("same-auc-pr", 720, 360, t(
        f"Precision-recall curves of the same two models when {pct(PR_PREV, 1)} of cases are positive. Model B keeps "
        f"high precision at low recall; model A starts low. Average precision: A {apa:.2f}, B {apb:.2f}. "
        f"Random ranking: {PR_PREV:.3f}.",
        f"同样两个模型在阳性占 {pct(PR_PREV, 1)} 时的 PR 曲线。模型 B 在 recall 低的地方 precision 很高；模型 A 一开始就低。"
        f"average precision：A {apa:.2f}，B {apb:.2f}。随机排序：{PR_PREV:.3f}。"))
    x0, yt, yb, w = 70, 50, 290, 420
    sx = lambda r: x0 + r * w
    sy = lambda p: yb - p * (yb - yt)
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        c.line(x0, sy(v), x0 + w, sy(v), "f-lnrule", 1)
        c.text(x0 - 8, sy(v) + 4.5, f"{v:g}", 12, "f-m", "end")
        c.line(sx(v), yb, sx(v), yb + 5, "f-lnrule", 1)
        c.text(sx(v), yb + 20, f"{v:g}", 12, "f-m", "middle")
    c.text(x0 - 40, yt - 16, "precision", 12, "f-m")
    c.text(x0 + w / 2, yb + 42, "recall", 12, "f-m", "middle")
    for mu, sig, cls in ((ma, sa, "f-lnhl"), (mb, sb, "f-lnalt")):
        pts = []
        for i in range(1, 400):
            r = i / 400
            th = th_at_tpr(r, mu, sig)
            f = fpr(th)
            p = PR_PREV * r / (PR_PREV * r + (1 - PR_PREV) * f)
            pts.append((sx(r), sy(p)))
        c.path(pts, cls, 2.2)
    c.line(x0, sy(PR_PREV), x0 + w, sy(PR_PREV), "f-lnrule", 1.5, ' stroke-dasharray="5 4"')
    # labels at the right
    lx = x0 + w + 24
    c.text(lx, 70, t(f"{pct(PR_PREV, 1)} positive", f"阳性占 {pct(PR_PREV, 1)}"), 13, "f-t", weight=600)
    for i, (cls, lab, v) in enumerate((("f-lnalt", t("model B", "模型 B"), apb), ("f-lnhl", t("model A", "模型 A"), apa))):
        y = 104 + i * 40
        c.line(lx, y - 4, lx + 18, y - 4, cls, 2.2)
        c.text(lx + 24, y, lab, 13, "f-t", weight=600)
        c.text(lx + 24, y + 18, t(f"average precision {v:.2f}", f"average precision {v:.2f}"), 12.5, "f-t")
    c.text(lx, 214, t("both: ROC-AUC 0.90", "两者 ROC-AUC 都是 0.90"), 12.5, "f-m")
    c.text(lx, sy(PR_PREV) - 6, t("random ranking", "随机排序"), 12, "f-m")
    # where each one wins
    rb = 0.3
    pb = PR_PREV * rb / (PR_PREV * rb + (1 - PR_PREV) * fpr(th_at_tpr(rb, mb, sb)))
    for i, line in enumerate(t("B: the top of its list|is almost all positives", "B：排在最前面的|几乎全是阳性").split("|")):
        c.text(sx(0.03), sy(0.9) + i * 16, line, 12.5, "f-t")
    return c


# ---- 5. average precision against prevalence ------------------------------------------
PREVS = [10 ** (-4 + 3.7 * i / 60) for i in range(61)]


def fig_prevalence() -> Canvas:
    (ma, sa), (mb, sb), (ms, ss) = MODELS["A"], MODELS["B"], MODELS["S"]
    c = Canvas("ap-vs-prevalence", 720, 380, t(
        "Average precision plotted against prevalence on a log axis for three models whose ROC-AUC does not change "
        f"with prevalence: the screening model of the previous note (ROC-AUC {auc(ms, ss):.2f}) and models A and B "
        f"(both {AUC_AB:.2f}). Average precision falls toward zero as positives get rarer; the dashed line is a random ranking.",
        f"三个模型的 average precision 随阳性占比（对数坐标）的变化，它们的 ROC-AUC 都不随阳性占比变化：上一篇的筛查模型"
        f"（ROC-AUC {auc(ms, ss):.2f}），以及模型 A、B（都是 {AUC_AB:.2f}）。阳性越少，average precision 越接近零；虚线是随机排序。"))
    xl, xr, yt, yb = 64, 560, 52, 296
    lo, hi = -4.0, math.log10(0.5)
    px = lambda p: xl + (math.log10(p) - lo) / (hi - lo) * (xr - xl)
    py = lambda v: yb - v * (yb - yt)
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        c.line(xl, py(v), xr, py(v), "f-lnrule", 1)
        c.text(xl - 8, py(v) + 4.5, f"{v:g}", 12, "f-m", "end")
    for p, lab in [(1e-4, "0.01%"), (1e-3, "0.1%"), (1e-2, "1%"), (1e-1, "10%"), (0.5, "50%")]:
        c.line(px(p), yb, px(p), yb + 5, "f-lnrule", 1)
        c.text(px(p), yb + 20, lab, 12, "f-m", "middle")
    c.text((xl + xr) / 2, yb + 42, t("prevalence (log scale)", "阳性占比（对数坐标）"), 12, "f-m", "middle")
    c.text(xl - 40, yt - 16, "average precision", 12, "f-m")
    grid = [10 ** (lo + i * (hi - lo) / 50) for i in range(51)]
    c.path([(px(p), py(p)) for p in grid], "f-lnrule", 1.5)
    series = [("S", "f-lnbar2", t(f"screening model, ROC-AUC {auc(ms, ss):.2f}", f"筛查模型，ROC-AUC {auc(ms, ss):.2f}")),
              ("B", "f-lnalt", t("model B, ROC-AUC 0.90", "模型 B，ROC-AUC 0.90")),
              ("A", "f-lnhl", t("model A, ROC-AUC 0.90", "模型 A，ROC-AUC 0.90"))]
    ends = {}
    for key, cls, _ in series:
        mu, sig = MODELS[key]
        vals = [(p, ap(mu, sig, p, 1500)) for p in grid]
        c.path([(px(p), py(v)) for p, v in vals], cls, 2.2)
        ends[key] = vals[-1][1]
    # direct labels at the right end
    ys = sorted(((ends[k], k) for k, _, _ in series), reverse=True)
    last = -99
    for v, k in ys:
        y = max(py(v) + 4.5, last + 30)
        last = y
        cls, lab = next((cl, lb) for kk, cl, lb in series if kk == k)
        c.line(xr + 6, y - 4, xr + 22, y - 4, cls, 2.2)
        words = lab.split(", ") if LANG == "en" else lab.split("，")
        c.text(xr + 28, y, words[0], 12.5, "f-t", weight=600)
        c.text(xr + 28, y + 15, words[1], 12, "f-m")
    # screening prevalence marker
    v = ap(ms, ss, SCREEN_PREV)
    c.line(px(SCREEN_PREV), yt, px(SCREEN_PREV), yb, "f-lnrule", 1, ' stroke-dasharray="3 3"')
    c.dot(px(SCREEN_PREV), py(v), 4.5, "f-dotink")
    c.text(px(SCREEN_PREV) - 8, py(v) + 4, f"AP {v:.2f}", 12.5, "f-t", "end", 600)
    c.text(px(SCREEN_PREV), yt - 8, t(f"screening: {pct(SCREEN_PREV, 2)}", f"筛查：{pct(SCREEN_PREV, 2)}"), 12, "f-m", "middle")
    c.text(px(0.16) - 8, py(0.16) + 4, t("random ranking", "随机排序"), 12, "f-m", "end")
    return c


# ---- numbers quoted in the note -----------------------------------------------------------
def numbers() -> str:
    out = []
    add = out.append
    good, total = toy_auc()
    pts = toy_ap()
    add("== toy ranking " + TOY)
    add(f"pairs {good}/{total} -> AUC {good / total:.2f};  precision at positives {[round(p, 3) for _, p in pts]} -> AP {sum(p for _, p in pts) / len(pts):.3f};  prevalence {NPOS / len(TOY):.3f}")
    trap = 0.0
    prev_r, prev_p = 0.0, 1.0
    for k, (_, p) in enumerate(pts, 1):
        r = k / NPOS
        trap += (r - prev_r) * (p + prev_p) / 2
        prev_r, prev_p = r, p
    add(f"trapezoid from (0,1) through the points: {trap:.3f}")
    add("== binormal models")
    for k, (mu, sig) in MODELS.items():
        add(f"{k}: mu {mu:.3f} sigma {sig:g}  ROC-AUC {auc(mu, sig):.4f}  Gini {2 * auc(mu, sig) - 1:.3f}")
    (ma, sa), (mb, sb) = MODELS["A"], MODELS["B"]
    for f in (0.0001, 0.001, 0.01, 0.1, 0.3):
        add(f"FPR {pct(f, 2)}: TPR A {pct(tpr(th_at_fpr(f), ma, sa), 1)}  B {pct(tpr(th_at_fpr(f), mb, sb), 1)}")
    for r in (0.5, 0.8, 0.9, 0.95):
        add(f"recall {pct(r)}: FPR A {pct(fpr(th_at_tpr(r, ma, sa)), 1)}  B {pct(fpr(th_at_tpr(r, mb, sb)), 1)}")
    # crossing point
    lo_, hi_ = 0.01, 0.99
    for _ in range(60):
        mid = (lo_ + hi_) / 2
        d = tpr(th_at_fpr(mid), ma, sa) - tpr(th_at_fpr(mid), mb, sb)
        if d < 0:
            lo_ = mid
        else:
            hi_ = mid
    add(f"ROC curves cross at FPR {pct(lo_, 1)}, recall {pct(tpr(th_at_fpr(lo_), ma, sa), 1)}")
    for fmax in (0.01, 0.1):
        ra, sa_ = pauc(ma, sa, fmax)
        rb, sb_ = pauc(mb, sb, fmax)
        add(f"partial AUC FPR<= {pct(fmax)}: raw A {ra:.5f} B {rb:.5f}; McClish A {sa_:.3f} B {sb_:.3f}")
    add("== average precision by prevalence")
    for p in (0.5, 0.1, 0.05, 0.01, 0.005, SCREEN_PREV, 0.001):
        add(f"prev {pct(p, 2)}: A {ap(ma, sa, p):.3f}  B {ap(mb, sb, p):.3f}  screening {ap(*MODELS['S'], p):.3f}  "
            f"(lift A {ap(ma, sa, p) / p:.1f}x)")
    for r in (0.1, 0.3, 0.5, 0.8):
        for name, (mu, sig) in (("A", (ma, sa)), ("B", (mb, sb))):
            f = fpr(th_at_tpr(r, mu, sig))
            add(f"  {name} at recall {pct(r)} & prev 0.5%: precision {pct(PR_PREV * r / (PR_PREV * r + (1 - PR_PREV) * f), 1)}")
    add("== Hanley-McNeil standard error")
    for a, npos, nneg in ((0.97, 25, 36), (0.96, 25, 36), (0.90, 25, 36), (0.90, 250, 360), (0.90, 50, 50000)):
        se = hanley_mcneil_se(a, npos, nneg)
        add(f"AUC {a} with {npos} pos / {nneg} neg: SE {se:.4f}  -> 95% {a - 1.96 * se:.3f} .. {min(1, a + 1.96 * se):.3f}")
    return "\n".join(out)


FIGURES = (fig_pairs, fig_ap, fig_roc_ab, fig_pr_ab, fig_prevalence)


def main() -> int:
    global LANG
    OUT.mkdir(parents=True, exist_ok=True)
    warnings = []
    for LANG, suffix in (("en", ""), ("zh", ".zh")):
        for make in FIGURES:
            fig = make()
            (OUT / f"{fig.name}{suffix}.svg").write_text(fig.svg(), encoding="utf-8")
            warnings += [f"[{LANG}] {w}" for w in fig.warnings]
        print(f"wrote {len(FIGURES)} figures ({LANG})")
    LANG = "en"
    print(numbers())
    for w in warnings:
        print("WARNING:", w)
    return 1 if warnings else 0


if __name__ == "__main__":
    sys.exit(main())

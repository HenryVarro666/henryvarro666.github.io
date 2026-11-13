#!/usr/bin/env python3
"""Figures and numbers for the note content/notes/precision-vs-recall/.

    python3 tools/notes/precision_vs_recall_figs.py

writes seven SVGs next to the note, in English (name.svg) and Chinese (name.zh.svg, which
Hugo serves to index.zh.md), and prints every number the text quotes, so the prose can be
checked against one source. Standard library only.

The SVGs carry classes, not colors: the colors come from assets/css/site.css
(.fig-svg .f-*), so the figures follow the light/dark toggle. They contain no ids,
because the page inlines them and the zoom dialog copies them.

Running example: U.S. screening mammography, Breast Cancer Surveillance Consortium,
2007-2013 (Lehman et al., Radiology 2017; counts from the BCSC benchmark tables).
"""
from __future__ import annotations

import math
import sys
from html import escape
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "figs"))
from figlib import text_width  # noqa: E402  (rough text widths, to catch overflow)

OUT = ROOT / "content" / "notes" / "precision-vs-recall"
ND = NormalDist()
LANG = "en"


def t(en: str, zh: str) -> str:
    """The label for the language being drawn. Metric names stay in English in both."""
    return zh if LANG == "zh" else en

# ---- the real matrix ---------------------------------------------------------------
N = 1_682_504          # screening mammograms
TP = 8_529             # cancers with an abnormal reading
FN = 1_283             # cancers read as normal
TN = 1_486_553         # no cancer, read as normal
FP = N - TP - FN - TN  # no cancer, called back

RECALL = TP / (TP + FN)
SPEC = TN / (TN + FP)
FPR = 1 - SPEC
PREC = TP / (TP + FP)
NPV = TN / (TN + FN)
PREV = (TP + FN) / N
ACC = (TP + TN) / N

# A two-Gaussian score model through that operating point: negatives ~ N(0, 1),
# positives ~ N(MU, 1), threshold T0. It reproduces the recall and specificity above.
T0 = ND.inv_cdf(SPEC)
MU = T0 + ND.inv_cdf(RECALL)
AUC = ND.cdf(MU / math.sqrt(2))


def phi(x: float) -> float:
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def rates(t: float) -> tuple[float, float]:
    """(recall, false positive rate) of the score model at threshold t."""
    return 1 - ND.cdf(t - MU), 1 - ND.cdf(t)


def ppv(recall: float, fpr: float, prev: float) -> float:
    return recall * prev / (recall * prev + fpr * (1 - prev))


# ---- a small SVG canvas ------------------------------------------------------------
class Canvas:
    def __init__(self, name: str, w: int, h: int, label: str):
        self.name, self.w, self.h, self.label = name, w, h, label
        self.parts: list[str] = []
        self.warnings: list[str] = []

    def raw(self, s: str):
        self.parts.append(s)

    def rect(self, x, y, w, h, cls, rx=6, sw=1.5):
        self.raw(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:g}" '
                 f'class="{cls}" stroke-width="{sw:g}"/>')

    def text(self, x, y, s, size=13.5, cls="f-t", anchor="start", weight=None, box=None):
        wpx = text_width(s, size)
        left = x if anchor == "start" else x - wpx if anchor == "end" else x - wpx / 2
        if left < 4 or left + wpx > self.w - 4:
            self.warnings.append(f"{self.name}: '{s}' leaves the canvas ({left:.0f}..{left + wpx:.0f})")
        if box is not None and wpx > box:
            self.warnings.append(f"{self.name}: '{s}' ~{wpx:.0f}px > {box:.0f}px")
        wt = f' font-weight="{weight}"' if weight else ""
        self.raw(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size:g}" class="{cls}" '
                 f'text-anchor="{anchor}"{wt}>{escape(s)}</text>')

    def line(self, x1, y1, x2, y2, cls="f-lnrule", sw=1.0, extra=""):
        self.raw(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}" '
                 f'stroke-width="{sw:g}" stroke-linecap="round"{extra}/>')

    def path(self, pts, cls, sw=2.0, close=False):
        d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts) + (" Z" if close else "")
        if close:
            self.raw(f'<path d="{d}" class="{cls}"/>')
        else:
            self.raw(f'<path d="{d}" class="{cls}" stroke-width="{sw:g}" stroke-linecap="round" '
                     f'stroke-linejoin="round"/>')

    def dot(self, cx, cy, r=5.0, cls="f-fillhl", ring=True):
        ring_attr = ' stroke-width="2"' if ring else ""
        ring_cls = " f-ring" if ring else ""
        self.raw(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:g}" class="{cls}{ring_cls}"{ring_attr}/>')

    def arrow(self, x1, y1, x2, y2, cls="f-ln", head_cls="f-head", sw=1.5, head=7.0):
        ang = math.atan2(y2 - y1, x2 - x1)
        ex, ey = x2 - math.cos(ang) * head * 0.9, y2 - math.sin(ang) * head * 0.9
        self.line(x1, y1, ex, ey, cls, sw)
        a1, a2 = ang + math.radians(152), ang - math.radians(152)
        pts = [(x2, y2), (x2 + math.cos(a1) * head, y2 + math.sin(a1) * head),
               (x2 + math.cos(a2) * head, y2 + math.sin(a2) * head)]
        self.raw('<polygon points="' + " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
                 + f'" class="{head_cls}"/>')

    def svg(self) -> str:
        return (f'<svg class="fig-svg" viewBox="0 0 {self.w} {self.h}" xmlns="http://www.w3.org/2000/svg" '
                f'role="img" aria-label="{escape(self.label, quote=True)}">'
                f'<rect x="0" y="0" width="{self.w}" height="{self.h}" rx="8" class="f-bg"/>'
                + "".join(self.parts) + "</svg>\n")


def pct(x: float, digits: int = 0) -> str:
    return f"{100 * x:.{digits}f}%"


# ---- 1. anatomy of the matrix ------------------------------------------------------
def fig_matrix() -> Canvas:
    c = Canvas("confusion-matrix", 720, 404, t(
        "A two-by-two confusion matrix. Rows are the truth, columns are the model's call. "
        "Recall and specificity are read along the rows; precision and negative predictive "
        "value are read down the columns.",
        "二乘二的混淆矩阵：行是真实情况，列是模型的判定。recall 和 specificity 横着读，"
        "precision 和 NPV 竖着读。"))
    x0, y0, cw, ch, gap = 162, 78, 166, 98, 6
    cx = [x0 + cw / 2, x0 + cw + gap + cw / 2]
    cy = [y0 + ch / 2, y0 + ch + gap + ch / 2]
    right, bottom = x0 + 2 * cw + gap, y0 + 2 * ch + gap

    pos, neg = t("positive", "阳性"), t("negative", "阴性")
    c.text((x0 + right) / 2, 26, t("the model says", "模型判定"), 12.5, "f-m", "middle")
    c.text(cx[0], 62, pos, 14, "f-t", "middle", 600)
    c.text(cx[1], 62, neg, 14, "f-t", "middle", 600)
    c.text(24, cy[0] - 14, t("the truth is", "真实情况"), 12.5, "f-m")
    c.text(24, cy[0] + 6, pos, 14, "f-t", weight=600)
    c.text(24, cy[1] - 14, t("the truth is", "真实情况"), 12.5, "f-m")
    c.text(24, cy[1] + 6, neg, 14, "f-t", weight=600)

    cells = [(0, 0, "TP", t("true positive", "真阳性"), t("a hit", "抓到了"), "f-hl"),
             (0, 1, "FN", t("false negative", "假阴性"), t("a miss", "漏报"), "f-altsoft"),
             (1, 0, "FP", t("false positive", "假阳性"), t("a false alarm", "误报"), "f-altsoft"),
             (1, 1, "TN", t("true negative", "真阴性"), t("correctly left alone", "正确放行"), "f-hl")]
    for r, k, abbr, name, nick, cls in cells:
        x, y = x0 + k * (cw + gap), y0 + r * (ch + gap)
        c.rect(x, y, cw, ch, cls, rx=6, sw=1.5)
        c.text(x + cw / 2, y + 38, abbr, 24, "f-t", "middle", 700)
        c.text(x + cw / 2, y + 60, name, 13.5, "f-t", "middle")
        c.text(x + cw / 2, y + 79, nick, 12.5, "f-m", "middle", box=cw - 12)

    # rows -> recall, specificity
    bx = right + 10
    for r, (name, formula, aka) in enumerate([
            ("Recall", "TP / (TP + FN)", t("sensitivity, TPR", "即 sensitivity、TPR")),
            ("Specificity", "TN / (TN + FP)", "FPR = 1 − specificity")]):
        top, bot = y0 + r * (ch + gap) + 8, y0 + r * (ch + gap) + ch - 8
        c.path([(bx, top), (bx + 6, top), (bx + 6, bot), (bx, bot)], "f-ln", 1.5)
        c.text(bx + 18, cy[r] - 12, name, 15, "f-t", weight=600)
        c.text(bx + 18, cy[r] + 8, formula, 13.5, "f-t")
        c.text(bx + 18, cy[r] + 27, aka, 12.5, "f-m")

    # columns -> precision, NPV
    by = bottom + 10
    for k, (name, formula, aka) in enumerate([
            ("Precision", "TP / (TP + FP)", t("positive predictive value", "阳性预测值（PPV）")),
            ("NPV", "TN / (TN + FN)", t("negative predictive value", "阴性预测值"))]):
        left, rt = x0 + k * (cw + gap) + 8, x0 + k * (cw + gap) + cw - 8
        c.path([(left, by), (left, by + 6), (rt, by + 6), (rt, by)], "f-ln", 1.5)
        c.text(cx[k], by + 28, name, 15, "f-t", "middle", 600)
        c.text(cx[k], by + 48, formula, 13.5, "f-t", "middle")
        c.text(cx[k], by + 67, aka, 12.5, "f-m", "middle")

    for i, row in enumerate(t("Across a row, the|denominator is set by|the world: these rates|ignore prevalence.",
                              "横着读：分母由|真实情况决定，|不受阳性占比影响。").split("|")):
        c.text(bx + 18, by + 22 + i * 16, row, 12.5, "f-m")
    for i, row in enumerate(t("Down a column, the|denominator is set|by the model: these|move with prevalence.",
                              "竖着读：分母由|模型的判定决定，|会随阳性占比变化。").split("|")):
        c.text(24, by + 22 + i * 16, row, 12.5, "f-m")
    return c


# ---- 1b. the four rates as slices of the matrix -------------------------------------
def fig_four_rates() -> Canvas:
    c = Canvas("four-rates", 720, 246, t(
        "Four small confusion matrices. Recall is TP over the top row, specificity is TN over the "
        "bottom row, precision is TP over the left column, NPV is TN over the right column. "
        f"For screening mammography they are {pct(RECALL, 1)}, {pct(SPEC, 1)}, {pct(PREC, 1)} and {pct(NPV, 2)}.",
        "四个小混淆矩阵：recall 是 TP 除以第一行，specificity 是 TN 除以第二行，precision 是 TP 除以第一列，"
        f"NPV 是 TN 除以第二列。乳腺筛查的数值依次是 {pct(RECALL, 1)}、{pct(SPEC, 1)}、{pct(PREC, 1)} 和 {pct(NPV, 2)}。"))
    cells = {"TP": (0, 0), "FN": (0, 1), "FP": (1, 0), "TN": (1, 1)}
    row = t("ignores prevalence", "不受阳性占比影响")
    col = t("moves with prevalence", "随阳性占比变化")
    panels = [
        ("Recall", "TP", "FN", pct(RECALL, 1), t("of real positives,|the share caught", "真实阳性里，|抓到了多少"), row),
        ("Specificity", "TN", "FP", pct(SPEC, 1), t("of real negatives,|the share left alone", "真实阴性里，|放过了多少"), row),
        ("Precision", "TP", "FP", pct(PREC, 1), t("of alarms,|the share that are real", "报警的里面，|真有问题的占多少"), col),
        ("NPV", "TN", "FN", pct(NPV, 2), t("of all-clears,|the share that are right", "放行的里面，|确实没事的占多少"), col),
    ]
    pw, gap, cw, ch, cg = 160, 16, 62, 38, 4
    for i, (name, num, other, value, question, note) in enumerate(panels):
        px0 = 16 + i * (pw + gap)
        mid = px0 + pw / 2
        gx, gy = mid - (2 * cw + cg) / 2, 44
        c.text(mid, 28, name, 15, "f-t", "middle", 600)
        for abbr, (r, k) in cells.items():
            x, y = gx + k * (cw + cg), gy + r * (ch + cg)
            if abbr == num:
                c.rect(x, y, cw, ch, "f-fillhl", rx=4, sw=0)
                c.text(x + cw / 2, y + ch / 2 + 5, abbr, 13.5, "f-onhl", "middle", 700)
            elif abbr == other:
                c.rect(x, y, cw, ch, "f-hl", rx=4, sw=1.5)
                c.text(x + cw / 2, y + ch / 2 + 5, abbr, 13.5, "f-t", "middle", 600)
            else:
                c.rect(x, y, cw, ch, "f-node", rx=4, sw=1)
                c.text(x + cw / 2, y + ch / 2 + 5, abbr, 13.5, "f-m", "middle")
        c.text(mid, gy + 2 * ch + cg + 32, value, 20, "f-t", "middle", 700)
        for j, line in enumerate(question.split("|")):
            c.text(mid, gy + 2 * ch + cg + 54 + j * 16, line, 12.5, "f-m", "middle", box=pw + gap - 4)
        c.text(mid, gy + 2 * ch + cg + 96, note, 12.5, "f-t", "middle", box=pw + gap - 4)
    return c


# ---- 2. one threshold on two score distributions -----------------------------------
def fig_threshold() -> Canvas:
    c = Canvas("threshold", 720, 352, t(
        "Two overlapping bell curves of model scores, one for actual negatives and one for "
        f"actual positives, cut by a threshold. Positives left of the threshold are misses, "
        f"{pct(1 - RECALL)} of positives; negatives right of it are false alarms, "
        f"{pct(FPR)} of negatives.",
        f"两条重叠的分数分布，一条是真实阴性，一条是真实阳性，中间一条阈值。阈值左边的阳性是漏报，"
        f"占阳性的 {pct(1 - RECALL)}；右边的阴性是误报，占阴性的 {pct(FPR)}。"))
    lo, hi, xl, xr, base, top = -3.4, 5.8, 40, 680, 250, 84
    sx = lambda s: xl + (s - lo) / (hi - lo) * (xr - xl)
    sy = lambda d: base - d / phi(0) * (base - top)
    grid = [lo + i * (hi - lo) / 320 for i in range(321)]

    tx = sx(T0)
    miss = [(sx(s), sy(phi(s - MU))) for s in grid if s <= T0] + [(tx, sy(phi(T0 - MU))), (tx, base), (xl, base)]
    alarm = [(tx, base), (tx, sy(phi(T0)))] + [(sx(s), sy(phi(s))) for s in grid if s >= T0] + [(xr, base)]
    c.path(miss, "f-altarea", close=True)
    c.path(alarm, "f-altarea", close=True)
    c.line(xl, base, xr, base, "f-lnrule", 1)
    c.path([(sx(s), sy(phi(s))) for s in grid], "f-lnbar2", 2)
    c.path([(sx(s), sy(phi(s - MU))) for s in grid], "f-lnhl", 2)
    c.line(tx, 58, tx, base + 8, "f-lnink", 1.5)

    c.text(tx, 48, t("threshold", "阈值"), 13.5, "f-t", "middle", 600)
    c.text(sx(0) - 14, 74, t("actual negatives", "真实阴性"), 13, "f-m", "middle")
    c.text(sx(MU) + 14, 74, t("actual positives", "真实阳性"), 13, "f-m", "middle")
    c.text(tx - 12, base + 24, t(f"misses (FN): {pct(1 - RECALL)} of positives",
                                 f"漏报（FN）：占阳性的 {pct(1 - RECALL)}"), 13, "f-t", "end")
    c.text(tx + 12, base + 24, t(f"false alarms (FP): {pct(FPR)} of negatives",
                                 f"误报（FP）：占阴性的 {pct(FPR)}"), 13, "f-t")
    c.text(tx - 12, base + 44, t("← called negative", "← 判为阴性"), 12.5, "f-m", "end")
    c.text(tx + 12, base + 44, t("called positive →", "判为阳性 →"), 12.5, "f-m")
    c.text(xl, base + 24, t("model score", "模型分数"), 12.5, "f-m")
    c.text((xl + xr) / 2, base + 80,
           t("Raise the threshold: fewer false alarms, more misses. Lower it: the reverse.",
             "阈值调高：误报变少，漏报变多。调低则相反。"),
           13, "f-m", "middle")
    return c


# ---- 3. 1,000 screening mammograms -------------------------------------------------
PER_1000 = {"TP": round(1000 * TP / N), "FN": round(1000 * FN / N), "FP": round(1000 * FP / N)}
PER_1000["TN"] = 1000 - sum(PER_1000.values())


def fig_thousand() -> Canvas:
    k = PER_1000
    flagged, cleared = k["TP"] + k["FP"], k["FN"] + k["TN"]
    c = Canvas("thousand-screens", 720, 388, t(
        f"One thousand screening mammograms drawn as dots. {flagged} are called back: "
        f"{k['TP']} cancers and {k['FP']} false alarms. {cleared} are cleared: "
        f"{k['TN']} correctly, and {k['FN']} cancer is missed.",
        f"把 1,000 次乳腺筛查画成点。{flagged} 次被叫回复查，其中 {k['TP']} 例是癌症，{k['FP']} 例是误报；"
        f"{cleared} 次被放行，其中 {k['TN']} 例正确，漏掉 {k['FN']} 例癌症。"))
    rows, pitch, sw, y0 = 20, 13, 8.4, 70

    def column(x, start, n, cls):
        """n dots down one column, drawn as one dashed round-capped line."""
        y1 = y0 + start * pitch
        c.raw(f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y1 + (n - 1) * pitch + 0.01:.2f}" '
              f'class="{cls}" stroke-width="{sw:g}" stroke-linecap="round" stroke-dasharray="0 {pitch}"/>')

    def block(x_first, counts):
        """Fill columns top to bottom with runs of (count, class); returns the last column's x."""
        col, row = 0, 0
        for n, cls in counts:
            while n > 0:
                take = min(n, rows - row)
                if cls is not None:
                    column(x_first + col * pitch, row, take, cls)
                n -= take
                row += take
                if row == rows:
                    col, row = col + 1, 0
        return x_first + (col if row else col - 1) * pitch

    xa = 28.5
    xa_end = block(xa, [(k["TP"], "f-lnhl"), (k["FP"], "f-lnalt")])
    xb = xa_end + 2 * pitch
    xb_end = block(xb, [(k["FN"], None), (k["TN"], "f-lnbar")])
    c.raw(f'<circle cx="{xb:.1f}" cy="{y0:.1f}" r="3.2" class="f-hollow" stroke-width="2"/>')

    for x1, x2, label in [(xa, xa_end, t(f"{flagged} called back", f"叫回复查 {flagged} 次")),
                          (xb, xb_end, t(f"{cleared} cleared", f"放行 {cleared} 次"))]:
        a, b = x1 - sw / 2, x2 + sw / 2
        c.path([(a, 52), (a, 46), (b, 46), (b, 52)], "f-ln", 1.5)
        c.text((a + b) / 2, 34, label, 13.5, "f-t", "middle", 600)

    ly = y0 + rows * pitch + 22
    items = [("f-fillhl", t(f"{k['TP']} cancers found (TP)", f"查出癌症 {k['TP']} 例（TP）")),
             ("f-alt", t(f"{k['FP']} false alarms (FP)", f"误报 {k['FP']} 例（FP）")),
             ("f-hollow", t(f"{k['FN']} cancer missed (FN)", f"漏掉癌症 {k['FN']} 例（FN）")),
             ("f-bar", t(f"{k['TN']} cleared (TN)", f"正确放行 {k['TN']} 例（TN）"))]
    x = 26
    for cls, label in items:
        extra = ' stroke-width="2"' if cls == "f-hollow" else ""
        r = 3.6 if cls == "f-hollow" else 4.6
        c.raw(f'<circle cx="{x + 5:.1f}" cy="{ly - 4.5:.1f}" r="{r}" class="{cls}"{extra}/>')
        c.text(x + 16, ly, label, 13, "f-t")
        x += 16 + text_width(label, 13) + 24
    c.text(26, ly + 26, t(f"Rounded to whole exams. Exact rates: recall {pct(RECALL, 1)}, "
                          f"specificity {pct(SPEC, 1)}, precision {pct(PREC, 1)}.",
                          f"按整数次取整。精确比率：recall {pct(RECALL, 1)}，specificity {pct(SPEC, 1)}，"
                          f"precision {pct(PREC, 1)}。"), 12.5, "f-m")
    return c


# ---- 4. precision against prevalence -----------------------------------------------
DOMAIN_PREV = [(("card fraud", "卡欺诈"), 0.000658, ("≈0.07% of volume", "约占交易额 0.07%")),
               (("screening", "乳腺筛查"), PREV, (pct(PREV, 2), pct(PREV, 2))),
               (("fake accounts", "虚假账户"), 0.04, ("≈4% of users", "约占用户 4%"))]
FPR_FAMILY = [(FPR, "f-lnhl1"), (0.01, "f-lnhl2"), (0.001, "f-lnhl")]


def fig_prevalence() -> Canvas:
    c = Canvas("precision-vs-prevalence", 720, 392, t(
        f"Precision plotted against prevalence on a log axis, at {pct(RECALL)} recall, for three "
        f"false positive rates: {pct(FPR)}, 1% and 0.1%. Every curve collapses toward zero as "
        f"positives get rarer; the screening mammography point sits at {pct(PREV, 2)} prevalence "
        f"and {pct(PREC, 1)} precision.",
        f"recall 固定在 {pct(RECALL)} 时，precision 随阳性占比（对数坐标）的变化，三条线对应假阳性率 "
        f"{pct(FPR)}、1% 和 0.1%。阳性越少，每条线都越接近零；乳腺筛查的点在阳性占比 {pct(PREV, 2)}、"
        f"precision {pct(PREC, 1)}。"))
    xl, xr, yt, yb = 64, 690, 70, 306
    lo, hi = -4.0, math.log10(0.5)
    px = lambda p: xl + (math.log10(p) - lo) / (hi - lo) * (xr - xl)
    py = lambda v: yb - v * (yb - yt)

    for v in (0, 0.25, 0.5, 0.75, 1.0):
        c.line(xl, py(v), xr, py(v), "f-lnrule", 1)
        c.text(xl - 8, py(v) + 4.5, pct(v), 12.5, "f-m", "end")
    for p, lab in [(1e-4, "0.01%"), (1e-3, "0.1%"), (1e-2, "1%"), (1e-1, "10%"), (0.5, "50%")]:
        c.line(px(p), yb, px(p), yb + 5, "f-lnrule", 1)
        c.text(px(p), yb + 21, lab, 12.5, "f-m", "middle")
    c.text((xl + xr) / 2, yb + 46, t("prevalence: share of cases that are truly positive (log scale)",
                                     "阳性占比（prevalence）：真实阳性在全部样本中的比例（对数坐标）"),
           12.5, "f-m", "middle")
    c.text(22, yt - 12, "precision", 12.5, "f-m")

    for name, p, val in DOMAIN_PREV:
        c.line(px(p), yt - 8, px(p), yb, "f-lnrule", 1)
        c.text(px(p), 26, t(*name), 12.5, "f-t", "middle", 600)
        c.text(px(p), 43, t(*val), 12.5, "f-m", "middle")

    grid = [10 ** (lo + i * (hi - lo) / 240) for i in range(241)]
    for fpr, cls in FPR_FAMILY:
        c.path([(px(p), py(ppv(RECALL, fpr, p))) for p in grid], cls, 2)
        # label left of the curve where it crosses 62% precision
        odds = 0.62 / 0.38 * fpr / RECALL
        p62 = odds / (1 + odds)
        lab = f"FPR {pct(fpr, 1 if fpr < 0.01 else 0)}"
        c.text(px(p62) - 10, py(0.62) + 4.5, lab, 13, "f-t", "end")
    c.dot(px(PREV), py(PREC), 5, "f-dotink")
    c.text(px(PREV) + 8, py(PREC) - 14, t(f"{pct(PREC, 1)} precision", f"precision {pct(PREC, 1)}"), 13, "f-t")
    return c


# ---- 5. ROC against precision-recall -----------------------------------------------
PR_PREVS = [(0.5, "f-lnhl", ("50% positive", "阳性占 50%")), (0.05, "f-lnhl2", ("5% positive", "阳性占 5%")),
            (PREV, "f-lnhl1", (f"{pct(PREV, 2)} positive", f"阳性占 {pct(PREV, 2)}"))]


def fig_roc_pr() -> Canvas:
    vals = ", ".join(pct(ppv(RECALL, FPR, p), 1 if p < 0.01 else 0) for p, _, _ in PR_PREVS)
    c = Canvas("roc-vs-pr", 720, 372, t(
        f"Left: one ROC curve with area {AUC:.2f}, the same at any prevalence. Right: the "
        f"precision-recall curves of the same model when 50%, 5% and {pct(PREV, 2)} of cases "
        f"are positive. At the marked operating point precision is {vals}.",
        f"左：一条 ROC 曲线，面积 {AUC:.2f}，阳性占比怎么变都一样。右：同一个模型在阳性占 50%、5% 和 "
        f"{pct(PREV, 2)} 时的 PR 曲线，标出的工作点上 precision 分别是 {vals}。"))
    yt, yb, size = 70, 310, 240
    ts = [-4 + i * (MU + 8) / 400 for i in range(401)]

    def frame(x0, title, yname, xname):
        c.text(x0, 28, title, 14, "f-t", weight=600)
        c.text(x0, 52, yname, 12.5, "f-m")
        for v in (0, 0.5, 1.0):
            y = yb - v * size
            c.line(x0, y, x0 + size, y, "f-lnrule", 1)
            c.text(x0 - 8, y + 4.5, f"{v:g}", 12.5, "f-m", "end")
            c.line(x0 + v * size, yb, x0 + v * size, yb + 5, "f-lnrule", 1)
            c.text(x0 + v * size, yb + 21, f"{v:g}", 12.5, "f-m", "middle")
        c.text(x0 + size / 2, yb + 44, xname, 12.5, "f-m", "middle")

    # ROC
    x0 = 56
    frame(x0, t("ROC: the same curve at any prevalence", "ROC：阳性占比怎么变都是这条线"),
          t("recall (true positive rate)", "recall（真阳性率）"), t("false positive rate", "假阳性率（FPR）"))
    c.line(x0, yb, x0 + size, yt, "f-lnrule", 1)
    c.path([(x0 + size * rates(th)[1], yb - size * rates(th)[0]) for th in ts], "f-lnhl", 2)
    ox, oy = x0 + size * FPR, yb - size * RECALL
    c.dot(ox, oy, 5, "f-dotink")
    c.text(ox + 12, oy + 22, t("operating point:", "工作点："), 12.5, "f-t")
    c.text(ox + 12, oy + 38, t(f"recall {pct(RECALL)}, FPR {pct(FPR)}", f"recall {pct(RECALL)}，FPR {pct(FPR)}"), 12.5, "f-t")
    c.text(x0 + size - 8, yb - 14, t(f"area under the curve {AUC:.2f}", f"曲线下面积 {AUC:.2f}"), 12.5, "f-m", "end")

    # precision-recall
    x0 = 414
    frame(x0, t("Precision–recall: it depends on prevalence", "PR 曲线：跟着阳性占比变"), "precision", "recall")
    for prev, cls, _ in PR_PREVS:
        pts = []
        for th in ts:
            r, f = rates(th)
            if r > 0.004:
                pts.append((x0 + size * r, yb - size * ppv(r, f, prev)))
        c.path(pts, cls, 2)
    for prev, cls, _ in PR_PREVS:
        v = ppv(RECALL, FPR, prev)
        c.dot(x0 + size * RECALL, yb - size * v, 5, "f-dotink")
        c.text(x0 + size + 9, yb - size * v + 4.5, pct(v, 1 if v < 0.1 else 0), 12.5, "f-t")
    ly = yb - 46
    for i, (_, cls, label) in enumerate(PR_PREVS):
        y = ly + i * 17
        c.line(x0 + 10, y - 4, x0 + 28, y - 4, cls, 2)
        c.text(x0 + 35, y, t(*label), 12.5, "f-t")
    return c


# ---- 6. several thresholds, graded actions -----------------------------------------
ACTIONS = [
    (("Screening mammogram", "乳腺筛查"),
     [(0.40, "f-bar", ("back to routine screening", "回到常规筛查")),
      (0.46, "f-fillhl1", ("call back for more imaging", "叫回来加做影像")),
      (0.14, "f-fillhl", ("biopsy", "活检"))]),
    (("Card payment", "刷卡支付"),
     [(0.40, "f-bar", ("approve", "放行")), (0.25, "f-fillhl1", ("one-time code check", "短信验证码")),
      (0.21, "f-fillhl2", ("hold for review", "人工审核")), (0.14, "f-fillhl", ("decline", "拒绝交易"))]),
    (("Account on a video platform", "短视频平台的账号"),
     [(0.40, "f-bar", ("no action", "不处理")), (0.25, "f-fillhl1", ("CAPTCHA / phone check", "验证码 / 手机验证")),
      (0.21, "f-fillhl2", ("limit reach, review", "限流并送审")), (0.14, "f-fillhl", ("remove", "封号"))]),
]


def fig_actions() -> Canvas:
    c = Canvas("graded-actions", 720, 372, t(
        "Three score bars, one each for a screening mammogram, a card payment and an account on a "
        "video platform. Each is cut by several thresholds into actions that get more costly "
        "to the right: from no action, through a cheap check, to biopsy, decline or removal.",
        "三条分数条：乳腺筛查、刷卡支付、短视频平台的账号。每条被几个阈值切成几段，越往右动作代价越高："
        "从不处理，到便宜的复核，再到活检、拒绝交易或封号。"))
    xl, xr, gap, bar_h = 24, 696, 2, 14
    y = 30
    for title, segs in ACTIONS:
        c.text(xl, y, t(*title), 14, "f-t", weight=600)
        x = xl
        for i, (share, cls, label) in enumerate(segs):
            w = share * (xr - xl)
            left = x + (gap / 2 if i else 0)
            right = x + w - (gap / 2 if i < len(segs) - 1 else 0)
            c.rect(left, y + 12, right - left, bar_h, cls, rx=3, sw=0)
            c.text((left + right) / 2, y + 46, t(*label), 13, "f-t", "middle", box=right - left - 6)
            x += w
        y += 88
    ay = y - 8
    c.arrow(xl, ay, xr, ay, "f-ln", "f-head", 1.5)
    c.text(xl, ay + 22, t("low score", "分数低"), 12.5, "f-m")
    c.text(xr, ay + 22, t("high score", "分数高"), 12.5, "f-m", "end")
    c.text((xl + xr) / 2, ay + 22, t("model score", "模型分数"), 12.5, "f-m", "middle")
    c.text(xl, ay + 46, t("Low thresholds guard cheap, reversible actions: tune them for recall.",
                          "低阈值管的是便宜、可撤销的动作：按 recall 来调。"), 13, "f-t")
    c.text(xl, ay + 65, t("High thresholds guard costly, hard-to-undo actions: tune them for precision.",
                          "高阈值管的是代价高、难撤销的动作：按 precision 来调。"), 13, "f-t")
    return c


# ---- numbers quoted in the note -----------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def numbers() -> str:
    out = []
    add = out.append
    add("== screening mammography (BCSC 2007-2013)")
    add(f"N {N:,}  TP {TP:,}  FN {FN:,}  FP {FP:,}  TN {TN:,}")
    add(f"prevalence {pct(PREV, 2)}  recall {pct(RECALL, 1)}  specificity {pct(SPEC, 1)}  FPR {pct(FPR, 1)}")
    add(f"precision {pct(PREC, 2)}  NPV {pct(NPV, 3)}  accuracy {pct(ACC, 2)}  always-negative accuracy {pct(1 - PREV, 2)}")
    f1 = 2 * PREC * RECALL / (PREC + RECALL)
    mcc = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    add(f"F1 {f1:.3f}  MCC {mcc:.3f}  LR+ {RECALL / FPR:.2f}  LR- {(1 - RECALL) / SPEC:.3f}")
    prior = PREV / (1 - PREV)
    add(f"prior odds {prior:.5f} (1 in {1 / prior:.0f})  posterior odds {prior * RECALL / FPR:.4f}  -> {pct(PREC, 2)}")
    add(f"false alarms per cancer found {FP / TP:.1f};  per 1,000: {PER_1000}")
    add(f"score model: threshold {T0:.3f}, separation {MU:.3f}, AUC {AUC:.3f}")
    add("== the four rates, one at a time")
    add(f"miss rate {pct(1 - RECALL, 1)}  false discovery rate {pct(1 - PREC, 1)}  call-backs per cancer found {(TP + FP) / TP:.1f}")
    add(f"one point of specificity = {0.01 * (TN + FP):,.0f} exams;  lift of a call-back {PREC / PREV:.1f}x")
    forate = FN / (FN + TN)
    add(f"before the test 1 in {1 / PREV:.0f};  after a normal reading 1 in {1 / forate:.0f} (false omission rate {pct(forate, 3)}): risk cut {PREV / forate:.1f}x")
    add(f"balanced accuracy {pct((RECALL + SPEC) / 2, 1)};  F1 with 'normal' as the positive class {2 * NPV * SPEC / (NPV + SPEC):.3f}")
    add(f"99% specificity at 0.1% prevalence: {0.01 * 999 / 1:.1f} false alarms per fraud in the stream")
    add("== precision at 87% recall, by prevalence and FPR")
    for (name, _), p, _ in DOMAIN_PREV:
        add(f"{name:14s} prev {pct(p, 3)}: " + "  ".join(f"FPR {pct(f, 1)} -> {pct(ppv(RECALL, f, p), 1)}" for f, _ in FPR_FAMILY))
    add("== same operating point on the PR curves")
    add("  ".join(f"{lab[0]}: {pct(ppv(RECALL, FPR, p), 1)}" for p, _, lab in PR_PREVS))
    add("== FPR needed for 50% precision: recall * prev / (1 - prev)")
    for p in (0.001, 0.0058, 0.04):
        add(f"prev {pct(p, 2)} at recall 80%: FPR <= {pct(0.8 * p / (1 - p), 3)}")
    add("== fraud example: 1,000,000 payments, 0.1% fraud, recall 80%")
    for f in (0.005, 0.001):
        fp = f * 999_000
        add(f"FPR {pct(f, 1)}: TP 800, FP {fp:,.0f}, precision {pct(800 / (800 + fp), 1)}")
    add("== fake-account example: 1e9 accounts, 4% fake, recall 90%, FPR 0.1%")
    tp, fp = 0.9 * 4e7, 0.001 * 9.6e8
    add(f"TP {tp:,.0f}  FP {fp:,.0f}  precision {pct(tp / (tp + fp), 1)}")
    add("== cost ratio -> probability threshold  p* = c_FP / (c_FP + c_FN)")
    for r in (1000, 100, 10, 1, 0.1, 0.01):
        add(f"c_FN/c_FP {r:g}: p* {1 / (1 + r):.4f}")
    add("== amount-dependent threshold: c_FN = A, c_FP = 0.02 A + 3")
    for a in (20, 200, 2000):
        cfp = 0.02 * a + 3
        add(f"A ${a}: p* {cfp / (cfp + a):.3f}")
    add("== test-set prevalence: recall 90%, FPR 5%")
    for p in (0.5, 0.01):
        add(f"prev {pct(p)}: precision {pct(ppv(0.9, 0.05, p), 1)}")
    add("== Wilson 95% interval")
    lo, hi = wilson(27, 30)
    add(f"27/30: {pct(lo, 1)} .. {pct(hi, 1)}")
    lo, hi = wilson(TP, TP + FN)
    add(f"{TP}/{TP + FN}: {pct(lo, 1)} .. {pct(hi, 1)}")
    lo, hi = wilson(22, 25)
    add(f"22/25: {pct(lo, 1)} .. {pct(hi, 1)}")
    add("== RadQNLI check: accuracy = prev*recall + (1-prev)*specificity, recall 0.53")
    for prev, acc in ((0.21, 0.81), (0.70, 0.6186)):
        spec = (acc - prev * 0.53) / (1 - prev)
        add(f"entailment share {pct(prev)}: implied specificity {spec:.3f}, precision {ppv(0.53, 1 - spec, prev):.2f}")
    return "\n".join(out)


FIGURES = (fig_matrix, fig_four_rates, fig_threshold, fig_thousand, fig_prevalence, fig_roc_pr, fig_actions)


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

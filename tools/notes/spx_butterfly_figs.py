#!/usr/bin/env python3
"""Rebuild the bilingual figures for the SPX butterfly note."""
from math import log, sqrt
from pathlib import Path
from statistics import NormalDist

from precision_vs_recall_figs import Canvas


OUT = Path(__file__).resolve().parents[2] / "content" / "notes" / "spx-butterfly"
CALL = (7550, 7575, 7600)
PUT = (7460, 7485, 7510)
CALL_DEBIT = 3.30
PUT_DEBIT = 2.00
CLOSE_QUOTE = 8.00
IV = 0.20
MINUTES_PER_YEAR = 365 * 24 * 60
OPEN_MINUTES, CLOSE_MINUTES = 231, 100
TIMES = (OPEN_MINUTES, CLOSE_MINUTES, 30, 5, 0)
NORMAL = NormalDist()


def expiration_value(spot, strikes):
    low, _, high = strikes
    return max(0, min(spot - low, high - spot))


def fly_value(spot, minutes, volatility=IV):
    if minutes == 0:
        return expiration_value(spot, CALL)

    # European Black–Scholes calls with zero interest and dividend rates.
    total_vol = volatility * sqrt(minutes / MINUTES_PER_YEAR)

    def call(strike):
        d1 = log(spot / strike) / total_vol + total_vol / 2
        return spot * NORMAL.cdf(d1) - strike * NORMAL.cdf(d1 - total_vol)

    low, middle, high = CALL
    return max(0, call(low) - 2 * call(middle) + call(high))


def spot_for_quote(quote, minutes):
    """Find the example point on the lower-spot side of the curve."""
    low, high = 7500, CALL[1]
    assert fly_value(low, minutes) < quote < fly_value(high, minutes)
    for _ in range(42):
        middle = (low + high) / 2
        if fly_value(middle, minutes) < quote:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def dollars(value, debit):
    return (value - debit) * 100


def plot_axes(canvas, domain, y_limits, x_ticks, y_ticks, x_label, y_label):
    left, right, top, bottom = 77, 692, 111, 361
    lo, hi = domain
    ymin, ymax = y_limits

    def x(spot):
        return left + (spot - lo) / (hi - lo) * (right - left)

    def y(pnl):
        return bottom - (pnl - ymin) / (ymax - ymin) * (bottom - top)

    canvas.text(left, 96, y_label, 12.5, "f-t")
    for tick in y_ticks:
        cls, width = ("f-ln", 1.25) if tick == 0 else ("f-lnrule", 0.85)
        canvas.line(left, y(tick), right, y(tick), cls, width)
        label = ("−$" if tick < 0 else "$") + f"{abs(tick):,.0f}"
        canvas.text(left - 9, y(tick) + 4, label, 11.5, "f-t", "end")
    canvas.line(left, top, left, bottom, "f-lnrule")
    canvas.line(left, bottom, right, bottom, "f-lnrule")
    for tick in x_ticks:
        canvas.line(x(tick), bottom, x(tick), bottom + 4, "f-ln")
        canvas.text(x(tick), bottom + 18, f"{tick:,}", 11.5, "f-t", "middle")
    canvas.text((left + right) / 2, 405, x_label, 12.5, "f-t", "middle")
    return x, y


def legend(canvas, left, label, color, dashes, width=2.4):
    canvas.line(left, 65, left + 23, 65, color, width, f' stroke-dasharray="{dashes}"')
    canvas.text(left + 30, 69, label, 12.5)


def curve(canvas, x, y, levels, value, color, dashes, width):
    canvas.raw(f'<g stroke-dasharray="{dashes}">')
    canvas.path([(x(spot), y(value(spot))) for spot in levels], color, width)
    canvas.raw("</g>")


def intraday_figure(zh):
    title = "7,575 Call 蝶式的盘中与到期盈亏" if zh else "The 7,575 call butterfly before and at expiration"
    label = (
        "五条曲线表示到期及剩余 5、30、100、231 分钟的理论盈亏。临近到期时峰值更高、曲线更窄。"
        "A、B 是模型中对应 3.30、8.00 点的两个假设情景。"
        if zh else
        "Five curves show modeled profit and loss with 231, 100, 30 and 5 minutes remaining and at expiration. "
        "The curve narrows and its peak rises as time runs out. A and B are example quotes of 3.30 and 8.00."
    )
    canvas = Canvas("intraday", 720, 426, label)
    canvas.text(26, 34, title, 22, weight=600)
    styles = (
        ("f-lnhl2", "4 4", 2.1), ("f-lnhl", "none", 2.7), ("f-lnalt", "7 4", 2.3),
        ("f-ln", "2 3", 2.4), ("f-lnink", "none", 2.65),
    )
    series = list(zip(TIMES, styles))
    for (minutes, style), left in zip(series, (78, 197, 315, 423, 515)):
        if minutes:
            label = f"{minutes} 分钟" if zh else f"{minutes} min"
        else:
            label = "到期" if zh else "At expiration"
        legend(canvas, left, label, *style)
    x, y = plot_axes(
        canvas, (7500, 7650), (-510, 2350), range(7500, 7651, 25),
        (-330, 0, 500, 1000, 1500, 2000),
        "SPX 点位（到期曲线使用结算值）" if zh else "SPX level (settlement value for the expiration curve)",
        "每组盈亏（美元）" if zh else "Profit or loss per spread (USD)",
    )
    for strike in CALL:
        canvas.line(x(strike), y(-510), x(strike), y(2350), "f-lnrule",
                    1, ' stroke-dasharray="3 4"')
    levels = [7500 + index / 2 for index in range(301)]
    for minutes, style in reversed(series):
        curve(canvas, x, y, levels, lambda spot: dollars(fly_value(spot, minutes), CALL_DEBIT), *style)

    for tag, quote, minutes, color in (
        ("A", CALL_DEBIT, OPEN_MINUTES, "f-fillhl2"),
        ("B", CLOSE_QUOTE, CLOSE_MINUTES, "f-fillhl"),
    ):
        cx = x(spot_for_quote(quote, minutes))
        cy = y(dollars(quote, CALL_DEBIT))
        canvas.dot(cx, cy, 4.8, color)
        canvas.line(cx - 4, cy - 4, cx - 12, cy - 12, "f-ln", 0.8)
        canvas.text(cx - 15, cy - 9, f"{tag} · {quote:.2f}", 12, "f-t", "end", 600)
    return canvas


def combined_figure(zh):
    title = "两组蝶式同时持有到期的盈亏" if zh else "Holding both butterflies to expiration"
    label = (
        "Call 和 Put 蝶式各一条曲线，实线是合计。7,510 至 7,550 点合计亏损 530 美元。"
        "合计盈亏平衡点是 7,465.30、7,504.70、7,555.30 和 7,594.70。"
        if zh else
        "The solid line adds both butterflies. The combined loss is $530 from 7,510 to 7,550. "
        "The combined break-even levels are 7,465.30, 7,504.70, 7,555.30 and 7,594.70."
    )
    canvas = Canvas("combined-expiry", 720, 426, label)
    canvas.text(26, 34, title, 22, weight=600)
    legend(canvas, 78, "7,485 Put 蝶式" if zh else "7,485 put butterfly", "f-lnhl", "6 4")
    legend(canvas, 269, "7,575 Call 蝶式" if zh else "7,575 call butterfly", "f-lnalt", "2 4")
    legend(canvas, 469, "两组合计" if zh else "Combined", "f-lnink", "none", 2.8)
    x, y = plot_axes(
        canvas, (7435, 7625), (-1050, 2700), range(7440, 7621, 30),
        (-500, 0, 500, 1000, 1500, 2000, 2500),
        "SPX 到期结算值（点）" if zh else "SPX settlement value (points)",
        "盈亏（美元）" if zh else "Profit or loss (USD)",
    )
    gap_left, gap_right = PUT[-1], CALL[0]
    canvas.rect(x(gap_left), y(0), x(gap_right) - x(gap_left), y(-1050) - y(0),
                "f-altsoft", rx=0, sw=0)
    levels = sorted((7435, *PUT, *CALL, 7625))

    def put_pnl(spot):
        return dollars(expiration_value(spot, PUT), PUT_DEBIT)

    def call_pnl(spot):
        return dollars(expiration_value(spot, CALL), CALL_DEBIT)

    curve(canvas, x, y, levels, put_pnl, "f-lnhl", "6 4", 2.35)
    curve(canvas, x, y, levels, call_pnl, "f-lnalt", "2 4", 2.65)
    curve(canvas, x, y, levels, lambda spot: put_pnl(spot) + call_pnl(spot), "f-lnink", "none", 2.8)
    total_debit = CALL_DEBIT + PUT_DEBIT
    for low, center, high in (PUT, CALL):
        canvas.text(x(center), y(2490), f"{center:,}", 13, "f-t", "middle", 600)
        for zero, side in ((low + total_debit, -1), (high - total_debit, 1)):
            canvas.dot(x(zero), y(0), 3.8, "f-dotink")
            canvas.line(x(zero) + side * 3, y(0) - 4, x(zero) + side * 12, y(0) - 13, "f-ln", 0.8)
            canvas.text(x(zero) + side * 15, y(0) - 12, f"{zero:,.2f}",
                        11.5, "f-t", "start" if side > 0 else "end", 600)
    gap_middle = (x(gap_left) + x(gap_right)) / 2
    canvas.text(gap_middle, y(-769), "7,510–7,550", 11.3, "f-t", "middle")
    canvas.text(gap_middle, y(-985), "合计 −$530" if zh else "−$530 total", 12.1, "f-t", "middle", 600)
    return canvas


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix, zh in (("", False), (".zh", True)):
        for figure in (intraday_figure(zh), combined_figure(zh)):
            if figure.warnings:
                raise ValueError("\n".join(figure.warnings))
            (OUT / f"{figure.name}{suffix}.svg").write_text(figure.svg(), encoding="utf-8")

    for quote, minutes in ((CALL_DEBIT, OPEN_MINUTES), (CLOSE_QUOTE, CLOSE_MINUTES)):
        print(f"{minutes} min, {quote:.2f} quote: illustrative SPX {spot_for_quote(quote, minutes):,.2f}")
    for spot in (7535, CALL[1]):
        values = [fly_value(spot, minutes) for minutes in TIMES]
        print(f"SPX {spot:,} spread values: " + ", ".join(f"{value:.2f}" for value in values))


if __name__ == "__main__":
    main()

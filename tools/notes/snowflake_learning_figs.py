#!/usr/bin/env python3
"""Rebuild the English and Chinese Snowflake note figures using site colors."""

from pathlib import Path

from precision_vs_recall_figs import Canvas

OUT = Path(__file__).resolve().parents[2] / "content" / "notes" / "snowflake-learning-guide"


def architecture(t):
    c = Canvas("architecture", 660, 356, t(
        "Cloud services plan queries. A reporting warehouse and a transformation warehouse compute independently against shared database storage.",
        "云服务规划查询。报表仓库和加工仓库独立计算，读取同一份数据库存储。",
    ))
    for x, y1, y2 in [(330, 48, 72), (172, 140, 176), (488, 140, 176),
                      (172, 244, 280), (488, 244, 280)]:
        c.arrow(x, y1, x, y2, "f-lnink", "f-dotink", sw=1.3)
    c.text(330, 37, "Snowsight / SQL", 16, "f-t", "middle", weight=600)
    c.rect(28, 72, 604, 68, "f-node")
    c.text(50, 101, t("03   Cloud services", "03   云服务"), 19, weight=650)
    c.text(50, 124, t("Authentication · permissions · query planning", "身份认证 · 权限检查 · 查询规划"), 16)
    for x, title, subtitle in [
        (28, t("02   Reporting warehouse", "02   报表仓库"), t("Dashboard reads", "执行仪表盘查询")),
        (344, t("02   Transform warehouse", "02   加工仓库"), t("Loads and transformations", "执行装载与加工")),
    ]:
        c.rect(x, 176, 288, 68, "f-node")
        c.text(x + 22, 205, title, 18, weight=650)
        c.text(x + 22, 228, subtitle, 16)
    c.rect(28, 280, 604, 56, "f-hl")
    c.text(50, 315, t("01   Shared storage", "01   共享存储"), 19, weight=650)
    c.text(608, 315, "CORE.ORDERS", 16, "f-h", "end")
    return c


def pipeline(t):
    c = Canvas("pipeline", 660, 404, t(
        "Nine input events pass into RAW. One invalid amount stays inspectable as a reject. Three obsolete or repeated rows are excluded, leaving five orders in CORE. MART counts paid orders by order date.",
        "九条事件进入原始层。一条坏金额可在原始层查找。过滤三条旧版本或重发记录后，CORE 有五笔订单，MART 汇总每日已付款订单。",
    ))
    for y1, y2 in [(84, 116), (192, 224), (300, 332)]:
        c.arrow(215, y1, 215, y2, "f-lnink", "f-dotink", sw=1.3)
    c.arrow(400, 154, 442, 154, "f-lnink", "f-dotink", sw=1.3)
    c.rect(28, 24, 372, 60, "f-node")
    c.text(50, 51, t("Input: orders.csv or INSERT", "输入：orders.csv 或 INSERT"), 19, weight=650)
    c.text(50, 72, t("9 events in batch A", "批次 A：9 条事件"), 15)
    c.rect(28, 116, 372, 76, "f-node")
    c.text(50, 147, "RAW.ORDER_EVENTS", 19, weight=650)
    c.text(50, 174, t("Retain input · validate types and status", "保留原值 · 校验类型与状态"), 16)
    c.rect(442, 116, 190, 76, "f-node")
    c.text(460, 147, t("1 rejected event", "1 条坏记录"), 18, weight=650)
    c.text(460, 174, t("O1005 stays in RAW", "O1005 留在 RAW"), 15)
    c.rect(28, 224, 372, 76, "f-hl")
    c.text(50, 255, "CORE.ORDERS", 19, weight=650)
    c.text(50, 282, t("Rank by version, then MERGE · 5 orders", "按版本排名，再 MERGE · 5 笔订单"), 16)
    c.rect(28, 332, 372, 52, "f-node")
    c.text(50, 365, "MART.DAILY_PAID_SALES", 18, weight=650)
    c.text(424, 353, t("Paid count and amount", "按下单日汇总"), 17)
    c.text(424, 376, t("by original order date", "已付款笔数与金额"), 16)
    return c


def pruning(t):
    c = Canvas("pruning", 660, 318, t(
        "Four illustrated micro-partitions have different date ranges. A filter for November 2 overlaps only partition P3; Snowflake can skip P1, P2 and P4.",
        "四个微分区的日期范围各不相同。查询 11 月 2 日时，只有 P3 的日期范围有交集，可以跳过 P1、P2 和 P4。",
    ))
    c.text(28, 35, t("Filter: November 2, 2026", "筛选：2026 年 11 月 2 日"), 19, weight=650)
    c.rect(300, 56, 48, 244, "f-fillsoft", sw=0)
    days = ["10/30", "10/31", "11/01", "11/02", "11/03", "11/04", "11/05"]
    positions = [180 + 48 * n for n in range(len(days))]
    for n, (x, day) in enumerate(zip(positions, days)):
        c.text(x, 78, day, 14, "f-h" if n == 3 else "f-t", "middle", weight=600)
        c.line(x, 92, x, 288, "f-lnrule", 0.8)
    for n, (first, last) in enumerate([(0, 1), (2, 2), (3, 4), (5, 6)], start=1):
        y = 113 + (n - 1) * 54
        reading = n == 3
        stroke = "f-lnhl" if reading else "f-lnink"
        fill = "f-fillhl" if reading else "f-dotink"
        c.text(30, y + 6, t(f"Partition P{n}", f"微分区 P{n}"), 17, weight=600)
        c.line(positions[first], y, positions[last], y, stroke, 3)
        for index in {first, last}:
            c.dot(positions[index], y, r=5, cls=fill, ring=False)
        label = t("Read, then filter", "读取后筛行") if reading else t("Skip", "跳过")
        c.text(505, y + 6, label, 16, "f-h" if reading else "f-t", weight=650 if reading else 500)
    return c


if __name__ == "__main__":
    for locale in ("en", "zh"):
        def label(en, zh):
            return zh if locale == "zh" else en

        suffix = ".zh" if locale == "zh" else ""
        for draw in (architecture, pipeline, pruning):
            canvas = draw(label)
            if canvas.warnings:
                raise SystemExit("\n".join(canvas.warnings))
            (OUT / f"{canvas.name}{suffix}.svg").write_text(canvas.svg(), encoding="utf-8")

#!/usr/bin/env python3
"""Rebuild the three theme-aware figures for the bilingual Databricks note."""

from pathlib import Path

from precision_vs_recall_figs import Canvas


OUT = Path(__file__).resolve().parents[2] / "content" / "notes" / "databricks-guide"


def platform(zh):
    def t(en, cn):
        return cn if zh else en

    c = Canvas("platform-map", 720, 328, t(
        "Notebooks, SQL queries and dashboards submit work to compute, which reads the "
        "Delta log and data files in object storage. Unity Catalog governs access. "
        "Lakeflow Jobs schedules work.",
        "Notebook、SQL 查询和仪表板将工作交给计算资源。计算资源读取对象存储里的 Delta 日志和数据文件。"
        "Unity Catalog 管理访问，Lakeflow Jobs 调度任务。",
    ))
    c.text(26, 37, t("What runs the work? What keeps the data?", "谁负责算？谁负责存？"), 23, weight=650)

    c.rect(24, 62, 672, 46, "f-hl")
    c.text(360, 91, t("Unity Catalog: names, permissions and lineage",
                      "Unity Catalog：对象命名、权限与数据血缘"), 16, "f-t", "middle", 600)
    for x in (358, 598):
        c.line(x, 108, x, 138, "f-ln", 1.4, ' stroke-dasharray="3 4"')

    for x, title, lines in (
        (24, t("Author & consume", "写查询、看结果"),
         ("Notebook", t("SQL editor", "SQL 查询编辑器"), t("Dashboard", "仪表板"))),
        (262, t("Run the work", "执行计算"),
         (t("Notebook compute", "Notebook 计算资源"), t("SQL warehouse", "SQL Warehouse"),
          t("Query and transform", "执行查询和转换"))),
        (500, t("Keep the data", "保存数据"),
         (t("Cloud object storage", "云对象存储"), t("Delta transaction log", "Delta 事务日志"),
          t("Parquet data files", "Parquet 数据文件"))),
    ):
        c.rect(x, 138, 196, 113, "f-node")
        c.text(x + 15, 166, title, 17, weight=650)
        for i, line in enumerate(lines):
            c.text(x + 15, 191 + i * 20, line, 13.5, "f-t", box=169)
    c.arrow(224, 193, 257, 193)
    c.arrow(462, 193, 495, 193)

    c.line(358, 252, 358, 272, "f-ln", 1.4, ' stroke-dasharray="3 4"')
    c.rect(24, 272, 672, 37, "f-node")
    c.text(360, 295, t("Lakeflow Jobs: runs dependent tasks on a schedule or trigger",
                       "Lakeflow Jobs：按计划或触发条件运行有依赖的任务"), 14, "f-t", "middle", 550)
    return c


def payments(zh):
    def t(en, cn):
        return cn if zh else en

    c = Canvas("payments-flow", 720, 326, t(
        "Bronze retains ten deliveries. Validation routes a malformed amount to rejects. "
        "Silver keeps five current payments. The Gold view groups by day and channel. "
        "Approved volume falls from 7999 to 7699 cents after batch two.",
        "Bronze 保留十条投递，其中一条因金额无效进入拒绝记录。Silver 保留五笔当前支付。"
        "Gold 按日期和渠道汇总。第二批以后，获批金额由 7,999 美分降为 7,699 美分。",
    ))
    c.text(26, 37, t("Follow a payment through the lab", "跟踪一笔支付经过三层表"), 23, weight=650)
    for x, title, subtitle, example in (
        (24, "Bronze", t("Keep raw deliveries", "原样保留投递"), "p01 v1 × 2"),
        (262, "Silver", t("Latest valid revision", "保留最新的有效版本"), "p01 v1 × 1"),
        (500, "Gold", t("Day × channel", "每行一个日期 × 渠道"), t("Counts and volume", "汇总笔数和获批金额")),
    ):
        c.rect(x, 70, 196, 105, "f-hl" if title == "Silver" else "f-node")
        c.text(x + 15, 100, title, 20, weight=650)
        c.text(x + 15, 125, subtitle, 14, box=169)
        c.text(x + 15, 149, example, 13.5, "f-m", box=169)
    c.arrow(224, 124, 257, 124)
    c.arrow(462, 124, 495, 124)
    c.path([(242, 128), (242, 197), (122, 197)], "f-lnalt", 1.6)
    c.arrow(122, 197, 122, 228, "f-lnalt", "f-alt")
    c.rect(24, 231, 196, 70, "f-altsoft")
    c.text(39, 256, t("Rejects: p_bad", "拒绝记录：p_bad"), 15, weight=650)
    c.text(39, 281, t("Invalid amount", "金额无法解析"), 13.5, box=169)

    c.arrow(598, 179, 598, 210)
    c.rect(262, 213, 434, 88, "f-node")
    c.line(479, 230, 479, 284)
    c.text(281, 245, t("Before batch 2", "第二批之前"), 15, weight=650)
    c.text(494, 245, t("After batch 2", "第二批之后"), 15, weight=650)
    c.text(281, 268, t("4 payments", "4 笔支付"), 13.5, "f-m")
    c.text(281, 288, t("7,999¢ approved", "获批 7,999 美分"), 14, weight=600)
    c.text(494, 268, t("5 payments", "5 笔支付"), 13.5, "f-m")
    c.text(494, 288, t("7,699¢ approved", "获批 7,699 美分"), 14, weight=600)
    return c


def availability(zh):
    def t(en, cn):
        return cn if zh else en

    c = Canvas("feature-availability", 720, 310, t(
        "For a decision at 10:00, use the purchase that occurred at 09:45 and arrived at 09:46. "
        "Exclude the purchase that occurred at 09:55 but did not arrive until 10:05.",
        "10:00 的支付决策可以用 09:45 发生且 09:46 已收到的交易，"
        "不能用 09:55 发生但 10:05 才收到的交易。",
    ))
    c.text(26, 39, t("Which purchase was known at 10:00?", "10:00 做决策时，已经知道哪一笔？"), 23, weight=650)
    c.text(26, 65, t("Availability time determines what the model can use.",
                     "模型能用什么，取决于数据何时可用。"), 14, "f-m")

    def x(minute_after_0945):
        return 199 + minute_after_0945 * 23

    cut = x(15)
    c.line(cut, 110, cut, 251, "f-lnink", 1.3, ' stroke-dasharray="3 5"')
    c.text(cut, 99, t("10:00 decision", "10:00 决策"), 14, "f-t", "middle", 650)
    for y, start, end, name, occurred, available, result in (
        (149, 0, 1, t("Purchase A", "交易 A"), "09:45", "09:46", t("Use", "可用")),
        (224, 10, 20, t("Purchase B", "交易 B"), "09:55", "10:05", t("Too late", "来不及")),
    ):
        color = "f-lnhl" if end < 15 else "f-lnalt"
        fill = "f-fillhl" if end < 15 else "f-alt"
        c.text(26, y - 4, name, 16, weight=650)
        c.text(26, y + 18, result, 14, "f-m")
        c.line(173, y, 684, y)
        c.line(x(start), y, x(end), y, color, 2.5)
        c.dot(x(start), y, r=5, cls="f-hollow", ring=False)
        c.dot(x(end), y, r=5, cls=fill)
        c.text(x(start) - 4, y - 13, occurred, 13.5, "f-t", "end")
        c.text(x(end), y + 22, available, 13.5, "f-t", "middle")

    c.dot(198, 277, r=5, cls="f-hollow", ring=False)
    c.text(212, 282, t("Event occurred", "事件发生"), 13.5, "f-m")
    c.dot(405, 277)
    c.text(419, 282, t("Record became available", "记录可用"), 13.5, "f-m")
    return c


def main():
    for zh in (False, True):
        for draw in (platform, payments, availability):
            c = draw(zh)
            if c.warnings:
                raise ValueError("\n".join(c.warnings))
            suffix = ".zh.svg" if zh else ".svg"
            (OUT / (c.name + suffix)).write_text(c.svg(), encoding="utf-8")


if __name__ == "__main__":
    main()

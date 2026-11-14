#!/usr/bin/env python3
"""Generate theme-aware diagrams for both versions of the Airflow note."""

from pathlib import Path

from precision_vs_recall_figs import Canvas

OUT = Path(__file__).resolve().parents[2] / "content" / "notes" / "airflow"


def architecture(t):
    c = Canvas("orchestration", 680, 317, t(
        "A Dag file is parsed. The scheduler uses the executor to start tasks. "
        "The task can submit computation to Spark, which reads and writes data independently. "
        "Airflow records orchestration state in its metadata database.",
        "Dag processor 读取定义。Scheduler 通过 executor 启动任务；任务可提交 Spark 作业，"
        "由 Spark 读写数据。Airflow 在元数据库记录调度状态。",
    ))
    c.text(24, 37, t("Airflow decides when the work runs", "Airflow 决定工作何时运行"), 23, weight=650)
    for x, title, detail, cls in (
        (26, t("Dag processor", "Dag processor"), t("Reads the Dag file", "解析 Dag 定义"), "f-node"),
        (250, t("Scheduler", "Scheduler"), t("Uses the executor", "由 executor 派发"), "f-hl"),
        (474, t("Task runtime", "任务执行环境"), t("Runs or submits work", "执行或提交计算"), "f-node"),
    ):
        c.rect(x, 67, 180, 68, cls)
        c.text(x + 90, 94, title, 18, anchor="middle", weight=650, box=163)
        c.text(x + 90, 117, detail, 14, "f-m", "middle", box=163)
    c.arrow(208, 101, 247, 101)
    c.arrow(432, 101, 471, 101)
    c.line(116, 136, 116, 185, "f-ln", 1.5, ' stroke-dasharray="3 4"')
    c.line(340, 136, 340, 185, "f-ln", 1.5, ' stroke-dasharray="3 4"')
    c.rect(26, 186, 404, 59, "f-node")
    c.text(228, 210, t("Metadata database", "元数据库"), 17, anchor="middle", weight=650)
    c.text(228, 231, t("Dag definitions, run and task states", "Dag 定义、运行与任务状态"), 14, "f-m", "middle")
    c.arrow(564, 137, 564, 182, "f-lnhl", "f-headhl")
    c.rect(474, 186, 180, 59, "f-hl")
    c.text(564, 210, t("Spark / SQL engine", "Spark / SQL 引擎"), 17, anchor="middle", weight=650)
    c.text(564, 231, t("Reads and writes data", "读取和写入业务数据"), 14, "f-m", "middle")
    c.arrow(564, 247, 564, 265)
    c.rect(474, 268, 180, 31, "f-node")
    c.text(564, 288, t("Object store / tables", "对象存储 / 数据表"), 14, anchor="middle")
    c.text(26, 285, t("The UI lets you inspect runs, task states and logs.", "在界面上查看运行记录、任务状态和日志。"), 14, "f-m", box=400)
    return c


def intervals(t):
    c = Canvas("cohort-clock", 680, 318, t(
        "The November 1 cohort spans November 1 to November 2 at midnight UTC. "
        "The last user's 24-hour window ends late on November 2. Airflow uses the "
        "November 2 to November 3 interval and starts the run after November 3 midnight.",
        "11 月 1 日队列在 11 月 1 日至 11 月 2 日 UTC 零点入组。最晚用户的 24 小时窗口在 "
        "11 月 2 日深夜结束。Airflow 用 11 月 2 日至 11 月 3 日这一数据区间，至 11 月 3 日零点后运行。",
    ))
    c.text(24, 37, t("One assignment day, one more day of follow-up", "入组一天，再多留一天追踪结果"), 22, weight=650)
    x0, x1, x2 = 204, 413, 622
    for x, label in ((x0, "11-01 00:00"), (x1, "11-02 00:00"), (x2, "11-03 00:00")):
        c.text(x, 77, label, 14, anchor="middle", weight=650)
        c.line(x, 87, x, 258, "f-ln", 1.0, ' stroke-dasharray="3 5"')
    c.text(26, 116, t("Assignment cohort", "入组日期范围"), 15, weight=650)
    c.rect(x0, 97, x1 - x0, 29, "f-hl", rx=3)
    c.text((x0 + x1) / 2, 117, t("Nov 1 arrivals", "11 月 1 日入组"), 14, anchor="middle")
    c.text(26, 166, t("Late user's 24 h", "最晚用户的 24 小时"), 15, weight=650)
    late = (x1 - x0) * (23 / 24)
    c.line(x0 + late, 162, x1 + late, 162, "f-lnink", 3)
    c.dot(x0 + late, 162, 5, "f-dotink", ring=False)
    c.dot(x1 + late, 162, 5, "f-hollow", ring=False)
    c.text((x0 + x1) / 2 + late, 187, "[23:00 → 23:00)", 13, "f-m", "middle")
    c.text(26, 224, t("Dag data interval", "Dag 数据区间"), 15, weight=650)
    c.rect(x1, 203, x2 - x1, 30, "f-node", rx=3)
    c.text((x1 + x2) / 2, 223, t("Nov 2 interval", "11 月 2 日的数据区间"), 14, anchor="middle")
    c.arrow(x2, 241, x2, 263, "f-lnhl", "f-headhl")
    c.rect(24, 267, 632, 32, "f-hl")
    c.text(340, 288, t("Start after Nov 3 midnight; publish only if source is complete.",
                       "11 月 3 日零点后调度；确认源数据完整才能发布。"),
           14, anchor="middle")
    return c


def tasks(t):
    c = Canvas("dag-recovery", 680, 332, t(
        "plan_window passes its window to compute_counts and publish. An incomplete source "
        "stops compute_counts. A transient write error can retry publish. The same "
        "cohort date always maps to the same output file.",
        "plan_window 把区间交给 compute_counts 和 publish。数据不完整会阻止计算，"
        "临时写入问题可重试 publish。同一天的队列始终写到同一路径。",
    ))
    c.text(24, 37, t("One Dag run, three task instances", "一次 Dag run，三个任务实例"), 23, weight=650)
    for x, code, line in (
        (25, "plan_window", t("Date and source cutoff", "确定日期及完整性时间")),
        (249, "compute_counts", t("Count assigned users", "按用户计算转化")),
        (473, "publish", t("Replace one output", "替换一个结果文件")),
    ):
        c.rect(x, 78, 182, 73, "f-hl" if code == "compute_counts" else "f-node")
        c.text(x + 91, 107, code, 17, anchor="middle", weight=650, box=171)
        c.text(x + 91, 132, line, 14, "f-m", "middle", box=171)
    c.arrow(209, 116, 246, 116)
    c.arrow(433, 116, 470, 116)
    c.path([(116, 77), (116, 58), (564, 58)], "f-ln", 1.4)
    c.arrow(564, 58, 564, 75)
    c.arrow(340, 153, 340, 181, "f-lnalt", "f-alt")
    c.rect(249, 184, 182, 59, "f-altsoft")
    c.text(340, 209, t("Source incomplete?", "源数据未到齐？"), 16, anchor="middle", weight=650)
    c.text(340, 232, t("Stop before publishing", "先停止，不发布"), 14, anchor="middle")
    c.arrow(564, 153, 564, 181)
    c.rect(473, 184, 182, 59, "f-node")
    c.text(564, 209, t("Temporary I/O error?", "临时写入错误？"), 16, anchor="middle", weight=650)
    c.text(564, 232, t("Retry the same output", "重试写同一结果"), 14, anchor="middle")
    c.rect(25, 268, 630, 44, "f-hl")
    c.text(340, 295, t("Output key: checkout-lab/2026-11-01.json", "输出键：checkout-lab/2026-11-01.json"),
           17, anchor="middle", weight=650)
    return c


def main():
    for zh in (False, True):
        def t(en, cn):
            return cn if zh else en

        for draw in (architecture, intervals, tasks):
            c = draw(t)
            if c.warnings:
                raise ValueError("\n".join(c.warnings))
            suffix = ".zh.svg" if zh else ".svg"
            (OUT / (c.name + suffix)).write_text(c.svg(), encoding="utf-8")


if __name__ == "__main__":
    main()

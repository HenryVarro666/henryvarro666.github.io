#!/usr/bin/env python3
"""Regenerate the bilingual dbt note figures and public lab download."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from precision_vs_recall_figs import Canvas

TOOLS = Path(__file__).resolve().parent
OUT = TOOLS.parents[1] / "content" / "notes" / "dbt-learning-guide"
LAB = TOOLS / "dbt_orders_lab"


def box(c, x, y, w, h, title, subtitle, focus=False):
    c.rect(x, y, w, h, "f-hl" if focus else "f-node")
    c.text(x + w / 2, y + (26 if subtitle else h / 2 + 5), title,
           15.5, anchor="middle", weight=650, box=w - 18)
    if subtitle:
        c.text(x + w / 2, y + 48, subtitle, 13.5, "f-m", "middle", box=w - 18)


def elt(t):
    c = Canvas("elt-boundary", 720, 255, t(
        "An ingestion tool copies data into raw tables. dbt compiles model files and sends "
        "SQL to the warehouse or DuckDB, which creates models read by reports.",
        "采集工具将源数据写进原始表。dbt 编译模型并提交 SQL，由数据仓库或 DuckDB 执行。报表读取建好的模型。",
    ))
    c.text(24, 38, t("Where dbt fits in ELT", "dbt 在 ELT 的哪一步？"), 24, weight=650)
    c.rect(198, 125, 340, 108, "f-node", sw=1)
    c.text(215, 216, t("Warehouse or local DuckDB", "数据仓库或本地 DuckDB"), 13.5, "f-m")
    box(c, 288, 60, 160, 62, "dbt", t("Compiles model code", "编译模型代码"), True)
    c.arrow(438, 125, 438, 144, "f-lnhl", "f-headhl")
    box(c, 22, 148, 144, 60, t("Ingestion", "采集工具"), t("Extract + load", "提取并加载"))
    box(c, 211, 148, 135, 50, t("Raw tables", "原始表"), "")
    box(c, 386, 148, 137, 50, t("Built models", "转换后的表"), "", True)
    box(c, 570, 148, 128, 60, t("Reports", "报表"), t("Read models", "查询模型"))
    c.arrow(170, 173, 207, 173)
    c.arrow(350, 173, 382, 173, "f-lnhl", "f-headhl")
    c.arrow(527, 173, 566, 173)
    return c


def lineage(t):
    c = Canvas("orders-lineage", 720, 357, t(
        "Payment deliveries go through staging, then latest revision per payment, "
        "then per-order totals joined with staged orders. The order fact feeds a daily mart. "
        "A status lookup seed also feeds per-order totals.",
        "支付投递先清洗，再选每笔支付的最新业务版本。按订单汇总后与订单关联，再由订单事实表生成日报。状态对照表参与支付汇总。",
    ))
    c.text(24, 36, t("The orders lab: from sources to the mart", "订单实验：从原始表到日报"), 23, weight=650)
    for y, title, sub in (
        (71, "raw.payment_events", t("One row per delivery", "每行一次投递")),
        (179, "raw.orders", t("One row per order", "每行一笔订单")),
    ):
        box(c, 24, y, 177, 64, title, sub)
    for y, title, sub in (
        (71, "stg_payments", t("Normalize status", "统一状态写法")),
        (179, "stg_orders", t("Normalize channel", "统一渠道写法")),
    ):
        box(c, 244, y, 174, 64, title, sub)
        c.arrow(205, y + 32, 240, y + 32)
    box(c, 477, 71, 219, 64, "int_payments_current", t("One row per payment", "每行一笔支付"))
    box(c, 477, 179, 219, 64, "int_payments_by_order", t("One row per order", "每行一笔订单"))
    box(c, 477, 270, 219, 64, "fct_orders", t("Keeps unpaid orders", "保留未入账的订单"), True)
    box(c, 244, 270, 174, 64, "mart_daily_orders", t("Day × channel", "每行一个日期 × 渠道"), True)
    c.arrow(422, 103, 473, 103)
    c.arrow(587, 139, 587, 175)
    c.rect(244, 143, 174, 28, "f-node", sw=1)
    c.text(331, 161, "payment_statuses", 13, anchor="middle", weight=550, box=156)
    c.path([(422, 157), (456, 157), (456, 195), (464, 195)], "f-ln", 1.5)
    c.arrow(464, 195, 473, 195)
    c.arrow(603, 247, 603, 266)
    c.path([(422, 211), (447, 211), (447, 283), (462, 283)], "f-ln", 1.5)
    c.arrow(462, 283, 473, 283)
    c.arrow(473, 302, 422, 302, "f-lnhl", "f-headhl")
    return c


def arrivals(t):
    c = Canvas("late-arrival", 720, 278, t(
        "Payment p02 occurred January 1. Revision 1 was loaded January 1, then a replay "
        "arrived January 4. Revision 2 arrived January 5, after the first build, and must "
        "replace revision 1 without changing the order's reporting date.",
        "p02 于 1 月 1 日发生；初版同日入库，1 月 4 日重送。更正于 1 月 5 日才入库，应替换旧版本，订单归属日期仍为 1 月 1 日。",
    ))
    c.text(24, 37, t("A January 1 payment changes on January 5", "1 月 1 日的支付，1 月 5 日才收到更正"), 23, weight=650)
    for y, label in ((108, t("Event date", "业务日期")), (188, t("Load date", "入库日期"))):
        c.text(24, y + 5, label, 15, weight=650)
        c.line(166, y, 685, y)
    for x, date in ((189, "01-01"), (308, "01-02"), (427, "01-03"), (546, "01-04"), (665, "01-05")):
        for y in (108, 188):
            c.line(x, y - 4, x, y + 4)
        c.text(x, 220, date, 13, "f-m", "middle")
    c.dot(189, 108, cls="f-fillhl", ring=False)
    c.text(204, 96, "p02", 15, weight=650)
    c.path([(665, 177), (665, 116), (211, 116)], "f-lnhl", 1.5)
    c.arrow(211, 116, 194, 116, "f-lnhl", "f-headhl")
    c.text(507, 99, t("Same business date", "业务日期不变"), 14, "f-t", "middle")
    for x, title in (
        (189, t("rev 1 · 1,500¢", "第 1 版 · 1,500¢")),
        (546, t("rev 1 replay", "第 1 版重送")),
        (665, t("rev 2 · 1,200¢", "第 2 版 · 1,200¢")),
    ):
        c.dot(x, 188, cls="f-fillhl" if x == 665 else "f-dotink", ring=False)
        c.text(x + (22 if x == 189 else 0), 169, title, 13.5, "f-t",
               "end" if x == 665 else "middle")
    c.text(24, 253, t("Use load time to find work and the source revision to pick the winner.",
                     "用入库时间找出发生变动的键，再用业务版本选出当前记录。"), 15, "f-m")
    return c


def release(t):
    c = Canvas("release-path", 720, 238, t(
        "Developers build in dev. A pull request builds in an isolated CI schema. "
        "Only after review do scheduled jobs write to production and expose approved marts.",
        "开发者在 dev 构建，变更先在独立 CI schema 验证；代码合并后，由正式作业写入生产并对下游开放通过检查的报表表。",
    ))
    c.text(24, 39, t("From a model change to a scheduled result", "SQL 修改怎样进入每天运行的作业"), 23, weight=650)
    for i, (title, subtitle, detail) in enumerate((
        (t("Develop", "开发"), t("Personal schema", "个人 schema"), t("Preview changed SQL", "预览变更结果")),
        (t("Check the PR", "验证变更"), t("Isolated CI schema", "独立 CI schema"), t("Build and test", "构建并测试")),
        (t("Run production", "生产运行"), t("Scheduled job", "定时作业"), t("Check freshness", "检查源表时效")),
        (t("Publish", "供下游使用"), t("Approved marts", "通过校验的表"), t("Audit the run", "保留运行记录")),
    )):
        x = 24 + i * 173
        c.rect(x, 80, 153, 108, "f-hl" if i == 1 else "f-node")
        c.text(x + 12, 107, title, 16, weight=650, box=129)
        c.text(x + 12, 136, subtitle, 13, box=129)
        c.text(x + 12, 160, detail, 12.5, "f-m", box=129)
        if i < 3:
            c.arrow(x + 156, 134, x + 170, 134)
    c.text(24, 215, t("Failed checks stop promotion; a failed job can still leave partial writes.",
                     "检查失败就停止发布；作业失败时，已经写入的表不一定会回滚。"), 14, "f-m")
    return c


def package_lab():
    files = [LAB / name for name in (
        ".gitignore", "README.md", "dbt_project.yml", "profiles.yml",
        "requirements.txt", "load_data.py",
    )]
    for folder, suffixes in (
        ("data", {".sql"}), ("models", {".sql", ".yml"}), ("seeds", {".csv"}),
        ("snapshots", {".yml"}), ("tests", {".sql"}),
    ):
        files.extend(p for p in (LAB / folder).rglob("*") if p.suffix in suffixes)
    with ZipFile(OUT / "dbt-orders-lab.zip", "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(files):
            name = f"dbt-orders-lab/{path.relative_to(LAB).as_posix()}"
            archive.writestr(ZipInfo(name, date_time=(2026, 11, 13, 0, 0, 0)),
                             path.read_bytes(), compress_type=ZIP_DEFLATED)


def main():
    for language in ("en", "zh"):
        def translate(en, zh):
            return zh if language == "zh" else en

        for draw in (elt, lineage, arrivals, release):
            c = draw(translate)
            if c.warnings:
                raise ValueError("\n".join(c.warnings))
            suffix = ".zh.svg" if language == "zh" else ".svg"
            (OUT / (c.name + suffix)).write_text(c.svg(), encoding="utf-8")
    package_lab()


if __name__ == "__main__":
    main()

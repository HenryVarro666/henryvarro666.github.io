#!/usr/bin/env python3
"""Regenerate the bilingual MLOps note diagrams and clean educational lab ZIP."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from precision_vs_recall_figs import Canvas

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "content/notes/mlops-learning-guide"
LAB = ROOT / "tools/notes/mlops_risk_lab"


def node(c, x, y, w, title, sub, focus=False, h=76):
    c.rect(x, y, w, h, "f-hl" if focus else "f-node")
    c.text(x + 14, y + 29, title, 16, weight=650, box=w - 28)
    c.text(x + 14, y + 55, sub, 14, "f-m", box=w - 28)


def lifecycle(t):
    c = Canvas("mlops-lifecycle", 800, 402, t(
        "Validated versioned data enters training, evaluation and an approved release. "
        "A serving system produces decisions. Operational signals and delayed labels "
        "feed investigation and a new candidate, which must pass the release gates again.",
        "校验并版本化的数据进入训练和评估，通过发布检查后用于推理。服务信号和延迟标签进入调查，新候选仍须重新通过发布检查。"))
    c.text(24, 38, t("A model runs inside a changing system", "模型运行在持续变化的系统里"), 25, weight=650)
    labels = [
        (t("Data contract", "数据约定"), t("Schema + snapshot", "字段规则与数据快照")),
        (t("Train + evaluate", "训练与评估"), t("Split + fit + compare", "划分、拟合、比较")),
        (t("Approve a release", "批准一个版本"), t("Bundle + checks", "完整发布包与检查")),
        (t("Serve decisions", "服务业务决策"), t("Batch or API", "批量任务或 API")),
    ]
    for i, (title, sub) in enumerate(labels):
        x = 24 + i * 194
        node(c, x, 89, 171, title, sub, i == 2)
        if i < 3:
            c.arrow(x + 175, 126, x + 190, 126)
    node(c, 485, 224, 291, t("Observe outcomes", "观察实际结果"),
         t("Errors, latency, inputs, mature labels", "错误、延迟、输入、成熟标签"))
    node(c, 24, 224, 344, t("Investigate before retraining", "先调查，再决定是否重训"),
         t("Data bug? Serving bug? A new population?", "数据错误？服务故障？人群变了？"), True)
    c.arrow(691, 170, 691, 219)
    c.arrow(480, 261, 373, 261)
    c.path([(196, 220), (196, 194), (303, 194)], "f-lnhl", 1.5)
    c.arrow(303, 194, 303, 170, "f-lnhl", "f-headhl")
    c.text(24, 351, t("A drift alert opens an investigation. It does not approve a replacement.",
                      "漂移告警触发调查；替换模型仍需独立评估与批准。"), 16, "f-m", box=750)
    c.text(24, 377, t("The release loop includes the data and the code that transforms it.",
                      "数据及其转换代码，也属于需要维护和追踪的对象。"), 15, "f-m", box=750)
    return c


def lineage(t):
    c = Canvas("release-lineage", 800, 388, t(
        "A release manifest binds data snapshot, code commit, environment and evaluation "
        "to a bundle of preprocessing, model, threshold and feature schema. The serving "
        "process resolves the approved alias to one immutable bundle and reports its version.",
        "发布清单将数据快照、代码版本、环境和评估结果绑定到同一个包，包中包含预处理、模型、阈值和特征约定。服务启动时解析获批别名，并返回实际版本。"))
    c.text(24, 38, t("What exactly did we deploy?", "我们到底部署了什么？"), 25, weight=650)
    inputs = [
        (t("Data snapshot", "数据快照"), "data hash / version"),
        (t("Code revision", "代码版本"), "commit / source hash"),
        (t("Runtime", "运行环境"), "lockfile / image digest"),
    ]
    for i, (title, sub) in enumerate(inputs):
        node(c, 24, 74 + i * 93, 217, title, sub, h=73)
        c.arrow(246, 110 + i * 93, 290, 110 + i * 93)
    c.rect(296, 74, 264, 259, "f-hl")
    c.text(313, 105, t("Immutable release bundle", "不可变发布包"), 18, weight=650, box=231)
    for i, label in enumerate([
        t("Preprocessing + model weights", "预处理与模型参数"),
        t("Feature schema + threshold", "特征约定与决策阈值"),
        t("Evaluation + approval evidence", "评估结果与批准依据"),
        t("Manifest binds the versions", "清单关联上述各项版本"),
    ]):
        c.text(313, 149 + i * 43, label, 14.5, box=231)
    node(c, 604, 103, 172, "champion", t("Approved alias", "获批版本的别名"))
    node(c, 604, 241, 172, t("Serving process", "推理服务进程"), t("Reports release ID", "返回实际版本号"))
    c.arrow(565, 142, 599, 142)
    c.arrow(690, 184, 690, 236)
    c.text(24, 367, t("Rollback restores a compatible bundle, including its decision rule.",
                      "回滚恢复兼容的完整包，包括当时使用的决策规则。"), 15, "f-m", box=750)
    return c


def time_rules(t):
    c = Canvas("time-and-leakage", 800, 388, t(
        "At prediction time, a feature must have both occurred and become available. "
        "Labels are usable for training only when mature and available before the training "
        "cutoff. Chronological validation and test windows follow training; group leakage "
        "needs a separate constraint.",
        "预测特征既要已经发生，也要已可用。训练标签须在训练截止前成熟且可用。训练、验证和测试沿时间排序，实体或患者泄漏仍须另行控制。"))
    c.text(24, 38, t("Two clocks can leak tomorrow into today", "两种时间都可能把未来泄漏到今天"), 25, weight=650)
    c.text(24, 91, t("One prediction", "一次预测"), 16, weight=650)
    c.line(177, 127, 755, 127)
    for x, label in [(217, t("Event", "事件发生")), (421, t("Prediction", "作出预测")),
                     (680, t("Outcome", "结果发生"))]:
        c.dot(x, 127, ring=False)
        c.text(x, 105, label, 14.5, anchor="middle")
    c.path([(217, 144), (217, 166), (480, 166)], "f-ln", 1.5)
    c.dot(490, 166, cls="f-dotink", ring=False)
    c.text(504, 172, t("Late arrival: exclude", "晚到的数据：不能用"), 14.5, box=272)
    c.text(24, 210, t("Training cutoff: use only labels already mature and available.",
                      "训练截止时，只纳入已成熟、已到达的标签。"), 16, "f-m", box=750)
    for x, w, title, sub, hl in [
        (24, 297, t("Train", "训练集"), t("Fit transformations here", "只在这里拟合预处理"), False),
        (339, 201, t("Validation", "验证集"), t("Choose threshold", "选择模型与阈值"), True),
        (558, 218, t("Test", "测试集"), t("Report held-out result", "最后报告保留集结果"), False),
    ]:
        node(c, x, 245, w, title, sub, hl)
    c.text(24, 361, t("Time order does not prevent the same patient appearing in both sets.",
                      "按时间切分，也不保证同一患者不会跨集合出现。"), 15, "f-m", box=750)
    return c


def release(t):
    c = Canvas("safe-release", 800, 354, t(
        "A candidate bundle passes independent gates before a staged rollout. Old and new "
        "versions remain available. A health failure stops the rollout and restores the old "
        "compatible bundle. Shadow evaluates behavior without executing candidate actions.",
        "候选包先通过独立检查，再分阶段上线。保留新旧版本；出现异常时停止放量并恢复旧兼容包。影子模式只观察候选结果，不执行候选业务动作。"))
    c.text(24, 38, t("Approval and rollout are separate decisions", "通过评估与扩大流量，是两次决定"), 25, weight=650)
    specs = [
        (t("Candidate", "候选包"), t("Fixed bundle", "固定内容与版本")),
        (t("Release gates", "发布检查"), t("Schema + quality + cost", "字段、质量、成本")),
        (t("Staged rollout", "分阶段上线"), t("Observe real traffic", "观察真实流量")),
    ]
    for i, (title, sub) in enumerate(specs):
        x = 24 + i * 264
        node(c, x, 89, 224, title, sub, i == 1)
        if i < 2:
            c.arrow(x + 230, 126, x + 258, 126)
    node(c, 552, 222, 224, t("Keep / expand", "保留并扩大"), t("Only if guardrails hold", "指标符合条件才继续"), True)
    node(c, 24, 222, 375, t("Previous compatible release", "上一份兼容的发布包"),
         t("Restore bundle + routing, then verify", "恢复完整包与路由，再验证服务"))
    c.arrow(664, 170, 664, 217)
    c.path([(550, 151), (507, 151), (507, 259)], "f-ln", 1.5)
    c.arrow(507, 259, 404, 259)
    c.text(431, 202, t("Failure", "异常"), 14, "f-m")
    c.text(24, 330, t("Shadow: no candidate actions. Canary: limited exposure. A/B: estimate an effect.",
                      "Shadow 不执行候选动作；Canary 控制风险；A/B 估计业务效果。"), 15, "f-m", box=750)
    return c


def monitoring(t):
    c = Canvas("monitoring-layers", 800, 412, t(
        "Monitor service health immediately, inputs and predictions soon after requests, "
        "and outcome quality only when labels mature. Input drift may or may not coincide "
        "with performance degradation; neither condition proves the other.",
        "服务健康可立即监控，输入与预测随后可观察，效果要等标签成熟。输入漂移与性能下降可能同时发生，也可能各自出现，不能互相证明。"))
    c.text(24, 38, t("What can we know before the labels arrive?", "标签回来前，我们能知道什么？"), 25, weight=650)
    rows = [
        (t("Immediately", "立即"), t("Availability, errors, latency, saturation", "可用性、错误、延迟、资源饱和")),
        (t("After requests", "请求之后"), t("Schema, missingness, input and score shifts", "字段、缺失率、输入与分数变化")),
        (t("After label maturity", "标签成熟后"), t("Recall, precision, calibration, business cost", "召回率、精确率、校准与业务成本")),
    ]
    for i, (label, detail) in enumerate(rows):
        y = 78 + i * 57
        c.rect(24, y, 752, 46, "f-hl" if i == 2 else "f-node")
        c.text(39, y + 29, label, 15, weight=650, box=205)
        c.text(259, y + 29, detail, 14.5, box=500)
    c.text(24, 292, t("Input drift", "输入漂移"), 16, weight=650)
    c.text(224, 292, t("may happen without", "可能出现，但没有"), 15, "f-m")
    c.text(485, 292, t("Quality loss", "效果下降"), 16, weight=650)
    c.arrow(184, 286, 214, 286)
    c.arrow(449, 286, 478, 286)
    c.text(24, 329, t("Quality loss", "效果下降"), 16, weight=650)
    c.text(224, 329, t("may happen without", "可能出现，但没有"), 15, "f-m")
    c.text(485, 329, t("Input drift", "输入漂移"), 16, weight=650)
    c.arrow(184, 323, 214, 323)
    c.arrow(449, 323, 478, 323)
    c.text(24, 383, t("Compare mature cohorts and slices. Every alert needs an owner and a response.",
                      "比较已成熟的同口径队列及切片；为告警指定负责人和处理动作。"), 15, "f-m", box=750)
    return c


def package_lab():
    """Explicit source allowlist keeps run artifacts and local credentials out of ZIP."""
    files = [".gitignore", "README.md", "requirements.txt", "requirements.lock", "lab.py", "api.py",
             "Dockerfile", "requirements-mlflow.txt", "requirements-mlflow.lock", "ci.example.yml"]
    sources = [LAB / name for name in files if (LAB / name).is_file()]
    sources.extend(sorted((LAB / "tests").glob("test_*.py")))
    if not (LAB / "lab.py").is_file():
        raise FileNotFoundError("Lab must exist before packaging")
    with ZipFile(OUT / "mlops-risk-lab.zip", "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(sources):
            info = ZipInfo(f"mlops-risk-lab/{path.relative_to(LAB).as_posix()}",
                           date_time=(2026, 11, 18, 0, 0, 0))
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=ZIP_DEFLATED)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for lang in ("en", "zh"):
        def translate(en, zh):
            return zh if lang == "zh" else en
        for draw in (lifecycle, lineage, time_rules, release, monitoring):
            canvas = draw(translate)
            if canvas.warnings:
                raise ValueError("\n".join(canvas.warnings))
            suffix = ".zh.svg" if lang == "zh" else ".svg"
            (OUT / (canvas.name + suffix)).write_text(canvas.svg(), encoding="utf-8")
    if (LAB / "lab.py").is_file():
        package_lab()


if __name__ == "__main__":
    main()

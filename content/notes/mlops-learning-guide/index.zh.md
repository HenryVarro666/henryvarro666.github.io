---
title: "MLOps 学习笔记：从第一个模型到可部署、可监控、可回滚的系统"
date: 2026-11-18T00:00:00-06:00
description: "面向初学者的 MLOps 完整学习笔记：用可下载的 CPU 本地实验，串起数据契约、无泄漏评估、模型血缘、API、测试、监控、发布与回滚，并延伸到医学影像和 RAG。"
topics: [ml-systems, data-science]
---

模型在 notebook 中得到一个好分数，只回答了整个系统中的一个问题。换一批数据后还能运行吗？线上传来的金额单位是否相同？这次请求使用了哪一版模型和阈值？如果新版本有问题，能否恢复上一版？**MLOps 要把这些问题变成有记录、能检查、可执行的工程流程。**

这份笔记从一个小型风险预测系统开始，提供可下载的本地实验，再解释生产环境中的设计取舍。数据完全虚构，只用于学习。文中的示意流量、费用、延迟目标和业务代价都会标明；实验结果只描述随文代码在指定环境中的运行，不能用于判断真实风控或临床效果。资料和版本说明核对至 **2026-11-18**。

## 先选一条阅读路线 {#reading-routes}

适合的起点是：会写 Python 函数，能使用终端和虚拟环境，知道特征、标签、训练集与测试集。如果还不熟悉，先用 [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course) 补分类、过拟合和评估，再回来运行实验。暂时不需要 GPU、云账号、Kubernetes 或分布式训练知识。

| 路线 | 建议顺序 | 应当带走的成果 |
|---|---|---|
| 第一次接触 MLOps | 概念 → 数据与时间 → 血缘 → 评估 → 本地实验 | 能解释“一次预测依赖什么”，自己恢复一版模型。 |
| 已经会训练模型 | 从[本地安装](#setup)开始，再读 API、测试、发布和监控 | 安装完成后预留约 60–90 分钟跑主实验。这是阅读与操作预算，不是性能测试结果。 |
| 正在做实际项目 | 数据契约 → 发布 → 监控 → 应用 → 排错 | 写出自己项目的发布清单、告警处理步骤和回滚边界。 |
| 准备系统学习 | 完成主实验，再按[六阶段路线](#learning-path)扩展 | 每个阶段交付一个可检查的产物。 |

**先做出一个小而完整的系统，再按实际问题增加工具。** 这里的学习顺序和检查表是教学设计，不是某家公司的统一标准。

## 1. MLOps 维护的是持续变化的预测系统 {#mental-model}

设想一个虚构的支付风险服务：订单到来时，根据金额、账户年龄、最近一小时尝试次数和渠道，估计风险，交给既定规则决定是否进入人工复核。模型只是其中一环。输入来自数据系统，复核有容量限制，真实结果会延迟到达，服务需要按时返回，模型与阈值以后还会变化。

传统软件主要通过代码改变行为；ML 系统还会因为训练数据、特征定义、标签政策、模型参数和决策阈值改变行为。代码完全没改，上游把美元改成美分，也足以使结果出错。因此，版本管理和检查范围必须覆盖这些依赖。

![数据约定经过训练与评估、版本批准后服务业务决策；实际结果回到调查和下一轮训练。](mlops-lifecycle.zh.svg "图 1：上方是一次交付路径，下方是运行后的反馈。告警先触发调查，调查结果才决定是否训练或修改系统。")

Google 的参考架构用 level 0、1、2 描述手工流程、自动化 ML 流水线、流水线的 CI/CD。它是一种解释成熟度的方法，原文最后审阅于 2024-08-28；不要把其中的全部平台组件当作入门前置条件。[Google MLOps 参考架构](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

### CI、CD、CT 分别改变什么？ {#ci-cd-ct}

| 缩写 | 本文中的含义 | 一个具体例子 | 通过后得到什么 |
|---|---|---|---|
| CI：Continuous Integration | 合并变更前检查代码及相关契约 | 改特征计算后运行单元测试、数据检查和小型流水线测试 | 一份有检查记录的实现。 |
| CD：Continuous Delivery / Deployment | 准备或发布已检查的实现、运行环境和模型包 | 将指定镜像与指定模型部署到验证环境，再批准生产发布 | 一个可识别、可恢复的发布版本。 |
| CT：Continuous Training | 满足触发条件后重复训练流程 | 每周对已成熟标签重新训练，产出候选模型 | 一份候选和评估报告，仍需发布门禁。 |

“Continuous”强调流程可以稳定地重复和自动执行。它不表示 GPU 永不停机，也不表示每次训练结束都应自动替换生产模型。团队可以先由人批准，再把明确且可验证的步骤自动化。

## 2. 先把常见名词放回具体对象 {#glossary}

| 术语 | 简明解释 | 在本实验中 |
|---|---|---|
| Feature / Label | 预测时的输入 / 希望预测的结果 | 金额等四个输入 / 虚构风险结果。 |
| Run | 一次带输入、配置与结果的执行 | `baseline` 对应一次训练。 |
| Experiment | 为一个问题组织的一组 runs | 比较不同正则化参数。 |
| Artifact | 程序产生并需要保留的文件 | 模型、清单、评估报告。 |
| Pipeline | 按顺序执行的一组处理步骤 | 编码、缩放、分类器；也可指完整训练流程，需看语境。 |
| Lineage / 血缘 | 一个产物由哪些输入和过程得到 | 数据摘要、代码、环境 → 训练 → 模型 → 发布。 |
| Registry | 管理版本及发布引用的位置 | 本地 JSON 指针，不是完整企业模型注册中心。 |
| Training-serving skew | 训练与推理的特征处理或输入分布出现不一致 | 训练用美元，线上误传美分。 |
| SLI / SLO | 实际测量的服务指标 / 为它设定的目标 | 可用响应比例 / 对该比例设定的目标。 |

把 **训练流水线**与 **推理流水线**分开想：前者可以读取训练标签并拟合参数；后者只加载已经拟合好的处理器与模型，对新输入给出结果。调用 `predict` 时重新 `fit` 一个编码器，会改变特征含义。

## 3. 先写决策，再写模型需求 {#requirements}

开工前写一页“决策说明”。它应该说明谁在什么时刻拿到预测，预测驱动什么动作，错报与漏报各有什么代价，最迟何时需要结果，失败时怎么办。先用规则或简单模型建立基线，便于分辨复杂模型到底改善了什么。[Google Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)

下面是**教学示例**，所有业务数字都是假设：

| 项目 | 示例定义 | 为什么会改变设计 |
|---|---|---|
| 预测机会 | 每笔支付授权之前 | 只能用授权前已可用的信息。 |
| 输出与动作 | 风险概率；超过阈值进入复核 | 模型分数与业务动作分开管理。 |
| 延迟预算 | 示例：网关观测的有效响应在 250 ms 内 | 总预算还包括排队、取特征、网络和序列化。 |
| 复核容量 | 示例：每天最多复核 500 笔 | 高 recall 不能以无限增加误报为代价。 |
| 失败路径 | 超时后执行事先批准的规则 | 是否拒绝、放行、延后，应由业务风险决定。 |
| 成功证据 | 成熟标签上的质量、损失、误拒、复核量 | 单一离线指标不足以回答业务是否改善。 |

一个模型可能在测试集上 recall 更高，却把人工队列塞满。优化目标、模型评估指标、服务目标和业务结果应分别写清，并在同一张发布报告中对照。

## 4. 数据契约包括类型，也包括单位与含义 {#data-contract}

本实验在预测时接收四个字段。一行代表一次独立预测机会，字段单位固定。下面的约束同时用于训练数据检查和 API 输入检查。

| 字段 | 类型与范围 | 业务含义 |
|---|---|---|
| `amount` | 有限数，0–100,000 | 金额，以美元为单位。 |
| `account_age_days` | 有限数，0–36,500 | 预测时账户年龄，单位为天。 |
| `attempts_1h` | 整数，0–1,000 | 预测前一小时已知的尝试次数。 |
| `channel` | `web`、`mobile`、`store` | 本实验固定的渠道枚举。 |

JSON 中的数字类型无法发现“120 美元被写成 12000 美分”。还要规定空值策略、时区、窗口边界、重复事件处理、可接受的延迟以及字段的生产者。可以加一个固定输入作为契约测试，检查特征值和最终输出是否落在预期范围。

未知类别也需要政策：本实验拒绝合同之外的渠道，便于暴露上游变化；真实系统可以选择映射到明确的 `unknown`，但应在训练与推理两端一致，并监控出现比例。不要一边静默丢列，一边把成功的 HTTP 响应当作输入正确。

## 5. 防泄漏的关键问题是“当时知道吗？” {#time-leakage}

数据泄漏（data leakage）指训练或评估使用了实际预测时不可获得的信息。先切分，再拟合标准化、缺失值填充、特征选择等步骤；用 sklearn `Pipeline` 能帮助把处理限制在训练范围内，但它无法修复从未来构造的字段。[scikit-learn：常见陷阱](https://scikit-learn.org/stable/common_pitfalls.html)

![事件发生时间、特征可用时间、预测时刻和标签成熟时间沿时间轴排列；未来信息不能回流到过去样本。](time-and-leakage.zh.svg "图 2：09:55 发生的事件若到 10:05 才可用，10:00 的预测仍不能使用它。时间切分之外，还需要检查可用时间和实体分组。")

请区分至少三个时间：

- **Event time**：业务事件什么时候发生。
- **Available time**：系统什么时候真正能读取它，可能晚于事件发生。
- **Label maturity time**：经过多长观察期后，结果才足够完整，可以用于训练或评估。

例如，一笔交易 09:55 发生，10:05 才入库，不能用于重放 10:00 的决策。退货、欺诈确认或疾病进展等标签也可能延迟。把“尚未观察到阳性”立即写成阴性，会混淆未知和真实负例。

### 切分方法必须对应未来使用方式 {#splits}

| 需要回答的问题 | 合理起点 | 仍需检查什么 |
|---|---|---|
| 新样本来自近似相同总体 | 随机训练/验证/测试切分 | 重复记录、同实体多样本和预处理泄漏。 |
| 模型未来一段时间表现如何 | 按时间切分，按需要滚动验证 | 标签成熟、时间间隔、季节性和可用时间。 |
| 对新患者或新客户是否有效 | 按实体分组切分 | 同一实体的扫描、切片或事件不可跨组。 |
| 能否推广到新站点/设备 | 留站点或独立外部测试 | 与实际目标群体、采集协议是否对应。 |

`GroupKFold`帮助隔离实体；`TimeSeriesSplit`帮助构造面向未来的验证窗口。二者各有假设，任何一个都不会自动同时解决分组、时间、标签延迟和代表性。[交叉验证指南](https://scikit-learn.org/stable/modules/cross_validation.html)

本实验使用按时间排序的 60% / 20% / 20% 切分，并在训练、验证边界排除尚未达到 **2 小时标签延迟**的样本。这个延迟是造数规则，不是现实风控的一般规律。若任务既要求预测未来、又要求新客户泛化，需要另外构造同时满足两项约束的切分。

### 用几行数据检查自己的理解 {#leakage-exercise}

下面是独立的虚构练习，不是下载包中的额外字段。假设任务是在申请提交时预测后续结果：

| 候选字段 | 当时可以使用吗？ | 要求的证据 |
|---|---|---|
| 申请提交时填写的收入 | 可以作为候选 | 明确采集时刻和缺失含义；后续更正不要覆盖历史值。 |
| 最终审核时更新的收入 | 不可以直接使用 | 它属于预测之后才得知的信息。 |
| 过去 30 天行为数 | 需要继续检查 | 窗口截止于预测时刻；行为当时已入库；不能包含本次结果。 |
| 当前数据库里的客户状态 | 不能只凭字段名判断 | 需要历史版本和当时可用记录，不能把今天状态回填到过去。 |
| 同一患者另一张扫描的标签 | 通常不能作为独立测试证据 | 先按患者分组，核查数据取得和标签来源。 |

动手时，给每个字段写一句“最早可用时刻”。再从测试集抽五行，手动追到原始记录，确认每个输入在该行预测时刻已经存在。这种小规模核查能暴露 SQL 条件、纠错回填和数据来源理解的问题，单看训练代码很难发现。

## 6. 可复现从保存完整依赖开始 {#lineage}

一个 `model.pkl` 或截图不能完整说明一个结果。最小发布包应把预处理、模型、输入契约、阈值和评估身份关联起来。这样才能从线上行为追到训练输入，也能判断恢复旧模型时还需要恢复什么。

![数据快照、代码版本与运行环境关联一个完整发布包；champion 别名指向它，推理服务返回实际版本。](release-lineage.zh.svg "图 3：发布包同时保存处理逻辑、模型、契约、阈值和评估身份。champion 可以移动，审计时需要它最终解析到的具体版本。")

本文建议维护下面这份清单，它是教学用设计，可分散存储在相互关联的系统里：

~~~yaml
run_id: baseline
data:
  snapshot: reference.csv
  sha256: "..."
  split: chronological-60-20-20
  label_delay_hours: 2
implementation:
  git_commit: "..."
  entrypoint: lab.py
  configuration: {C: 1.0, seed: 42}
environment:
  python: "3.12.x"
  dependency_lock: requirements.lock
prediction:
  pipeline_sha256: "..."
  schema_version: "risk-v1"
  threshold: "从验证集选择后冻结"
evaluation:
  report: evaluations/baseline.json
  metric: average_precision_score
~~~

这个 YAML 用来解释清单内容，不是本实验文件格式的逐字复制。实际字段以下载包中的 `manifest.json`、`environment.json` 和评估 JSON 为准。

三个承诺应分开：**可追溯**表示知道产物从哪里来；**可重跑**表示程序能再次执行；**可复现**表示结果在事先定义的容差内一致。逐比特一致是更强的要求。随机种子有帮助，但不能保证跨版本、跨 CPU/GPU、跨平台完全相同。[PyTorch 可复现性说明](https://docs.pytorch.org/docs/2.15/notes/randomness.html)

数据也必须能恢复。记录一个路径或 digest 不等于保存了原始字节。小项目可以先保留不可覆盖的文件和校验和；数据变大、多人协作后再考虑 DVC。DVC 用 Git 保存指针，用 cache/remote 保存实际数据，推送 Git 本身不会上传数据对象。[DVC 入门](https://doc.dvc.org/start)

## 7. 评估要同时检查排序、决策与概率 {#evaluation}

分类器通常先产生一个分数或概率，再由阈值决定动作。**阈值属于发布内容**，不能看完测试集后反复调整。先在验证集选规则，再冻结，最后用保留测试集报告一次无偏于该选择过程的结果。[scikit-learn：决策阈值](https://scikit-learn.org/stable/modules/classification_threshold.html)

| 评估问题 | 可用指标 | 要避免的误读 |
|---|---|---|
| 能否把阳性排在前面 | ROC-AUC、Average Precision（AP） | 排序好不代表当前阈值合适。 |
| 当前阈值下会发生什么 | Precision、Recall、混淆矩阵、告警量 | 总体 accuracy 在不平衡数据中可能掩盖漏报。 |
| 概率是否可信 | Brier score、校准曲线 | 排序不变时，概率仍可能失准。 |
| 不同群体表现如何 | 按渠道、站点、年龄段等切片 | 小样本切片波动很大，需要样本量和不确定性。 |
| 动作是否有价值 | 成本、复核量、用户结果 | 离线关联不等于干预后的因果收益。 |

Precision = TP / (TP + FP)，Recall = TP / (TP + FN)。TP 是正确识别的阳性，FP 是错报，FN 是漏报。先看一个阈值下的混淆矩阵，再画阈值变化曲线，通常比只记一个公式更容易理解。

本实验报告 `average_precision_score` 计算的 **AP**。它用 recall 增量对 precision 加权；用梯形法积分 precision-recall 曲线会得到另一种数值，二者不应混称为同一个“PR-AUC”。概率、连续分数和硬分类标签适用的指标也不同。[scikit-learn：指标定义](https://scikit-learn.org/stable/modules/model_evaluation.html)

选择指标前，先写清正类、评估总体、时间窗口、样本数、标签覆盖率和阈值。一个标注为 0.90 的分数，若没有这些条件，几乎无法比较。本站的[Precision 与 Recall]({{< relref path="/notes/precision-vs-recall" lang="zh" >}})、[ROC-AUC 与 PR-AUC]({{< relref path="/notes/roc-auc-vs-pr-auc" lang="zh" >}})可作为补充。

### 用一张业务表解释为什么选择这个阈值 {#threshold-exercise}

下面是另一组**虚构数字**，用于说明决策，不是实验测量：1,000 笔中有 50 个阳性。假设每个漏报损失 10 美元，每个错报复核成本 1 美元。

| 规则 | TP / FP / FN / TN | Precision / Recall | 进入复核 | 假设成本 |
|---|---|---|---:|---:|
| 阈值较高 | 30 / 20 / 20 / 930 | 0.60 / 0.60 | 50 | 20×10 + 20×1 = 220 美元。 |
| 阈值较低 | 40 / 80 / 10 / 870 | 0.333 / 0.80 | 120 | 10×10 + 80×1 = 180 美元。 |

在这组代价里，较低阈值的期望成本较小；若一天只能处理 50 条复核，它却不可执行。可以重新定义规则或容量，而不能只把 recall 最大的版本写成胜者。实际代价还可能随金额、用户、干预方式变化，应将这些假设写入评估方案。

概率校准回答另一个问题：给 100 个案例都预测约 0.8，最后是否大约有 80 个阳性？这个例子仍是假设。样本不足时，校准图会很不稳定；可以按合理分箱汇总并报告样本量，不要把单个案例的概率解释成确定承诺。切片报告也一样：总体改善但一个重要渠道退化时，应说明该渠道样本数和可接受范围。

## 8. 本地安装：先跑一条完整流程 {#setup}

[下载完整 MLOps 风险实验（ZIP）](/notes/mlops-learning-guide/mlops-risk-lab.zip)。解压后进入 `mlops-risk-lab` 目录。实验只需要 **Python 3.12、CPU 和本地文件系统**，不用账号。安装依赖时需要网络。请用解压后的完整目录，避免只复制单个脚本而遗漏锁文件和测试。

~~~bash
cd mlops-risk-lab
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python lab.py generate
python lab.py train --run-id baseline
python lab.py evaluate --run-id baseline
python lab.py promote --run-id baseline
python lab.py predict --json '{"amount":120.0,"account_age_days":90.0,"attempts_1h":2,"channel":"web"}'
python -m pytest -q
~~~

Windows PowerShell 可用 `py -3.12 -m venv .venv` 建环境，用 `.\.venv\Scripts\Activate.ps1` 激活。其余 Python 命令相同；若系统的脚本执行策略限制激活，也可以直接调用虚拟环境内的 Python。

实验在 **macOS arm64、Python 3.12.16** 实测，锁定 sklearn 1.7.2、pandas 2.3.3、NumPy 2.3.4、SciPy 1.16.3、FastAPI 0.121.1、Uvicorn 0.38.0 和 pytest 8.4.2；完整依赖以下载包为准。锁文件固定版本，没有额外提供安装包哈希校验。这些是**实验版本**，不表示它们是本文日期下最新版本。Linux/Windows 的命令是迁移指引，本文没有在这两种系统上执行验证。

默认输出写入 `workspace/`。想使用另一处输出目录，把全局参数放在子命令之前，例如 `python lab.py --root workspace-two generate`。每一个后续命令都必须使用同一个 root。不要把两个不同实验的输出混在一起。

## 9. 看懂实验产物，再比较两个候选 {#lab}

每个命令都承担一个明确职责：

| 命令 | 应当检查的东西 |
|---|---|
| `generate` | 生成 reference 与 shifted 两份合成数据，保留时间与标签可用信息。 |
| `train` | 读取既定训练范围，拟合预处理和分类器，保存一次不可覆盖的 run bundle。 |
| `evaluate` | 在固定验证/测试流程下形成评估报告和发布门禁结果。 |
| `promote` | 仅将满足门禁的具体 run 设为生产引用。 |
| `predict` | 读取当前生产包，返回概率、冻结阈值、版本和包摘要。 |
| `monitor` | 比较输入、预测和有标签表现，生成监控报告。 |
| `rollback` | 恢复记录中的上一生产版本。 |

~~~text
workspace/
  data/reference.csv
  data/shifted.csv
  runs/baseline/
    model.joblib
    manifest.json
    environment.json
    lab.py
    api.py
    requirements.lock
  evaluations/baseline.json
  registry/production.json
  reports/monitor-....json
~~~

训练包里保存的是**已拟合的同一个 sklearn Pipeline**，离线与 API 都加载它；API 不重新拟合编码器。训练命令拒绝覆盖已有 run，评估命令也拒绝覆盖同一 run 的报告。这是程序约定，不是文件系统强制的不可变存储。SHA-256 摘要能发现内容变化，不能验证发布者身份，也不是数字签名。

这里的 registry 是供可信单人写入的本地 JSON 实现，缺少并发控制、权限审批、审计和高可用，不能当作生产注册中心的现成替代品。除了数据文件摘要，包里还记录切分身份，并在评估时核对测试样本 ID 与数量；相同数据文件若换了一种切分规则，比较的就可能已不是同一批测试样本。

训练核心可以缩成下面几行。完整代码还会先验证数据、切分、选择阈值并写清单，不能只复制这几行就跳过那些条件：

~~~python
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

features = ["amount", "account_age_days", "attempts_1h", "channel"]
preprocess = ColumnTransformer([
    ("numeric", StandardScaler(), features[:3]),
    ("categorical", OneHotEncoder(handle_unknown="error"), ["channel"]),
])
pipeline = Pipeline([
    ("features", preprocess),
    ("classifier", LogisticRegression(C=1.0, max_iter=500, random_state=42)),
])
pipeline.fit(training[features], training["label"])
probabilities = pipeline.predict_proba(validation[features])[:, 1]
~~~

这里的 `training` 和 `validation` 是已经按时间与标签可用性切好的 DataFrame。`fit`只在 training 调用；验证、测试和推理调用同一对象的变换与预测。`[:, 1]`在这个二分类标签约定下取正类概率。

### 一次参考运行得到什么？ {#measured-results}

下面是随文实验在锁定环境中的**实测参考值**，四舍五入展示。数据生成规则人为设置风险关系，阳性率接近 20%，并非真实欺诈比例。

| 项目 | baseline 参考运行 |
|---|---:|
| 标签延迟清理后的训练行数 | 3,481 |
| 标签延迟清理后的验证行数 | 1,081 |
| 测试行数 | 1,200 |
| 测试阳性率 | 0.1975 |
| 验证集选出的阈值 | 0.20 |
| 测试 AP | 0.435033 |
| 测试 ROC-AUC | 0.756225 |
| 测试 Precision / Recall | 0.397403 / 0.645570 |
| 测试 F1 / Brier score | 0.491961 / 0.136691 |
| TN / FP / FN / TP | 731 / 232 / 84 / 153 |
| 教学发布门禁 | 通过 |

在该参考包中，前面那条输入的风险概率约 **0.311307**，超过 0.20，所以 `flagged` 为 true。返回中还包括 `model_version`、`bundle_sha256`、`threshold` 和 `synthetic_only`。自己的运行应以生成文件为准；若差异较大，先比较依赖、数据摘要、配置和切分，不要直接改测试阈值让它通过。

### 再做一次候选发布与回滚 {#rollback-lab}

~~~bash
python lab.py train --run-id candidate --C 0.2
python lab.py evaluate --run-id candidate
python lab.py promote --run-id candidate
python lab.py predict --json '{"amount":120.0,"account_age_days":90.0,"attempts_1h":2,"channel":"web"}'
python lab.py rollback
python lab.py predict --json '{"amount":120.0,"account_age_days":90.0,"attempts_1h":2,"channel":"web"}'
~~~

`C`是这个逻辑回归模型的正则化参数，较小值对应更强正则化。候选参考运行的阈值约 0.17，测试 AP 约 0.435109、recall 约 0.704641，而 F1 约 0.469101，低于 baseline 的 0.491961。这里的目的在于演练候选、门禁、版本切换和恢复；不能据此称候选全面更优。如果某候选没有通过门禁，应保留失败报告，生产引用继续指向现有版本。

这个教学演练对 baseline 和 candidate 使用同一保留测试集来观察版本流程，**不构成每次迭代都独立的新统计结论**。真实项目在验证集或滚动验证上选择 `C` 和阈值，最终保留一份未参与选择的测试集；不要根据反复看到的测试分数继续选参。每个 run 只能写一份评估报告，与整个研究过程只看一次测试集，是两种不同约束。

对照两次 `predict` 的具体版本和包摘要，确认恢复的是原包。一个已经加载模型的 API 进程不会因为 JSON 指针变化而自动刷新，需要重启并重新检查 readiness；后面会详细说明这一点。

还可以故意提高门槛，观察训练成功与发布获准之间的区别：

~~~bash
python lab.py train --run-id blocked
python lab.py evaluate --run-id blocked --min-average-precision 0.999
python lab.py promote --run-id blocked
~~~

最后一条应拒绝发布并以退出码 2 结束。评估仍会写报告，并因为“生成报告成功”而退出 0；自动化流程必须检查 `gate_passed`，不能只看前一条进程退出码。内置门槛是 AP ≥ 0.25、ROC-AUC ≥ 0.65、recall ≥ 0.45，全是教学选择，未表达真实损失、容量、公平性或审批要求。

## 10. Tracking 解决记录问题，Registry 解决版本引用问题 {#tracking}

当手工比较文件开始困难时，再引入 MLflow。**Tracking** 记录 runs、参数、指标和 artifacts；**Model Registry** 管理已注册模型版本及引用。它们相互关联，但“有实验记录”不表示“已发布”。

主实验不依赖 MLflow。需要观察同一个训练过程的实验记录时，在已激活的环境中安装可选依赖：

~~~bash
python -m pip install -r requirements-mlflow.txt
python lab.py train --run-id tracked --mlflow
~~~

可选依赖锁定 **mlflow-skinny 3.6.0、SQLAlchemy 2.0.44 和 Alembic 1.17.1**，使用显式 SQLite tracking URI，避免依赖默认行为。MLflow 3.7.0 起，新建本地 tracking 的默认存储改为 SQLite；旧教程和既有 file store 的行为可能不同。[MLflow 自托管说明](https://mlflow.org/docs/latest/self-hosting/)

可以先通过 Python API 检查记录，无需再运行一个服务器：

~~~python
from pathlib import Path
from mlflow import MlflowClient

database = (Path("workspace") / "mlflow.db").resolve()
if not database.exists():
    raise FileNotFoundError("先运行带 --mlflow 的训练，并确认 root")
client = MlflowClient(tracking_uri=f"sqlite:///{database}")
for experiment in client.search_experiments():
    runs = client.search_runs([experiment.experiment_id])
    print(experiment.name, [(run.info.run_id, run.data.metrics) for run in runs])
~~~

具体 SQLite 文件位置以实验 README 为准。这个查询示例假设数据库位于 `workspace/mlflow.db`，不要在不同工作目录中无意创建另一个空数据库。Tracking 数据中的数据集名字、digest 和 source 是血缘元数据，不能代替数据文件备份。

未来接入真正的 MLflow Registry 时，可以用 `models:/risk-model/7` 指定版本，或用 `models:/risk-model@champion` 读取别名。别名可移动，重新执行加载才会解析新版本；已在内存中的模型不会自动热更新。部署记录应保存解析后的具体身份。旧教程的 Model Stages 从 2.9.0 起已弃用，应查当前 alias 工作流。[MLflow Registry 工作流](https://mlflow.org/docs/latest/ml/model-registry/workflow/)

## 11. API 必须同时说明输入、输出和加载状态 {#serving}

先完成至少一次 promotion，再在第一个终端运行：

~~~bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --no-access-log
~~~

在第二个终端激活同一个虚拟环境，进入同一个项目目录：

~~~bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{"amount":120.0,"account_age_days":90.0,"attempts_1h":2,"channel":"web"}'
~~~

`/healthz`回答进程是否仍能响应；`/readyz`回答当前发布包是否已经加载、能够接收预测；`/predict`执行输入检查并使用已加载模型。它们都不能单独证明业务预测是正确的。

FastAPI 的 lifespan 适合在启动阶段加载共用资源。这样每个进程加载一次模型，而不是每次请求都从磁盘读模型。[FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/)

对 API 做三个观察：合法请求应返回概率与版本；负金额或缺字段应得到受控的客户端错误；切换生产引用后，重启服务，确认 readiness 与版本都变成目标值。服务返回具体版本，可以帮助把线上的一条预测关联到离线评估报告。

参考验证中，readiness 和合法预测为 HTTP 200，负金额为 HTTP 422；没有可用发布包时 liveness 仍为 200，readiness 和预测为 503。如果 CLI 使用了自定义 root，启动服务时也设置同一处路径，例如 `MLOPS_LAB_ROOT=/tmp/risk-demo-2 python -m uvicorn api:app --host 127.0.0.1 --port 8000 --no-access-log`。PowerShell 用 `$env:MLOPS_LAB_ROOT`设置环境变量。

生产中还要决定请求体大小、超时、并发、认证、速率限制和敏感信息日志政策。`async def`不会让 CPU 推理自动并行，也不会减少模型占用。每个 worker 通常有独立内存，增加 worker 之前先测模型大小、峰值内存和吞吐。

## 12. 测试按风险和成本分层 {#testing}

这里采用的测试层次是**教学建议**，根据失败后果和运行成本选择频率。ML Test Score 的原始论文提出 28 项与生产 ML 有关的测试和监控需求，并明确补充普通软件测试；它不是一个用模型 accuracy 换算的成熟度证书。[ML Test Score 原论文](https://storage.googleapis.com/gweb-research2023-media/pubtools/4156.pdf)

| 层次 | 本项目应能发现的错误 | 适合何时运行 |
|---|---|---|
| 快速代码与契约检查 | 缺列、非法类别、负值、非有限数、单位或窗口计算错误 | 每次改动。 |
| 小样本集成 | 训练 → 保存 → 加载 → 预测失败；离线/API 不一致 | 每次相关变更。 |
| 候选评估 | 指标未达标、重要切片明显退化、阈值导致工作量超限 | 每次候选发布。 |
| 部署 smoke test | 包缺失、启动失败、无法 ready、返回错误版本 | 每次部署。 |
| 运行期监控与恢复演练 | 数据过期、延迟升高、成熟标签质量下降、回滚失败 | 按系统风险持续运行或定期演练。 |

实验的测试可以用 `python -m pytest -q` 运行。核心环境的参考结果为 **21 passed、1 skipped**，跳过项是可选 MLflow 集成；安装可选依赖后的参考结果为 **22 passed**。读测试时，追问它挡住哪种真实错误。例如，“模型文件存在”比“能重新加载并对固定输入给出一致概率”弱得多；“请求返回 200”比“模型身份、输入处理与输出契约正确”弱得多。

不必为了覆盖率数字给每一行写机械断言。优先覆盖时间边界、空值、重复事件、非法输入、序列化、训练/推理一致性和恢复步骤。故意破坏一个规则，再确认测试确实失败，比仅看绿色标记更有学习价值。

## 13. Docker 固定运行环境，CI 固定检查与交付身份 {#packaging}

Docker 把应用和运行依赖组织成镜像。它能减少“电脑上能跑，服务器上缺包”的问题，但没有验证模型质量，也不会自动提供扩缩容、权限和告警。**下载包提供 Dockerfile 扩展示例，其构建与运行没有执行验证。** 示例在构建时只安装依赖并复制明确的源文件，训练另在容器环境执行，避免训练与服务环境不一致。

下面的扩展命令使用同一个 named volume 保存训练产物。它们是待读者验证的示例，不属于前面的实测结果：

~~~bash
docker build -t mlops-risk-lab .
docker run --rm -v risk-lab-demo:/app/workspace mlops-risk-lab python lab.py generate
docker run --rm -v risk-lab-demo:/app/workspace mlops-risk-lab python lab.py train --run-id baseline
docker run --rm -v risk-lab-demo:/app/workspace mlops-risk-lab python lab.py evaluate --run-id baseline
docker run --rm -v risk-lab-demo:/app/workspace mlops-risk-lab python lab.py promote --run-id baseline
docker run --rm -p 127.0.0.1:8000:8000 -v risk-lab-demo:/app/workspace mlops-risk-lab
~~~

重做造数时换一个 volume 名称；切换版本后重启服务容器。生产镜像还需要团队自己的基础镜像、最小权限、漏洞与资源检查，这个示例没有提供生产部署保证。

写容器时，可先复制依赖文件再安装，利用缓存；用 exec-form `CMD`启动进程，让关闭信号到达服务。FastAPI 官方指南从基础 Python 镜像构建，旧的预打包 FastAPI/Gunicorn 基础镜像已弃用。[FastAPI 容器部署](https://fastapi.tiangolo.com/deployment/docker/)

CI 的核心目标是：**对一个确定的变更运行检查，产出一个确定的发布包，后续部署同一份包。** 下载包包含未启用的 CI 示例，**没有在 GitHub 执行**；其中的本地检查命令经过验证。一个入门工作流可以先安装锁定依赖，运行快速测试，再做小型端到端演练，显式检查 `gate_passed`，保留命名明确的报告和包摘要。`evaluate`能完成并写出失败报告，不表示候选通过；promotion 还会拒绝未过门禁的候选。

GitHub Actions 的 workflow artifact 与容器 registry 中的镜像是两类对象。`upload-artifact@v4`的产物不可原地修改，但下载时自动发现摘要不一致目前只会告警；若要作为完整性门禁，需要显式比较并失败退出。[GitHub workflow artifacts](https://docs.github.com/en/actions/tutorials/store-and-share-data)

一次可靠的交付应能回答：检查对应哪个 commit？哪个依赖锁？哪个数据快照？哪个包或 image digest？部署的是否就是检查过的内容？如果回答只剩 `latest`，就需要补充身份记录。

## 14. 部署方式由决策时间决定 {#serving-patterns}

先问多久需要一次结果，再决定服务形式。以下是本文设计示例，模式之间可以组合，不是成熟度排名。

| 方式 | 适用例子 | 主要运行契约 |
|---|---|---|
| Batch inference / 批量推理 | 夜间生成次日复核名单 | 截止时间、完整性、输入新鲜度、重跑不重复写。 |
| Synchronous API / 同步推理 | 页面或支付流程等待结果 | 端到端超时、尾延迟、并发和失败路径。 |
| Asynchronous / 异步推理 | 一个较大的三维图像处理作业 | job ID、状态、队列年龄、重试与幂等。 |
| Event-stream inference / 事件流推理 | 持续计算交易窗口特征并评分 | 事件时间、迟到、去重、状态恢复与积压。 |

**在线推理不等于在线学习。** 一个每天重训的固定模型完全可以逐请求提供服务。**事件流也不等于 LLM token streaming**：前者处理持续到来的事件与状态，后者逐步发送一个回答。

重试是另一个容易漏掉的边界。若一个预测作业还会写付款、发通知或提交复核，给这些副作用设置持久的幂等键。消息处理的 exactly-once 保证有明确范围，不能自动覆盖任意外部 API 调用。[Kafka Streams 概念与保证范围](https://kafka.apache.org/44/streams/core-concepts/)

## 15. 发布应当有门禁、观察窗口和完整回滚 {#release}

离线胜出只是候选发布的一个条件。还要检查输入兼容性、环境、系统性能、业务限制和恢复能力。发布前保存上一版，写清谁批准、何时扩大流量、出现什么信号立即停止。

![候选包经过发布检查、分阶段上线与观察后扩大使用；异常时恢复上一份兼容发布和路由。](safe-release.zh.svg "图 4：通过离线检查与扩大真实流量是两次决定。影子、金丝雀和 A/B 的区别写在图下；异常分支需要恢复完整旧版本。")

| 方法 | 候选输出是否影响用户动作 | 主要目的 | 不能由它单独证明什么 |
|---|---|---|---|
| Shadow / 影子测试 | 应用忽略候选响应 | 用真实形状的请求观察运行和分歧 | 候选干预后的业务收益。 |
| Canary / 金丝雀 | 小部分流量会受影响 | 限制故障暴露并观察能否扩量 | 短窗口覆盖所有稀有错误或延迟结果。 |
| A/B experiment | 按分配单元影响动作 | 比较预先定义的处理效果 | 仅凭两个路由权重就得到有效因果结论。 |

影子系统也必须禁止真实副作用；忽略第二套系统的响应，并不能阻止它重复扣款。金丝雀在监控触发时保留恢复旧版本的路径。A/B 则还需稳定分配、曝光记录、样本量、观察周期和保护指标。[AWS Shadow](https://docs.aws.amazon.com/sagemaker/latest/dg/shadow-tests.html) · [AWS Canary](https://docs.aws.amazon.com/sagemaker/latest/dg/deployment-guardrails-blue-green-canary.html) · [本站 A/B 实验笔记]({{< relref path="/notes/ab-testing" lang="zh" >}})

### SLO 把“运行正常”写成可判断条件 {#slo}

例如，**教学目标**可以写成：“滚动 28 天内，99.5% 的合格请求在网关测量的 250 ms 内返回有效结果。”先定义合格请求、有效结果、超时和 fallback 如何计入。SLI 是实际比例，SLO 是目标；比例型目标的 error budget 为 1 减去目标。[Google SRE：实施 SLO](https://sre.google/workbook/implementing-slos/)

p95 延迟指所选窗口内约 95% 请求的延迟不超过该值。比较时同时报告时间窗、流量、输入大小、冷热状态和错误；单请求 stopwatch 无法说明最大稳定 QPS。延迟要包括排队、特征读取、预处理、模型执行和返回路径。

回滚需要恢复 **模型 + 预处理 + schema/特征契约 + 阈值 + 运行环境**的兼容组合。若线上特征已经被破坏性地改成新格式，移动模型别名无法修复。恢复后重新检查 readiness、版本和固定请求，再观察真实服务指标。实验中的 API 需重启才能加载指针所指的新包。

### 再测容量和费用，别把单请求耗时当作系统能力 {#capacity}

扩展练习时，先固定输入集合，分别测试冷启动、预热后的单请求、稳定并发与短时突发。每轮记录成功数、错误数、总运行时长、成功请求延迟分位数、队列等待与峰值内存。保持输入大小和请求组合一致，才能比较两次版本。压测与主实验分开进行；本文没有提供生产 QPS 或延迟承诺。

如果模型计算只需几毫秒，主要时间却花在远端取特征，增加推理 worker 未必解决问题；若队列持续增长，短时响应正常也不能证明可持续。先定位时间花在哪里，再决定缓存、批处理、并发或硬件变化。测试 fallback 路径同样有价值：它可能快速返回，但业务结果质量与正常路径不同。

费用也用同一份边界定义。**算术示例**：一小时总成本 2 美元，完成 40,000 个可用结果，则每 1,000 个结果为 0.05 美元。如果只有一半请求真正得到可用结果，成本变为每 1,000 个 0.10 美元。比较时注明是否包含闲置资源、特征查询、存储、网络、重试和影子流量，不能只报模型前向计算费用。

## 16. 监控要覆盖五个不同层面 {#monitoring}

![上方是立即可见的服务信号，中间是请求输入与分数，下面是等待标签成熟后的质量与成本。](monitoring-layers.zh.svg "图 5：从上到下，信号通常需要更长时间才能完整出现。输入变化和效果下降可以分别发生，需要独立调查。")

| 层面 | 记录什么 | 告警后先问什么 |
|---|---|---|
| 技术运行 | 请求量、错误/超时、p50/p95/p99、队列年龄、内存/CPU/GPU | 用户能否按时获得可用结果？ |
| 数据契约 | 缺失率、越界、未知类别、单位、新鲜度、批次完整性 | 上游是否交付了预期输入？ |
| 预测行为 | 分数分布、阳性率、拒答率、不同切片的变化 | 模型动作是否发生变化？ |
| 有标签质量 | AP、recall、校准、切片指标、标签覆盖与成熟度 | 结果是否真的变差，证据覆盖谁？ |
| 业务与成本 | 损失、误拒、复核量、完成率、每个可用结果的费用 | 系统是否达到原来的目的？ |

技术监控可以从 latency、traffic、errors、saturation 四类信号开始，并关注成功请求与失败请求的不同延迟。平均值可能掩盖尾部问题。[Google SRE：监控分布式系统](https://sre.google/sre-book/monitoring-distributed-systems/)

一次预测最好有稳定的 `prediction_id`，记录预测时刻、具体模型与阈值、特征版本，再用它关联后到标签。评价“上周预测”时，按预测时间形成 cohort，并等到约定成熟度；不要直接把本周到达的标签视为本周模型表现。还要监控关联成功率：若只匹配到容易标注的样本，指标可能很漂亮但总体不完整。

例如，**虚构标签政策**要求预测后观察七天，周一发出的预测到下周一才进入成熟评估。周五看本周结果时，缺失不能自动填成负例。报告可以分成“近期输入与服务情况”和“已成熟 cohort 质量”，并写明后者覆盖的日期。若标签后来更正，保留标签版本及评估重算记录，避免同名报表悄悄改变。

每条告警都补齐对象、阈值、窗口、负责人和第一步动作。比如“渠道未知值比例超过自定阈值”应能追到原始数据、查看何时开始、决定暂停哪条路径。一个每天都响却没有人知道怎么处理的告警，无法提供有效的运行保护。

实验运行 `python lab.py monitor` 默认比较 `workspace/data/shifted.csv`。也可显式指定 `python lab.py monitor --data workspace/data/shifted.csv`。参考运行同时触发输入变化、分数变化和质量下降告警，shifted AP 约为 **0.194730**，ROC-AUC 约为 **0.359483**。输入告警使用简单汇总启发式，不能当作正式的分布假设检验。这是刻意同时植入输入变化与关系变化的演练，不能由此推出“输入变化必然使模型下降”。

为了让历史示例不随现实日期变化，监控默认 `--as-of 2025-02-01T00:00:00Z`。所有时间统一为 UTC，未来事件被排除，只在 `label_ready_time`不晚于快照时刻时计入质量；不足 100 个成熟样本或只有一个类别时，报告说明原因并将 performance 设为 null。无标签也应明确记录无法评估，而不是写成“模型没有下降”。运行 `python lab.py monitor --as-of 2025-01-10`，可以观察当时只有一个事件、没有成熟标签时的行为。

## 17. Drift 是调查线索，不是自动重训命令 {#drift}

“漂移”在不同文章中含义不完全相同。这里把输入分布变化写成 `P(X)` 改变，把条件关系变化写成 `P(Y|X)` 改变，把阳性率变化写成 `P(Y)` 改变。需要标签才能直接计算的性能，不应只靠输入统计替代。

两个**数学构造示例**足够说明原因。设二值输入 x，模型始终预测 `f(x)=x`：

| 场景 | 改了什么 | 模型 accuracy |
|---|---|---|
| 原关系始终为 `y=x`，但 `P(x=1)` 从 0.5 变成 0.9 | 输入明显变化 | 仍为 1。 |
| `P(x)` 不变，但真实关系变为 `y=1-x` | 输入统计看不出变化 | 从 1 变为 0。 |

所以，输入变化既不是性能下降的充分条件，也不是必要条件。PSI、KS 或分布距离的告警阈值需要结合窗口、样本量、季节性、切片和误报成本校准，没有一个通用数字能证明“必须重训”。

遇到告警，先核监控是否正确、数据是否新鲜；然后查 schema、单位、丢列和窗口逻辑；再看流量组成、成熟标签和切片。最后才决定修数据、恢复旧路径、调整经批准的规则、训练候选、补标注或继续观察。节假日客户组成变化与金额单位错误，需要的处理显然不同。

## 18. 重训应当产生候选，发布另有决定 {#retraining}

适合触发训练的事件包括新一批成熟标签到齐、预定周期、经调查确认的性能问题，或特征/标签定义修订。每次触发都记录原因、输入快照和评估总体。不要让一个上游列全为空的事故自动启动“学习新现实”。

建议让流水线明确分成：等待数据与标签 → 数据检查 → 建训练快照 → 训练候选 → 固定评估 → 发布检查 → 批准/拒绝 → 受控部署。训练成功只意味着计算步骤完成；候选仍可能比简单基线差、对某切片退化，或超过推理资源预算。

自动化前，先演练三种失败：训练中断后能安全续跑；同一批次重跑不会重复发布；候选未通过时旧版本继续服务。把失败报告保留下来，避免后续只看到成功 run，误以为所有训练都值得发布。

## 19. 调度器和 Feature Store 在痛点明确后再加入 {#orchestration}

当流程每天重复、依赖多个上游、需要重试/回填/超时和执行记录时，调度器才真正开始省事。把脚本步骤整理成任务 DAG，为每步规定输入、输出、幂等性和失败行为。调度器会安排执行，却不会替你定义正确标签，也不会自动消除数据泄漏。可结合[Airflow 笔记]({{< relref path="/notes/airflow" lang="zh" >}})、[dbt 数据建模实验]({{< relref path="/notes/dbt-learning-guide" lang="zh" >}})理解各自边界。

Feature Store 在多模型复用特征、历史取数与在线查询容易不一致时更有价值。**Point-in-time join**为一个历史预测时刻查找当时有效的特征。例如 10:00 的样本不能直接关联训练当天“最新”的 11:00 账户消费总额。Feast 的历史关联会按实体行的时间向前查找，并应用 TTL。[Feast Point-in-time joins](https://docs.feast.dev/getting-started/concepts/point-in-time-joins)

还需单独处理可用时间：event time 为 09:55、10:10 才到达的数据，对 10:00 的决策仍不可用。正确工具无法从错误或回填覆盖的时间戳恢复真实历史。先把这些语义写清，再评估离线/在线特征服务、物化新鲜度和回滚兼容性。

## 20. 同一套运行流程在三类应用中长什么样？ {#applications}

### 表格风控：反馈是延迟且有选择的 {#fraud}

真实欺诈研究中，调查员快速反馈通常只覆盖部分告警，更广的结果会稍后到达；两类反馈的选择机制不同。[Dal Pozzolo 等：延迟监督下的欺诈检测](https://boracchi.faculty.polimi.it/docs/2015_04_Credit_Card_Fraud_Detection_DalPozzolo_Boracchi_Caelen_Alippi_Bontempi.pdf)

应用时，把近期复核 precision 与成熟 cohort 的总体质量分开；记录阈值和人工容量；明确“暂时没有拒付”何时才能形成负标签。若系统已经拒绝某笔交易，其未被执行的后果也不一定可观察。这里的合成实验只教完整工程流程，不模拟完整业务因果关系。

### 医学影像：先守住患者、采集与预处理边界 {#imaging}

假设一个三维分割研究管道需要接收不同站点的 MRI。除了模型版本，清单还应包括输入序列、空间方向、spacing、重采样、强度处理、裁剪、标签定义和后处理。按患者切分，避免同一受试者的重复扫描或切片跨训练/测试；根据问题再做站点、设备、时间或外部队列验证。

运行监控可先检查文件完整性、体积尺寸、预处理失败率、队列到结果的时间；模型质量另需有参考标签的病例与合适指标。高体素 accuracy 可能掩盖小结构分割差，Dice 与表面距离也应说明定义、病例聚合和标注质量。可以把代表性、独立测试、相关使用环境和持续监控作为设计自查项；进一步阅读 FDA 对 2025 年 IMDRF GMLP 正式原则的介绍及原文入口。[FDA GMLP 介绍与 IMDRF 原文入口](https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles)

这是一种**研究管道设计**，不意味着完成实验就能用于临床。预期用途、实际工作流、人机协作、验证和变更控制仍需独立建立。

### LLMOps / RAG：发布包还包括知识与提示 {#llmops}

一个 RAG 系统通常要保存生成模型、prompt 模板、解析/切块规则、embedding 模型、语料与索引快照、reranker、访问控制和评估集版本。替换 embedding 时，需要确认索引是否仍兼容；只恢复 prompt，未必恢复原来的回答行为。

把失败拆开看：文档解析漏表格；检索找错文档；上下文被截断；生成没有依据。Tracing 记录一次请求内部的步骤，有助于定位问题，而不是只看最终回答。[MLflow Tracing](https://mlflow.org/docs/latest/genai/tracing/)

评估时分开检查检索质量、来源是否正确且当前有效、回答是否支持、任务是否完成。Groundedness 关注回答是否被给定上下文支持；它不证明上下文本身真实、最新或有权限使用。LLM judge 也要用人工样本校准并记录其配置。[MLflow RetrievalGroundedness](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/rag/groundedness)

监控总完成时间、首 token 时间、错误/重试、输入输出 tokens、费用和抽样质量。首 token 快不代表完整回答快。日志和 traces 应过滤敏感内容，可结合[LangChain 与 LangGraph 笔记]({{< relref path="/notes/langchain-langgraph-guide" lang="zh" >}})继续实践。

## 21. 把安全作为输入、产物和权限的具体约束 {#security}

只加载可信来源的 Python 序列化模型；`pickle` / `joblib`加载可能执行代码，摘要只能检查内容身份，不能证明内容安全。[scikit-learn 模型持久化说明](https://scikit-learn.org/stable/model_persistence.html) 模型包与镜像应来自受控构建，给生产服务只授予必要读取权限。API 输入要限制大小、检查有限数与范围，避免把内部异常或文件路径直接返回给调用者。

原始样本、用户标识、提示、检索内容和标签都可能包含敏感信息。不要因为调试方便，把完整请求写进公开日志或 CI artifact。优先保留请求 ID、版本、必要统计和经授权的脱敏样本；为数据与模型设定访问、保留和删除政策。

CI 的 token 用最小权限，部署身份优先短期凭证；不在仓库保存密钥。对于外部提交的代码，谨慎处理带高权限的 workflow 触发器和 secrets。日志脱敏不是绝对保证，依赖和 action 版本也应验证并锁定。[GitHub Actions 安全使用](https://docs.github.com/en/actions/reference/security/secure-use)

## 22. 排错时先找证据，再决定修哪一层 {#troubleshooting}

| 症状 | 先找的证据 | 可能的修复 |
|---|---|---|
| 离线很好，API 输出异常 | 同一输入的字段、单位、编码结果、包摘要 | 使用同一已拟合 Pipeline；修 schema 或单位转换。 |
| `readyz`失败 | 是否 promotion、root 是否一致、包是否完整 | 先完成评估和 promotion；确认路径与权限。 |
| promotion 后 API 仍旧版本 | 服务启动时间、返回版本、当前 registry 指针 | 重启加载；生产做受控 rollout 并验证具体版本。 |
| 跨机器分数不同 | Python/依赖、数据摘要、随机源、硬件 | 恢复锁定环境，说明可接受容差并排查非确定性。 |
| 输入 drift 告警但指标稳定 | 成熟标签、季节性、切片、窗口样本量 | 调整监控基线或继续观察，不急于重训。 |
| 输入无变化但质量下降 | 标签政策、条件关系、关联覆盖率 | 检查概念变化或标签问题，形成新候选。 |
| 训练一直成功，生产越来越差 | 候选门禁、评估数据是否重复使用、阈值记录 | 修评估流程，避免测试集调参和无条件自动发布。 |
| CPU/GPU 不忙，用户仍等很久 | 排队、取特征、序列化、外部调用、重试 | 先优化实测瓶颈；测端到端和尾延迟。 |
| Batch 显示成功，但少一半结果 | 输入清单、行数、键覆盖、输出提交方式 | 加完整性门禁和原子发布策略。 |
| 回滚模型后仍错 | 当前特征/schema、预处理、阈值、环境 | 恢复兼容的完整发布，保留旧特征路径。 |

把每次事故写成短记录：现象、影响、证据、根因、处理、增加哪项检查。这样下一个人能从记录复现问题，也能看见测试为何存在。

## 23. 用六个阶段安排四到六周学习 {#learning-path}

下面是可调整的学习安排，时间是本文估计。已有 ML 基础可以合并前两阶段；每周时间少就延长。以完成产物为准，不用把读过多少工具当进度。

| 阶段 | 重点 | 可检查的交付物 |
|---|---|---|
| 1. 模型与数据基础 | 标签、基线、切分、泄漏、指标 | 一页数据契约；一张无泄漏切分图；解释 AP 与阈值。 |
| 2. 可重复本地训练 | CLI、依赖锁、数据摘要、Pipeline | 在新目录重跑；加载保存模型；比对结果和环境。 |
| 3. 比较与血缘 | 两个候选、评估、tracking | 一份带版本、数据、指标和选择理由的实验报告。 |
| 4. 推理与测试 | API 或 batch、输入契约、测试 | 合法/非法输入演示；序列化一致性测试；故意破坏一条测试。 |
| 5. 发布与恢复 | 门禁、具体版本、shadow/canary 思路 | 一次 promotion 与 rollback；恢复后核版本和固定请求。 |
| 6. 运行与扩展 | 监控、延迟标签、调度、领域应用 | 一次注入故障的调查记录；一份自己的系统设计。 |

完成后可以扩展一个小项目：把合成数据换成许可明确的公开数据，保留基线与冻结评估；选择一个真正需要的服务形式；加一项可触发且有处理步骤的告警。先写清数据与失败边界，再考虑云部署、Feature Store 或集群。

## 24. 外部资料按“读后做什么”来选择 {#resources}

以下资料与前文的 API 参考配合使用。课程公开不等于云计算免费；课程版本较早时，思想可学，依赖按项目文档迁移。无需按表全部学完，先挑与当前阶段有关的一项。

| 资料 | 最适合用来学什么 | 阅读后的练习 |
|---|---|---|
| [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course) | 补分类、过拟合、指标等基础。 | 解释一张混淆矩阵，训练简单基线。 |
| [Google MLOps 参考架构](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) | 区分实现交付与新数据训练。 | 画出自己的手工步骤和批准点。 |
| [scikit-learn 常见陷阱](https://scikit-learn.org/stable/common_pitfalls.html) | 看懂预处理泄漏与不一致。 | 故意在全数据上拟合 scaler，再改回正确 Pipeline。 |
| [ML Test Score](https://storage.googleapis.com/gweb-research2023-media/pubtools/4156.pdf) | 从 2017 年原始论文理解生产检查。 | 选三项适用检查，实现最有风险的一项。 |
| [DVC 入门](https://doc.dvc.org/start) | 理解数据指针、cache 与 remote。 | 同时恢复旧代码和旧数据，验证摘要。 |
| [Made With ML](https://madewithml.com/) | 串联设计、数据、训练、测试与运行。 | 把一个 notebook 训练路径拆成脚本与测试。 |
| [MLOps Zoomcamp](https://datatalksclub.github.io/docs/courses/mlops-zoomcamp/) | 跟随完整项目学习 tracking、部署、调度、监控。 | 先复现一个模块，再迁移到自己的小数据。 |
| [Full Stack Deep Learning 2022](https://fullstackdeeplearning.com/course/2022/) | 深度学习项目的测试、数据与交付取舍。 | 做一个测试或部署 lab，记录旧 API 的替换。 |
| [Stanford CS329S](https://stanford-cs329s.github.io/) | 从需求出发做 ML 系统设计。 | 写一页延迟、数据、评估与失败权衡。 |
| [Designing Machine Learning Systems，作者页面](https://huyenchip.com/books/) | 系统性理解数据、重训和监控决策。书籍为商业出版物。 | 用一章的问题审查自己的设计。 |

按 2026-11-18 的课程说明，Zoomcamp 可自学，2026 年没有计划 live cohort；不要把历史 cohort 的证书说明当成本年度承诺。FSDL 是 2022 年课程，公开讲义与实验可用于学习；CS329S 的公开材料也不等于注册、学分或证书。

完成主实验后，用一次独立演练验收：换一个新目录，从下载包重新生成数据和训练；提交一条非法请求；发布 candidate 后恢复 baseline；在输出中找出实际模型身份；运行 shifted 监控并说明为什么告警。把命令、结果与解释留在自己的实验记录中。

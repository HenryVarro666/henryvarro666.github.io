---
title: "Apache Airflow 学习笔记：从第一个 Dag 到可重跑的数据管道"
date: 2026-11-13T19:49:00-06:00
description: "从任务依赖和数据区间入门，在本地运行结账指标流水线，练习补数、重试、幂等写入与故障排查。"
topics: [data-science, ml-systems]
---

每天出一张结账转化表，真正难答的是两个问题：**这次算的是哪天入组的人？失败后再跑一遍，结果会不会多一倍？** 例如，一位用户 11 月 1 日 23:00 入组。等到 11 月 2 日零点时只观察了一个小时，不能把还没下单当成 24 小时未下单。再比如，任务写完表却在汇报成功前断线，Airflow 会重试。如果脚本每次都追加，报表就会重复计数。

这篇笔记用 Python 和一份六人模拟数据，练习把日期、依赖和写入规则写进同一条流水线。会写 Python 函数就可以跟做。初次使用建议先看[任务怎么运行](#mental-model)，再[装 Airflow 跑实验](#lab)；日常维护者可以直接看[补数与排障](#recovery)。

## Airflow 管什么，计算引擎管什么 {#mental-model}

**Airflow 是工作流编排器。** 一个 Dag（directed acyclic graph，有向无环图）规定任务之间的依赖。比如先确认原始数据齐备，再运行查询，最后发布。每次触发会形成一个 **Dag run**；其中每个步骤的这一次执行叫 **task instance**。两个日期的运行共享同一份 Dag 定义，各有自己的任务状态和日志。[Airflow 架构](https://airflow.apache.org/docs/apache-airflow/3.4.0/core-concepts/overview.html) · [Dag run 与状态](https://airflow.apache.org/docs/apache-airflow/3.4.0/core-concepts/dag-run.html)

![Dag processor 解析定义；Scheduler 通过 executor 派发任务；任务可提交 Spark 作业；Airflow 元数据库保存运行状态。](orchestration.zh.svg "图 1：Airflow 保存调度状态。大批业务数据留在表或对象存储中，由计算引擎处理。")

| 名词 | 实际职责 | 遇到故障先看什么 |
|---|---|---|
| Dag processor | 导入 Python 定义，把 Dag 和任务结构序列化进元数据库。 | 文件是否放在配置的目录或 bundle，导入是否报错。 |
| Scheduler（调度器） | 根据 timetable、上游状态和并发限制决定何时运行。 | Dag 是否暂停、数据区间是否到期、上游是否成功。 |
| Executor（执行器） | 接收调度器派发的任务，按配置在本机、worker 或容器等环境执行。 | 排队时间、执行环境容量、依赖和任务日志。 |
| Triggerer | 为支持 deferral 的任务异步等待外部条件，等待期间可以释放 worker 执行槽。 | 是否部署了 triggerer，外部作业是否完成，等待是否超时。 |
| API server 与元数据库 | 前者提供界面和接口；后者保存任务、运行和配置等状态。 | Graph、运行历史、任务实例日志；业务行到目标表查。 |

**Operator** 是一种任务模板，例如运行 Python、提交 SQL 或等待外部作业。**TaskFlow** 的 `@task` 把 Python 函数声明成任务，`counts = compute_counts(window)` 在定义 Dag 时传的是结果引用，也声明了依赖；函数体要等任务实际运行才执行。小结果由 **XCom（cross-communication）** 传递。几百万行数据存共享存储，XCom 只传路径、分区名、行数或外部作业编号。[TaskFlow 教程](https://airflow.apache.org/docs/apache-airflow/3.4.0/tutorial/taskflow.html) · [XCom](https://airflow.apache.org/docs/apache-airflow/3.4.0/core-concepts/xcoms.html)

对一个短小脚本，cron 加日志就可能够用。需要看跨步骤依赖、补跑历史日期、定位哪个步骤失败时，Airflow 更有价值。大表关联交给 [Spark／PySpark]({{< relref "/notes/spark-pyspark" >}})，连续流的逐条处理交给流处理引擎；Airflow 可以定期提交它们的作业或检查结果。

## 数据日期不等于任务的开跑时间 {#time}

本例统计结账提醒实验：A 组保持原流程，B 组获分配提醒。用户入组后 24 小时内至少完成一笔购买，记作转化。即使 B 组用户没看到或没点开提醒，也留在分母。这与 [A/B Testing 笔记]({{< relref "/notes/ab-testing" >}}) 的按分配分析使用同一口径。

![11 月 1 日入组窗口，最晚用户的观察窗口，以及 11 月 2 日至 3 日的 Airflow 数据区间。](cohort-clock.zh.svg "图 2：分组日期、24 小时观察窗和 Airflow 的数据区间是三件不同的事。图中日期均为 UTC。")

| 时间 | 本例 | 用途 |
|---|---|---|
| 入组区间 | `[11-01 00:00, 11-02 00:00)` UTC | 选出 11 月 1 日入组的人。 |
| 个体观察窗 | 用户 `u06` 在 `[11-01 23:00, 11-02 23:00)` | 左端含入组时刻；恰好 24 小时的购买不计。 |
| Dag 的数据区间 | `[11-02 00:00, 11-03 00:00)` UTC | 此处代表用于等待队列成熟的那一天。调度到期时间是 11 月 3 日零点，实际开始可能更晚。 |
| 源数据完整到 | 至少 `11-03 00:00` UTC（不含该时刻） | 确认对应观察窗的事件都可查询。源可能晚到，经过零点本身不能证明完整。 |

**Logical date** 是运行的逻辑标记。本例的定时区间从 11 月 2 日零点开始，因此该次定时运行的 logical date 是 11 月 2 日。要查询哪批数据，代码直接用 `data_interval_start` 和 `data_interval_end` 计算，不从实际开跑时间或 `datetime.now()` 推算。手动运行则要求明确传 `cohort_date`，因为手动触发的区间取决于 timetable 和触发方式。

Airflow 3 的裸 cron 字符串（如 `@daily`）默认使用触发型 timetable，起止相同。想得到一个完整的日期区间，直接指定 `CronDataIntervalTimetable("0 0 * * *", timezone="UTC")`，示例不会依赖安装机器的默认配置。业务如果以本地日历日为准，另写清时区和夏令时；“本地下一天零点”和“经过 24 小时”在换时日可能不是同一时刻。[Timetables](https://airflow.apache.org/docs/apache-airflow/3.4.0/authoring-and-scheduling/timetable.html) · [手动运行与数据区间](https://airflow.apache.org/docs/apache-airflow/3.4.0/core-concepts/dag-run.html#manual-triggering-and-data-intervals)

## 动手：为六个用户跑一次转化日报 {#lab}

### 1. 建独立环境，下载 Dag {#setup}

实验固定 **Airflow 3.4.0、Python 3.12**，用本地 SQLite 保存元数据，无需云账号、Docker 或 Spark。macOS、Linux 直接在终端运行；Windows 可在 Windows Subsystem for Linux（WSL2）中操作。下载 [checkout_airflow_dag.py](/notes/airflow/checkout_airflow_dag.py) 放到终端的当前目录，然后执行下面的命令。版本约束固定 Airflow 已测试的依赖组合。[官方安装步骤](https://airflow.apache.org/docs/apache-airflow/3.4.0/start.html)

```bash
LAB_ROOT="$(mktemp -d)"
python3.12 -m venv "$LAB_ROOT/.venv"
source "$LAB_ROOT/.venv/bin/activate"
python -m pip install "apache-airflow==3.4.0" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.4.0/constraints-3.12.txt"

export AIRFLOW_HOME="$LAB_ROOT/airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$AIRFLOW_HOME/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
mkdir -p "$AIRFLOW__CORE__DAGS_FOLDER"
cp checkout_airflow_dag.py "$AIRFLOW__CORE__DAGS_FOLDER/"
airflow db migrate
```

`AIRFLOW_HOME` 把数据库、配置和日志放到这次实验的目录。若开新终端，先用原目录的路径重新激活虚拟环境、设置这些变量。先确认 `airflow version` 是 `3.4.0`。

### 2. 看依赖，跑一次 Dag {#first-run}

把下面三条当成运行顺序图。任务函数的调用在**定义**依赖，传给下游的是一个结果引用：

```python
window = plan_window()
counts = compute_counts(window)
publish(window, counts)
```

![plan_window 把区间传给 compute_counts 和 publish；compute_counts 只在数据完整时执行；publish 按入组日期写结果。](dag-recovery.zh.svg "图 3：二者通过小的 XCom 值传区间和计数，最后为给定日期写一个文件。")

`plan_window` 取日期和模拟源系统的完整性时间，`compute_counts` 检查完整性，再去重并数人，`publish` 校验计数后写 JSON（JavaScript Object Notation）。任何一个前置任务失败，默认依赖会阻止发布。本地用 `airflow dags test` 就能跑通这三个任务，无需启动调度器：

```bash
airflow dags test checkout_cohort --conf '{"cohort_date":"2026-11-01"}'
cat "$AIRFLOW_HOME/checkout-lab/2026-11-01.json"
```

输出的 `variants` 应该是：

```json
{
  "A": {"assigned": 3, "converted": 1},
  "B": {"assigned": 3, "converted": 2}
}
```

| 用户 | 入组后发生什么（UTC） | 算转化？ |
|---|---|---|
| `u01`（A） | 在入组时刻下单，随后又下一笔 | 是，只算一位用户 |
| `u02`（A） | 购买发生在入组之前 | 否 |
| `u03`（A） | 无购买 | 否，仍留在分母 |
| `u04`（B） | 次日 00:15 的购买在窗口内，同一事件投递两次 | 是，只算一位用户 |
| `u05`（B） | 距窗口右端还差一秒时购买 | 是 |
| `u06`（B） | 恰好在入组后 24 小时购买 | 否 |
| `u99` | 有购买、没有分配记录 | 整行不进入实验 |

这里的 1/3 和 2/3 是为验证边界构造的例子。它们和 [PySpark 实验]({{< relref "/notes/spark-pyspark" >}}) 应得的计数相同，不能用来判断提醒是否有用。要看任务图和日志，可以在另一终端重新进入同一环境、导出上面的变量，再启动本地服务：

```bash
AIRFLOW__API__HOST=127.0.0.1 airflow standalone
```

用终端提示的本地凭据打开 `http://127.0.0.1:8080`。这个命令会启动调度组件；若只想看手动测试，先保持 Dag 暂停。[调试 Dag](https://airflow.apache.org/docs/apache-airflow/3.4.0/core-concepts/debug.html)

### 3. 重跑与故障演练 {#failure-lab}

第一次运行后做两个试验：

```bash
# 同一天再跑：sha256 应该相同。
shasum -a 256 "$AIRFLOW_HOME/checkout-lab/2026-11-01.json"
airflow dags test checkout_cohort --conf '{"cohort_date":"2026-11-01"}'
shasum -a 256 "$AIRFLOW_HOME/checkout-lab/2026-11-01.json"

# 故意缩短完整性范围：这条命令预期失败。
airflow dags test checkout_cohort \
  --conf '{"cohort_date":"2026-11-01","simulate_incomplete":true}'
```

第二个试验应在 `compute_counts` 报 `Purchase source is incomplete for this cohort.`，`publish` 不运行，先前的 JSON 保持原样。**还没到齐**与**确认没买**是两回事。这里完整性由模拟输入直接给出；上线后从数据摄取系统或源表的提交清单读取。不要用“我看到了最新一条事件”的时间戳来推断之前都没有缺口。

本地文件先写在同目录的临时文件，再替换目标。同一输入写同一份结果，这叫**幂等（idempotency）**。即使写完后在上报成功前挂了，再跑也不会多出一批结果。对象存储不一定提供相同的重命名语义；生产写入可使用支持事务的表提交、带唯一键的 upsert，或写不可变版本再更新发布指针。幂等解决重试重复写；输入确有更正时，要允许同一日期产出有记录的新版本。[Airflow 任务编写实践](https://airflow.apache.org/docs/apache-airflow/3.4.0/best-practices.html)

## 什么时候重试，什么时候补数 {#recovery}

| 操作 | 适用情况 | 先确认 |
|---|---|---|
| **Retry**（重试） | 某次任务遇到短暂连接或写入故障。 | 保留业务键和输出路径，设有限次数、间隔和超时。永久错误直接报错。 |
| **Clear**（清除任务实例状态） | 已修好一个指定历史 run 中的失败步骤，要再执行它。 | 明确是否同时清理受影响的下游，已经发布的输出如何更正。 |
| **Catchup**（追赶） | 开启调度后，需要调度器创建缺失的历史区间运行。 | 要填的历史范围和负载；每个历史运行是否只读指定区间。 |
| **Backfill**（补数） | 主动为指定日期范围创建运行，包括按策略重算失败或已完成区间。 | 先预览哪些 logical date 会运行，再检查事件是否仍可读和产出是否可替换。 |

本例 `catchup=False`，所以恢复调度时不会自动把所有过去的日子补一遍。补数日期指 **Dag logical date**，比输出队列日期晚一天。下面只预览，不创建运行；实际候选也受已有运行和重算策略影响：

```bash
airflow backfill create --dag-id checkout_cohort \
  --from-date 2026-11-02 --to-date 2026-11-04 \
  --reprocess-behavior failed --max-active-runs 1 --dry-run
```

在空历史中，上面区间对应 11 月 1、2、3 日三个输出。已有运行时，以 dry run 打印出的日期为准：`none` 仅补缺失，`failed` 包含失败，`completed` 也允许对已完成的日期创建新运行；正在排队或运行中的实例不会再创建同日期的 backfill run。要执行补数，启动 `airflow standalone`，在界面取消暂停这个 Dag，再去掉 `--dry-run` 运行。这三个日期会使用同一套模拟事件相对时间。[补数参数和并发规则](https://airflow.apache.org/docs/apache-airflow/3.4.0/core-concepts/backfill.html)

### 排障顺序：先找没过的一关 {#debug}

| 看到的现象 | 具体检查 |
|---|---|
| 找不到 Dag | Dag 文件路径与运行环境；用 `airflow dags list-import-errors --local` 查当前文件导入错误。Dag 文件顶层只定义依赖，别在导入时联网或查业务表。 |
| 没出现计划运行 | Dag 是否暂停、start date、timetable、logical date、scheduler 和 dag processor 是否在运行。 |
| 任务一直 queued | 查看 executor/worker、pool 和任务并发限制，检查对应队列是否有人消费。 |
| 任务跑了又重试 | 查**第一条**失败日志，分清可重试的瞬时问题和字段、权限等确定性问题。调试先复用同一分区。 |
| Dag 成功但产出不对 | 核对任务决策的日期、源版本、行数和分母。Dag 状态按叶节点判断；一个总会运行并成功的清理任务（`all_done`）有可能掩盖上游失败。 |

## 从练习走到实际管道 {#production}

先用这个练习把时间定义写清，再把其中两处替换掉：`plan_window` 之后查真实源清单，`compute_counts` 提交 Spark 或数据仓库作业。可先用一个任务提交计算，等外部作业**真正完成**再校验和发布。如果等待时间长，可用提供者支持的可延迟模式和 triggerer。把外部作业编号和表分区放进日志或 XCom，分群明细放表。Spark 的示例脚本写到新目录；编排时可为每次尝试另建暂存位置，校验后再把该日期指向已提交的结果。

上线前写一张任务约定表：

| 约定 | 结账示例 |
|---|---|
| 输入 / 版本 | 首次有效分组记录、去重后的购买源；记录对象版本或表快照。 |
| 数据区间 / 门槛 | 协调世界时（UTC）自然日入组，逐人 24 小时窗口，购买源的完整性至少到入组区间结束 + 24 小时。 |
| 业务唯一键 | `experiment_id` × `cohort_date` × `variant`；需要行级追溯则另保留用户层。 |
| 验证 / 可观测 | 组别齐全、分配用户唯一、购买去重、`0 ≤ converted ≤ assigned`。记录输入版本、起止时间、组别总数和产出目标。 |
| 重跑 / 资源 | 同一输入重复执行结果一致；固定重算策略，用 pool 保护共享数据库。分开部署环境，用 Connections 和 secrets backend 管凭据。 |
| 分析下游 | 分开的分析步骤检查样本比例及实验口径，再计算置信区间。每天产出已成熟数据，不代表每天都可以用普通固定样本检验决定停不停。 |

需要等上游产出时，可以考虑 **Asset 调度**：上游发布约定的数据资产更新后触发 Dag。它能表达“何时有新数据”；消费者仍要校验更新涉及哪个分区、是否完整。需要按运行时发现的日期或文件创建若干任务，可以用 **Dynamic Task Mapping**，但不要给每行事件创建一个任务实例。[Asset 调度](https://airflow.apache.org/docs/apache-airflow/3.4.0/authoring-and-scheduling/asset-scheduling.html) · [动态任务映射](https://airflow.apache.org/docs/apache-airflow/3.4.0/authoring-and-scheduling/dynamic-task-mapping.html)

## 自测：改一个变量，再看任务怎么变 {#practice}

1. **11 月 4 日 00:00 UTC 的计划运行会写哪天的队列？** 数据区间是 11 月 3 日到 4 日，输出 11 月 2 日入组用户。直接看运行开始日期会错两天。
2. **把 `u06` 的购买提前一秒，哪个结果变？** 它从窗口外变成窗口内，B 变为 3/3；A 仍 1/3。修改本地 Dag 文件的样例事件，重跑验证。
3. **重试时从源系统读取“最新一批”，为什么不稳？** 新文件可能改变行数和结果，同一次运行就难以复核。固定日期和输入版本；确需接纳更正时，明确产生新版本并记录来源。
4. **为什么不能在 Spark 提交任务拿到作业编号后马上更新仪表板？** 作业编号只说明请求已提交。要等计算成功、检查分区和数据合同，再更新读取目标。

## 外部学习路线 {#resources}

| 什么时候读 | 资料 | 建议动手做什么 |
|---|---|---|
| 第一次装 | {{< new-tab "Apache Airflow Quick Start" "https://airflow.apache.org/docs/apache-airflow/3.4.0/start.html" >}} | 装好本地环境，找到 Graph、任务日志和运行历史。 |
| 第一次写 Dag | {{< new-tab "Airflow TaskFlow 教程" "https://airflow.apache.org/docs/apache-airflow/3.4.0/tutorial/taskflow.html" >}} | 用一个字典作为任务输出，观察依赖何时自动形成。 |
| 日期总差一天 | {{< new-tab "Airflow Dag Runs" "https://airflow.apache.org/docs/apache-airflow/3.4.0/core-concepts/dag-run.html" >}} · {{< new-tab "Timetables" "https://airflow.apache.org/docs/apache-airflow/3.4.0/authoring-and-scheduling/timetable.html" >}} | 并排写出 cron 触发时间、logical date 和实际处理区间。 |
| 试着运维 | {{< new-tab "Airflow Backfill" "https://airflow.apache.org/docs/apache-airflow/3.4.0/core-concepts/backfill.html" >}} · {{< new-tab "Debugging Dags" "https://airflow.apache.org/docs/apache-airflow/3.4.0/core-concepts/debug.html" >}} | 对两个日期做 dry run，再用 `dags test` 制造一次数据不全的失败。 |
| 要接生产 | {{< new-tab "Airflow Best Practices" "https://airflow.apache.org/docs/apache-airflow/3.4.0/best-practices.html" >}} · {{< new-tab "Deferrable Operators & Triggers" "https://airflow.apache.org/docs/apache-airflow/3.4.0/authoring-and-scheduling/deferring.html" >}} | 检查顶层导入、存储边界、重试和等待资源；把一次外部作业编号沿日志追到实际产出。 |

同系列：[因果推断]({{< relref "/notes/causal-inference" >}}) · [A/B Testing]({{< relref "/notes/ab-testing" >}}) · [Spark／PySpark]({{< relref "/notes/spark-pyspark" >}})。

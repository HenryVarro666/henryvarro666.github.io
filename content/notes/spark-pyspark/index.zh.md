---
title: "Spark 与 PySpark 学习笔记：从第一个 DataFrame 到批处理作业"
date: 2026-11-13T19:49:01-06:00
description: "理解 Spark SQL、分区、Shuffle 和 Join，用 PySpark 把结账事件变成实验指标，再读执行计划定位慢在哪里。"
topics: [data-science, ml-systems]
---

假设结账团队想知道，用户被分配到提醒策略后，有多少人在 24 小时内下单。同一个人可能买两次，同一笔购买也可能重复投递，晚上分配的用户还可能在第二天购买。直接数事件行会把指标算错。这个作业要保留整个分配人群，为每人算一条结果。下面用 PySpark 把过程跑通，先手算几条边界样例，再读 Spark 的执行计划。

读者只需会基本的 Python，能读懂一点结构化查询语言（SQL）。本地实验可预留 45–60 分钟。如何分配用户、量化不确定性，以及据此决定是否上线，见同一案例的 [A/B Testing 笔记]({{< relref "/notes/ab-testing" >}})。

## 什么时候该用 Spark {#mental-model}

Apache Spark 是一个计算引擎。它能把一次分析拆成多个数据分区，有需要时让多台机器并行处理。**PySpark** 是 Spark 的 Python 应用程序接口（API）。这篇笔记通过 DataFrame 使用 Spark SQL 引擎。DataFrame 描述了行、带类型的列以及要执行的运算。同样的关系运算可以用 SQL 或 DataFrame 表达，交给同一套优化器。**弹性分布式数据集（RDD）**是较底层的分布式集合抽象，遇到有表结构的数据，建议先学 DataFrame。[Spark SQL 指南](https://spark.apache.org/docs/4.0.1/sql-programming-guide.html) · [RDD 指南](https://spark.apache.org/docs/4.0.1/rdd-programming-guide.html)

| 工具 | 负责什么 | 例子 |
|---|---|---|
| pandas | 单个 Python 进程里的表格分析 | 在内存里检查一份导出的实验结果。 |
| Apache Spark / PySpark | 并行执行数据转换 | 在多台机器上连接较大的事件表和分组表。 |
| Parquet、对象存储或表格式 | 持久化数据 | 用 Parquet 保存结果。Delta Lake 等表格式还通过事务日志管理一致的表版本。 |
| Databricks | 托管计算、工作区和数据服务 | 在 Notebook 里写 Spark 转换，再交给托管任务运行。 |
| Apache Airflow | 工作流的定时、依赖和重试 | 等源分区到齐，提交 Spark 作业，检查后发布。 |

处理小数据时，可以先测一下简单工具。pandas 如果几秒就能完成，启动 Spark 和调度任务也要花时间。这次在本地学习 Spark 的执行方式，为以后判断何时需要集群做准备。托管环境见 [Databricks 笔记]({{< relref "/notes/databricks-guide" >}})，任务依赖和补跑见 [Airflow 笔记]({{< relref "/notes/airflow" >}})。

## DataFrame 是怎样执行的 {#execution}

![Python 程序描述 DataFrame 计划。Action 触发一个或多个 Job。集群管理器分配资源，Driver 调度 Stage，Executor 执行处理分区的 Task。](execution.zh.svg "图 1。Driver 负责规划和协调，Executor 运行分配到各分区的 Task。")

- **Driver（驱动进程）**运行协调程序，生成和优化计划，调度计算，并接收程序主动取回的结果。
- 在集群模式下，Kubernetes 等**集群管理器**负责分配资源。应用的 **Executor（执行进程）**运行 Task，也可能保存中间数据。
- 写入、计数等 **Action（行动操作）**可能触发一个或多个 **Job（作业）**。Spark 将工作划成多个 **Stage（阶段）**，Shuffle 常是阶段分界。一个 **Task（任务）**通常处理某个阶段的一个分区。重试会为该分区再执行一次。
- 本地实验用 `local[2]`，在一台机器上提供两个工作线程。可以用它验证计算口径和读计划。实际集群的网络开销和数据分布需要另外测量。[Spark 集群模式及术语](https://spark.apache.org/docs/4.0.1/cluster-overview.html)

### Transformation、Action 与列类型 {#dataframe-basics}

`select`、`filter`、`groupBy(...).agg(...)` 生成转换计划。`show`、`count`、写 Parquet 等 Action 会执行计算。没有持久化时，下一次 Action 可能重算同样的转换。`explain("formatted")` 可以在执行输出操作前打印物理计划。[DataFrame 快速入门](https://spark.apache.org/docs/4.0.1/api/python/getting_started/quickstart_df.html) · [Explain 的几种模式](https://spark.apache.org/docs/4.0.1/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.explain.html)

读数据时就规定列类型。字符串时间按字典顺序排序，格式和时区偏移也可能混用。应该解析成时间戳、检查无效值，并明确会话时区。金额用整数分或明确精度的小数类型。做连接前先确认两张表各自一行代表什么，特别是同一个键存在多行时。Spark SQL 中，普通比较遇到空值会得出 unknown，过滤只保留条件为 true 的行。[Spark SQL 空值语义](https://spark.apache.org/docs/4.0.1/sql-ref-null-semantics.html)

## 动手算结账转化 {#lab}

下列命令使用 **Python 3.12**、**Java 开发工具包（JDK）17** 和 **PySpark 4.0.1**。固定版本是为了能够复现实验。用其他版本时先看对应的安装要求。把 [Python 实验脚本](/notes/spark-pyspark/checkout-pyspark-lab.py) 下载到单独目录。脚本自己生成输入，在本机执行并写出 Parquet。Windows 可以在装好 Python 和 Java 的 WSL 等 Linux 环境中运行这些命令。[PySpark 4.0.1 安装说明](https://spark.apache.org/docs/4.0.1/api/python/getting_started/install.html)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install pyspark==4.0.1
python --version
java -version
python checkout-pyspark-lab.py --output spark-checkout-output --explain
```

macOS 上如果有多套 Java，可以用 `JAVA_HOME=$(/usr/libexec/java_home -v 17)` 指向 JDK 17，再将 `$JAVA_HOME/bin` 放到 `PATH` 开头。如果 Spark 报 Driver 与 Python worker 的次版本不一致，先查是否有旧的 `PYSPARK_PYTHON` 设置。脚本默认用启动它的解释器启动本地 worker。每次运行请指定一个新的输出目录。

想先试第一个 DataFrame，可以在同一个环境启动 `python`，输入：

```python
from pyspark.sql import SparkSession, functions as F

spark = SparkSession.builder.master("local[2]").appName("first-frame").getOrCreate()
purchases = spark.createDataFrame(
    [("u01", 1299), ("u01", 500), ("u02", 800)],
    "user_id STRING, amount_cents LONG",
)
totals = purchases.groupBy("user_id").agg(F.sum("amount_cents").alias("total_cents"))
totals.explain("formatted")
totals.orderBy("user_id").show()  # u01: 1799, u02: 800
spark.stop()
```

### 定义入组人群和观察窗 {#cohort}

这里读取 **2026-11-01 协调世界时（UTC）入组**的模拟数据。分组已经记在 A/B 实验日志里。每个入组用户都对应一行，包括从未购买的人。转化条件是 `assigned_at <= purchased_at < assigned_at + 24 hours` 时至少有一笔购买。用户实际看到或打开提醒，可以另外分析。

| 输入 | 一行是什么，列有哪些 | 必须检查什么 |
|---|---|---|
| 分组日志 | 每个入组用户一行。使用 `user_id`、`variant`、`assigned_at`。 | 每位用户恰好有一条 A 或 B 的分组记录。 |
| 购买投递 | 每条收到的记录一行。使用 `event_id`、`user_id`、`purchased_at`。 | 内容完全相同的重复投递只留一条。同一个事件编号对应不同内容时停止。 |

入组区间在 **11 月 2 日 00:00 UTC** 结束，该时刻本身不入组。样例的购买源声明事件时间覆盖到 **11 月 3 日 00:00 UTC**，同样是右端点不含。作业按 `入组区间结束 + 24 小时` 检查覆盖，所以最后入组的人也有完整观察窗。试着使用新输出目录并传 `--complete-through 2026-11-02T23:00:00Z`，脚本会报覆盖不足。实际工作中应从上游清单或入库协议确认覆盖。仅看文件中最大的时间戳，无法判断前面某个小时有没有漏数据。

![按 UTC 标出入组和购买数据区间。u05 在个人观察窗结束前一秒购买，计转化。u06 恰在结束时购买，落在观察窗之外。](checkout-window.zh.svg "图 2。每个人的观察窗从入组时刻算起。实验要求购买源覆盖到入组日结束后 24 小时。")

### 去重、左连接，最后回到一人一行 {#join}

样例用 `purchases.dropDuplicates()` 删除所有列都相同的记录。再按事件编号分组，检查是否还有冲突记录。真实的数据若允许更正，先定好使用哪个上游版本号识别更新。`dropDuplicates(["event_id"])` 在同一个编号对应不同内容时没有实现这个版本规则。

分组 DataFrame 起别名 `a`，购买 DataFrame 起别名 `p`。把用户键和观察窗条件都写在**左连接**的条件里。如果在连接后按右表时间字段过滤，没有购买的行会因为空值被删掉，只有窗外匹配的用户也可能消失。`count("p.event_id")` 跳过左连接产生的空值。`count("*")` 会把那一行计进去。[PySpark Join 参考](https://spark.apache.org/docs/4.0.1/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.join.html) · [聚合中的空值](https://spark.apache.org/docs/4.0.1/sql-ref-null-semantics.html#builtin-aggregate-expressions)

```python
condition = (
    (F.col("a.user_id") == F.col("p.user_id"))
    & (F.col("p.purchased_at") >= F.col("a.assigned_at"))
    & (F.col("p.purchased_at") < F.col("a.assigned_at") + F.expr("INTERVAL 24 HOURS"))
)
outcomes = (
    a.join(p, condition, "left")
    .groupBy("a.user_id", "a.variant", "a.assigned_at")
    .agg(F.count("p.event_id").alias("purchase_events"))
    .withColumn("converted", (F.col("purchase_events") > 0).cast("int"))
)
```

每对匹配的记录都会产出一行。同一个键若有 2 条分组记录和 3 条符合条件的购买，连接可能得到 6 行。先在应该唯一的表上核对键，再处理预期的一对多。这里允许一个人购买多次，聚合后恢复一人一行。实验分母是已分组用户数，分子是这些用户的 0／1 结果之和。

| 用户 | 组别 | 窗内购买事件数 | 是否转化 | 核查什么 |
|---|---:|---:|---:|---|
| u01 | A | 2 | 1 | 刚分组就购买计入，两笔购买仍只算一人。 |
| u02 | A | 0 | 0 | 分组前一秒的购买在窗外。 |
| u03 | A | 0 | 0 | 没有购买的用户仍在分母。 |
| u04 | B | 1 | 1 | 次日凌晨的合法购买投递两次，只留一次。 |
| u05 | B | 1 | 1 | 观察窗结束前一秒的购买计入。 |
| u06 | B | 0 | 0 | 观察窗结束时刻的购买在窗外。 |

脚本还造了一笔未入组用户 `u99` 的购买，其用户结果表里没有对应行。输出路径下有 `user_outcomes`（每人一行）和 `variant_summary`（每组一行）两个 Parquet 数据集，每个数据集的目录下可能有多个分片文件。比较两次运行时先按键排序。预期 A 组 **1/3**，B 组 **2/3**。这六行用于核对取数口径。[A/B Testing 笔记]({{< relref "/notes/ab-testing#readout" >}})里另有样本量规划与正式读数示例。[因果推断笔记]({{< relref "/notes/causal-inference" >}})介绍历史分配策略带来的混杂。

### 读回 Parquet 结果 {#read-output}

在实验目录开启新的 Python 会话。如果启动实验时指定了其他 `--output`，修改读取路径。下面用 Spark SQL 复算两组转化率。

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[2]").appName("read-outcomes").getOrCreate()
spark.read.parquet("spark-checkout-output/user_outcomes").createOrReplaceTempView("outcomes")

spark.sql("""
    SELECT variant, COUNT(*) AS assigned_users, SUM(converted) AS converted_users,
           AVG(converted) AS conversion_rate
    FROM outcomes
    GROUP BY variant
    ORDER BY variant
""").show(truncate=False)
spark.stop()
```

对照脚本里的 DataFrame 聚合和 `variant_summary` 数据集。`createOrReplaceTempView` 将 DataFrame 注册为当前会话的 SQL 视图。会话结束后临时视图就消失。

## 在执行计划里追踪 Shuffle {#shuffle}

**分区（partition）**是一份 Task 可以处理的数据切片。选列和过滤通常各分区可以独立做，这类操作称为**窄依赖**操作。按键分组时，同一个键可能散在多个输入分区里。**Shuffle** 会重分布记录或局部聚合结果，让同一个键去同一个目标分区。跨机器传输、序列化、排序和溢写磁盘都可能抬高成本。[Shuffle 操作](https://spark.apache.org/docs/4.0.1/rdd-programming-guide.html#shuffle-operations)

![两个输入分区分别更新 A、B 的局部计数。Shuffle 把相同组别送到同一个分区，最后得到 A 组 1/3 和 B 组 2/3。](shuffle.zh.svg "图 3。Spark 可以先在分区内聚合，再按组别交换局部结果。图中为便于阅读把 A 和 B 画在不同分区，实际一个哈希分区可以容纳多个键。")

在脚本打印的格式化计划里，找到 `SortMergeJoin LeftOuter`、连接前按 `user_id` 重分区的 `Exchange`，以及组别聚合前按 `variant` 重分区的 `Exchange`。局部和最终聚合可能都显示为 `HashAggregate`。`AdaptiveSparkPlan` 表示启用了**自适应查询执行（AQE）**。运行中有了统计信息，优化器可能改用另一种连接策略，或合并小分区。Action 前打印的是初始计划，运行后到 SQL 界面看实际采用的计划。[Spark SQL 调优](https://spark.apache.org/docs/4.0.1/sql-performance-tuning.html) · [Spark Web UI](https://spark.apache.org/docs/4.0.1/web-ui.html#sql-tab)

### 根据证据调优 {#diagnosis}

| 现象 | 先检查和尝试什么 |
|---|---|
| 读取字节数大，打开了太多文件 | 只读必要日期和列。在 Parquet 扫描里找 `PartitionFilters`、`PushedFilters` 和 `ReadSchema`。按经常过滤且取值数量合适的列组织存储，例如事件日期。 |
| 连接占据大部分时间 | 先查键重复次数和两个输入的大小。如果用大事实表连小维表，Spark 在合适的连接类型上可以广播小表。手工加广播提示前，检查构建侧大小和内存。 |
| 一两个 Task 比其他慢很多 | 在 Stages 界面比较 Task 的 Shuffle 读量和耗时。少数高频键可能让一个 Task 得到远多于其他 Task 的数据。查键频次和实际计划是否处理了倾斜。口径允许时可以先聚合或单独处理热点键。 |
| 小 Task 或小文件太多 | 分别检查运行时分区数和存储布局。按取值极多的标识符做目录分区容易生成小文件。按下游访问方式决定压实或重分区。数据分布合适时，可用 `coalesce` 减少 Task 分区。 |
| 花在 Python 上的时间多 | 优先用 `when`、时间戳函数和聚合等内置列函数。确需 Python 用户定义函数（UDF）时，先测量成本，再看操作能否使用 Arrow 支持的向量化 pandas UDF。 |
| Driver 内存不足 | 查是否用了 `collect()` 或 `toPandas()`。明细保留在分布式数据集中，只取有大小上限的样本或汇总。 |

`repartition(n, "user_id")` 可以 Shuffle 数据生成新的运行时分区。`write.partitionBy("event_date")` 则按日期组织 Parquet 输出目录，让读端可以跳过无关日期。一个目录可含多个文件，运行时也可能把多个小文件交给同一个 Task，或拆分较大的输入。只有计算代价高且会反复读取的 DataFrame 才值得考虑缓存，用完调用 `unpersist()`。[Parquet 分区发现](https://spark.apache.org/docs/4.0.1/sql-data-sources-parquet.html#partition-discovery) · [SQL 调优参考](https://spark.apache.org/docs/4.0.1/sql-performance-tuning.html) · [Arrow 与 Python UDF](https://spark.apache.org/docs/4.0.1/api/python/tutorial/sql/arrow_pandas.html)

Spark 运行时，Driver 的本地网页通常在 `http://localhost:4040`。到 SQL 页找较慢的查询，再点进相关 Stage 看任务耗时、Shuffle 读写量和磁盘溢写。实验脚本写完并显示结果后就退出。要现场查看，可用交互会话让应用继续运行。要事后查看，可启用 Spark 事件日志并通过 History Server 读取。[监控与事件日志](https://spark.apache.org/docs/4.0.1/monitoring.html)

## 从样例变成定期运行的作业 {#production}

1. **写清输入约定。** 定义分组主键和跨日的首次合格入组规则，明确事件编号、版本规则、时区及每次读取的源快照覆盖范围。产出使用已记录的分组。
2. **放大之前先查边界。** 给没有购买、窗首、窗尾、重复投递、事件编号冲突写出预期行。要求各组分母与分组表对得上，且 `converted_users <= assigned_users`。[PySpark 测试工具](https://spark.apache.org/docs/4.0.1/api/python/getting_started/testing_pyspark.html)
3. **同一个区间可以重算。** 每次传入明确的入组区间和数据源版本。先写临时结果，校验后按目的地的提交方式发布一个入组分区。若每次重试都无条件追加，报表会重复计算。普通 Parquet 文件负责保存数据，发布和并发规则要单独定。
4. **提交到集群并记录应用 ID。** 为 Driver 和 Executor 准备兼容的 Python 和 Spark 依赖。把应用 ID 与产出关联起来，用有代表性的数据看输入扫描量、Shuffle 体积和总耗时，再决定改哪里。[提交 Spark 应用](https://spark.apache.org/docs/4.0.1/submitting-applications.html)

### 批处理与 Structured Streaming {#streaming}

如果决策只需要每天一批已经结束观察的用户，可以先用批处理。**Structured Streaming（结构化流处理）**让 DataFrame 运算随数据到达而增量执行。带状态的操作可用 **watermark（水位线）**声明允许迟到多久，供 Spark 管理状态。**Checkpoint（检查点）**保存运行进度和状态，便于恢复。超过迟到阈值的记录可能无法进入结果，重试是否导致输出重复还取决于写入端。流处理的水位线无法证明整个实验观察窗的事件都已到齐。上线前应定好迟到／更正策略，并挑一个已经结束的区间与批处理结果核对。[Structured Streaming 概览](https://spark.apache.org/docs/4.0.1/streaming/index.html) · [水位线和检查点](https://spark.apache.org/docs/4.0.1/streaming/apis-on-dataframes-and-datasets.html)

## 练习和参考答案 {#exercises}

**1. 改成内连接有什么后果？** 把连接方式改成 `inner`，使用新路径运行。剩哪些用户，两组转化率怎么算？

> **答案。** 只剩 u01、u04、u05。A 组变成 1/1，B 组是 2/2。没有窗内匹配的用户已经从分母消失。改回左连接，对比每位用户的结果和组汇总。

**2. 哪种重复需要报错？** 复制一条 `p04`，将购买时间改为 00:16 UTC，再运行会怎样？

> **答案。** 作业停止。全行去重后，`p04` 仍有两条不同时间的记录。真实更正需要可靠的上游版本或将冲突隔离处理的规则。

**3. 广播能省掉所有 Shuffle 吗？** 在微型购买表上加广播提示，对照执行计划和结果。

```python
p = F.broadcast(unique_purchases).alias("p")
```

> **答案。** 此处有等值用户键与附加时间筛选，连接可改用 `BroadcastHashJoin`。分组和去重检查仍可能需要 Shuffle。先比对每行结果，再讨论性能。

**4. 为什么重跑后多了一批行？** 生产作业把同一入组区间的结果再追加到存储。怎样避免自动报表重复计算？

> **答案。** 为输出设唯一键，例如 `(experiment_id, cohort_day, variant)`。先校验暂存结果，再用表事务或具有唯一运行清单的发布流程，每个区间只发布一次。更正重算时，记录哪个快照替换旧版。

## 按顺序读这些资料 {#resources}

| 资料 | 怎么用 |
|---|---|
| {{< new-tab "PySpark DataFrame 快速入门" "https://spark.apache.org/docs/4.0.1/api/python/getting_started/quickstart_df.html" >}} | 跑建表、选列、聚合几节，再把一个 DataFrame 查询改成临时视图上的 SQL。 |
| {{< new-tab "Spark 集群模式" "https://spark.apache.org/docs/4.0.1/cluster-overview.html" >}} | 对照 Driver、集群管理器和 Executor，在 Spark UI 找到对应的 Job、Stage、Task。 |
| {{< new-tab "Spark SQL 性能调优" "https://spark.apache.org/docs/4.0.1/sql-performance-tuning.html" >}} | 读连接策略和 AQE，再做本文的广播练习。 |
| {{< new-tab "PySpark 测试指南" "https://spark.apache.org/docs/4.0.1/api/python/getting_started/testing_pyspark.html" >}} | 把六位用户的预期结果写成 DataFrame，与实际输出逐行比较。 |
| {{< new-tab "Spark SQL 论文" "https://people.csail.mit.edu/matei/papers/2015/sigmod_spark_sql.pdf" >}} | 先读 DataFrame 和 Catalyst 两节，理解接口与优化器怎样配合。 |
| {{< new-tab "Structured Streaming API 指南" "https://spark.apache.org/docs/4.0.1/streaming/apis-on-dataframes-and-datasets.html" >}} | 要实现有状态的流式聚合时，先读事件时间和故障恢复。 |

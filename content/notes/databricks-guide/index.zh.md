---
title: "Databricks 学习笔记：从第一条 SQL 到支付数据流水线"
date: 2026-11-13T18:30:00-06:00
description: "在 Free Edition 上用模拟数据练习 Delta Lake 分层、增量更新和历史查询，再接到仪表板、定时作业和 MLflow。"
topics: [data-science, ml-systems]
---

一笔支付可能被重复投递，金额更正也可能在日报发出以后才到。如果按收到的记录逐条相加，重复记录会被算两次，更正金额还会叠加到原金额上。这个练习会在 Databricks 中保留原始事件，找出每笔支付最新的有效版本，再让报表从清洗后的表取数。先用结构化查询语言（SQL）走一遍，再用 PySpark 写同一条聚合。跑通以后，可以把结果做成仪表板、交给定时任务，或沿着同一条数据链准备模型训练集。

适合能读懂基本 SQL、第一次用 Databricks 的读者。第一次来可以直接[开工作区](#setup)，再[动手跑实验](#lab)；已经有工作区的话，跳到[增量合并](#lab-merge)或[应用章节](#applications)。

## 一条查询经过哪些组件 {#mental-model}

Databricks 是一个托管数据平台，用来处理数据、提供报表和训练模型。工作区是你写代码的地方，计算资源负责执行，表和文件有独立的存储位置。

比如在 Notebook 里查询 `samples.nyctaxi.trips` 时，**Unity Catalog** 解析表名并检查访问权限，Notebook 所连接的**计算资源**执行查询。如果读的是 **Delta 表**，执行引擎会看 Delta 事务日志，找出当前快照包含哪些文件。计算资源停掉之后，数据文件还在，下次连上计算资源就可以继续查。[Databricks 查询数据](https://docs.databricks.com/aws/en/query) · [Delta 事务](https://docs.databricks.com/aws/en/lakehouse/acid)

![Notebook、查询和仪表板把工作交给计算资源。计算资源访问对象存储，Unity Catalog 管理访问权限，Lakeflow Jobs 调度任务。](platform-map.svg "图 1：计算资源负责执行，Delta 表保存在对象存储中。Unity Catalog 管理访问，Lakeflow Jobs 调度任务。")

| 组件 | 做什么 | 在哪里遇到 |
|---|---|---|
| 对象存储 | 保存数据文件与元数据文件。Parquet 按列存储，查询可以只读取用到的列。 | 托管表或卷背后的存储 |
| Delta Lake | 在数据文件之上加事务日志。每次提交记录文件变更并形成表的新版本，读者可以在写入期间读取一致的快照。 | `MERGE`、`DESCRIBE HISTORY`、查询历史版本 |
| Apache Spark 与 PySpark | Spark 分布式执行数据处理。PySpark 是它的 Python 应用程序接口（API）。 | 在 Notebook 里用 DataFrame，例如 `spark.table(...).groupBy(...)` |
| Databricks SQL Warehouse | 给查询编辑器、仪表板或 SQL Notebook 提供计算资源。 | SQL 编辑器顶部的计算资源选择器 |
| Unity Catalog | 用 `catalog.schema.table` 这样的名称组织表、视图、卷和模型，管理权限，并为支持的操作记录数据血缘。 | Catalog Explorer 和权限授予页面 |
| Lakeflow Jobs | 按依赖关系运行任务并记录结果。任务可以是 Notebook、SQL 文件或流水线。 | 作业的依赖图与运行历史 |

开源 Spark 和 Delta Lake 也可以脱离 Databricks 运行。Databricks 将它们和托管计算、工作区、权限管理及调度工具放在一起。**工作区（workspace）**是打开编辑器、创建作业的地方。**目录（catalog）**是数据对象地址的一部分。Unity Catalog 负责托管表的存储生命周期，外部表对应的底层数据则由团队管理。用**卷（volume）**管理逗号分隔值（CSV）、图像这类文件，用**表**保存需要按列查询的记录。[Databricks 概览](https://docs.databricks.com/aws/en/introduction) · [Unity Catalog](https://docs.databricks.com/aws/en/data-governance/unity-catalog) · [表的类型](https://docs.databricks.com/aws/en/tables/types) · [卷中的文件](https://docs.databricks.com/aws/en/volumes/volume-files)

### 为什么分 Bronze、Silver 和 Gold

**Lakehouse（湖仓）**让分析和机器学习可以共用对象存储里的数据。**Medallion 分层**约定每层的数据可以用来做什么。Bronze 保留收到的事件，方便审计和重放。Silver 转换类型、处理重复和更正，明确一行代表一笔支付。Gold 为下游定义指标，例如按支付日期和渠道各汇总一行。建表前先写清楚**数据粒度**（一行是什么）和**键**（靠哪些字段识别这一行）。[Medallion 架构](https://docs.databricks.com/aws/en/lakehouse/medallion) · [湖仓设计论文](https://www.cidrdb.org/cidr2021/papers/cidr2021_paper17.pdf)

## 打开工作区并运行第一条 SQL {#setup}

Databricks Free Edition 面向个人学习和非商业用途，提供有用量限制的无服务器（serverless）计算资源。面向企业的免费试用是另一种注册选项，提供有时限的额度。这篇笔记只用公开样例和模拟数据。步骤以 Amazon Web Services（AWS）版 Databricks 文档为准，菜单和功能会随云平台及工作区配置变化。[开通 Free Edition](https://www.databricks.com/learn/free-edition) · [对比注册选项](https://docs.databricks.com/aws/en/getting-started/free-trial-vs-free-edition) · [Free Edition 限制](https://docs.databricks.com/aws/en/getting-started/free-edition-limitations)

点 **New → Notebook**，语言选 **SQL**，在顶部连接 **Serverless** 或 SQL Warehouse。要用 Python 或 PySpark 时，选择 **Serverless**。每次按 **Shift+Enter** 运行一个单元格。先跑下面这条确认计算资源连通。也可以在 **New → SQL query** 里选一个 Warehouse 运行它。[Notebook 快速入门](https://docs.databricks.com/aws/en/getting-started/quick-start) · [Notebook 可用的计算资源](https://docs.databricks.com/aws/en/notebooks/notebook-compute)

```sql
SELECT 1 AS ready;
```

如果想先看一眼公开数据，可以读内置的纽约出租车样例。这里 `fare_amount` 是车费字段。要定义报表口径，先打开 Catalog Explorer 看字段和说明。[内置数据集](https://docs.databricks.com/aws/en/discover/databricks-datasets)

```sql
SELECT pickup_zip, trip_distance, fare_amount
FROM samples.nyctaxi.trips
LIMIT 10;
```

如果第一条能跑、出租车表却读不了，先检查可见的目录。下面的主实验自己生成数据，只需要一个有建表权限的 schema。在共享工作区里，可以使用团队分配给你的练习 schema。[创建第一张表并授予权限](https://docs.databricks.com/aws/en/getting-started/create-table)

如果 `SELECT 1` 都运行不了，先查计算资源是否已连接、有无使用权限，或 Free Edition 是否已到配额。能查询样例却不能建表时，去 Catalog Explorer 核实当前 schema 和建表权限。

## 搭建支付流水线并验证重跑结果 {#lab}

**实验目标：**按日期和渠道回答“有多少笔支付获批，获批金额是多少？”再看一批更正到达后，报表怎样变化。第一次做预留 45–60 分钟。整个练习只有 10 条模拟投递记录，无需下载数据集。

![原始投递先经过校验和去重，再汇总成日报。无效金额另走一条支路。](payments-flow.svg "图 2：Bronze 保留收到的记录，Silver 保留每笔支付最新的有效版本，Gold 按事件日期和渠道汇总。")

### 1. 选择 catalog 和 schema {#lab-setup}

打开 **New → Notebook**，语言选 **SQL**，连接 notebook 的 serverless compute。下面按顺序执行。需要查看各步结果时，把每条语句放在单独的 cell。也可以[下载 SQL notebook 源文件](/notes/databricks-guide/payments-lab.sql)，在 **Workspace** 里导入，按顺序运行已经拆好的单元格。[导入 notebook](https://docs.databricks.com/aws/en/notebooks/notebook-export-import)

```sql
SHOW CATALOGS;
```

把下面的 `workspace` 换成你有权写入的 catalog。建 schema 需要 catalog 上的 `USE CATALOG` 和 `CREATE SCHEMA`。如果管理员已经分配了练习 schema，把 `databricks_notes_lab` 换成那个名字，省略 `CREATE SCHEMA`；要有 `USE CATALOG`、`USE SCHEMA` 和 `CREATE TABLE`。请用专门的练习空间，下面的 `CREATE OR REPLACE` 会覆盖同名对象。

```sql
USE CATALOG `workspace`;
CREATE SCHEMA IF NOT EXISTS databricks_notes_lab;
USE SCHEMA databricks_notes_lab;
SELECT current_catalog() AS catalog_name, current_schema() AS schema_name;
```

### 2. Bronze：按收到的样子保留投递记录 {#lab-bronze}

一笔支付有固定的 `payment_id`，每次投递有自己的 `delivery_id`。数据源给同一笔支付的变更编号 `event_version`，较大的编号覆盖较小的。这是**业务版本**，与 Delta 表自己的版本号无关。本例金额都用美元，校验后存为整数美分。日期直接给出，避免依赖会话时区。

```sql
CREATE OR REPLACE TABLE payments_bronze USING DELTA AS
SELECT * FROM VALUES
  (1, 1, 'p01', 1, DATE '2026-11-01', 'web',   'approved', '2499'),
  (1, 2, 'p01', 1, DATE '2026-11-01', 'web',   'approved', '2499'),
  (1, 3, 'p02', 1, DATE '2026-11-01', 'store', 'declined', '1800'),
  (1, 4, 'p03', 2, DATE '2026-11-02', 'web',   'approved', '3100'),
  (1, 5, 'p04', 1, DATE '2026-11-02', 'store', 'approved', '2400'),
  (1, 6, 'p_bad', 1, DATE '2026-11-02', 'web', 'approved', 'oops')
AS v(batch_id, delivery_id, payment_id, event_version, event_day,
     channel, status, amount_raw);

SELECT * FROM payments_bronze ORDER BY delivery_id;
```

这里有 **6 条投递**。`p01` 的第 1 版投递了两次，还有一条记录的金额格式不对。原样保留这些记录，才能回头核对数据源发了什么。实验用的建表语句可以重置样例；实际管道一般按约定向 Bronze 追加新记录。

### 3. Silver：先校验，再挑选有效版本 {#lab-silver}

先给坏记录写明原因，再把它们从有效数据里筛掉。`TRY_CAST` 遇到 `oops` 这样的金额时返回空值。下面的规则会把这条记录送进拒绝记录视图。每行保留第一个命中的规则。

```sql
CREATE OR REPLACE VIEW payments_checked AS
WITH typed AS (
  SELECT *, TRY_CAST(amount_raw AS BIGINT) AS amount_cents
  FROM payments_bronze
)
SELECT *,
  CASE
    WHEN payment_id IS NULL OR TRIM(payment_id) = ''
      OR event_version IS NULL OR event_version < 1
      OR event_day IS NULL THEN 'invalid_key_version_or_day'
    WHEN channel IS NULL OR channel NOT IN ('web', 'store', 'mobile')
      THEN 'invalid_channel'
    WHEN status IS NULL OR status NOT IN ('approved', 'declined')
      THEN 'invalid_status'
    WHEN amount_cents IS NULL OR amount_cents <= 0
      THEN 'invalid_amount'
  END AS reject_reason
FROM typed;

CREATE OR REPLACE VIEW payments_rejects AS
SELECT * FROM payments_checked WHERE reject_reason IS NOT NULL;

SELECT payment_id, amount_raw, reject_reason FROM payments_rejects;
```

预期拒绝记录只有 **`p_bad | oops | invalid_amount`**。接着用 `ROW_NUMBER` 为每个批次的每笔支付挑出最高版本。样例中相同版本的正文完全一致，取较大的投递编号只是为了固定结果。实际数据如果同一个键、同一个版本却有不同内容，要记录冲突，或请上游提供可信的排序规则。

```sql
CREATE OR REPLACE VIEW payments_batch_latest AS
SELECT batch_id, payment_id, event_version, event_day, channel, status, amount_cents
FROM (
  SELECT *, ROW_NUMBER() OVER (
    PARTITION BY batch_id, payment_id
    ORDER BY event_version DESC, delivery_id DESC
  ) AS row_num
  FROM payments_checked
  WHERE reject_reason IS NULL
) AS ranked
WHERE row_num = 1;

CREATE OR REPLACE TABLE payments_silver USING DELTA AS
SELECT payment_id, event_version, event_day, channel, status, amount_cents
FROM payments_batch_latest
WHERE batch_id = 1;

DESCRIBE HISTORY payments_silver;
```

Silver 中有 **4 笔支付**。在做更正前，从 `DESCRIBE HISTORY` 记下第一行的 **Delta 表版本号**。这就是稍后要用来读取旧答案的版本；重置实验后要重新查看。

### 4. Gold：先说清楚一行代表什么 {#lab-gold}

一行 Gold 代表一个**事件日期 × 渠道**。`payment_count` 包含所有当前支付，包括被拒绝的；金额只统计获批支付。这份样例报表衡量的是批准情况，不能拿来当银行结算报表。

```sql
CREATE OR REPLACE VIEW payments_daily AS
SELECT
  event_day,
  channel,
  COUNT(*) AS payment_count,
  COUNT_IF(status = 'approved') AS approved_count,
  SUM(CASE WHEN status = 'approved' THEN amount_cents ELSE 0 END) AS approved_cents
FROM payments_silver
GROUP BY event_day, channel;

SELECT * FROM payments_daily ORDER BY event_day, channel;

SELECT SUM(payment_count) AS payments,
       SUM(approved_count) AS approved,
       SUM(approved_cents) AS approved_cents
FROM payments_daily;
```

此时合计为 **4 笔支付、3 笔获批、7,999 美分**（79.99 美元）。Gold 在这里是普通视图，每次查询都读当时的 Silver，无需单独刷新。

### 5. 应用增量、重放，再查旧版本 {#lab-merge}

追加第二批，**只执行一次 INSERT**。它更正 `p02` 和 `p04`，带来一条迟到的 `p03` 旧版本，还新增 `p05`。

```sql
INSERT INTO payments_bronze VALUES
  (2, 7,  'p02', 2, DATE '2026-11-01', 'store',  'approved', '2100'),
  (2, 8,  'p04', 2, DATE '2026-11-02', 'store',  'declined', '2400'),
  (2, 9,  'p03', 1, DATE '2026-11-02', 'web',    'declined', '3000'),
  (2, 10, 'p05', 1, DATE '2026-11-02', 'mobile', 'declined', '1200');
```

执行增量合并。用业务主键写 `ON`，把版本判断放在 `WHEN MATCHED`。如果把版本判断写到 `ON` 里，迟到的旧记录可能被当成“没匹配到”，从而插入重复的支付。同一批次内，每个 ID 最多有一条记录进入合并。[Delta MERGE 示例](https://docs.databricks.com/aws/en/delta/merge)

```sql
MERGE INTO payments_silver AS target
USING (SELECT * FROM payments_batch_latest WHERE batch_id = 2) AS incoming
ON target.payment_id = incoming.payment_id
WHEN MATCHED AND incoming.event_version > target.event_version THEN
  UPDATE SET
    event_version = incoming.event_version,
    event_day = incoming.event_day,
    channel = incoming.channel,
    status = incoming.status,
    amount_cents = incoming.amount_cents
WHEN NOT MATCHED THEN
  INSERT (payment_id, event_version, event_day, channel, status, amount_cents)
  VALUES (incoming.payment_id, incoming.event_version, incoming.event_day,
          incoming.channel, incoming.status, incoming.amount_cents);

SELECT payment_id, event_version, status, amount_cents
FROM payments_silver ORDER BY payment_id;
```

接着**只重跑 MERGE，不重复执行 Bronze 的 INSERT**。核对每笔记录和报表合计：`p03` 仍是第 2 版、获批、3,100 美分；`p02` 改成获批，`p04` 改成拒绝。获批笔数仍是 3，获批总额比之前少 300 美分。

| 检查点 | Bronze 投递数 | Silver 支付数 | 获批笔数 | 获批美分 |
|---|---:|---:|---:|---:|
| 第一批 | 6 | 4 | 3 | 7,999 |
| 第二批合并后 | 10 | 5 | 3 | 7,699 |
| 重跑第二批的 `MERGE` | 10 | 5 | 3 | 7,699 |

把下面的 `0` 换成先前记下的**实际表版本号**，再读取旧快照：

```sql
SELECT COUNT(*) AS payments,
       COUNT_IF(status = 'approved') AS approved,
       SUM(CASE WHEN status = 'approved' THEN amount_cents ELSE 0 END) AS approved_cents
FROM payments_silver VERSION AS OF 0;
```

旧快照返回 **4 / 3 / 7,999**，当前 Gold 视图返回 **5 / 3 / 7,699**。Time travel 是一次历史读取，不会把当前表回滚过去。旧日志和对应的数据文件还在才能读取。长期归档要单独设计。[Delta 历史与保留策略](https://docs.databricks.com/aws/en/tables/history)

### 选做：用 PySpark 写一次 Gold {#lab-pyspark}

另开一个连到 notebook compute 的 **Python** notebook。先换掉三段式名称里的 `workspace`。下面用 DataFrame 表达同样的分组和条件统计。

```python
from pyspark.sql import functions as F

silver = spark.table("`workspace`.databricks_notes_lab.payments_silver")
approved = F.col("status") == "approved"
daily = silver.groupBy("event_day", "channel").agg(
    F.count("*").alias("payment_count"),
    F.sum(F.when(approved, 1).otherwise(0)).alias("approved_count"),
    F.sum(F.when(approved, F.col("amount_cents")).otherwise(0)).alias("approved_cents"),
)
display(daily.orderBy("event_day", "channel"))
```

`groupBy` 先构建计算计划，`display` 才要求 Databricks 执行并展示结果。和 SQL 视图逐行对照：第二批之后应当都有五组，数字也相同。真实数据量大时，把连接和聚合留在 Spark，只把小结果取到本地 Python。[PySpark DataFrame 入门](https://spark.apache.org/docs/latest/api/python/getting_started/quickstart_df.html)

## 把结果接到报表、定时任务和模型 {#applications}

### 用 Gold 视图做仪表板 {#dashboard}

点 **New → Dashboard → Data → Add SQL dataset**，创建 Databricks 的人工智能与商业智能（AI/BI）仪表板。把下方 SQL 的 catalog 换成自己的，用 SQL Warehouse 执行。在画布上加一个柱状图，横轴选 `event_day`，纵轴对 `approved_cents` 求和，按 `channel` 分组，再加日期筛选。展示美元金额时用美分除以 100，并在图表上注明单位。[创建 AI/BI 仪表板](https://docs.databricks.com/aws/en/dashboards/tutorials/create-dashboard)

```sql
SELECT event_day, channel, payment_count, approved_count, approved_cents
FROM `your_catalog`.databricks_notes_lab.payments_daily;
```

在 **Data** 页打开数据集，点 **Add custom calculation** 添加[计算指标](https://docs.databricks.com/aws/en/dashboards/manage/data-modeling/custom-calculations)：`TRY_DIVIDE(SUM(approved_count), SUM(payment_count))`。直接平均各渠道的通过率会给交易量悬殊的渠道相同权重。跑完第二批后，选中所有日期与渠道应该看到 **3 / 5 = 60%** 的支付通过率，以及总计 **7,699 美分**的通过金额。这些数字反映实验中每笔支付的最新状态。给报表写上底层数据更新时间，别人才能看出是查询刚刷新，还是源数据刚入库。分享仪表板时，确认它采用发布者的权限还是每位查看者自己的权限。[仪表板的数据权限](https://docs.databricks.com/aws/en/dashboards/share/share) · [定时刷新](https://docs.databricks.com/aws/en/dashboards/share/schedule-subscribe)

### 把增量更新交给作业 {#job}

打开 **Jobs & Pipelines → Create → Job**，添加两个任务。`merge_silver` 运行一个只含实验中 `MERGE` 的 SQL notebook。把 `payments_silver` 和 `payments_batch_latest` 改成带 catalog、schema 的完整名称。`check_gold` 使用下方保存好的 SQL 查询，并将 `merge_silver` 设为前置任务。[Lakeflow Jobs 入门](https://docs.databricks.com/aws/en/jobs/jobs-quickstart) · [SQL 任务](https://docs.databricks.com/aws/en/jobs/tasks/sql)

Bronze 的追加语句留在实验 notebook 里，这样重试作业只会处理已经写入的数据。用于验证的数字只针对第二批：

```sql
SELECT assert_true(
  COALESCE(SUM(payment_count), 0) = 5
    AND COALESCE(SUM(approved_count), 0) = 3
    AND COALESCE(SUM(approved_cents), 0) = 7699,
  'Batch 2 totals differ from 5 payments, 3 approvals, 7699 cents'
)
FROM `your_catalog`.databricks_notes_lab.payments_daily;
```

用同一批已接收的数据运行两次，核对结果是否一致。还可以把预期支付数从 5 改成 6，让第二步故意失败，到 **Runs** 查看失败信息，修正后再观察修复运行。每天接收新数据的作业可以记录批次号、隔离行数、最新事件时间，以及报表读取的 Silver 版本。[任务依赖](https://docs.databricks.com/aws/en/jobs/control-flow) · [修复失败运行](https://docs.databricks.com/aws/en/jobs/repair-job-failures)

### 记录训练与发布的模型版本 {#mlflow}

**MLflow Tracking** 可以存下训练输入、参数、指标和产物。把模型作为产物记录下来，就可以重新加载并推断。需要按版本管理发布与访问权限时，把模型注册进 Unity Catalog，并附上输入示例或显式的模型签名，让下游知道模型接受哪些字段。Databricks 的 Unity Catalog 注册表地址写作 `databricks-uc`，注册需要对应目录和 schema 的权限。[MLflow Tracking 入门](https://mlflow.org/docs/latest/ml/getting-started/quickstart/) · [Unity Catalog 模型管理](https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle)

**在 Python notebook 里试一次。** 连接 serverless compute，确认环境中有 MLflow 3、pandas 和 scikit-learn。如果缺包或 MLflow 版本较旧，在 notebook 的 **Environment** 面板加上对应依赖。下面的鸢尾花分类在中央处理器（CPU）上就能跑，与支付实验无关。用它练习保存和加载模型，准确率不能用来说明风控效果。[Databricks 上的 MLflow 3](https://docs.databricks.com/aws/en/mlflow/mlflow-3-install)

```python
import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=7
)
model = LogisticRegression(max_iter=300).fit(X_train, y_train)

with mlflow.start_run(run_name="iris-logistic-baseline"):
    mlflow.log_params({"max_iter": 300, "split_seed": 7})
    mlflow.log_metric("test_accuracy", accuracy_score(y_test, model.predict(X_test)))
    example = X_train.head(2)
    info = mlflow.sklearn.log_model(
        model,
        name="iris_classifier",
        signature=infer_signature(example, model.predict(example)),
        input_example=example,
    )

reloaded = mlflow.pyfunc.load_model(info.model_uri)
print("Model:", info.model_uri)
print("Predictions:", reloaded.predict(X_test.head(3)))
```

在 **Experiments** 里找到这次运行，查看参数与 `test_accuracy`；单元格应该打印重新加载后的三个预测。若有 `CREATE MODEL` 和对应 catalog、schema 的使用权限，可以选做下面的注册步骤，把 `your_catalog` 换成自己的 catalog。能重新加载模型就已经完成 Tracking 练习。输入示例会随模型一起保存，请使用公开或模拟的样本。

```python
mlflow.set_registry_uri("databricks-uc")
version = mlflow.register_model(
    info.model_uri, "your_catalog.databricks_notes_lab.iris_classifier"
)
print("Registered version:", version.version)
```

下面第一个地址固定读取第 3 版，第二个读取 `Champion` 当时指向的版本：

```text
models:/prod.risk.payment_model/3
models:/prod.risk.payment_model@Champion
```

做批量打分时，把实际使用的模型版本和输入表快照一同记下来，日后才能复现。应用需要立即得到预测时，可以通过 Model Serving 的接口调用已部署模型。若线上端点配置的是具体版本，调整别名以后还需要更新并验证该端点的版本，才能切换流量。[模型别名](https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle#use-model-aliases) · [服务端点配置](https://docs.databricks.com/aws/en/machine-learning/model-serving/create-manage-serving-endpoints)

### 延伸到支付风控与影像元数据

做**支付欺诈检测**时，可以加入稍后送达的争议交易数据，为每次支付决策构造一条训练记录。10:00 决策可以使用 09:45 发生、09:46 已收到的交易。另一笔交易虽然 09:55 就发生了，却到 10:05 才收到，当时无法用于打分。记录事件时间和数据可用时间，才能还原线上在每个时点可取到的特征。后续争议结果用作标签。等标签经过足够长的确认期，再用更晚的时间段评估。用平均精确率衡量排序表现，也检查在固定人工审核容量下能找出多少欺诈。[时点正确的特征关联](https://docs.databricks.com/aws/en/machine-learning/feature-store/time-series) · [平均精确率定义](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)

![两笔交易都发生在 10:00 决策前，但只有第一笔及时到达。](feature-availability.svg "图 3：发生时间和可用时间回答不同的问题。10:05 才收到的记录不能进入 10:00 的模型特征。")

做**磁共振成像（MRI）**数据管理时，可以先造一张模拟文件清单，包含 `subject_id`、`session_id`、`modality`、`file_uri`、校验和以及质控状态。影像保留为文件，清单存成表。用视图筛掉质控不合格的扫描，并统计各采集中心入选的受试者人数。训练集和测试集按受试者划分，让同一人的多次扫描留在一边。每次训练，把清单的表版本和预处理代码版本写进 MLflow 运行记录。[脑影像数据结构（BIDS）的 MRI 规范](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/magnetic-resonance-imaging-data.html) · [按组划分验证集](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)

## 从个人实验走到生产 {#production}

如果只是临时分析一个能放进内存的本地数据集，从现成的本地 SQL 或 DataFrame 工具开始更省事。需要多人访问同一批表、重复运行依赖任务，或用分布式计算处理超出单机能力的负载时，再考虑 Databricks。

### 按数据源选择摄取和重放方式

| 源端提供什么 | 从什么工具开始 | 还需要确定什么 |
|---|---|---|
| 定时落下一小批新文件 | [SQL `COPY INTO`](https://docs.databricks.com/aws/en/sql/language-manual/delta-copy-into) | 文件名保持唯一，落盘后不覆盖。同一条业务记录仍可能出现在两个文件里。 |
| 对象存储里持续到达的文件 | [Auto Loader](https://docs.databricks.com/aws/en/ingestion/cloud-object-storage/auto-loader) | 为每条流保留自己的 checkpoint，并测试上游加一列后的行为。 |
| 受支持数据库的变更日志 | [Lakeflow Connect](https://docs.databricks.com/aws/en/ingestion/lakeflow-connect/cdc-overview) | 初始快照怎么取，网络怎么连，谁维护连接器。 |
| 变更流里有插入、更新和删除，且有可靠的排序字段 | [`AUTO CDC`](https://docs.databricks.com/aws/en/ldp/cdc) | 确定业务键和版本顺序，决定目标表只留当前状态还是保留历史版本。 |

变更数据捕获（CDC）用来记录源系统的变化。Lakeflow Connect 可以从支持的数据库读取变更。变更乱序到达时，`AUTO CDC` 可以按排序字段将它们应用到目标表。**Lakeflow pipelines** 基于 Apache Spark Declarative Pipelines（SDP），可以声明目标表、转换和质量规则。旧教程里常见的名字是 Delta Live Tables（DLT）。质量规则可以记录、丢弃或因无效记录让流程失败。需要检查和修复的坏记录可以另存。**Lakeflow Jobs** 再负责调度流水线及其上下游任务。[Lakeflow 与开源 SDP](https://docs.databricks.com/aws/en/ldp/concepts/spark-declarative-pipelines) · [质量规则和坏数据隔离](https://docs.databricks.com/aws/en/ldp/expectation-patterns)

流式 checkpoint 保存处理进度和状态，供查询恢复使用。保留它并验证重启行为。自定义 `foreachBatch` 写入时，失败后同一个批次可能再次送到，所以业务键和版本检查仍然需要保留。用 **watermark（水位线）**控制流式聚合或连接要维护多长时间的状态时，同时决定超出这个时间范围的数据如何补算。文件被发现的时间、实际到达顺序和业务事件时间要分别记录和解释。[流式 checkpoint](https://docs.databricks.com/aws/en/structured-streaming/checkpoints) · [`foreachBatch`](https://docs.databricks.com/aws/en/structured-streaming/foreach) · [水位线](https://docs.databricks.com/aws/en/structured-streaming/watermarks)

### 配置访问权限并记录发布版本

给报表读者上级目录的 `USE CATALOG`、schema 的 `USE SCHEMA`，再给指定 Gold 对象的 `SELECT`。作业的 **Run as** 身份需要自己的读写权限。可以用服务主体（service principal）作为程序身份，避免定时任务依赖某个人的凭据。发现指标变化时，在 **Catalog Explorer → Lineage** 顺着血缘找源表，再对照表历史和对应的作业运行。[Unity Catalog 权限参考](https://docs.databricks.com/aws/en/data-governance/unity-catalog/access-control/privileges-reference) · [作业权限](https://docs.databricks.com/aws/en/jobs/privileges) · [数据血缘](https://docs.databricks.com/aws/en/data-governance/unity-catalog/data-lineage)

把代码和作业配置放进 Git。**Declarative Automation Bundles**（旧称 Databricks Asset Bundles）允许将任务和流水线定义与代码一起维护。在 `databricks.yml` 中用不同的目标配置区分测试和生产环境。先验证配置，在练习目录中跑一遍，再为生产设置调度。[Bundle 配置](https://docs.databricks.com/aws/en/dev-tools/bundles/settings)

### 先看查询实际花了多久，再调计算资源

从 **Query Profile** 查起。若大部分时间在等待 Warehouse 启动或排队，先检查计算资源的运行状态和并发。若扫描占大头，查读取了哪些列、日期过滤是否生效、读了多少字节。若连接后行数突然膨胀，回查两侧表的粒度。慢阶段有大量跨执行器的数据交换（shuffle）或溢写磁盘时，先找数据倾斜和多带的列。用 `EXPLAIN FORMATTED` 看引擎计划执行什么，用 Query Profile 看实际花在哪里。[查询性能剖析](https://docs.databricks.com/aws/en/sql/user/queries/query-profile) · [EXPLAIN](https://docs.databricks.com/aws/en/sql/language-manual/sql-ref-syntax-qry-explain)

试着在出租车样例上分析一条带日期过滤的聚合，记录耗时、扫描字节和 Warehouse 状态。冷启动和缓存会让两次查询难以直接比较。十行支付数据适合验逻辑。等测到文件过小或扫描量过大，再考虑合并文件或调整聚簇布局。付费工作区可以按 Warehouse 或作业汇总计费量，再对照相应时段的价格，另外考虑适用的云基础设施费用。[优化文件布局](https://docs.databricks.com/aws/en/tables/operations/optimize) · [监控计算费用](https://docs.databricks.com/aws/en/admin/usage/system-tables)

### 实验卡住时怎么排查

| 遇到的问题 | 先检查什么 |
|---|---|
| 换了 notebook 或 SQL 查询后找不到表 | 查 `current_catalog()` 和 `current_schema()`，再用三段式名称验证访问的是哪张表。 |
| MERGE 提示多个源记录匹配同一条目标记录 | 用 `GROUP BY payment_id HAVING COUNT(*) > 1` 查这一批。实验应从 `payments_batch_latest` 取数，每批每个键至多留一行有效记录。 |
| 重跑后报表翻倍 | 确认 Gold 读的是 Silver，且合并的 `ON` 只按支付键匹配。直接检查 Silver 的重复键或空键。 |
| 旧快照读不了 | 在 `DESCRIBE HISTORY` 核对合并前的实际版本。只有日志还在、旧数据文件已经被清理时，无法靠日志还原。 |

## 四次练习与对应的资料 {#learning-path}

| 第几次 | 做什么 | 怎样算做完 |
|---|---|---|
| 1. 查询和建表 | 查出租车数据，找到所属目录，再在练习 schema 建一张表。 | 在 Catalog Explorer 找到两张表，说明谁能读取它们。 |
| 2. 搭流水线并重放 | 跑支付实验，检查隔离记录，再重跑同一批增量。 | 给出前后两组指标，解释为什么第二次合并没有改变结果。 |
| 3. 调度和画图 | 建两个有依赖的任务和一张仪表板，故意让检查失败一次并修复。 | 从图上一个数字追到 Gold 查询，再定位修改输入的那次作业。 |
| 4. 选一个方向 | 写出报表的分母口径，补算一天的数据，或复现一次模型批量打分。 | 留下查询或代码、输入样例、预期结果和实测记录。 |

| 读什么 | 读完动手做什么 |
|---|---|
| [Databricks Academy 课程和岗位学习路线](https://www.databricks.com/learn/training/home) | 先学 Fundamentals，再从数据分析、数据工程和机器学习中选与自己项目最贴近的一条。 |
| [PySpark DataFrame 快速入门](https://spark.apache.org/docs/latest/api/python/getting_started/quickstart_df.html) 与 [Delta Lake Quickstart](https://docs.delta.io/quick-start/) | 用 DataFrame 改写一条 SQL 聚合，再做一次 Delta 合并并读回合并前的版本。 |
| [Lakeflow Jobs 快速入门](https://docs.databricks.com/aws/en/jobs/jobs-quickstart) 与 [Spark Structured Streaming 入门例子](https://spark.apache.org/docs/latest/streaming/getting-started.html) | 先从运行历史定位一次任务失败，再给流式实验设置 checkpoint，停下来重启一次。 |
| [Databricks 的 MLflow 端到端示例](https://docs.databricks.com/aws/en/mlflow/end-to-end-example) | 对比两次训练记录，加载选定版本做预测。 |
| [Lakehouse 论文（2021）](https://www.cidrdb.org/cidr2021/papers/cidr2021_paper17.pdf) 与 [Delta Lake 论文（2020）](https://www.vldb.org/pvldb/vol13/p3411-armbrust.pdf) | 先看湖仓的架构图，再看 Delta 如何根据日志在对象存储上构建一致快照。 |
| [Data Engineer Associate 认证](https://www.databricks.com/learn/certification/data-engineer-associate) | 如果要备考，从页面打开当期考试大纲，将考点逐一对应到能亲手完成的任务。 |

> 实验的行数、金额、重放和历史读取已用本地 Apache Spark 4.0.1 与 Delta Lake 4.0.0 验证，MLflow 保存和加载也在本地跑过。Databricks 工作区的菜单与权限步骤按官方文档编写。

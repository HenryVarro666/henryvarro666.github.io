---
title: "Learn Databricks by building a payment pipeline"
date: 2026-11-13T18:30:00-06:00
description: "Start with Free Edition, then build and replay a Delta Lake pipeline. Turn it into a dashboard or scheduled job and try a model experiment in MLflow."
topics: [data-science, ml-systems]
---

A payments feed can send the same record twice, then deliver a correction after the daily report has gone out. Summing the received rows counts the duplicate and adds the correction on top of the original amount. In this lab, you'll keep the raw events, resolve each payment to its latest valid revision, and build a report on the current state. You'll use Structured Query Language (SQL) for the first pass and run the same aggregation in PySpark. From there, you can turn the result into a dashboard, schedule it as a job, or use the same ideas to build reproducible training data.

This guide assumes you can read basic SQL and are new to Databricks. Start by [opening a workspace](#setup) and [running the lab](#lab), or skip to the [incremental merge](#lab-merge) or [applications](#applications) if your workspace is ready.

## What happens when you run a query {#mental-model}

Databricks is a managed platform for processing data, publishing reports and training models. The workspace is where you write code; compute runs it, and storage holds the tables and files.

Consider a notebook query against `samples.nyctaxi.trips`. **Unity Catalog** resolves the table name and checks access. The attached **compute** reads the table and runs the query. If it's a **Delta** table, the Delta log tells the reader which files belong to the snapshot it should see. Turn off the compute and the stored files remain available when you start another session. [Query data](https://docs.databricks.com/aws/en/query) · [Delta transactions](https://docs.databricks.com/aws/en/lakehouse/acid)

![A map of authoring tools, compute and storage, with Unity Catalog for governance and Lakeflow Jobs for orchestration.](platform-map.svg "Figure 1. Editors and dashboards submit work to compute. Delta tables persist in object storage, while Unity Catalog governs access and Lakeflow Jobs schedules work.")

| Component | What it does | Where you'll see it |
|---|---|---|
| Object storage | Holds data and metadata files. Parquet stores data by column so a reader can select the columns it needs. | The storage behind a managed table or volume |
| Delta Lake | Adds a transaction log to data files. A commit records the file changes that define a new table version. This gives readers a consistent snapshot even while another job is writing. | `MERGE`, `DESCRIBE HISTORY`, and reads of an older table version |
| Apache Spark and PySpark | Spark executes data transformations across compute resources. PySpark is its Python application programming interface (API). | A notebook using DataFrames, for example `spark.table(...).groupBy(...)` |
| Databricks SQL warehouse | Executes SQL for the query editor, dashboards, or a SQL notebook. | The compute selector in the SQL editor |
| Unity Catalog | Organizes tables, views, volumes and models under names such as `catalog.schema.table`. Controls access and records lineage for supported operations. | Catalog Explorer and permission grants |
| Lakeflow Jobs | Runs tasks in order and records their status. A task can run a notebook, a SQL file or a pipeline. | The dependency graph and run history of a job |

Open-source Spark and Delta Lake also run outside Databricks. Databricks brings them together with managed compute, workspaces, governance and tools for running recurring work. A **workspace** is where you open an editor or a job. A **catalog** is part of the address of the data that work reads. In a Unity Catalog managed table, the catalog manages the storage lifecycle. With an external table, your team owns that lifecycle. Use a **volume** for files, such as incoming comma-separated values (CSV) or images, and a **table** for records you intend to query by column. [Databricks overview](https://docs.databricks.com/aws/en/introduction) · [Unity Catalog](https://docs.databricks.com/aws/en/data-governance/unity-catalog) · [Table types](https://docs.databricks.com/aws/en/tables/types) · [Volumes](https://docs.databricks.com/aws/en/volumes/volume-files)

### Why keep three layers?

A **lakehouse** lets teams use data in object storage for analytics and machine learning without maintaining a separate copy for every tool. The **medallion** pattern organizes that data by what a reader can assume about it. Bronze preserves what arrived so you can replay and audit it. Silver assigns types and resolves duplicates or revisions so consumers can agree on what one row means. Gold defines a result for a particular consumer, such as one row per payment day and channel. Before building each layer, write down **the grain**, what a row represents, and **the key**, how you identify it. [Medallion architecture](https://docs.databricks.com/aws/en/lakehouse/medallion) · [Lakehouse design paper](https://www.cidrdb.org/cidr2021/papers/cidr2021_paper17.pdf)

## Open a workspace and run your first query {#setup}

Databricks Free Edition is for personal, noncommercial practice. It gives you a workspace with serverless compute and usage quotas. The business free trial is a separate offering with credits that expire. Use public or synthetic data for the exercises here. The instructions follow Databricks on Amazon Web Services (AWS). Menus and feature availability can vary by cloud and workspace. [Sign up for Free Edition](https://www.databricks.com/learn/free-edition) · [Compare signup options](https://docs.databricks.com/aws/en/getting-started/free-trial-vs-free-edition) · [Free Edition limits](https://docs.databricks.com/aws/en/getting-started/free-edition-limitations)

Choose **New → Notebook**, set the language to **SQL** and attach **Serverless** compute or a SQL warehouse at the top. For Python or PySpark, attach **Serverless**. Run each cell with **Shift+Enter**. Confirm compute is working with a quick query. You can also run this check in **New → SQL query** on a warehouse. [Notebook quickstart](https://docs.databricks.com/aws/en/getting-started/quick-start) · [Notebook compute options](https://docs.databricks.com/aws/en/notebooks/notebook-compute)

```sql
SELECT 1 AS ready;
```

For an optional read of a real dataset, run this preview. `fare_amount` is a fare column. Check the schema in Catalog Explorer before deciding what a metric includes. [Sample datasets](https://docs.databricks.com/aws/en/discover/databricks-datasets)

```sql
SELECT pickup_zip, trip_distance, fare_amount
FROM samples.nyctaxi.trips
LIMIT 10;
```

If the first query runs but the sample does not, inspect which catalogs you can see. The main lab builds its own data, so you can continue once you have a writable schema. In a shared workspace, a personal or team scratch schema with table creation rights is enough. [Create a table and grant privileges](https://docs.databricks.com/aws/en/getting-started/create-table)

If `SELECT 1` fails, check the compute attachment, permission to use it and whether you have hit the Free Edition quota. If you can read the sample but cannot create tables, check the current schema and its privileges in Catalog Explorer.

## Build and replay a payments pipeline {#lab}

**Goal:** answer “how many payments were approved, and for how much, on each day and channel?” Correct the answer when a newer event arrives. Allow 45–60 minutes the first time. The inputs are ten fabricated delivery records; you can run the full lab without downloading a dataset.

![The raw deliveries become clean payment records and then daily metrics. Invalid amounts take a separate path.](payments-flow.svg "Figure 2. Bronze remembers what arrived. Silver remembers the latest valid revision of each payment. Gold reports by event day and channel.")

### 1. Set the catalog and schema {#lab-setup}

Open **New → Notebook**, choose **SQL** as the notebook language, and attach serverless notebook compute. Run the statements in order. Put each statement in its own cell to examine its result, or import the supplied notebook to start with the cells already separated. You can also [download the SQL notebook source](/notes/databricks-guide/payments-lab.sql), import it under **Workspace**, and run its cells in order. [Import a notebook](https://docs.databricks.com/aws/en/notebooks/notebook-export-import)

```sql
SHOW CATALOGS;
```

Replace `workspace` below with a catalog you can write to. Check the returned names before proceeding. Creating a schema needs `USE CATALOG` and `CREATE SCHEMA` on the catalog. In an existing practice schema you need `USE CATALOG`, `USE SCHEMA` and `CREATE TABLE`. If an administrator gives you a different practice schema, substitute that name and omit `CREATE SCHEMA`. Run this in a dedicated sandbox: the lab's `CREATE OR REPLACE` statements overwrite objects with these names.

```sql
USE CATALOG `workspace`;
CREATE SCHEMA IF NOT EXISTS databricks_notes_lab;
USE SCHEMA databricks_notes_lab;
SELECT current_catalog() AS catalog_name, current_schema() AS schema_name;
```

### 2. Bronze: keep the delivery as received {#lab-bronze}

A *payment* has a stable `payment_id`. Each delivery has a `delivery_id`. The sender supplies `event_version` for each payment; a larger version supersedes a smaller one. It is a business revision and has no connection to the Delta table's version number. The exercise uses US dollars (USD) and stores amounts as integer cents after validation. Dates are supplied directly so the exercise does not depend on timestamp or session time zone.

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

Expect **6 deliveries**. Two carry the same revision of `p01`; one has a malformed amount. Keep both duplicates and the bad input in Bronze so you can see what the source sent. The seed statement resets the sandbox. In a real pipeline, ingest into Bronze by appending new deliveries under a defined ingestion contract.

### 3. Silver: validate, then keep one revision per payment {#lab-silver}

Give invalid rows a reason before you filter them out. `TRY_CAST` turns a malformed value such as `oops` into a null, which the rule below sends to a rejects view. The first matching rule wins.

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

The rejects view should return **`p_bad | oops | invalid_amount`**. Next, select the highest revision for each payment within each batch. Here a tie means identical payloads, so the larger delivery ID gives a stable winner. If payloads with the same key and revision disagree in a real feed, flag the conflict or ask the source for an authoritative ordering rule.

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

Silver has **4 payments**. Write down the newest **table version** from `DESCRIBE HISTORY` now, before you run the correction. This is the version you will use to reconstruct the first batch's answer later. If you reset the exercise, read the version again.

### 4. Gold: make the reporting grain explicit {#lab-gold}

Each Gold row represents one **event day × channel**. `payment_count` counts every current payment, including declines. The amount column sums approved payments only. This is a demonstration of approvals, not a measure of bank settlement.

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

The totals are **4 payments, 3 approvals, 7,999 approved cents** (79.99 USD). Gold is an ordinary view here, so a later query reads the current Silver state without a separate refresh step.

### 5. Apply an increment, replay it, and read the old answer {#lab-merge}

Append the second batch once. It corrects `p02` and `p04`, delivers an old revision of `p03` late, and adds `p05`.

```sql
INSERT INTO payments_bronze VALUES
  (2, 7,  'p02', 2, DATE '2026-11-01', 'store',  'approved', '2100'),
  (2, 8,  'p04', 2, DATE '2026-11-02', 'store',  'declined', '2400'),
  (2, 9,  'p03', 1, DATE '2026-11-02', 'web',    'declined', '3000'),
  (2, 10, 'p05', 1, DATE '2026-11-02', 'mobile', 'declined', '1200');
```

Now apply the batch. Join on the business key in `ON` and put the version check in `WHEN MATCHED`. If the version check were part of `ON`, an old event could enter the insert branch. The source view already ensures at most one row per payment in the selected batch. [Delta MERGE examples](https://docs.databricks.com/aws/en/delta/merge)

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

Run the `MERGE` again **without rerunning the Bronze insert**. The rows and totals should stay the same. Check both: `p03` stays approved at version 2 and 3,100 cents; `p02` moves to approved and `p04` moves to declined, so the number of approvals stays at three while the total drops by 300 cents.

| Checkpoint | Bronze deliveries | Silver payments | Approvals | Approved cents |
|---|---:|---:|---:|---:|
| Batch 1 | 6 | 4 | 3 | 7,999 |
| After batch 2 | 10 | 5 | 3 | 7,699 |
| Replay the second `MERGE` | 10 | 5 | 3 | 7,699 |

Now compare with the previous Silver state. Replace `0` with the **actual version number** you recorded above, then run this query:

```sql
SELECT COUNT(*) AS payments,
       COUNT_IF(status = 'approved') AS approved,
       SUM(CASE WHEN status = 'approved' THEN amount_cents ELSE 0 END) AS approved_cents
FROM payments_silver VERSION AS OF 0;
```

The old snapshot returns **4 / 3 / 7,999**, while the current Gold view returns **5 / 3 / 7,699**. Time travel reads an old snapshot without rolling the table back. It requires the old log and its data files to still be available; treat long-term retention and backups separately. [Delta history and retention](https://docs.databricks.com/aws/en/tables/history)

### Optional: write Gold in PySpark {#lab-pyspark}

Open a **Python** notebook on notebook compute. Replace `workspace` in the three-part name before running the cell. This version expresses the same grouping and conditional counts in the DataFrame API.

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

`groupBy` builds a plan. `display` asks Databricks to run it and show the result. Compare each row with the SQL view; both should produce the same five groups after batch 2. In production, keep joins and aggregation in Spark and bring only small results into local Python. [PySpark DataFrame quickstart](https://spark.apache.org/docs/latest/api/python/getting_started/quickstart_df.html)

## Use the results in a report, a job or a model {#applications}

### Put the Gold view on a dashboard {#dashboard}

Create an artificial intelligence and business intelligence (AI/BI) dashboard via **New → Dashboard → Data → Add SQL dataset**. Select a SQL warehouse and run the query below with your catalog name. In the canvas, add a bar chart with `event_day` on the horizontal axis, summed `approved_cents` on the vertical axis, and `channel` as the series. Add a date filter. For an amount in US dollars (USD), divide cents by 100 and label the units on the chart. [Create an AI/BI dashboard](https://docs.databricks.com/aws/en/dashboards/tutorials/create-dashboard)

```sql
SELECT event_day, channel, payment_count, approved_count, approved_cents
FROM `your_catalog`.databricks_notes_lab.payments_daily;
```

In the **Data** tab, choose **Add custom calculation** and define a [calculated measure](https://docs.databricks.com/aws/en/dashboards/manage/data-modeling/custom-calculations) with `TRY_DIVIDE(SUM(approved_count), SUM(payment_count))`. An average of per-channel rates would give small and large channels equal weight. With every date and channel selected after the second batch, you should see **3 / 5 = 60%** approvals and **7,699 cents** in approved volume. This measures the lab's latest payment states. Record when the underlying data was updated so readers can distinguish a fresh dashboard query from a fresh input load. When sharing a dashboard, check whether it runs under the publisher's data permissions or each viewer's. [Dashboard data permissions](https://docs.databricks.com/aws/en/dashboards/share/share) · [Dashboard refresh schedules](https://docs.databricks.com/aws/en/dashboards/share/schedule-subscribe)

### Schedule the incremental step {#job}

Under **Jobs & Pipelines → Create → Job**, add two tasks. For `merge_silver`, make a SQL notebook with just the `MERGE` from the lab. Qualify `payments_silver` and `payments_batch_latest` with your catalog and schema. For `check_gold`, save the query below and add it as a SQL task with `merge_silver` as its dependency. [Lakeflow Jobs quickstart](https://docs.databricks.com/aws/en/jobs/jobs-quickstart) · [SQL task](https://docs.databricks.com/aws/en/jobs/tasks/sql)

The Bronze insert stays in the lab notebook, so retries apply already ingested data. These expected values are specific to batch 2:

```sql
SELECT assert_true(
  COALESCE(SUM(payment_count), 0) = 5
    AND COALESCE(SUM(approved_count), 0) = 3
    AND COALESCE(SUM(approved_cents), 0) = 7699,
  'Batch 2 totals differ from 5 payments, 3 approvals, 7699 cents'
)
FROM `your_catalog`.databricks_notes_lab.payments_daily;
```

Run the job twice on the same already ingested batch and compare the results. To test failure handling, change the expected payment count from 5 to 6 and locate its output in **Runs**. Then fix the check and inspect the repaired run. For a job that receives fresh data every day, log the batch identifier, rejected row count, latest event timestamp, and the Silver version used by the report. [Job task dependencies](https://docs.databricks.com/aws/en/jobs/control-flow) · [Repair job runs](https://docs.databricks.com/aws/en/jobs/repair-job-failures)

### Track an experiment and identify the deployed model {#mlflow}

**MLflow Tracking** records the inputs, settings, metrics and artifacts of a training run. A logged model contains what you need to reload it for inference. Register it in Unity Catalog when you need to grant access to a managed release and identify it as a named version. Include an input example or explicit model signature describing the expected schema. On Databricks, the Unity Catalog registry URI is `databricks-uc`. Registration requires the appropriate catalog and schema privileges. [MLflow Tracking quickstart](https://mlflow.org/docs/latest/ml/getting-started/quickstart/) · [Manage models in Unity Catalog](https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle)

**Try it in a Python notebook.** Connect to serverless compute and use a notebook environment with MLflow 3, pandas and scikit-learn. If a package is missing or MLflow is older, add the required dependency in the notebook's **Environment** panel. This tiny iris example runs on the central processing unit (CPU) and has no dependency on the payments lab. It exercises saving and reloading a model; its accuracy does not tell you anything about payment fraud. [MLflow 3 on Databricks](https://docs.databricks.com/aws/en/mlflow/mlflow-3-install)

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

Open **Experiments** and find the run's parameters and `test_accuracy`; the cell should print three predictions from the reloaded model. If you have permission to create a model in your lab schema (`CREATE MODEL` plus the parent usage privileges), you can also register the artifact. Replace `your_catalog` with your own catalog. This step is optional: the tracking exercise is complete once you can reload the model. The input example becomes part of the artifact, so use a public or synthetic example.

```python
mlflow.set_registry_uri("databricks-uc")
version = mlflow.register_model(
    info.model_uri, "your_catalog.databricks_notes_lab.iris_classifier"
)
print("Registered version:", version.version)
```

Version 3 is a fixed release. The `Champion` alias resolves to whichever version currently holds that label. The two model addresses look like this:

```text
models:/prod.risk.payment_model/3
models:/prod.risk.payment_model@Champion
```

For a batch scoring job, save the resolved version and input snapshot with each run so you can reproduce it later. For prediction inside an app, Model Serving exposes a deployed model over a web endpoint. Promoting an alias does not itself update an endpoint configured with a specific version. Update the endpoint's served version and check it before routing live traffic. [Model aliases](https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle#use-model-aliases) · [Serving endpoint configuration](https://docs.databricks.com/aws/en/machine-learning/model-serving/create-manage-serving-endpoints)

### Extend the exercise to payment risk or imaging data

For **payment fraud**, add a late-arriving dispute feed and build a training row for each decision. A decision at 10:00 can use a purchase made at 09:45 and received at 09:46. A purchase made at 09:55 and received at 10:05 was still missing when the decision was made. Store both event and availability timestamps so historical features reflect what the scoring system could access. Use later dispute outcomes as labels, choose a maturity window for those labels, and evaluate on a later time period. Report average precision and how many fraud cases you catch at a fixed review capacity. [Point-in-time feature joins](https://docs.databricks.com/aws/en/machine-learning/feature-store/time-series) · [Average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)

![Two purchases occurred before the 10:00 decision, but only the first arrived in time to affect it.](feature-availability.svg "Figure 3. Time of occurrence and time of availability answer different questions. The later delivery must be excluded from the features used at 10:00.")

For **magnetic resonance imaging (MRI)**, make a synthetic file manifest containing `subject_id`, `session_id`, `modality`, `file_uri`, `checksum` and a quality status. Store images as files and the manifest as a table. Build a cohort view that excludes failed quality checks and lists subject counts by collection site. Split by subject so repeated scans from the same person stay in one split. Record the manifest version and preprocessing code revision alongside each MLflow run. [Brain Imaging Data Structure (BIDS) MRI specification](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/magnetic-resonance-imaging-data.html) · [Grouped validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)

## Take the pipeline into production {#production}

For a one-off analysis of a local dataset that fits in memory, a local SQL or DataFrame tool is usually enough. Databricks becomes useful when a team needs to share governed tables, rerun dependent jobs, or distribute work that exceeds one machine's capacity.

### Choose how to ingest and replay

| What the source sends | Tool to start with | What still needs a decision |
|---|---|---|
| A small scheduled batch of new files | [SQL `COPY INTO`](https://docs.databricks.com/aws/en/sql/language-manual/delta-copy-into) | Keep file names immutable. The same business record can still appear in two different files. |
| Files arriving regularly in object storage | [Auto Loader](https://docs.databricks.com/aws/en/ingestion/cloud-object-storage/auto-loader) | Use a stable checkpoint for each stream and test how it handles new columns. |
| A supported database's change log | [Lakeflow Connect](https://docs.databricks.com/aws/en/ingestion/lakeflow-connect/cdc-overview) | Plan initial snapshot, network access, and ownership of the source connection. |
| A stream of inserts, updates and deletes with a reliable sequence field | [`AUTO CDC`](https://docs.databricks.com/aws/en/ldp/cdc) | Choose a key and sequence. Decide whether to keep current state or retain each historical revision. |

Change data capture (CDC) records changes in a source system. Lakeflow Connect can capture them from supported databases. `AUTO CDC` uses the sequence field to apply changes to the target in the intended order even when inputs arrive out of order. Lakeflow pipelines, built on Apache Spark Declarative Pipelines (SDP), let you declare target tables, transformations and quality rules. Older material calls this product Delta Live Tables (DLT). A quality rule can track, discard, or fail on invalid rows. Keep a separate quarantine output if those rows need repair. Schedule the pipeline and related tasks with **Lakeflow Jobs**. [Lakeflow and open-source SDP](https://docs.databricks.com/aws/en/ldp/concepts/spark-declarative-pipelines) · [Expectations and quarantine](https://docs.databricks.com/aws/en/ldp/expectation-patterns)

A streaming checkpoint keeps progress and state so a query can resume. Protect that directory and test restart behavior. A custom `foreachBatch` write can receive a batch again after a failure, so the stable key and revision check still matter. If you use a **watermark** to limit how long a streaming join or aggregation keeps state, decide how you'll reconcile data that arrives after that window. File discovery, delivery order and business event time are three separate concerns. [Streaming checkpoints](https://docs.databricks.com/aws/en/structured-streaming/checkpoints) · [`foreachBatch`](https://docs.databricks.com/aws/en/structured-streaming/foreach) · [Watermarks](https://docs.databricks.com/aws/en/structured-streaming/watermarks)

### Assign access and keep a release record

Give a report reader `USE CATALOG` and `USE SCHEMA` on the parent objects and `SELECT` on the intended Gold asset. The job's **Run as** identity needs its own read and write permissions. A service principal, an identity for software, lets the job run without a person's credentials. When totals change, follow **Lineage** in Catalog Explorer to the source objects, then check their Delta history and the corresponding job run. [Unity Catalog privileges](https://docs.databricks.com/aws/en/data-governance/unity-catalog/access-control/privileges-reference) · [Job permissions](https://docs.databricks.com/aws/en/jobs/privileges) · [Data lineage](https://docs.databricks.com/aws/en/data-governance/unity-catalog/data-lineage)

Keep code and job configuration in Git. **Declarative Automation Bundles**, formerly Databricks Asset Bundles, define jobs and pipelines alongside their code. Put the test and production workspace settings in separate targets in `databricks.yml`. Validate configuration and run the job against a scratch catalog before setting up a production schedule. [Bundle configuration](https://docs.databricks.com/aws/en/dev-tools/bundles/settings)

### Look at the query before tuning compute

In **Query Profile**, first distinguish time spent waiting for a warehouse from time spent executing. If a scan dominates, check the selected columns, date predicates and bytes read. If a join produces far more rows than its inputs, verify that both sides have the grain you assumed. If the slow stage shuffles a large amount of data between executors or spills to disk, check for skew and excess columns before increasing compute size. Use `EXPLAIN FORMATTED` for the planned operators and Query Profile for observed costs. [Query Profile](https://docs.databricks.com/aws/en/sql/user/queries/query-profile) · [EXPLAIN](https://docs.databricks.com/aws/en/sql/language-manual/sql-ref-syntax-qry-explain)

Try profiling a date-filtered aggregation on the taxi sample and note query duration, bytes read and warehouse state. Cold starts and cached results can make two runs hard to compare. The ten-row payment fixture is for checking correctness. Consider file compaction or clustering once measured scans and file sizes justify it. In a paid workspace, group billable usage by warehouse or job and use the pricing records for the applicable period. Add cloud infrastructure charges where applicable. [Optimize file layout](https://docs.databricks.com/aws/en/tables/operations/optimize) · [Monitor compute cost](https://docs.databricks.com/aws/en/admin/usage/system-tables)

### Debug common lab failures

| What you see | What to check |
|---|---|
| A table is missing in a new notebook or SQL query | Check `current_catalog()` and `current_schema()`. Use the three-part name to test which object you are reaching. |
| The first merge reports several source rows matching one target row | Inspect the selected batch with `GROUP BY payment_id HAVING COUNT(*) > 1`. Use `payments_batch_latest` as the source; it produces at most one valid row per key in the batch. |
| The same batch doubles the report count | Verify that Gold reads Silver and that the merge's `ON` only matches the payment key. Check Silver for duplicate or null keys instead of masking them with a dashboard filter. |
| A historical query fails | Check the actual pre-merge version in `DESCRIBE HISTORY`. The log entry alone cannot recover old data files that have been removed. |

## Four practice sessions and what to read {#learning-path}

| Session | What to work through | Check your result |
|---|---|---|
| 1. Query and save | Run the taxi preview, find the owning catalog and create a scratch table. | Open both tables in Catalog Explorer and explain who can read each. |
| 2. Build and replay | Run the payments lab, inspect rejected records, then replay the update. | Show the before and after values and why the second merge preserves the result. |
| 3. Schedule and visualize | Build the job and dashboard. Deliberately fail one check, then repair the run. | Trace a number on the chart to its Gold query and the job run that changed the input. |
| 4. Pick a specialization | Add a dashboard with a defined denominator, backfill one day of data, or reproduce a batch of model scores. | Save the SQL or code, a small input fixture, the expected result and what you measured. |

| Read next | What to do with it |
|---|---|
| [Databricks Academy training and role paths](https://www.databricks.com/learn/training/home) | Start with Fundamentals, then pick analyst, data engineer or machine learning material according to the project you want to build. |
| [PySpark DataFrame quickstart](https://spark.apache.org/docs/latest/api/python/getting_started/quickstart_df.html) and [Delta Lake quickstart](https://docs.delta.io/quick-start/) | Express a SQL aggregation as a DataFrame, then merge into a Delta table and read its earlier snapshot. |
| [Lakeflow Jobs quickstart](https://docs.databricks.com/aws/en/jobs/jobs-quickstart) and [Spark Structured Streaming example](https://spark.apache.org/docs/latest/streaming/getting-started.html) | Observe an intentionally failed task before adding a stream. For the stream, try a restart with the same checkpoint. |
| [Databricks end-to-end MLflow example](https://docs.databricks.com/aws/en/mlflow/end-to-end-example) | Compare two training runs and reload the exact model you chose for scoring. |
| [Lakehouse (CIDR 2021)](https://www.cidrdb.org/cidr2021/papers/cidr2021_paper17.pdf) and [Delta Lake (PVLDB 2020)](https://www.vldb.org/pvldb/vol13/p3411-armbrust.pdf) | Read the Lakehouse architecture diagram, then read how Delta constructs a consistent snapshot in object storage. |
| [Data Engineer Associate exam guide](https://www.databricks.com/learn/certification/data-engineer-associate) | If you want a certification, open the current guide from this page and map its objectives to tasks you can already perform. |

> The row counts, sums, replay and historical reads were verified on local Apache Spark 4.0.1 with Delta Lake 4.0.0. Model logging and reloading were also exercised locally. Workspace menus and privilege steps follow the official Databricks guides.

---
title: "Snowflake Notes: from your first SQL query to an orders pipeline"
date: 2026-11-13T18:08:00-06:00
description: "A hands-on Snowflake guide for beginners. Load synthetic orders, validate and deduplicate them, replay a batch safely, and diagnose a slow query. Includes diagrams, a downloadable lab, and a reading path."
topics: [data-science, ml-systems]
---

An order feed sends nine rows. One order appears twice, an older version arrives late, a customer gets a refund, and one amount reads `oops`. What would `SUM(amount)` give you? This note starts with that small problem and builds a Snowflake pipeline whose daily totals you can check by hand.

**Where to start.** If you know a little `SELECT`, you can follow the examples. Spend about 20 minutes on the mental model and first query, then allow 45–60 minutes for the lab. It uses synthetic orders. [Download the SQL lab](/notes/snowflake-learning-guide/orders-lab.sql) · [Download the nine orders as comma-separated values (CSV)](/notes/snowflake-learning-guide/orders.csv).

## 1. The mental model {#mental-model}

Snowflake is a managed cloud data platform. Start by thinking of it as an analytical database. An application database might record a payment and update the order in one transaction. Snowflake can combine historical records from several systems and answer which sales channel led last month. These are typical examples of online transaction processing (OLTP) and online analytical processing (OLAP), respectively. This guide uses standard tables; Snowflake also offers Hybrid Tables for low-latency transactional access.

![Cloud services handle login and authorization. Reporting and transformation run on separate warehouses that read the same persisted orders table.](architecture.svg "Figure 1. Two virtual warehouses can query one table. Suspending the reporting warehouse leaves the stored orders intact.")

| Layer | Job | Where you see it in the lab |
|---|---|---|
| Cloud services | Handles authentication, permissions, metadata, and query planning | Your active role determines whether the query is allowed |
| Compute | Virtual warehouses execute queries, data loads, and transformations | `SF_LEARN_ORDERS_WH` provides the compute |
| Storage | Persists standard tables in compressed columnar form and manages their micro-partitions | `SF_LEARN_ORDERS.CORE.ORDERS` remains available after the warehouse is suspended and resumed |

A **database** is a namespace for objects. A **virtual warehouse** is compute. Point a query at the same table from a different warehouse and you change the resources and which warehouse accrues usage. Giving reports and batch transformations separate warehouses can isolate their compute workloads. [Snowflake architecture and core objects](https://docs.snowflake.com/en/user-guide/intro-key-concepts)

| Object | How to recognize it |
|---|---|
| Account | Boundary for sign-in, account-level roles, and usage |
| Database → schema → table or view | A fully qualified name looks like `SF_LEARN_ORDERS.MART.DAILY_PAID_SALES`. A regular view stores a query definition |
| Role | A bundle of privileges. The same SQL can work under one role and fail under another |
| Stage / file format | A stage tells Snowflake where files are; a file format tells it how to read CSV, JavaScript Object Notation (JSON), or Parquet |

Snowflake normally resolves unquoted identifiers in uppercase. A double-quoted identifier preserves its case and usually needs an exact match. The lab uses unquoted names and single quotes for string literals.

## 2. Set up and run your first query {#first-query}

Use a personal trial or an approved sandbox. The [trial documentation](https://docs.snowflake.com/en/user-guide/admin-trial-account) says a trial ends after 30 days or when its usage balance is exhausted, whichever comes first. Check the remaining balance before starting. In Snowsight, open **Projects → Workspaces → My Workspace**, create a SQL file, and check the role and warehouse in the editor. Use [Workspaces](https://docs.snowflake.com/en/user-guide/ui-snowsight/workspaces#run-your-first-sql-query) when older tutorials refer to Legacy Worksheets.

### Make a small sandbox

With the `SYSADMIN` role in your personal sandbox, create the objects below. In a shared account, ask for an approved lab database, warehouse, and role, then substitute those names. Run one section at a time in the same SQL session. If either `CREATE` finds a pre-existing object, stop and verify that you own it or choose fresh names.

```sql
USE ROLE SYSADMIN;

CREATE WAREHOUSE sf_learn_orders_wh
  WAREHOUSE_TYPE = 'STANDARD'
  WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE
  STATEMENT_TIMEOUT_IN_SECONDS = 120;

CREATE DATABASE sf_learn_orders DATA_RETENTION_TIME_IN_DAYS = 1;
USE WAREHOUSE sf_learn_orders_wh;
USE DATABASE sf_learn_orders;
CREATE SCHEMA raw;
CREATE SCHEMA core;
CREATE SCHEMA mart;

SELECT CURRENT_ROLE(), CURRENT_WAREHOUSE(), CURRENT_DATABASE();
```

The warehouse starts suspended. It can resume when a statement needs compute and suspend after it has been idle for about a minute. The statement timeout limits the time for an individual statement; it is not a budget. If an object is missing or access is denied, check the active role, database, and schema. Try the fully qualified name, then check privileges on the warehouse, the parent objects, and the table or view.

### Query an existing table

If your account has installed `SNOWFLAKE_SAMPLE_DATA` and you have access, count how often each order priority appears:

```sql
SELECT o_orderpriority AS priority, COUNT(*) AS orders
FROM snowflake_sample_data.tpch_sf1.orders
GROUP BY o_orderpriority
ORDER BY orders DESC, priority
LIMIT 5;
```

`TPCH_SF1` is a scale-factor-one sample for the TPC-H benchmark from the Transaction Processing Performance Council. Start with the smaller sample. The engine still has to aggregate before `LIMIT 5` restricts the returned rows. You can skip this query if the sample database is unavailable; the orders lab is self-contained. [Sample data and access](https://docs.snowflake.com/en/user-guide/sample-data)

## 3. Lab: from raw orders to daily paid sales {#orders-lab}

Define the **grain** and metric before writing SQL. The core table has one row per `order_id`, representing the latest valid version. The daily view counts orders that are currently `PAID` and sums their amounts in US dollars, grouped by original order date. A later refund removes the order from the earlier day's metric. This is a current-state report by order date. A finance ledger for daily cash movement would use a separate fact table at the payment or refund transaction grain and group by transaction time.

![Nine CSV or INSERT rows enter RAW. Check failures stay traceable in RAW; valid rows are deduplicated and merged into CORE. MART aggregates paid sales by date.](pipeline.svg "Figure 2. RAW keeps the received values. CORE has one row per order. MART has one row per date with paid sales.")

### Step A: load the raw rows

**Use SQL for the first run.** Open the downloaded script, run Section 1, and use the `CREATE TABLE` and `INSERT` in Section 2. The raw date, amount, and version columns keep the original strings, including invalid ones.

**To practice a CSV load instead of the INSERT**, create the same empty raw table, then run the first two statements below. In Snowsight choose **Ingestion → Add Data → Load files into a Stage** and upload `orders.csv` to the root of the named internal stage `SF_LEARN_ORDERS.RAW.ORDERS_INBOX`. The web SQL editor cannot read a local path through `PUT`; use the stage upload screen or run `PUT` through the Snowflake command-line interface (CLI), for example with `snow sql`. [Stage a file in Snowsight](https://docs.snowflake.com/en/user-guide/data-load-local-file-system-stage-ui) · [PUT reference](https://docs.snowflake.com/en/sql-reference/sql/put)

```sql
CREATE STAGE raw.orders_inbox;
CREATE FILE FORMAT raw.orders_csv
  TYPE = CSV
  SKIP_HEADER = 1
  FIELD_OPTIONALLY_ENCLOSED_BY = '"';

-- Upload orders.csv to the named stage before running the following.
LIST @raw.orders_inbox;

COPY INTO raw.order_events
  FROM @raw.orders_inbox
  FILES = ('orders.csv')
  FILE_FORMAT = (FORMAT_NAME = 'SF_LEARN_ORDERS.RAW.ORDERS_CSV')
  VALIDATION_MODE = RETURN_ALL_ERRORS;

COPY INTO raw.order_events
  FROM @raw.orders_inbox
  FILES = ('orders.csv')
  FILE_FORMAT = (FORMAT_NAME = 'SF_LEARN_ORDERS.RAW.ORDERS_CSV')
  ON_ERROR = 'ABORT_STATEMENT';

SELECT COUNT(*) AS raw_rows FROM raw.order_events;
```

Expect no validation errors and nine rows loaded. File validation accepts `oops` in a text column; business validation comes next. Snowflake uses load metadata to avoid reloading known files into the same table. A duplicate order can still arrive in another file, and `FORCE = TRUE` can reload known files. Keep file-level load tracking separate from order-level deduplication. [COPY INTO table](https://docs.snowflake.com/en/sql-reference/sql/copy-into-table)

### Step B: validate, deduplicate, then merge

Run the rest of Section 2. The `raw.order_events_checked` view uses `TRY_TO_DATE` and `TRY_TO_DECIMAL` so bad conversions yield `NULL`. Format checks prevent silently rounding amounts with more than two decimal places or parsing fractional version numbers. The lab accepts non-negative amounts and positive integer versions. Expect a single rejection: row 8, order `O1005`, reason `BAD_AMOUNT`. In a real pipeline, record rejection rates and keep a way to repair and replay rows.

| Order | What arrived in RAW | What belongs in CORE |
|---|---|---|
| `O1001` | Version 2, 120.00, arrived before version 1, 100.00 | Version 2. Arrival order does not establish source order |
| `O1003` | Paid, then refunded | Version 2, `REFUNDED`. Keep the order out of paid sales |
| `O1004` | The same version and payload arrived twice | One order row |

Here is Section 3's key statement. The source subquery picks the highest version per order. A unique `ingest_id` breaks ties for identical replays. The match condition then prevents an older version from replacing a newer target row.

```sql
SET batch_first = 1;
SET batch_last = 9;

MERGE INTO core.orders AS t
USING (
  SELECT order_id, order_date, status, amount, source_version
  FROM raw.order_events_checked
  WHERE reject_reason IS NULL AND ingest_id BETWEEN $batch_first AND $batch_last
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY order_id ORDER BY source_version DESC, ingest_id DESC
  ) = 1
) AS s ON t.order_id = s.order_id
WHEN MATCHED AND s.source_version > t.source_version THEN UPDATE SET
  t.order_date = s.order_date,
  t.status = s.status,
  t.amount = s.amount,
  t.source_version = s.source_version
WHEN NOT MATCHED THEN INSERT (order_id, order_date, status, amount, source_version)
  VALUES (s.order_id, s.order_date, s.status, s.amount, s.source_version);
```

`WHERE` filters rows before the window function assigns ranks; `QUALIFY` filters after it. `MERGE` does not choose the winner for you. Multiple source rows matching one target can cause an update error, and duplicate unmatched rows can all be inserted. On standard tables, primary, unique, and foreign keys are not enforced. Test the grain explicitly. If the editor reconnects, set the batch variables again before running the `MERGE`. [QUALIFY](https://docs.snowflake.com/en/sql-reference/constructs/qualify) · [MERGE](https://docs.snowflake.com/en/sql-reference/sql/merge) · [Constraints](https://docs.snowflake.com/en/sql-reference/constraints-overview)

The lab assumes one writer and identical payloads for the same key and source version. Check for conflicts across batches in production. Since invalid rows are filtered before ranking, an invalid new version can leave the last valid state visible while a correction is pending. Make that outcome visible to operators and decide whether the business should continue serving the older value.

### Step C: check the numbers, then replay a batch

The `mart.daily_paid_sales` view reads paid orders from `core.orders` and groups `COUNT(*)` and `SUM(amount)` by `order_date`.

```sql
SELECT COUNT(*) AS order_rows FROM core.orders;

SELECT order_id, COUNT(*) AS n
FROM core.orders
GROUP BY order_id
HAVING COUNT(*) > 1;

SELECT * FROM mart.daily_paid_sales ORDER BY order_date;
```

| Order date | Batch A: count / amount (USD) | After batch B: count / amount (USD) |
|---|---:|---:|
| 2026-11-01 | 2 / 195.50 | 2 / 190.50 |
| 2026-11-02 | 1 / 80.00 | 2 / 105.00 |

After batch A, expect five core rows and no duplicate keys. November 1 adds 120.00 and 75.50; November 2 counts `O1004` once. Now run Section 4: **insert batch B once**, then run the `MERGE` twice. The first run updates `O1002` to 70.50 and inserts `O1007` at 25.00, while ignoring the old version of `O1001`. The second run should report zero updates and inserts. This demonstrates **idempotence** in the transformation. Repeating the raw `INSERT` would still add rows. In production, track stable event identifiers and advance a batch checkpoint only after successful processing.

## 4. SQL you will use again {#useful-sql}

### JSON: expand one order into its line items

`VARIANT` stores semi-structured values. `PARSE_JSON` turns a string into a structured value; use a colon to reach a key and cast it to the type you need. `LATERAL FLATTEN` can produce several rows for every input order.

```sql
WITH orders AS (
  SELECT PARSE_JSON('{
    "order_id": "O1001",
    "items": [{"sku":"A","qty":2,"price":40.00}, {"sku":"B","qty":1,"price":40.00}]
  }') AS payload
)
SELECT o.payload:order_id::VARCHAR AS order_id,
       f.value:sku::VARCHAR AS sku,
       f.value:qty::NUMBER(10, 0) * f.value:price::NUMBER(12, 2) AS line_amount
FROM orders AS o, LATERAL FLATTEN(INPUT => o.payload:items) AS f
ORDER BY sku;
```

Expect 80.00 for A and 40.00 for B. Summing a repeated order total over these two rows would yield 240.00 instead of 120.00. Sum line amounts or regain order grain before joining to another fact. Empty arrays produce no rows by default; see `OUTER => TRUE` if you need to preserve the left row. Decide separately what empty arrays, missing keys, and JSON null mean for your metric. [Semi-structured queries](https://docs.snowflake.com/en/user-guide/querying-semistructured) · [FLATTEN](https://docs.snowflake.com/en/sql-reference/functions/flatten)

### Dates: decide where midnight falls

The lab receives an order date, so `DATE` fits. With timestamped events, save the original event time and record when the data became available. Define the reporting date in the business time zone. For example, if a timestamp without a time zone (`TIMESTAMP_NTZ`) represents Coordinated Universal Time (UTC), this query returns `2026-11-01`:

```sql
SELECT CONVERT_TIMEZONE(
  'UTC', 'America/Chicago', '2026-11-02 02:00:00'::TIMESTAMP_NTZ
)::DATE AS reporting_date;
```

A named time zone such as `America/Chicago` accounts for changes to the local offset. For historical model features, the difference between occurrence time and availability time matters as well. [CONVERT_TIMEZONE](https://docs.snowflake.com/en/sql-reference/functions/convert_timezone)

## 5. Query performance and costs {#performance-cost}

![A date predicate eliminates micro-partitions whose date ranges are fully outside the requested range.](pruning.svg "Figure 3. A filter for November 2 can skip three of these illustrative micro-partitions. Scanning the matching partition still requires filtering its rows.")

Snowflake automatically divides standard tables into micro-partitions and records metadata such as column ranges. With a predicate like `WHERE order_date >= '2026-11-02'::DATE AND order_date < '2026-11-03'::DATE`, Snowflake can skip partitions whose dates cannot match. This is **pruning**. If every partition has some records for November 2, the filter might eliminate little at the partition level. The nine lab rows teach grain and correctness; choose a larger sample to study pruning. [Micro-partitions and clustering](https://docs.snowflake.com/en/user-guide/tables-clustering-micropartitions)

### Find the slow part in Query Profile

In Snowsight, find the statement in Query History and open Query Profile. Record the query ID and warehouse. Separate queue time, compilation, and execution, then inspect the expensive scans, joins, or aggregations.

| Symptom | First check |
|---|---|
| Long queue, short execution | Check whether reports share compute with batch jobs. Split workloads or evaluate a multi-cluster warehouse for concurrency; check edition requirements |
| Most partitions scanned for a narrow question | Check the filter and projected columns. Compare partitions scanned, total partitions, and bytes scanned. Consider clustering and its maintenance cost if a large table sees this continually |
| A join emits far more rows than expected | Check the grain of both sides and look for missing join conditions or non-unique keys |
| Sort or aggregate spills heavily to remote storage | Reduce the intermediate data first. Then compare a larger warehouse if the query still spills |
| An identical rerun suddenly gets much faster | Separate persisted result reuse and warehouse data caching. In a benchmark session, `ALTER SESSION SET USE_CACHED_RESULT = FALSE;` bypasses result reuse. Record the warehouse's cache state too |

[Query History and Profile](https://docs.snowflake.com/en/user-guide/ui-snowsight-activity) · [Warehouse considerations](https://docs.snowflake.com/en/user-guide/warehouses-considerations) · [Persisted query results](https://docs.snowflake.com/en/user-guide/querying-persisted-results)

**Measure the difference.** If you can access the sample data, run the January 1995 query below. Then run it without the `WHERE` clause. Compare partitions and bytes in the respective TableScan operators. These queries answer different questions; the experiment illustrates scan scope. When benchmarking a rewrite, verify that the outputs agree first.

```sql
ALTER SESSION SET USE_CACHED_RESULT = FALSE;

SELECT SUM(o_totalprice) AS total_price
FROM snowflake_sample_data.tpch_sf1.orders
WHERE o_orderdate >= DATE '1995-01-01'
  AND o_orderdate < DATE '1995-02-01';

-- After measuring both queries, restore the default behavior.
ALTER SESSION UNSET USE_CACHED_RESULT;
```

### Check credits alongside latency

An active virtual warehouse is billed by the second, with a one-minute minimum each time it starts or resumes. At an **illustrative** rate of one credit per hour, 12 continuous minutes cost about 0.20 credits. Twelve 10-second runs that each start a suspended warehouse also cost at least 0.20 credits. The example excludes other services and does not quote a price. Storage and serverless features can incur costs independently of the lab warehouse. [Compute cost](https://docs.snowflake.com/en/user-guide/cost-understanding-compute)

Compare several runs under the same role, data range, and cache conditions. Check usage as well as latency. A resource monitor applies to user-managed warehouses; check separate usage and budgets for serverless features such as Snowpipe. [Resource monitors](https://docs.snowflake.com/en/user-guide/resource-monitors)

## 6. Beyond the lab: pipelines, access, and recovery {#production}

### Match the tool to the input and the failure mode

| Need | Start with | Check before adopting |
|---|---|---|
| Scheduled CSV or Parquet batches | Stage + `COPY INTO` | Batch identity, reject handling, and load history |
| Files arriving throughout the day | Snowpipe | File sizes, freshness, duplicates, and serverless usage |
| A producer sends rows directly | Snowpipe Streaming | Channels, source offsets, and replay policy |
| A frequently queried derived table defined by a `SELECT` | Dynamic Table | `TARGET_LAG` expresses a freshness target. Check the supported SQL, refresh mode, actual lag, and cost |
| Custom processing of changes, such as upserts and deletes | Stream + Task | A stream tracks a change offset. Reading it with `SELECT` does not advance it; committed consumption through a write does. Separate consumers usually need separate streams |
| A team maintains models and shared metrics | dbt with Snowflake | dbt organizes models and tests; Snowflake runs the SQL. Add unique order-key, not-null, and accepted-status checks |

[Data loading options](https://docs.snowflake.com/en/user-guide/data-load-overview) · [Dynamic Tables decision guide](https://docs.snowflake.com/en/user-guide/dynamic-tables/decision-guide) · [Introduction to streams](https://docs.snowflake.com/en/user-guide/streams-intro) · [dbt's Snowflake setup](https://docs.getdbt.com/docs/local/connect-data-platform/snowflake-setup)

### Give readers access to the view

Role-based access control (RBAC) lets you keep report consumption separate from pipeline writes. An analyst needs warehouse `USAGE`, database and schema `USAGE`, and `SELECT` on the report view. Grant those permissions through an authorized role. Leave raw-table access and writes to the pipeline role. Test with the actual reporting role: the report should open and a direct read of RAW should fail. [Access control](https://docs.snowflake.com/en/user-guide/security-access-control-overview)

### Inspect history before restoring data

Time Travel can query historical data within its retention window. A zero-copy clone initially shares the underlying storage; changes can add storage over time. In a sandbox, replace the placeholder with the query ID of a change you want to inspect:

```sql
SELECT order_id, status, amount
FROM sf_learn_orders.core.orders
BEFORE (STATEMENT => '<query_id_of_the_change>')
ORDER BY order_id;
```

`BEFORE` returns the state preceding the specified change. Retention depends on configuration, object type, and edition. Long-term audit or recovery outside that window needs its own plan. [Time Travel](https://docs.snowflake.com/en/user-guide/data-time-travel) · [AT / BEFORE](https://docs.snowflake.com/en/sql-reference/constructs/at-before)

## 7. Apply it to reporting, data science, or fraud {#applications}

| Scenario | A useful first project | First validation |
|---|---|---|
| Business intelligence (BI) | A read-only dashboard on `MART.DAILY_PAID_SALES` | Pick a day, trace its amount to source orders, and see where a refund changes the report |
| Data science | Compute historical aggregates with Snowpark Python and pull back a small result | A feature may use only records available at the label cutoff. A current order snapshot can leak a future refund |
| Real-time fraud | Prepare historical features, evaluate models, and inspect drift in the warehouse | Evaluate an online feature service or low-latency store if the request has a tight tail-latency target |

To run this locally, install `snowflake-snowpark-python` in a virtual environment, then configure a named connection called `orders_lab` with an approved warehouse and a role that can read MART. Follow the authentication instructions in the session guide. The example needs no credentials in the source. [Set up Snowpark Python](https://docs.snowflake.com/en/developer-guide/snowpark/python/setup)

```python
from snowflake.snowpark import Session

session = Session.builder.config("connection_name", "orders_lab").create()
try:
    rows = (
        session.table("SF_LEARN_ORDERS.MART.DAILY_PAID_SALES")
        .sort("ORDER_DATE")
        .limit(10)
        .collect()
    )
    for row in rows:
        print(row)
finally:
    session.close()
```

Snowpark DataFrame transformations are generally lazy until an action such as `collect()`. Limit and aggregate in Snowflake first. `collect()` brings rows back to Python, so this is suitable for a small daily report. [Create a Snowpark session](https://docs.snowflake.com/en/developer-guide/snowpark/python/creating-session) · [Snowpark DataFrames](https://docs.snowflake.com/en/developer-guide/snowpark/python/working-with-dataframes)

## 8. Try these next, then clean up {#practice}

1. Send `O1007` as a newer refund and an older paid record. Predict which day's sales change, then replay the batch and check the amount.
2. Give `O1001` two line items. Sum its order total after joining and explain the result. Rewrite the query to sum at order grain.
3. Run two equivalent scans over a large sample with different filter strategies, or compare with and without a date range. Record bytes, partitions, and runtime. Explain whether extra compute addresses what you saw.
4. Hand someone the metric definition, the batch boundaries, where to find rejects, and the steps for a read-only role. Have them reconcile one day independently.

Suspend the lab warehouse when you finish, or skip the first statement if it has already suspended. If the database and warehouse are disposable, check their names and drop them one statement at a time. The downloadable lab does not execute these operations.

```sql
ALTER WAREHOUSE sf_learn_orders_wh SUSPEND;
-- Only after checking these are your lab's disposable objects:
DROP DATABASE sf_learn_orders;
DROP WAREHOUSE sf_learn_orders_wh;
```

## 9. A reading path {#further-reading}

| Order | Resource | What to do with it |
|---|---|---|
| 1 | [Snowflake in 20 minutes](https://docs.snowflake.com/en/user-guide/tutorials/snowflake-in-20minutes) | Run the load and query yourself. Use the [Workspaces guide](https://docs.snowflake.com/en/user-guide/ui-snowsight/workspaces-working) if the interface looks different |
| 2 | [Snowflake University: Data Warehousing Workshop](https://learn.snowflake.com/en/courses/OD-ESS-DWW/) | Load CSV and JSON yourself and check what reaches the table |
| 3 | [Official hands-on tutorials](https://docs.snowflake.com/en/learn-tutorials) | Pick one cloud source such as Amazon S3 or Azure and practice a staged load |
| 4 | [Query History and Query Profile](https://docs.snowflake.com/en/user-guide/ui-snowsight-activity) | Use operator timings and scan metrics to explain a slow query and verify a change |
| 5 | [Snowflake University: Hands-On Essentials](https://learn.snowflake.com/en/pages/hands-on-essentials-track/) | After Badge 1, pick Data Engineering, Data Science, or Data Applications to fit your goal |
| 6 | [The Snowflake Elastic Data Warehouse, SIGMOD 2016 (paper)](https://pages.cs.wisc.edu/~remzi/Classes/739/Papers/p215-dageville-snowflake.pdf) | Read why the authors chose shared storage and multi-cluster compute. Use current documentation for today's product behavior |

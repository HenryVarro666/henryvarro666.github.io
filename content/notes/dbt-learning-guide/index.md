---
title: "Learning dbt: From a First SQL Model to a Production Orders Report"
date: 2026-11-13T19:46:15-06:00
description: "Build a complete orders project with dbt v2 and local DuckDB. Practice model layers, data tests, late arrivals, snapshots, and production releases."
topics: [data-science, ml-systems]
---

Four orders total 116 dollars. Join them directly to payment deliveries and sum the order amounts, and the total becomes 166 dollars. The database reports no error: split payments and a replay make one order appear three times. Fixing this is a good way to learn dbt. First ask **what one row represents**, then decide **which conditions downstream readers can rely on**.

This note assumes you can read `SELECT`, `JOIN`, and `GROUP BY` in Structured Query Language (SQL). Allow 45–60 minutes for the main lab. Start with [local setup](#setup) to follow along, or jump to [late data and replay](#incremental) or [production releases](#production).

## Where dbt fits in a data pipeline {#mental-model}

dbt manages **transformation code and dependencies between models**. A SQL model is typically a named `SELECT`. You put the query in Git, use `ref()` to declare what it depends on, and write tests for the result. dbt parses the project, expands templates, orders the work, and sends SQL to the target engine, such as DuckDB, Snowflake, BigQuery, or Databricks.

![An ingestion tool loads raw tables. dbt compiles and submits transformation SQL. The data engine builds the models that reports query.](elt-boundary.svg "Figure 1. In ELT, ingestion handles E and L, dbt manages T, and the target engine executes the queries.")

**Extract, load, transform (ELT)** loads raw data into the target before transforming it there. Extract, transform, load (ETL) transforms it before loading it into the target. A connector or ingestion job moves the data; dbt turns raw tables into datasets; an orchestrator starts the work when inputs are ready; a business intelligence (BI) tool displays the results. A dbt seed can load a small comma-separated values (CSV) file, such as a fixed lookup table. Periodic database syncs and data collection from an application programming interface (API) belong in an upstream job.

### dbt v2, dbt OSS, and older tutorials {#versions}

| Name in the material | What it refers to | Which to use |
|---|---|---|
| **dbt v2**, called Fusion in early material | The primary distribution of the current Rust engine. Runs locally; some features require a dbt platform sign-in. | This lab uses `dbt==2.0.11` and its built-in DuckDB adapter. |
| **dbt OSS**, called dbt Core v2 in early material | The Apache 2.0 open source distribution of the same generation. | Check the installation guide and feature table if your team requires an open source runtime. |
| **dbt v1**, often called dbt Core v1 | The Python engine. Older tutorials install `dbt-core` and a separate adapter. | Use a separate environment with the project's pinned engine and adapter when maintaining a v1 project. |
| **dbt platform**, often called dbt Cloud in older material | A hosted development and execution environment with tools such as an editor, jobs, and a catalog. | Try it for hosted jobs and team workflows. The lab needs no account. |

Version and adapter information was checked on **November 13, 2026**. Run `dbt --version` before copying commands from a tutorial. In particular, check how test arguments are nested, which incremental strategies the adapter supports, and what the documentation commands produce. [Official v2 introduction](https://docs.getdbt.com/docs/introduction?version=2) · [Distribution comparison](https://docs.getdbt.com/blog/comparing-dbt-and-dbt-oss) · [Supported data platforms](https://docs.getdbt.com/docs/supported-data-platforms?version=2)

## Run the project locally {#setup}

[Download the complete orders lab (ZIP)](/notes/dbt-learning-guide/dbt-orders-lab.zip). Extract it and open a terminal in `dbt-orders-lab`. All input data is fictional, and the lab writes to a local file named `orders_lab.duckdb`. The following steps use Python 3.12 on macOS or Linux.

```bash
cd dbt-orders-lab
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
dbt --version

python load_data.py initial
dbt debug --profiles-dir .
dbt build --profiles-dir .
```

On Windows PowerShell, create the environment with `py -3.12 -m venv .venv` and activate it with `.\.venv\Scripts\Activate.ps1`, then run the remaining commands. `dbt --version` should report **2.0.11**. The first build should report **26 successful tasks**: six models, 17 data tests, one unit test, one seed, and one snapshot. The tested setup uses Python 3.12 and dbt 2.0.11. The loader uses `duckdb==1.5.0` to write the same database file. The dbt v2 adapter is already built in. [Install dbt](https://docs.getdbt.com/docs/local/install-dbt) · [DuckDB setup for v2](https://docs.getdbt.com/docs/local/connect-data-platform/duckdb-setup)

```text
dbt-orders-lab/
├── dbt_project.yml                 # Project name, model paths, build defaults
├── profiles.yml                    # Connection: DuckDB file and output schema
├── data/                           # Initial data and later batches
├── load_data.py                     # Simulates a loader writing to raw
├── seeds/payment_statuses.csv       # Small lookup table tracked in Git
├── models/
│   ├── staging/                    # Declare and clean up inputs
│   ├── intermediate/               # Pick the latest version and aggregate
│   ├── marts/                      # Order fact table and daily report
│   └── _properties.yml             # Descriptions and tests
├── tests/no_order_overcapture.sql
└── snapshots/orders_history.yml
```

The `profile: orders_lab` setting in `dbt_project.yml` points to the profile below. YAML is the text format used for configuration. `--profiles-dir .` tells dbt to read `profiles.yml` from this directory. A **target** is a named connection and output configuration. Here `dev` writes models into `analytics_dev`, while inputs stay in `raw`. Run commands from the lab directory because the database path is relative. [Profiles reference](https://docs.getdbt.com/docs/local/profiles.yml)

```yaml
# profiles.yml
orders_lab:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: orders_lab.duckdb
      schema: analytics_dev
      threads: 2
```

> **Find the first failing step.** If `dbt --version` is wrong, check which environment and executable you are using. If `debug` passes but dbt cannot find `raw.orders`, check that you ran the loader and that both programs use the same file. Extract the archive into a fresh directory to start over. Reloading `initial` replaces only the input tables; existing models and snapshot history remain.

## Build the report from two raw tables {#lab}

All amounts in this shop use integer US cents. **Ordered amount** is taken from the order. **Captured amount** counts payments whose current status is `captured`. The `paid_order_count` counts any order with positive capture, including partially paid orders. Capture is attributed to the **order date**. A cash flow report grouped by payment date would answer a different question. The lab has no refunds; a real pipeline needs rules for refunds, disputes, currencies, and exchange rates.

![Staged payments are reduced to one version per payment, aggregated by order, and joined to staged orders. The order fact table feeds the daily report.](orders-lineage.svg "Figure 2. Arrows point from inputs to consumers. The order fact table brings the two branches together. The diagram omits the snapshot and tests.")

### Declare dependencies with source() and ref() {#sources}

`source('shop', 'orders')` refers to a table that **already exists outside dbt**. `shop` is a logical name, and `schema: raw` says where it lives. This declaration does not load `raw.orders`. Use `ref()` for a dbt model, seed, or snapshot. dbt resolves the reference to a relation in the target and records the dependency. The directed acyclic graph (DAG) determines the order of execution, regardless of file names. [source()](https://docs.getdbt.com/reference/dbt-jinja-functions/source) · [ref()](https://docs.getdbt.com/reference/dbt-jinja-functions/ref)

```yaml
# models/staging/_sources.yml (excerpt)
version: 2
sources:
  - name: shop
    schema: raw
    tables:
      - name: orders
      - name: payment_events
```

```sql
-- models/staging/stg_orders.sql
select
    order_id,
    ordered_on,
    lower(trim(channel)) as channel,
    order_total_cents
from {{ source('shop', 'orders') }}
```

The double braces contain a **Jinja** template expression. Look at the compiled SQL to separate a template problem from a problem with the target's SQL dialect:

```bash
dbt compile --profiles-dir . --select fct_orders
# Open target/compiled/orders_lab/models/marts/fct_orders.sql
```

### Declare the grain before writing the join {#grain}

| Table or model | What one row means | Key |
|---|---|---|
| `raw.orders` / `stg_orders` | One order | `order_id` |
| `raw.payment_events` / `stg_payments` | One delivered record, possibly a replay | `delivery_id` |
| `int_payments_current` | One payment at its current source revision | `payment_id` |
| `int_payments_by_order` / `fct_orders` | Payment totals for an order / a complete order | `order_id` in each |
| `mart_daily_orders` | Metrics for one order date and channel | `(ordered_on, channel)` |

Order `o100` has two payments, one of which was replayed. Joining orders to the five original payment deliveries puts the same 2,500 cents on three rows and adds 5,000 cents to the order total. Reproduce this just after the initial batch:

```bash
dbt show --profiles-dir . --inline "
select count(*) as joined_rows, sum(o.order_total_cents) as wrong_cents
from {{ ref('stg_orders') }} as o
left join {{ ref('stg_payments') }} as p on o.order_id = p.order_id"
```

Expect **6 rows and 16,600 cents**. Pick the current revision by `payment_id`, aggregate those rows by `order_id`, then join at order grain. `sum(distinct order_total_cents)` would combine two independent orders that happened to have the same price. [Kimball: Declaring the grain](https://www.kimballgroup.com/2003/03/declaring-the-grain/)

### Choose a layer and a materialization {#materializations}

**Staging** usually mirrors the source systems, with one model per input relation. Standardize column names, types, and text values there. **Intermediate** models do work such as selecting a current version or aggregating to order grain. **Marts** serve business processes and readers: `fct_orders` supports reconciliation, and `mart_daily_orders` goes to a dashboard. These layers are conventions. A simple transformation can live in one clear model.

| Materialization | What dbt produces | When to start with it |
|---|---|---|
| `view` | A stored query definition, evaluated when read | Lightweight staging. Measure query costs if views are deeply nested or read often. |
| `table` | Stored query results, normally recomputed on each build | A fact or report that is read often and inexpensive enough to rebuild. |
| `incremental` | A table built initially, with subsequent writes governed by model logic and the adapter strategy | Use it when a full rebuild is too expensive. Define what changed, the merge key, and how backfills will work. |
| `ephemeral` | A common table expression (CTE) in downstream queries | Small logic shared by a few models. There is no separate relation to query. |

This project uses views for staging and intermediate models, and tables for marts. The current payment model overrides its folder default with the incremental materialization. Other options, such as materialized views, depend on the target platform. Project folders are separate from warehouse schemas; if you introduce custom schemas, confirm the naming rule keeps development outputs isolated. [Project structure guide](https://docs.getdbt.com/best-practices/how-we-structure/1-guide-overview) · [Materializations](https://docs.getdbt.com/docs/build/materializations)

The fact table starts from orders and uses a left join to keep `o103`, which has no payment attempts:

```sql
-- models/marts/fct_orders.sql
select
    o.order_id,
    o.ordered_on,
    o.channel,
    o.order_total_cents,
    coalesce(p.captured_cents, 0) as captured_cents
from {{ ref('stg_orders') }} as o
left join {{ ref('int_payments_by_order') }} as p on o.order_id = p.order_id
```

```bash
dbt show --profiles-dir . --inline "
select * from {{ ref('mart_daily_orders') }}
order by ordered_on, channel"
```

| Order date | Channel | Orders | Paid orders | Ordered (cents) | Captured (cents) |
|---|---|---:|---:|---:|---:|
| 2026-01-01 | web | 2 | 1 | 4,300 | 2,500 |
| 2026-01-02 | store | 1 | 1 | 3,200 | 3,200 |
| 2026-01-02 | web | 1 | 0 | 4,100 | 0 |

## Make quality failures reproducible {#testing}

A **data test** queries real data for violations. An ordinary test passes when it returns zero failures. The built-in `not_null`, `unique`, `accepted_values`, and `relationships` tests cover nulls, duplicate keys, valid values, and missing referenced keys. Check uniqueness and nulls separately. A relationship test is also separate from checking that the referenced key is unique. The lab's two-row seed makes the status vocabulary reviewable in Git. In v2, nest test inputs under `arguments`.

```yaml
# models/_properties.yml (excerpt)
- name: payment_status
  data_tests:
    - not_null
    - relationships:
        arguments:
          to: ref('payment_statuses')
          field: payment_status
```

Use a stand-alone SQL test, also called a singular test, for rules involving several columns. This shop forbids capture above the order total. Select rows that violate that rule. A **unit test** instead supplies a few rows to exercise a branch of model logic. The lab gives the fact model two orders and totals for just one, then expects the unpaid order to survive with zero capture. This catches an accidental change from a left join to an inner join, even before a real order without payment appears. [Data tests](https://docs.getdbt.com/docs/build/data-tests) · [Unit tests](https://docs.getdbt.com/docs/build/unit-tests)

```sql
-- tests/no_order_overcapture.sql
select order_id, order_total_cents, captured_cents
from {{ ref('fct_orders') }}
where captured_cents < 0 or captured_cents > order_total_cents
```

```yaml
# models/_properties.yml (excerpt)
unit_tests:
  - name: includes_order_without_payment
    model: fct_orders
    given:
      - input: ref('stg_orders')
        rows:
          - {order_id: a, ordered_on: '2026-01-01', channel: web, order_total_cents: 1000}
          - {order_id: b, ordered_on: '2026-01-01', channel: web, order_total_cents: 500}
      - input: ref('int_payments_by_order')
        rows:
          - {order_id: a, captured_cents: 700}
    expect:
      rows:
        - {order_id: a, captured_cents: 700}
        - {order_id: b, captured_cents: 0}
```

```bash
dbt test --profiles-dir . --select 'test_type:unit'
```

Run the late batch in the next section before injecting the bad data. The full build includes the relationship test for allowed payment statuses. Selecting only `int_payments_current+` does not bring in that upstream check.

## Find late changes, then pick the current revision {#incremental}

A delivery can arrive several days late, and an old revision can arrive after a newer one. This lab has two timestamps: `event_at` says when the payment happened, and `loaded_at` says when the delivery entered the database. The source's `revision` determines which version of a payment takes precedence.

![Payment p02 occurred January 1, revision 1 was replayed January 4, and revision 2 arrived January 5.](late-arrival.svg "Figure 3. The correction arrived on January 5 and changes the orders report for January 1.")

### What an incremental build does {#incremental-sql}

The first build reads every delivery. An incremental run uses a load-time window to find **changed payment keys** and then looks at all revisions for those keys. Payment `p04` has revision 2 outside the window and a late revision 1 inside it. Ranking only rows in the window would turn an existing capture of 3,200 cents into zero. The model uses `merge` on `payment_id` to write the winning row. The `unique_key` config tells dbt how to match rows. The model and its tests are responsible for making that key unique. [Incremental models](https://docs.getdbt.com/docs/build/incremental-models) · [Adapter strategy support](https://docs.getdbt.com/docs/build/incremental-strategy?version=2)

```sql
-- models/intermediate/int_payments_current.sql (excerpt)
{{ config(materialized='incremental', incremental_strategy='merge',
          unique_key='payment_id') }}

{% if is_incremental() %}
-- This filter in changed_keys picks the payment keys to recompute:
where loaded_at >= (
    select coalesce(max(loaded_at), timestamp '1900-01-01')
    from {{ this }}
) - interval '2 days'
{% endif %}
```

`{{ this }}` points to the current model's relation. The first build and a full refresh take the branch that does not use an existing target table. The two-day overlap in the lab assumes an append-only feed with trustworthy load timestamps and revisions. The maximum from a table of winning revisions is not an exact processing cursor. For production, consider a watermark based on the last completed load batch. Decide how to backfill outside the overlap, handle deletes and conflicting revisions, and retry failures. On large inputs, measure whether selecting changed keys and reading their history reduces scans.

### Apply the late batch, replay it, and recover from a failure {#changes}

```bash
python load_data.py late
dbt build --profiles-dir . --select 'int_payments_current+'
dbt show --profiles-dir . --inline "
select * from {{ ref('mart_daily_orders') }}
order by ordered_on, channel"
```

Three things happen. The correction drops `p02` from 1,500 to 1,200 cents. A late failure for `p04` is an older revision and does not replace its capture. Order `o101` gains 1,800 cents. The **January 1 web capture moves from 2,500 to 4,000 cents, and paid orders go from one to two**. January 2 store stays at 3,200 cents. Reload `late` and repeat the same build; the result holds. The loader deduplicates by `delivery_id`; the model merges by `payment_id`.

Now inject a delivery with an unrecognized status:

```bash
python load_data.py bad
dbt build --profiles-dir .    # Expected to fail
```

The log should report one failing row for the status relationship test and show `mart_daily_orders` as skipped. Meanwhile, dbt **already wrote** `int_payments_current`, which does not depend on the status seed. The test spans multiple parents and blocks the downstream branch that depends on the relevant inputs. A build is not one atomic transaction across all models. The previous report table still exists. After deleting the bad fixture, rebuild current payments from source; this merge does not delete keys that disappear from the input:

```bash
python load_data.py fix
dbt build --profiles-dir . --select 'int_payments_current+' --full-refresh
```

The January 1 web row should again show **4,000 cents**. Before using `--full-refresh` in production, check whether the full input history still exists, which relations will be rewritten, and what the window and cost mean for readers. For large tables, plan a backfill for the affected partitions or keys. [Build and failure propagation](https://docs.getdbt.com/reference/commands/build)

## Add history, reuse logic, and read lineage {#toolkit}

### Snapshots record what the source looked like {#snapshots}

If the source overwrites order status in place, the latest row cannot answer whether an order was open three days ago. A **snapshot** stores multiple versions per business key with validity intervals. This is also called a Type 2 slowly changing dimension (SCD Type 2). The lab uses `order_id` as the key and a reliable `updated_at` column. If the source cannot provide one, consider a `check` strategy against specific columns. Decide how to represent source deletions in either case.

The first build captured the initial version of `o100`. Change its status and take another snapshot:

```bash
python load_data.py order-status
dbt snapshot --profiles-dir . --select orders_history
dbt show --profiles-dir . --inline "
select order_id, order_status, dbt_valid_from, dbt_valid_to
from {{ ref('orders_history') }}
where order_id = 'o100'
order by dbt_valid_from"
```

| order_id | Status | Valid from | Valid to |
|---|---|---|---|
| o100 | open | 2026-01-01 09:00 | 2026-01-06 10:00 |
| o100 | complete | 2026-01-06 10:00 | `NULL` (current) |

A snapshot only sees states visible at run time. If an order changes twice in the morning and the job runs at night, the middle version can be missing. Keep an event log or consider change data capture (CDC) when every source mutation matters. A snapshot keeps observed history; an incremental current-state model keeps one current output per key. [Snapshots](https://docs.getdbt.com/docs/build/snapshots) · [Incremental or snapshot?](https://docs.getdbt.com/best-practices/how-we-handle-cdc/2-choosing-incremental-or-snapshots)

### Macros and documentation keep decisions in the project {#macros-docs}

A macro is a reusable Jinja template. If several staging models use the same status cleanup, put this macro under `macros/` and use `{{ clean_status('payment_status') }}`. Start with readable SQL and extract a macro once the shared behavior is clear. When moving warehouses, verify how the expanded SQL handles types, dates, and nulls.

```sql
{% macro clean_status(column_name) -%}
  lower(trim({{ column_name }}))
{%- endmacro %}
```

Packages such as dbt-utils also supply macros and tests. Check engine and adapter compatibility, declare a package version, and install it with `dbt deps`. Document metric definitions in model and column descriptions. A **model contract** can pin the shared columns and data types; tests still check business rules. Evaluate a semantic layer when multiple tools need to slice the same metric consistently. [Packages](https://docs.getdbt.com/docs/build/packages) · [Model contracts](https://docs.getdbt.com/docs/mesh/govern/model-contracts?version=2) · [Semantic Layer](https://docs.getdbt.com/docs/use-dbt-semantic-layer/dbt-sl)

```bash
dbt docs generate --profiles-dir .
dbt docs serve
# Open the address printed in the terminal; press Ctrl+C to stop serving.
```

Local dbt Docs v2 builds a static site backed by Parquet metadata. `manifest.json` describes dependencies and supports state comparisons; `run_results.json` reports outcomes and timings for a run. Read the run's log alongside the compiled SQL when a query fails. Treat internal documentation as a data asset: the `target/` directory can contain compiled SQL and sensitive relation names. [Generate documentation](https://docs.getdbt.com/docs/build/documentation?version=2) · [Artifacts](https://docs.getdbt.com/reference/artifacts/dbt-artifacts)

## Commands and a debugging order {#commands}

The examples below omit the lab's usual `--profiles-dir .` argument. The graph operator `+` includes ancestors as a prefix, or descendants as a suffix, in addition to the named node. Use `dbt ls` to preview selection before a build, including any tests selected indirectly.

| Goal | Command | What to check |
|---|---|---|
| Validate project structure | `dbt parse` | Fix broken references, YAML, and project settings at this step. |
| Inspect generated SQL | `dbt compile --select fct_orders` | Open `target/compiled/`. Compile does not build models; some cases still query warehouse metadata. |
| Preview a model query | `dbt show --select mart_daily_orders --limit 5` | Preview the current definition; query a `ref()` with `show --inline` to inspect the stored report. |
| Rebuild a report and its dependencies | `dbt ls --select '+mart_daily_orders'`, then substitute `dbt build` | Raw tables must already be available. The selector includes seeds, models, and relevant tests. |
| Find the downstream impact of a change | `dbt ls --select 'int_payments_current+'` | Running a smaller graph assumes the unbuilt upstream inputs are ready. |
| Rebuild a model on its own | `dbt run --select fct_orders` | Runs models without seeds, snapshots, or tests. Upstream relations must already be readable. |
| Test already built data | `dbt test --select 'test_type:data'` | After editing SQL, make sure the tests target outputs built from the new code. |
| Check freshness of a configured input | `dbt freshness --select 'source:shop.payment_events'` | Give the source a threshold first. The lab uses fixed dates and cannot stay fresh relative to the current clock. |

| Symptom | First check |
|---|---|
| `relation does not exist` | Check the profile, the output schema, and whether inputs and upstream models exist. Read the full relation name in the compiled SQL. |
| SQL compiles, but the warehouse rejects it | Check the dialect, column types, and permissions of the execution identity. |
| Every replay raises the total | Compare row counts, distinct keys, and totals across steps. Inspect append behavior, merge keys, and join grain. |
| The dashboard is stale despite passing tests | Check which relation it queries and when it was loaded and built. Valid rows can be missing a batch. |

[Command reference](https://docs.getdbt.com/reference/dbt-commands) · [Graph operators](https://docs.getdbt.com/reference/node-selection/graph-operators) · [How show works](https://docs.getdbt.com/reference/commands/show)

## Take the project into production {#production}

![Develop locally or in a personal schema, test the proposed change in isolation, and run a scheduled job before releasing the marts to readers.](release-path.svg "Figure 4. Separate development, validation, and production outputs. Decide what happens when a check fails before opening the report to readers.")

1. **Isolate outputs and credentials.** A developer writes into a personal schema. Continuous integration (CI) uses temporary schemas isolated by change and reads approved samples or production copies as needed. The production job runs as a service identity. Inject secrets from the environment or a secret manager; use a separate production profile.
2. **Build and test pull requests (PRs).** Run a full `dbt build` while that is cheap enough. For a large project, consider `dbt build --select 'state:modified+' --defer --state ./prod-state`. Keep the production manifest in `./prod-state`, outside the current run's `target/`. Deferral can resolve unselected upstream models to existing production relations; review cross-environment tests and permissions. The state comparison selects code changes, not all source data changes.
3. **Check availability and completeness.** For the real `payment_events` feed, configure `loaded_at_field: loaded_at` and set `warn_after` and `error_after` to match its delivery schedule. Run `dbt freshness` in v2 after the load succeeds and decide whether a warning or error should block a build. Older v1 jobs often use `dbt source freshness`. A recent maximum timestamp proves that some recent data arrived. Use a batch manifest, partition check, or reconciliation to establish completeness.
4. **Decide when readers get the new result.** A failed build can leave some models updated. If consumers need consistency, build in a candidate schema and switch a stable view or release pointer after checks pass, using the warehouse's supported mechanism. Record the code revision, batch, build time, and failing tests. Assign someone to respond and document the backfill procedure. A successful run should tell you which data went into the report.

For an hourly source, you might start by warning after two hours and raising an error after four. Merge the settings into your real source declaration. These thresholds are unsuitable for the lab's fixed dates.

```yaml
sources:
  - name: shop
    schema: raw
    tables:
      - name: payment_events
        config:
          loaded_at_field: loaded_at
          freshness:
            warn_after: {count: 2, period: hour}
            error_after: {count: 4, period: hour}
```

[Continuous integration](https://docs.getdbt.com/docs/deploy/continuous-integration) · [Defer](https://docs.getdbt.com/reference/node-selection/defer) · [Freshness configuration](https://docs.getdbt.com/reference/resource-configs/freshness) · [Freshness command](https://docs.getdbt.com/reference/commands/freshness)

### Apply it to another dataset {#applications}

| Application | Start with | Check first |
|---|---|---|
| Commerce reporting or reconciliation | One fact row per order, plus a separate fact per payment | Attribution dates for orders, capture, and refunds. Record the currency and timezone alongside the definition. |
| Product funnels or subscription retention | One row per event or per version of a subscription state | Event deduplication, user identity, and time boundaries. Exclude cohorts that have not completed the observation window. |
| Machine learning (ML) training features | One row per scoring opportunity, keyed by entity and decision time | Use information that both occurred and was available by the decision. A 09:55 payment loaded at 10:05 cannot inform a 10:00 decision. Record input and code versions. |
| An imaging study cohort | A file manifest keyed by subject, session, and modality | Use dbt for manifests, cohorts, and quality results. Image registration belongs in a separate program. Keep repeated scans from the same subject in the same split. |

When moving to Snowflake or Databricks, keep the business grain and tests. After changing the profile and adapter, verify SQL dialect, data types, permissions, cost, concurrency, and the incremental strategy. See the [Snowflake learning guide]({{< relref path="/notes/snowflake-learning-guide" lang="en" >}}) and the [Databricks payments lab]({{< relref path="/notes/databricks-guide" lang="en" >}}) for the platform side: storage, compute, and access control.

## Exercises and external resources {#resources}

| Stage | Resource | What to do after reading |
|---|---|---|
| 1. Try the workflow | {{< new-tab "Official dbt v2 introduction" "https://docs.getdbt.com/docs/introduction?version=2" >}} · {{< new-tab "dbt Learn: Fundamentals (VS Code)" "https://learn.getdbt.com/courses/dbt-fundamentals-vs-code" >}} | Build a model and add a test in a development schema. Locate its input in the lineage. The course interface may differ from the local lab. |
| 2. Practice SQL and grain | {{< new-tab "DuckDB SQL documentation" "https://duckdb.org/docs/stable/sql/introduction" >}} · {{< new-tab "Kimball's four design steps" "https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/dimensional-modeling-techniques/four-4-step-design-process/" >}} | Write the grain of orders and payments separately. Use `row_number()` to find the current revision, then verify totals with `GROUP BY`. |
| 3. Read a complete project | {{< new-tab "dbt Labs: Jaffle Shop" "https://github.com/dbt-labs/jaffle-shop" >}} · {{< new-tab "Official project structure guide" "https://docs.getdbt.com/best-practices/how-we-structure/1-guide-overview" >}} | Trace a mart upstream through `ref()` until you reach a source. Mark where the grain changes. Follow the repository's version instructions. |
| 4. Expand test coverage | {{< new-tab "dbt-utils package and source" "https://hub.getdbt.com/dbt-labs/dbt_utils/latest/" >}} · {{< new-tab "dbt unit tests" "https://docs.getdbt.com/docs/build/unit-tests" >}} | Check daily uniqueness on `(ordered_on, channel)`. Add a unit test case for partial payment and one for no payment. |
| 5. Practice scheduling and recovery | {{< new-tab "dbt workflow guide" "https://docs.getdbt.com/best-practices/best-practice-workflows" >}} · {{< new-tab "Dagster and dbt integration" "https://dagster.io/docs/integrations/libraries/dbt" >}} | Schedule load, check, and build in order. Simulate an overdue input and a failing quality test. Check when the job stops and which report version readers can see. |

For a portfolio exercise, build reports for capture by **order date** and capture by **payment date**. Add an order paid on a later day and show why the two results should disagree. Document each table's grain and attribution rule, give expected values, and run the lab from a fresh directory to check that someone else can reproduce it.

---
title: "Apache Airflow: From Your First Dag to a Rerunnable Pipeline"
date: 2026-11-13T19:49:00-06:00
description: "Learn task dependencies and data intervals, then run a checkout metrics pipeline locally and practice retries, backfills, idempotent writes and debugging."
topics: [data-science, ml-systems]
---

A daily checkout report has to answer two questions: **which day's users did we count, and what happens if we run it again?** A user assigned at 23:00 on November 1 has only an hour of follow-up by midnight. Marking that user as a non-converter would cut a 24-hour observation window short. A publisher can also write its output successfully and crash before reporting success. If each retry appends the same rows, the next report counts them twice.

This guide uses Python and a six-user fixture to make dates, dependencies and writes explicit in one pipeline. Basic Python functions are enough. Start with [how tasks run](#mental-model) and [try the lab](#lab), or skip to [backfills and debugging](#recovery) if you already maintain Dags.

## What Airflow runs and where computation happens {#mental-model}

**Airflow orchestrates workflows.** A Dag (directed acyclic graph) specifies which tasks depend on one another: check the source, run a query, then publish. Each execution creates a **Dag run**. Each task's execution within a run is a **task instance**. Runs for two dates share a Dag definition and have their own task states and logs. [Architecture](https://airflow.apache.org/docs/apache-airflow/3.4.0/core-concepts/overview.html) · [Dag runs and their states](https://airflow.apache.org/docs/apache-airflow/3.4.0/core-concepts/dag-run.html)

![The Dag processor reads the definition. The scheduler uses the executor to start tasks, which can submit jobs to Spark. The metadata database records run state.](orchestration.svg "Figure 1. Airflow records orchestration state. The compute engine reads and writes business data in tables or object storage.")

| Component | What it does | First thing to check |
|---|---|---|
| Dag processor | Imports Python definitions and serializes the Dag and task graph into the metadata database. | Is the file inside a configured folder or bundle? Does it import? |
| Scheduler | Decides when to run based on the timetable, upstream state and concurrency limits. | Is the Dag paused? Is an interval due? Did the upstream tasks succeed? |
| Executor | Dispatches the work using the configured execution environment, such as local processes, workers or containers. | Queue time, capacity, worker dependencies and logs. |
| Triggerer | Waits asynchronously on external conditions for tasks that support deferral. Those tasks can release their worker slot while waiting. | Is a triggerer running? Did the remote job finish before the timeout? |
| API server and metadata database | The server provides the interface and API; the database holds run, task and configuration state. | Check the graph, run history and task logs here. Inspect business rows in the target table. |

An **operator** is a task template, such as running Python, submitting SQL or waiting for a remote job. With **TaskFlow**, `@task` declares a Python function as a task. A call such as `counts = compute_counts(window)` during Dag definition returns a reference to the future result and establishes a dependency. The body runs later, when that task instance executes. **XCom (cross-communication)** passes small values between tasks. Put millions of rows in shared storage; pass their path, partition, row count or remote job identifier through XCom. [TaskFlow tutorial](https://airflow.apache.org/docs/apache-airflow/3.4.0/tutorial/taskflow.html) · [XCom](https://airflow.apache.org/docs/apache-airflow/3.4.0/core-concepts/xcoms.html)

Cron and a log file may be enough for a short script. Airflow earns its place when you need to inspect dependencies, rerun history and identify the failed step. Use [Spark and PySpark]({{< relref "/notes/spark-pyspark" >}}) for large joins and a streaming engine for continuous per-event processing. Airflow can submit their jobs or check their results on a schedule.

## Separate the data date from the run time {#time}

The checkout test compares the existing flow (A) with assignment to an optional reminder (B). A user converts if they make at least one purchase in the first 24 hours after assignment. A B user who never sees or clicks the reminder still belongs in the denominator. This matches the assignment-based definition in the [A/B testing note]({{< relref "/notes/ab-testing" >}}).

![The November 1 assignment cohort, the last user's 24-hour window, and Airflow's November 2 to November 3 interval.](cohort-clock.svg "Figure 2. The assignment cohort, the outcome window and the Dag data interval have different jobs. All times shown are UTC.")

| Time | Example | What it controls |
|---|---|---|
| Assignment interval | `[Nov 1 00:00, Nov 2 00:00)` UTC | Select people who entered on November 1. |
| User's outcome window | For `u06`: `[Nov 1 23:00, Nov 2 23:00)` | Include purchase at assignment. Exclude one exactly 24 hours later. |
| Dag data interval | `[Nov 2 00:00, Nov 3 00:00)` UTC | In this design it covers the extra day of follow-up. The run is due after midnight on November 3; it can start later. |
| Source completeness | Events available through at least `Nov 3 00:00` UTC, exclusive | Confirms that all 24-hour windows are observable. A late source can still be incomplete after the clock passes midnight. |

A **logical date** is a run's scheduling label. This scheduled run has a logical date of November 2 because its data interval begins then. Compute the input dates from `data_interval_start` and `data_interval_end`, rather than the actual start time or `datetime.now()`. For a manual run the lab requires an explicit `cohort_date`. Its inferred Airflow interval can depend on the timetable and trigger path.

In Airflow 3, a bare cron schedule such as `@daily` uses a trigger timetable by default, with equal interval start and end. Specify `CronDataIntervalTimetable("0 0 * * *", timezone="UTC")` to make full-day intervals explicit. For business dates in a local timezone, account for daylight saving time. Local midnight tomorrow and 24 elapsed hours can be different instants on a clock-change day. [Timetables](https://airflow.apache.org/docs/apache-airflow/3.4.0/authoring-and-scheduling/timetable.html) · [Manual triggers and data intervals](https://airflow.apache.org/docs/apache-airflow/3.4.0/core-concepts/dag-run.html#manual-triggering-and-data-intervals)

## Lab: run a conversion report for six users {#lab}

### 1. Set up an isolated environment {#setup}

The lab pins **Airflow 3.4.0 and Python 3.12**, with local SQLite for metadata. It needs no cloud account, Docker or Spark. Use a terminal on macOS or Linux, or Windows Subsystem for Linux (WSL2) on Windows. Download [checkout_airflow_dag.py](/notes/airflow/checkout_airflow_dag.py) into your current directory and run the following. The constraints file fixes the dependency versions tested by Airflow. [Official quick start](https://airflow.apache.org/docs/apache-airflow/3.4.0/start.html)

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

`AIRFLOW_HOME` keeps the database, configuration and logs in this lab directory. If you open another terminal, reactivate the same environment and set those variables using the original path. `airflow version` should print `3.4.0`.

### 2. Read the dependencies and run the Dag {#first-run}

Read these three lines as a dependency graph. The function calls declare tasks; their outputs are references that downstream tasks resolve at runtime.

```python
window = plan_window()
counts = compute_counts(window)
publish(window, counts)
```

![plan_window passes the date to compute_counts and publish. A failed completeness check stops the flow. publish writes one output for the cohort date.](dag-recovery.svg "Figure 3. Small XCom values pass the interval and counts, and the last task writes one file for the requested cohort.")

`plan_window` selects the date and a simulated source cutoff. `compute_counts` checks completeness, deduplicates purchases and counts users. `publish` validates the counts and writes JSON (JavaScript Object Notation). The default task dependency prevents publication if an upstream task fails. Run all three locally without starting a scheduler:

```bash
airflow dags test checkout_cohort --conf '{"cohort_date":"2026-11-01"}'
cat "$AIRFLOW_HOME/checkout-lab/2026-11-01.json"
```

The output should contain these `variants`:

```json
{
  "A": {"assigned": 3, "converted": 1},
  "B": {"assigned": 3, "converted": 2}
}
```

| User | Purchase behavior (UTC) | Conversion? |
|---|---|---|
| `u01` (A) | Purchases at assignment and again later | Yes, counts once |
| `u02` (A) | Purchases before assignment | No |
| `u03` (A) | Makes no purchase | No, stays in the denominator |
| `u04` (B) | Purchases at 00:15 next day, still in the window; event delivered twice | Yes, counts once |
| `u05` (B) | Purchases a second before the right endpoint | Yes |
| `u06` (B) | Purchases exactly 24 hours after assignment | No |
| `u99` | Purchases with no assignment record | Outside the experiment |

The 1/3 and 2/3 results are fixtures for checking boundaries. They should match the [PySpark lab]({{< relref "/notes/spark-pyspark" >}}), and they are too small and artificial for a decision about the reminder. To explore the UI, reactivate the same environment in another terminal, set the same variables, and start the lab server:

```bash
AIRFLOW__API__HOST=127.0.0.1 airflow standalone
```

Open `http://127.0.0.1:8080` with the local credentials described in the terminal. This starts the scheduling components; leave the Dag paused if you only want to inspect the manual test. [Debugging Dags](https://airflow.apache.org/docs/apache-airflow/3.4.0/core-concepts/debug.html)

### 3. Rerun and simulate a failure {#failure-lab}

Try two checks after a successful run:

```bash
# Rerun the same date. The sha256 should match.
shasum -a 256 "$AIRFLOW_HOME/checkout-lab/2026-11-01.json"
airflow dags test checkout_cohort --conf '{"cohort_date":"2026-11-01"}'
shasum -a 256 "$AIRFLOW_HOME/checkout-lab/2026-11-01.json"

# Move the source cutoff earlier. Expect a failed run.
airflow dags test checkout_cohort \
  --conf '{"cohort_date":"2026-11-01","simulate_incomplete":true}'
```

The second check should stop at `compute_counts` with `Purchase source is incomplete for this cohort.` The publish task never runs, so the earlier JSON stays intact. A missing observation is different from an observed non-purchase. In this fixture completeness is supplied directly. In a real pipeline, read it from the ingestion service or a committed source manifest. Seeing one recent event does not tell you whether earlier events are missing.

The example writes a temporary file in the destination directory, then replaces the target on the same local filesystem. Writing the same result with the same input is **idempotency**. If a task dies after publication but before it reports success, rerunning it does not append another copy. Object stores do not necessarily support the same rename semantics. Use a transactional table write, an upsert on a unique key or immutable versions with a publishing pointer. Late corrections should create a traceable revision, even when the business dates are unchanged. [Airflow task best practices](https://airflow.apache.org/docs/apache-airflow/3.4.0/best-practices.html)

## When to retry and when to backfill {#recovery}

| Action | Use it when | Check first |
|---|---|---|
| **Retry** | A task hit a transient connection or write error. | Keep the business key and output path. Bound the attempts and timeout. Fail directly on known permanent errors. |
| **Clear** | You fixed a step in a particular historical run and want to execute it again. | Choose whether to clear affected downstream tasks and how to repair published output. |
| **Catchup** | After enabling a schedule, the scheduler should create runs for missed historical intervals. | Confirm the historical load and that each run reads only its own interval. |
| **Backfill** | You want runs for an explicit date range, with a policy for reprocessing failed or completed dates. | Preview the logical dates, confirm source retention and check that output can be replaced. |

This Dag sets `catchup=False`, so resuming it does not create every previous interval automatically. Backfill flags take **Dag logical dates**, one day later than the output cohort dates in this example. Preview the candidates without creating any runs:

```bash
airflow backfill create --dag-id checkout_cohort \
  --from-date 2026-11-02 --to-date 2026-11-04 \
  --reprocess-behavior failed --max-active-runs 1 --dry-run
```

With no run history, the logical dates in this command correspond to cohorts for November 1, 2 and 3. Check the printed dates after testing. Existing run states and the reprocessing policy affect what Airflow will create: `none` allows missing runs, `failed` also allows failed runs, and `completed` also allows reprocessing completed runs. A queued or running latest execution prevents another backfill run for that logical date. To execute the previewed backfill, start `airflow standalone` if needed, unpause the lab Dag in the UI, and rerun the command without `--dry-run`. The lab uses the same relative event times for each synthetic date. [Backfill options and concurrency](https://airflow.apache.org/docs/apache-airflow/3.4.0/core-concepts/backfill.html)

### Find the first failed dependency {#debug}

| Symptom | Where to look |
|---|---|
| Dag is missing | Check the file path and runtime environment. Run `airflow dags list-import-errors --local`. Keep network calls and business queries out of top-level imports. |
| Scheduled run is missing | Check pause state, start date, timetable and logical date, then verify the scheduler and Dag processor are running. |
| Task stays queued | Look at the executor and workers, pool and concurrency limits, and whether a worker can consume the relevant queue. |
| Task is retrying | Find the first error in its log. Separate transient errors from deterministic failures such as schema or permissions. Debug against the same input partition. |
| Run succeeds but output looks wrong | Check the chosen dates, input version, row counts and denominators. Airflow uses the leaf tasks to determine final run status. A successful cleanup leaf with `all_done` can hide an upstream failure. |

## Turn the lab into a production workflow {#production}

Keep the time contract, then replace the fixture with a source completeness check and a Spark or warehouse job. A task that submits remote work should wait for its actual completion before the Dag validates and publishes. For long waits, use a provider's deferrable mode and a triggerer where supported. Log the external job identifier. Pass that and the partition through XCom, and keep data rows in shared storage. The Spark lab writes to a fresh output directory. An orchestrated version can stage each attempt separately and publish a pointer after validation.

A contract for the checkout pipeline could read as follows:

| Decision | Checkout example |
|---|---|
| Input and version | First valid assignment and deduplicated purchases, with an object version or table snapshot recorded. |
| Window and readiness | Coordinated Universal Time (UTC) assignment day, 24-hour user windows, and purchase source complete through cohort end plus 24 hours. |
| Business key | `experiment_id` × `cohort_date` × `variant`. Keep a separate user-level table if analysts need to audit individual flags. |
| Validation and logs | Both arms present, unique assignments, unique purchase events and `0 ≤ converted ≤ assigned`. Record source versions, window bounds, arm counts and destination. |
| Reprocessing and resources | Rerun safely for the same input; define how to replace corrected results and cap shared database load with a pool. Separate environments and use Connections and a secrets backend for credentials. |
| Downstream analysis | Check allocation and metric definitions before computing uncertainty. Publishing a new mature cohort each day does not justify a new fixed-horizon stopping test each day. |

For data-driven execution, **asset scheduling** can start a Dag when an upstream producer reports a data asset update. The consumer still needs to check which partition arrived and whether it is complete. **Dynamic task mapping** can create tasks for a list of dates or files discovered at runtime. Keep one task per meaningful unit of work rather than per event row. [Asset scheduling](https://airflow.apache.org/docs/apache-airflow/3.4.0/authoring-and-scheduling/asset-scheduling.html) · [Dynamic task mapping](https://airflow.apache.org/docs/apache-airflow/3.4.0/authoring-and-scheduling/dynamic-task-mapping.html)

## Practice: change an input and predict the run {#practice}

1. **Which cohort does the scheduled run due on November 4 at midnight UTC publish?** Its interval is November 3 through November 4, so it counts November 2 assignments. Using the run's wall clock date would select a cohort two days too late.
2. **What changes if `u06` purchases one second earlier?** The purchase moves inside the window. B becomes 3/3; A remains 1/3. Edit the toy Dag in your local Dags folder and verify.
3. **Why is reading the latest source on retry risky?** A new file can change what the task sees, making the same run harder to reproduce. Pin an interval and input version. Record any intentional revision separately.
4. **Why should a dashboard wait after Spark returns a job ID?** Submission only confirms that the job was requested. Wait for completion, validate the expected partition and check the data contract before switching readers to the output.

## Continue learning {#resources}

| When | Resource | What to try |
|---|---|---|
| First install | {{< new-tab "Apache Airflow Quick Start" "https://airflow.apache.org/docs/apache-airflow/3.4.0/start.html" >}} | Start a local environment and find the graph, task logs and run history. |
| First Dag | {{< new-tab "Airflow TaskFlow tutorial" "https://airflow.apache.org/docs/apache-airflow/3.4.0/tutorial/taskflow.html" >}} | Pass a small dictionary to a downstream task and see how the dependency is inferred. |
| Date looks off by one | {{< new-tab "Airflow Dag Runs" "https://airflow.apache.org/docs/apache-airflow/3.4.0/core-concepts/dag-run.html" >}} · {{< new-tab "Timetables" "https://airflow.apache.org/docs/apache-airflow/3.4.0/authoring-and-scheduling/timetable.html" >}} | Write the trigger time, logical date and input bounds side by side. |
| Operations practice | {{< new-tab "Airflow Backfill" "https://airflow.apache.org/docs/apache-airflow/3.4.0/core-concepts/backfill.html" >}} · {{< new-tab "Debugging Dags" "https://airflow.apache.org/docs/apache-airflow/3.4.0/core-concepts/debug.html" >}} | Preview two dates, then use `dags test` to reproduce missing input. |
| Production design | {{< new-tab "Airflow Best Practices" "https://airflow.apache.org/docs/apache-airflow/3.4.0/best-practices.html" >}} · {{< new-tab "Deferrable Operators & Triggers" "https://airflow.apache.org/docs/apache-airflow/3.4.0/authoring-and-scheduling/deferring.html" >}} | Trace a remote job ID to its committed output, then check imports, write behavior and waiting costs. |

In this series: [Causal inference]({{< relref "/notes/causal-inference" >}}) · [A/B testing]({{< relref "/notes/ab-testing" >}}) · [Spark and PySpark]({{< relref "/notes/spark-pyspark" >}}).

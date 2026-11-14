---
title: "Spark and PySpark: From Your First DataFrame to a Batch Job"
date: 2026-11-13T19:49:01-06:00
description: "Learn Spark SQL, partitions, shuffles and joins. Build and diagnose a PySpark pipeline that turns checkout events into experiment metrics."
topics: [data-science, ml-systems]
---

Suppose a checkout team wants to count how many people purchased within 24 hours of being assigned a reminder. Its logs can contain several purchases from the same user, duplicate event deliveries, and purchases that arrived after midnight. Counting event rows gives the wrong metric. The job needs to preserve the assigned population while producing exactly one outcome per user. You'll build that job in PySpark, check the rows by hand, and then use Spark's execution plan to see where the work goes.

You can start with basic Python and a little Structured Query Language (SQL). Allow about 45–60 minutes for the local lab. The same metric appears in the [A/B testing guide]({{< relref "/notes/ab-testing" >}}), which covers assignment, uncertainty and the experiment decision.

## When to use Spark {#mental-model}

Apache Spark is a compute engine. It runs a single analysis across partitions of data and can distribute those partitions across machines. **PySpark** is its Python application programming interface (API). Here we'll use the Spark SQL engine through DataFrames, which describe rows, typed columns, and operations on them. SQL and the DataFrame API feed the same optimizer. A **Resilient Distributed Dataset (RDD)** is a lower-level abstraction for a distributed collection. Start with DataFrames when the work has a tabular schema. [Spark SQL guide](https://spark.apache.org/docs/4.0.1/sql-programming-guide.html) · [RDD programming guide](https://spark.apache.org/docs/4.0.1/rdd-programming-guide.html)

| Tool | What it owns | Example |
|---|---|---|
| pandas | Tabular analysis in one Python process | Inspect an experiment export that fits comfortably in memory. |
| Apache Spark / PySpark | Parallel execution of data transformations | Join large event and assignment tables across machines. |
| Parquet, object storage, or a table format | Persistent data | Store outputs in Parquet files. A table format such as Delta Lake adds a transaction log for consistent table versions. |
| Databricks | Managed compute, workspaces and data services | Develop Spark transforms in a notebook and run them as a managed job. |
| Apache Airflow | Workflow scheduling, dependencies and retries | Wait for a source partition, submit Spark, then check the output before publishing. |

On a small dataset, first measure a simpler tool. A pandas workflow that runs in seconds has little to gain from Spark startup and scheduling overhead. The local Spark lab teaches the execution model so you can recognize when cluster execution is useful. See the [Databricks guide]({{< relref "/notes/databricks-guide" >}}) for a managed environment and the [Airflow guide]({{< relref "/notes/airflow" >}}) for orchestration.

## How Spark runs a DataFrame {#execution}

![A Python program describes a DataFrame plan. An action runs one or more jobs. A cluster manager allocates resources, the driver schedules stages, and executors run tasks on partitions.](execution.svg "Figure 1. The driver plans and coordinates the work. Executors process the partitions assigned to each task.")

- The **driver** runs the coordinating program. It builds and optimizes the plan, schedules work, and receives results requested by the program.
- In cluster mode, a **cluster manager** such as Kubernetes allocates resources. The application's **executors** are processes that run tasks and can keep intermediate data.
- An **action** such as a write or a count can trigger one or more **jobs**. Spark divides the work into **stages**, often separated by shuffles. A **task** usually processes one partition for one stage. Task retries give that partition another attempt.
- The lab uses `local[2]`, which makes two worker threads available on one machine. Use it to check semantics and inspect a plan. Cluster networking and the distribution of real data are outside this measurement. [Cluster mode and terminology](https://spark.apache.org/docs/4.0.1/cluster-overview.html)

### Transformations, actions and data types {#dataframe-basics}

`select`, `filter`, and `groupBy(...).agg(...)` return plans for transformations. `show`, `count`, and a Parquet write materialize results. Without persistence, a later action may compute the same transformations again. `explain("formatted")` prints the planned physical operators before you run the output action. [DataFrame quickstart](https://spark.apache.org/docs/4.0.1/api/python/getting_started/quickstart_df.html) · [Explain modes](https://spark.apache.org/docs/4.0.1/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.explain.html)

Set types at ingestion. A time stored as a string sorts lexicographically and may carry offsets or mixed formats. Parse it as a timestamp, check invalid inputs, and set the session time zone. Use integers in cents or an explicit decimal type for money. Check the row grain on both sides of a join, especially when several rows share the same key. Spark SQL treats ordinary comparisons with a null as unknown, and a filter keeps only rows for which the condition is true. [Spark SQL null semantics](https://spark.apache.org/docs/4.0.1/sql-ref-null-semantics.html)

## Run the checkout conversion lab {#lab}

Use **Python 3.12**, **Java Development Kit (JDK) 17**, and **PySpark 4.0.1** for the commands below. The version pin makes the lab reproducible. Consult the install guide before using a different release. Download the [Python lab](/notes/spark-pyspark/checkout-pyspark-lab.py) to its own folder. It generates all input rows, runs locally, and writes Parquet. For Windows, the shell commands below can be run from a Linux environment such as WSL with Python and Java installed there. [PySpark 4.0.1 installation](https://spark.apache.org/docs/4.0.1/api/python/getting_started/install.html)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install pyspark==4.0.1
python --version
java -version
python checkout-pyspark-lab.py --output spark-checkout-output --explain
```

On macOS with multiple Java versions installed, set `JAVA_HOME=$(/usr/libexec/java_home -v 17)` and put `$JAVA_HOME/bin` at the front of `PATH` before running the lab. If the Python driver and worker minor versions differ, check whether an old `PYSPARK_PYTHON` override is active. The script defaults local worker processes to the interpreter that started it. Choose a new output directory for each run.

For a first DataFrame, open `python` in that environment and try:

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

### Set the population and the event window {#cohort}

The script reads a simulated **2026-11-01 cohort in Coordinated Universal Time (UTC)**. Assignment comes from an existing A/B log. Every assigned user contributes one row, including users with no purchase. A purchase qualifies at `assigned_at <= purchased_at < assigned_at + 24 hours`. Exposure and opening a reminder are separate events from assignment.

| Input | Grain and columns | Required check |
|---|---|---|
| Assignments | One row per user in the cohort. `user_id`, `variant`, `assigned_at`. | Each user has one recorded A or B assignment. |
| Purchase deliveries | One delivered record per row. `event_id`, `user_id`, `purchased_at`. | Repeated deliveries with identical fields collapse to one event. Conflicting fields for the same event ID stop the job. |

The cohort ends at **Nov 2 00:00 UTC** (exclusive). The fixture declares complete purchase data for event times from cohort start through **Nov 3 00:00 UTC** (exclusive), so every user's observation window has ended within the covered period. The job checks the conservative bound `cohort_end + 24 hours`. Try passing `--complete-through 2026-11-02T23:00:00Z` with a new output path for a coverage error. In production, get the bound from a verified source manifest or ingestion contract. The maximum timestamp in a file cannot tell you whether an earlier hour is missing.

![Three UTC day boundaries frame the assignment and purchase windows. u05's purchase one second before its upper bound qualifies. u06's purchase exactly at its upper bound falls outside.](checkout-window.svg "Figure 2. Each user's window starts at assignment. The lab waits for purchase coverage through 24 hours after the cohort end.")

### Deduplicate, join and restore user grain {#join}

For the toy data, `purchases.dropDuplicates()` removes identical rows, and a grouped check catches remaining duplicate event IDs. With a real revision feed, define how to choose a current record using a version from the source. `dropDuplicates(["event_id"])` can discard conflicting data without implementing that version rule.

The script aliases the assignment DataFrame as `a` and the purchase DataFrame as `p`. The full condition belongs in the **left join**. Filtering on the right timestamp after the join would drop rows with null purchase fields and could also lose users whose only matched purchases are outside the window. `count("p.event_id")` ignores the null left by the join. `count("*")` would count that row. [PySpark join reference](https://spark.apache.org/docs/4.0.1/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.join.html) · [Spark SQL nulls in aggregates](https://spark.apache.org/docs/4.0.1/sql-ref-null-semantics.html#builtin-aggregate-expressions)

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

A join produces one row for each matching pair. Two assignment rows and three purchase rows for the same key could create six pairs. Check uniqueness where the data contract promises uniqueness. The left join here intentionally allows multiple purchases, then aggregation collapses them to one user outcome. The experiment denominator is the number of assigned users, and the numerator is their sum of binary outcomes.

| User | Group | Events in window | Converted | What it checks |
|---|---:|---:|---:|---|
| u01 | A | 2 | 1 | Purchase at assignment counts. Two purchases still mean one converter. |
| u02 | A | 0 | 0 | Purchase one second before assignment falls outside. |
| u03 | A | 0 | 0 | A user without a purchase stays in the denominator. |
| u04 | B | 1 | 1 | A valid event after midnight delivered twice counts once. |
| u05 | B | 1 | 1 | A purchase one second before the upper bound counts. |
| u06 | B | 0 | 0 | A purchase exactly at the upper bound falls outside. |

The script also includes a purchase from an unassigned user, `u99`. No outcome row is produced for that user. The output directories are `user_outcomes` and `variant_summary` under the path passed to `--output`. Each directory represents a Parquet dataset and may contain more than one part file. Sort by key when comparing results across runs. A has **1/3** and B has **2/3**. Those six rows check the bookkeeping. They are separate from the fictional experiment readout in the [A/B testing guide]({{< relref "/notes/ab-testing#readout" >}}). Historical assignment rules and confounding are covered in the [causal inference guide]({{< relref "/notes/causal-inference" >}}).

### Read one of the Parquet outputs {#read-output}

Open a new Python session in the lab directory. Replace the path if you used a different `--output` value. This query computes the displayed group rates again using Spark SQL.

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

Compare the query with the `variant_summary` dataset and the DataFrame aggregation in the script. `createOrReplaceTempView` registers the DataFrame for SQL in this session. The view disappears when the session ends.

## Follow a shuffle in the plan {#shuffle}

A **partition** is a slice of data that a task can process. When you select a column or filter its values, each partition can usually do its own work. These are typical **narrow** operations. Grouping by a key may need rows from many input partitions. A **shuffle** redistributes records or partial aggregates so matching keys reach the same destination partition. Network transfer, serialization, sorting and disk spill can make it expensive. [Shuffle operations](https://spark.apache.org/docs/4.0.1/rdd-programming-guide.html#shuffle-operations)

![Two input partitions compute partial A/B counts. A shuffle sends all partial A results to one destination and all B results to another, giving A 1/3 and B 2/3.](shuffle.svg "Figure 3. For the variant count, Spark can aggregate inside each input partition and exchange partial counts by key. The drawing separates A and B for clarity. A real hash partition can hold multiple keys.")

In the lab's formatted plan, find `SortMergeJoin LeftOuter`, an `Exchange` for repartitioning by `user_id` before the join, and another `Exchange` for grouping by `variant`. `HashAggregate` can show up in both partial and final aggregation. `AdaptiveSparkPlan` indicates that **Adaptive Query Execution (AQE)** can revise some choices as runtime statistics arrive. The plan displayed before an action is the starting point. Check the SQL view for the executed plan. [Spark SQL performance tuning](https://spark.apache.org/docs/4.0.1/sql-performance-tuning.html) · [Spark web interface](https://spark.apache.org/docs/4.0.1/web-ui.html#sql-tab)

### Diagnose first, then tune {#diagnosis}

| What you see | What to check and try |
|---|---|
| High input bytes or too many file opens | Read the needed dates and columns. In a Parquet scan inspect `PartitionFilters`, `PushedFilters`, and `ReadSchema`. Use a storage partition column that matches frequent filters and has manageable cardinality, often an event date. |
| Join takes most of the time | Check key multiplicity and input sizes. For a large fact table and small dimension, Spark can broadcast the small side for a supported join. Check the build-side size and available memory before adding a broadcast hint. |
| One or two tasks run much longer | Compare shuffle read and duration across tasks in the Stages tab. A few unusually frequent keys can put disproportionate work on one task. Check key counts and the executed plan for skew handling. Consider preaggregation or isolating a hot key when the metric permits it. |
| Many tiny tasks or part files | Check Spark partition counts and the storage layout separately. Output partitioning by high-cardinality identifiers can create many small files. Compact or repartition to fit downstream reads. Use `coalesce` to reduce task partitions when it preserves a usable distribution. |
| Python execution dominates | Prefer built-in column functions, such as `when`, timestamp functions and aggregations. If a Python user-defined function (UDF) is required, profile its cost and consider a vectorized pandas UDF using Arrow for compatible operations. |
| Driver runs out of memory | Look for `collect()` or `toPandas()`. Keep detailed rows distributed and return a bounded sample or a small summary. |

`repartition(n, "user_id")` can shuffle the rows into new runtime partitions. `write.partitionBy("event_date")` organizes Parquet output into directories that readers can prune. A file, a directory partition, and a runtime partition are different units. One directory can contain many files, and Spark can group small files or split larger inputs across tasks. Cache only when a DataFrame is reused and recomputation is expensive enough to justify it. Unpersist it when done. [Parquet partition discovery](https://spark.apache.org/docs/4.0.1/sql-data-sources-parquet.html#partition-discovery) · [SQL tuning reference](https://spark.apache.org/docs/4.0.1/sql-performance-tuning.html) · [Arrow and Python UDFs](https://spark.apache.org/docs/4.0.1/api/python/tutorial/sql/arrow_pandas.html)

The driver's local web interface is usually at `http://localhost:4040` while Spark is running. In the SQL tab, find the slow query, then inspect the associated stages for task durations, shuffle read and write, and spills. The lab script exits after writing and printing the results. Use an interactive session to keep the UI open, or enable Spark event logs and use the History Server to investigate after exit. [Monitoring and event logs](https://spark.apache.org/docs/4.0.1/monitoring.html)

## Move from a fixture to a recurring job {#production}

1. Define an input contract. Record the assignment key and first eligible assignment rule across days, the event identity and version rule, the time zone, and the coverage of each input snapshot. Build the output from the stored assignment groups.
2. Test boundaries and bad data before increasing scale. Use exact expected rows for no purchase, an inclusive start, an exclusive end, duplicate delivery and a conflicting event ID. Check that both group denominators reconcile to the assignment table and that `converted_users <= assigned_users`. [PySpark testing utilities](https://spark.apache.org/docs/4.0.1/api/python/getting_started/testing_pyspark.html)
3. Make replay predictable. Read a declared cohort interval and source version. Stage output, validate it, then publish a single cohort partition using the destination's commit procedure. An unconditional append would duplicate the cohort after retry. Plain Parquet files provide storage, so define the publication and concurrency semantics explicitly.
4. Submit the application to the cluster and record its application ID with the output. Prepare compatible Python and Spark dependencies for the driver and executors. Review input scan volume, shuffle size and total runtime on a representative cohort before changing resources. [Submitting applications](https://spark.apache.org/docs/4.0.1/submitting-applications.html)

### Batch versus Structured Streaming {#streaming}

Start with batch if the decision only needs a complete daily cohort. **Structured Streaming** applies DataFrame operations incrementally as new data arrives. In stateful streaming, a **watermark** expresses how late event-time data may arrive for the purpose of bounded state. A **checkpoint** lets a query resume from recorded progress and state. Very late records can miss a result, and sink behavior determines whether a retried output can appear twice. A streaming watermark does not attest that every source event has arrived for experiment reporting. Set an explicit late-event or correction policy and verify parity against batch for a closed interval. [Structured Streaming overview](https://spark.apache.org/docs/4.0.1/streaming/index.html) · [Watermarks and checkpointing](https://spark.apache.org/docs/4.0.1/streaming/apis-on-dataframes-and-datasets.html)

## Exercises with answers {#exercises}

**1. Which denominator survives?** Change the join to `inner` and run with a new output path. Which users survive, and how do the observed rates change?

> **Answer.** u01, u04 and u05 survive. A becomes 1/1 and B becomes 2/2 because the job dropped users without a matched purchase. Change back to the left join and compare the per-user outcomes as well as the group summary.

**2. Which duplicate should raise an error?** Add a second `p04` row with purchase time 00:16 UTC. What does the lab do?

> **Answer.** It stops. Full-row deduplication leaves two records for `p04` with different timestamps. A real correction flow would need source versioning or a rule to quarantine the conflict.

**3. Can a broadcast remove every shuffle?** Add a temporary broadcast hint on the tiny purchase side. Compare the plans and output.

```python
p = F.broadcast(unique_purchases).alias("p")
```

> **Answer.** A `BroadcastHashJoin` can replace the sort-merge join for this equijoin plus time condition. The grouping and duplicate checks can still redistribute rows. Match output rows before using plan changes to judge performance.

**4. Why did a rerun add rows?** A production job appends another set of variant summaries after retrying the same cohort interval. How would you prevent an automated report from double counting?

> **Answer.** Define a unique output key such as `(experiment_id, cohort_day, variant)`. Validate staged results and publish once for that interval using a table transaction or a publication protocol with a unique run manifest. For a corrected replay, specify which snapshot supersedes the prior one.

## Reading and practice {#resources}

| Resource | What to do with it |
|---|---|
| {{< new-tab "PySpark DataFrame quickstart" "https://spark.apache.org/docs/4.0.1/api/python/getting_started/quickstart_df.html" >}} | Run the schema, select and aggregation cells. Translate one DataFrame query into SQL over a temporary view. |
| {{< new-tab "Spark cluster mode" "https://spark.apache.org/docs/4.0.1/cluster-overview.html" >}} | Trace driver, cluster manager and executors. Locate a job, stage and task in the corresponding Spark UI. |
| {{< new-tab "Spark SQL performance tuning" "https://spark.apache.org/docs/4.0.1/sql-performance-tuning.html" >}} | Read the join strategy and adaptive execution sections, then try the lab's broadcast exercise. |
| {{< new-tab "PySpark testing guide" "https://spark.apache.org/docs/4.0.1/api/python/getting_started/testing_pyspark.html" >}} | Write an expected DataFrame for the six users and compare it with the computed result. |
| {{< new-tab "Spark SQL and DataFrame paper" "https://people.csail.mit.edu/matei/papers/2015/sigmod_spark_sql.pdf" >}} | Read the sections on DataFrames and Catalyst for the design behind the API and optimizer. |
| {{< new-tab "Structured Streaming API guide" "https://spark.apache.org/docs/4.0.1/streaming/apis-on-dataframes-and-datasets.html" >}} | Read the event-time and recovery sections before implementing a stateful aggregation. |

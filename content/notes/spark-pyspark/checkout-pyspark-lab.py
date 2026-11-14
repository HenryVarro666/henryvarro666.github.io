#!/usr/bin/env python3
"""Build one 24-hour conversion outcome per assigned user from synthetic data."""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StringType, StructField, StructType, TimestampType


def utc(value):
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() != timedelta(0):
        raise ValueError("Use an ISO 8601 timestamp with Z or +00:00.")
    return parsed.astimezone(timezone.utc)


COHORT_START = utc("2026-11-01T00:00:00Z")
COHORT_END = utc("2026-11-02T00:00:00Z")
FIXTURE_COMPLETE_THROUGH = utc("2026-11-03T00:00:00Z")

ASSIGNMENT_SCHEMA = StructType([
    StructField("user_id", StringType(), nullable=False),
    StructField("variant", StringType(), nullable=False),
    StructField("assigned_at", TimestampType(), nullable=False),
])
PURCHASE_SCHEMA = StructType([
    StructField("event_id", StringType(), nullable=False),
    StructField("user_id", StringType(), nullable=False),
    StructField("purchased_at", TimestampType(), nullable=False),
])


def fixture(spark):
    assignments = spark.createDataFrame([
        ("u01", "A", utc("2026-11-01T00:00:00Z")),
        ("u02", "A", utc("2026-11-01T10:00:00Z")),
        ("u03", "A", utc("2026-11-01T18:00:00Z")),
        ("u04", "B", utc("2026-11-01T09:00:00Z")),
        ("u05", "B", utc("2026-11-01T12:00:00Z")),
        ("u06", "B", utc("2026-11-01T23:00:00Z")),
    ], ASSIGNMENT_SCHEMA)
    purchases = spark.createDataFrame([
        ("p01", "u01", utc("2026-11-01T00:00:00Z")),  # inclusive lower edge
        ("p02", "u01", utc("2026-11-01T14:00:00Z")),  # same converter, second purchase
        ("p03", "u02", utc("2026-11-01T09:59:59Z")),  # before assignment
        ("p04", "u04", utc("2026-11-02T00:15:00Z")),
        ("p04", "u04", utc("2026-11-02T00:15:00Z")),  # exact duplicate delivery
        ("p05", "u05", utc("2026-11-02T11:59:59Z")),  # just inside the window
        ("p06", "u06", utc("2026-11-02T23:00:00Z")),  # exclusive upper edge
        ("p99", "u99", utc("2026-11-01T12:00:00Z")),  # never assigned
    ], PURCHASE_SCHEMA)
    return assignments, purchases


def user_outcomes(assignments, purchases):
    cohort = assignments.where(
        (F.col("assigned_at") >= F.lit(COHORT_START))
        & (F.col("assigned_at") < F.lit(COHORT_END))
    )
    # These are actions. Fail on inconsistent source keys before counting.
    if cohort.groupBy("user_id").count().where("count > 1").take(1):
        raise ValueError("Expected one assignment per user in the cohort.")
    if cohort.where(~F.col("variant").isin("A", "B")).take(1):
        raise ValueError("Expected a recorded A or B assignment.")

    unique_purchases = purchases.dropDuplicates()
    if unique_purchases.groupBy("event_id").count().where("count > 1").take(1):
        raise ValueError("The same purchase event ID has conflicting contents.")

    a = cohort.alias("a")
    p = unique_purchases.alias("p")
    in_window = (
        (F.col("a.user_id") == F.col("p.user_id"))
        & (F.col("p.purchased_at") >= F.col("a.assigned_at"))
        & (F.col("p.purchased_at") < F.col("a.assigned_at") + F.expr("INTERVAL 24 HOURS"))
    )
    return (
        a.join(p, in_window, "left")
        .groupBy("a.user_id", "a.variant", "a.assigned_at")
        .agg(F.count("p.event_id").alias("purchase_events"))
        .withColumn("converted", (F.col("purchase_events") > 0).cast("int"))
    )


def variant_summary(outcomes):
    return (
        outcomes.groupBy("variant")
        .agg(
            F.count("*").alias("assigned_users"),
            F.sum("converted").alias("converted_users"),
        )
        .withColumn("conversion_rate", F.col("converted_users") / F.col("assigned_users"))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("spark-checkout-output"))
    parser.add_argument("--complete-through", type=utc, default=FIXTURE_COMPLETE_THROUGH)
    parser.add_argument("--explain", action="store_true", help="Print the physical plan.")
    args = parser.parse_args()

    # The fixture assumes the purchase source covers the cohort start onward.
    # The supplied time is an exclusive source coverage bound, not the run time.
    if not COHORT_END + timedelta(days=1) <= args.complete_through <= FIXTURE_COMPLETE_THROUGH:
        parser.error("Purchase coverage must reach the cohort end plus 24h and stay within the fixture.")
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Choose a new output directory. Already exists: {output}")

    # In local mode use the driver environment for Python worker processes.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    spark = (
        SparkSession.builder
        .appName("checkout-reminder-lab")
        .master("local[2]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        assignments, purchases = fixture(spark)
        outcomes = user_outcomes(assignments, purchases)
        summary = variant_summary(outcomes)
        if args.explain:
            summary.explain("formatted")

        output.mkdir(parents=True, exist_ok=False)
        outcomes.write.mode("errorifexists").parquet(str(output / "user_outcomes"))
        summary.write.mode("errorifexists").parquet(str(output / "variant_summary"))
        outcomes.orderBy("user_id").show()
        summary.orderBy("variant").show(truncate=False)
        print(f"Wrote results to {output}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

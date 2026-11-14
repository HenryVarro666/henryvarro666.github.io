"""Airflow 3.4 tutorial: materialize a complete checkout cohort as one JSON file."""

from datetime import UTC, date, datetime, time, timedelta
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import pendulum

from airflow.configuration import conf
from airflow.sdk import CronDataIntervalTimetable, Param, dag, get_current_context, task
from airflow.sdk.exceptions import AirflowFailException


DAY = timedelta(days=1)


def utc_midnight(day):
    return datetime.combine(day, time.min, tzinfo=UTC)


def calculate(cohort_start):
    """Synthetic source for any UTC cohort day; Spark uses the same six users."""
    def at(hour, minute=0, second=0):
        return cohort_start + timedelta(hours=hour, minutes=minute, seconds=second)

    assignments = {
        "u01": ("A", at(0)), "u02": ("A", at(10)), "u03": ("A", at(18)),
        "u04": ("B", at(9)), "u05": ("B", at(12)), "u06": ("B", at(23)),
    }
    purchases = [
        ("p01", "u01", at(0)),             # inclusive lower bound
        ("p02", "u01", at(14)),            # second purchase, still one conversion
        ("p03", "u02", at(9, 59, 59)),     # before assignment
        ("p04", "u04", at(24, 15)),        # next day, inside 24 hours
        ("p04", "u04", at(24, 15)),        # identical delivery
        ("p05", "u05", at(35, 59, 59)),   # one second before the upper bound
        ("p06", "u06", at(47)),            # exact upper bound, excluded
        ("p99", "u99", at(12)),             # unassigned user
    ]
    events = {}
    for event_id, user_id, purchased_at in purchases:
        value = (user_id, purchased_at)
        if events.setdefault(event_id, value) != value:
            raise AirflowFailException(f"Conflicting purchase event: {event_id}")

    converted = {
        user_id for user_id, purchased_at in events.values()
        if user_id in assignments
        and assignments[user_id][1] <= purchased_at < assignments[user_id][1] + DAY
    }
    return {
        variant: {
            "assigned": sum(v == variant for v, _ in assignments.values()),
            "converted": sum(assignments[u][0] == variant for u in converted),
        }
        for variant in ("A", "B")
    }


@dag(
    dag_id="checkout_cohort",
    start_date=pendulum.datetime(2026, 11, 2, tz="UTC"),
    schedule=CronDataIntervalTimetable("0 0 * * *", timezone="UTC"),
    catchup=False,
    max_active_runs=1,
    params={
        "cohort_date": Param(None, type=["null", "string"], format="date"),
        "simulate_incomplete": Param(False, type="boolean"),
    },
    tags=["tutorial", "experiments"],
)
def checkout_cohort():
    @task
    def plan_window():
        ctx = get_current_context()
        params = ctx["params"]
        if ctx["dag_run"].run_type == "manual":
            if not params["cohort_date"]:
                raise AirflowFailException("Manual runs require cohort_date (YYYY-MM-DD).")
            cohort_start = utc_midnight(date.fromisoformat(params["cohort_date"]))
            # Only the synthetic fixture can declare its own source completeness.
            complete_through = cohort_start + 2 * DAY
        else:
            interval_start = ctx["data_interval_start"]
            interval_end = ctx["data_interval_end"]
            if (
                interval_start is None
                or interval_end is None
                or interval_start != utc_midnight(interval_start.date())
                or interval_end - interval_start != DAY
            ):
                raise AirflowFailException("Expected a one-day UTC data interval.")
            cohort_start = interval_start - DAY
            complete_through = interval_end
        if params["simulate_incomplete"]:
            complete_through = cohort_start + DAY
        return {
            "cohort_start": cohort_start.isoformat(),
            "cohort_end": (cohort_start + DAY).isoformat(),
            "complete_through": complete_through.isoformat(),
        }

    @task
    def compute_counts(window):
        cohort_start = datetime.fromisoformat(window["cohort_start"])
        cohort_end = datetime.fromisoformat(window["cohort_end"])
        complete_through = datetime.fromisoformat(window["complete_through"])
        if complete_through < cohort_end + DAY:
            raise AirflowFailException("Purchase source is incomplete for this cohort.")
        return calculate(cohort_start)

    @task(retries=2, retry_delay=timedelta(seconds=20), execution_timeout=timedelta(minutes=2))
    def publish(window, counts):
        if any(not 0 <= row["converted"] <= row["assigned"] for row in counts.values()):
            raise AirflowFailException("Invalid counts.")
        cohort_date = datetime.fromisoformat(window["cohort_start"]).date().isoformat()
        root = Path(conf.get("core", "dags_folder")).parent / "checkout-lab"
        root.mkdir(parents=True, exist_ok=True)
        target = root / f"{cohort_date}.json"
        payload = {"cohort_date": cohort_date, **window, "variants": counts}
        data = json.dumps(payload, indent=2, sort_keys=True) + "\n"

        # Stage and replace on the same local filesystem. Never append on retry.
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, delete=False) as stage:
            staged = Path(stage.name)
            try:
                stage.write(data)
                stage.close()
                staged.replace(target)
            finally:
                staged.unlink(missing_ok=True)
        print(f"Published {target}")

    window = plan_window()
    counts = compute_counts(window)
    publish(window, counts)


checkout_cohort()

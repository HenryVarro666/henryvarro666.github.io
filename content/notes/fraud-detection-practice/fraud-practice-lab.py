"""Fraud modeling practice lab.

Runs the coding exercises in the note "Fraud detection modeling: practice questions with
worked answers" and prints every number the note quotes.

    python3 fraud-practice-lab.py

Needs Python 3.10 or later, NumPy and pandas. If scikit-learn or DuckDB is installed, the
AUC and the SQL queries are cross-checked against them. There are no data files: every input
is generated here with a fixed seed.
"""

import math
from collections import defaultdict

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# Exercise 1. Point-in-time velocity features
# --------------------------------------------------------------------------------------
def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """df columns: account_id, ts (datetime64), amount.

    Every feature uses only that account's EARLIER transactions.
    """
    df = df.sort_values(["account_id", "ts"]).reset_index(drop=True)
    g = df.set_index("ts").groupby("account_id")["amount"]
    # closed="left" -> the window is [t - 24h, t): the current row is excluded
    df["cnt_24h"] = g.rolling("24h", closed="left").count().fillna(0).to_numpy()
    df["sum_24h"] = g.rolling("24h", closed="left").sum().fillna(0).to_numpy()
    df["secs_since_prev"] = df.groupby("account_id")["ts"].diff().dt.total_seconds()
    prior_mean = df.groupby("account_id")["amount"].transform(
        lambda s: s.shift().expanding().mean()
    )
    df["amt_to_prior_mean"] = df["amount"] / prior_mean
    return df


VELOCITY_SQL = """
SELECT account_id, ts, amount,
       COUNT(*) OVER w AS cnt_24h,
       COALESCE(SUM(amount) OVER w, 0) AS sum_24h
FROM transactions
WINDOW w AS (
  PARTITION BY account_id
  ORDER BY ts
  RANGE BETWEEN INTERVAL 24 HOUR PRECEDING
            AND INTERVAL 1 MICROSECOND PRECEDING
)
ORDER BY account_id, ts
"""


# --------------------------------------------------------------------------------------
# Exercise 2. Metrics at a fixed alert budget
# --------------------------------------------------------------------------------------
def fraud_metrics(y, score, amount, alert_rate=0.01):
    """Flag the top `alert_rate` share of transactions by score."""
    y, score, amount = map(np.asarray, (y, score, amount))
    k = max(1, int(np.ceil(alert_rate * len(y))))
    flagged = np.zeros(len(y), dtype=bool)
    flagged[np.argsort(-score, kind="stable")[:k]] = True
    tp = int((flagged & (y == 1)).sum())
    fp = k - tp
    return {
        "precision": tp / k,
        "recall": tp / int((y == 1).sum()),
        "value_detection_rate": float(amount[flagged & (y == 1)].sum() / amount[y == 1].sum()),
        "false_positive_ratio": fp / max(tp, 1),
    }


# --------------------------------------------------------------------------------------
# Exercise 3. ROC-AUC from ranks
# --------------------------------------------------------------------------------------
def roc_auc(y, score):
    """Mann-Whitney form: P(a random fraud outranks a random non-fraud), ties count half."""
    y = np.asarray(y)
    ranks = pd.Series(score).rank(method="average").to_numpy()
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# --------------------------------------------------------------------------------------
# Exercise 4. Population stability index
# --------------------------------------------------------------------------------------
def psi(expected, actual, bins=10, eps=1e-6):
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


# --------------------------------------------------------------------------------------
# Exercise 5. Undo negative down-sampling
# --------------------------------------------------------------------------------------
def correct_for_downsampling(p_sampled, beta):
    """beta = share of negatives kept in training. Returns the probability on the real population."""
    p_sampled = np.asarray(p_sampled, dtype=float)
    return beta * p_sampled / (beta * p_sampled + 1 - p_sampled)


# --------------------------------------------------------------------------------------
# Exercise 6. Near-duplicate transactions
# --------------------------------------------------------------------------------------
def flag_near_duplicates(df: pd.DataFrame, seconds: int = 60) -> pd.DataFrame:
    """Flag a transaction that repeats the previous one (same account, merchant and amount)
    within `seconds`. The first of a run is not flagged."""
    keys = ["account_id", "merchant_id", "amount"]
    df = df.sort_values(keys + ["ts"]).reset_index(drop=True)
    gap = df.groupby(keys)["ts"].diff().dt.total_seconds()
    df["near_duplicate"] = gap.le(seconds)  # the first row of each group has no gap -> False
    return df


DUPLICATES_SQL = """
SELECT account_id, merchant_id, amount, ts
FROM (
  SELECT t.*,
         LAG(ts) OVER (PARTITION BY account_id, merchant_id, amount ORDER BY ts) AS prev_ts
  FROM transactions AS t
) AS x
WHERE prev_ts IS NOT NULL
  AND ts - prev_ts <= INTERVAL 60 SECOND
ORDER BY account_id, merchant_id, amount, ts
"""


# --------------------------------------------------------------------------------------
# Exercise 7. Amounts that are unusual for the account
# --------------------------------------------------------------------------------------
def robust_z(s: pd.Series, min_history: int = 5, mad_floor: float = 0.05) -> pd.Series:
    """Modified z-score of each value against the EARLIER values of the same series."""
    med = s.expanding(min_history).median().shift()          # shift: the current row is excluded
    mad = s.expanding(min_history).apply(
        lambda w: np.median(np.abs(w - np.median(w))), raw=True).shift()
    return 0.6745 * (s - med) / mad.clip(lower=mad_floor)


def flag_unusual_amounts(df: pd.DataFrame, threshold: float = 3.5,
                         min_history: int = 5) -> pd.DataFrame:
    df = df.sort_values(["account_id", "ts"]).reset_index(drop=True)
    log_amount = np.log1p(df["amount"])                      # amounts are heavily right-skewed
    df["robust_z"] = log_amount.groupby(df["account_id"]).transform(
        robust_z, min_history=min_history)
    df["unusual"] = df["robust_z"].abs() > threshold
    return df


# --------------------------------------------------------------------------------------
# Exercise 8. Warm-ups
# --------------------------------------------------------------------------------------
def two_sum(nums, target):
    """Indices of two numbers that add up to target, or None."""
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return seen[target - x], i
        seen[x] = i
    return None


def group_anagrams(words):
    """Group words that use the same letters."""
    groups = defaultdict(list)
    for w in words:
        groups["".join(sorted(w))].append(w)
    return list(groups.values())


def max_subarray(nums):
    """Largest sum of a contiguous run (Kadane)."""
    best = cur = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best


def max_profit(prices):
    """Best total profit when buying and selling as often as you like, one unit at a time."""
    return sum(max(b - a, 0) for a, b in zip(prices, prices[1:]))


def min_abs_difference(nums):
    """Smallest gap between any two values."""
    s = sorted(nums)
    return min(b - a for a, b in zip(s, s[1:]))


def longest_unique_substring(s):
    """Length of the longest substring without a repeated character (sliding window)."""
    start, best, last = 0, 0, {}
    for i, ch in enumerate(s):
        if ch in last and last[ch] >= start:
            start = last[ch] + 1
        last[ch] = i
        best = max(best, i - start + 1)
    return best


# --------------------------------------------------------------------------------------
# Checks and printed numbers
# --------------------------------------------------------------------------------------
def heading(text):
    print(f"\n== {text} ==")


def drills():
    heading("Drill 1. Precision at a low base rate")
    prev, recall, fpr = 0.001, 0.90, 0.01
    n = 1_000_000
    frauds = n * prev
    caught = frauds * recall
    false_alerts = n * (1 - prev) * fpr
    print(f"per 1,000,000 transactions: {frauds:,.0f} frauds, {caught:,.0f} caught, "
          f"{false_alerts:,.0f} false alerts")
    print(f"precision = {caught / (caught + false_alerts):.3f}; "
          f"false alerts per fraud caught = {false_alerts / caught:.1f}")
    fpr2 = 0.001
    print(f"with FPR 0.1%: precision = "
          f"{prev * recall / (prev * recall + (1 - prev) * fpr2):.3f}")
    print(f"accuracy of 'everything is legitimate' = {1 - prev:.3f}")

    heading("Drill 2. Interval for recall")
    r, n_fraud = 0.80, 400
    se = math.sqrt(r * (1 - r) / n_fraud)
    print(f"SE = {se:.3f}; 95% interval = {r - 1.96 * se:.3f} to {r + 1.96 * se:.3f}")
    print(f"frauds needed for +/- 2 points = {math.ceil(1.96 ** 2 * r * (1 - r) / 0.02 ** 2):,}")

    heading("Drill 3. Champion against challenger")
    p1, p2, n1, n2 = 0.10, 0.12, 5000, 5000
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    z = (p2 - p1) / math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    print(f"z = {z:.2f}; two-sided p = {math.erfc(abs(z) / math.sqrt(2)):.4f}")

    heading("Drill 4. Threshold from costs")
    cost_fp = 5
    for loss in (50, 500, 5000):
        print(f"amount ${loss:>5,}, false-positive cost ${cost_fp}: "
              f"p* = {cost_fp / (loss + cost_fp):.4f}")

    heading("Drill 5. Undo down-sampling")
    print(f"beta = 0.01, sampled score 0.5 -> {float(correct_for_downsampling(0.5, 0.01)):.4f}")
    print(f"beta = 0.01, sampled score 0.9 -> {float(correct_for_downsampling(0.9, 0.01)):.4f}")
    base, beta = 0.002, 0.02
    sampled_rate = base / (base + beta * (1 - base))
    print(f"base rate {base:.3%}, keep {beta:.0%} of negatives: rate in the sample = "
          f"{sampled_rate:.3%}; corrected back = "
          f"{float(correct_for_downsampling(sampled_rate, beta)):.3%}")

    heading("Drill 6. Transactions until a fraud is likely")
    print(f"n for a 95% chance of at least one fraud = "
          f"{math.ceil(math.log(0.05) / math.log(0.999)):,}")
    print(f"chance of at least one fraud in 1,000 transactions = {1 - 0.999 ** 1000:.3f}")

    heading("Class weight")
    print(f"scale_pos_weight at a 0.1% fraud rate = {(1 - prev) / prev:.0f}")

    heading("System design. An illustrative latency budget")
    budget = {"network and parsing": 10, "feature lookups": 30, "model scoring": 10,
              "rules and decision": 10, "headroom for the tail": 40}
    print(", ".join(f"{k} {v} ms" for k, v in budget.items()), f"-> total {sum(budget.values())} ms")

    heading("Case 3. Where a spike sits")
    usual = {"all": 50_000, "merchant X under $2": 300}
    now = {"all": 70_000, "merchant X under $2": 19_300}
    rise_all = now["all"] - usual["all"]
    rise_seg = now["merchant X under $2"] - usual["merchant X under $2"]
    print(f"total rise {rise_all:,} ({rise_all / usual['all']:.0%}); in the one segment {rise_seg:,}; "
          f"everything else {rise_all - rise_seg:,} "
          f"({(rise_all - rise_seg) / (usual['all'] - usual['merchant X under $2']):.1%})")
    print(f"share of the rise in the one segment = {rise_seg / rise_all:.0%}")


def exercises():
    heading("Exercise 1. Velocity features on six transactions")
    df = pd.DataFrame({
        "account_id": ["A", "A", "A", "A", "B", "B"],
        "ts": pd.to_datetime([
            "2026-12-01 09:00", "2026-12-01 09:30", "2026-12-02 08:59", "2026-12-02 09:31",
            "2026-12-01 10:00", "2026-12-03 10:00"]),
        "amount": [20.0, 30.0, 500.0, 40.0, 100.0, 100.0],
    })
    out = add_velocity_features(df)
    print(out.round({"amt_to_prior_mean": 2}).to_string(index=False))
    for _, row in out.iterrows():  # brute force: recount each window by hand
        window = ((out.account_id == row.account_id) & (out.ts < row.ts)
                  & (out.ts >= row.ts - pd.Timedelta(hours=24)))
        assert row.cnt_24h == window.sum()
        assert math.isclose(row.sum_24h, out.amount[window].sum())
    print("brute-force recount: OK")
    try:
        import duckdb
        transactions = df  # noqa: F841  (DuckDB reads the local DataFrame by name)
        sql = duckdb.sql(VELOCITY_SQL).df()
        assert (sql.cnt_24h.to_numpy() == out.cnt_24h.to_numpy()).all()
        assert np.allclose(sql.sum_24h.to_numpy(), out.sum_24h.to_numpy())
        print("SQL version (DuckDB): matches pandas")
    except ImportError:
        print("SQL version: DuckDB not installed, skipped")

    heading("Exercise 2. Metrics at a fixed alert budget")
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    score = np.array([.9, .8, .1, .7, .2, .3, .1, .05, .6, .4])
    amount = np.array([500, 20, 30, 100, 10, 10, 10, 10, 10, 10.])
    toy = fraud_metrics(y, score, amount, alert_rate=0.2)
    print("ten transactions, top 20% flagged:", {k: round(v, 2) for k, v in toy.items()})

    rng = np.random.default_rng(0)
    n = 100_000
    y_sim = (rng.random(n) < 0.002).astype(int)
    score_sim = rng.normal(0, 1, n) + 2.5 * y_sim
    amount_sim = rng.lognormal(4, 1, n)
    sim = fraud_metrics(y_sim, score_sim, amount_sim, alert_rate=0.01)
    print(f"simulated 100,000 transactions with {int(y_sim.sum())} frauds, top 1% flagged:",
          {k: round(v, 3) for k, v in sim.items()})

    heading("Exercise 3. ROC-AUC from ranks")
    print(f"toy list: {roc_auc(y, score):.4f} (15 of 16 pairs ordered correctly = {15 / 16})")
    auc = roc_auc(y_sim, score_sim)
    print(f"simulated data: {auc:.6f}")
    try:
        from sklearn.metrics import roc_auc_score
        assert math.isclose(auc, roc_auc_score(y_sim, score_sim), rel_tol=1e-12)
        ties_y, ties_s = np.array([0, 0, 1, 1, 0, 1]), np.array([.1, .4, .4, .8, .2, .4])
        assert math.isclose(roc_auc(ties_y, ties_s), roc_auc_score(ties_y, ties_s))
        print("scikit-learn agrees, including on tied scores")
    except ImportError:
        print("scikit-learn not installed, cross-check skipped")

    heading("Exercise 4. PSI against a mean shift")
    base = rng.normal(0, 1, 50_000)
    for shift in (0.0, 0.25, 0.5, 1.0):
        print(f"shift {shift:.2f} standard deviations: PSI = "
              f"{psi(base, rng.normal(shift, 1, 50_000)):.2f}")
    print(f"two bins, 50/50 moving to 60/40: PSI = "
          f"{(0.6 - 0.5) * math.log(0.6 / 0.5) + (0.4 - 0.5) * math.log(0.4 / 0.5):.4f}")

    heading("Exercise 6. Near-duplicate transactions")
    dup = pd.DataFrame({
        "account_id": ["A", "A", "A", "A", "A", "B", "A"],
        "merchant_id": ["M1", "M1", "M1", "M1", "M2", "M1", "M1"],
        "amount": [25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 40.0],
        "ts": pd.to_datetime([
            "2026-12-01 10:00:00", "2026-12-01 10:00:20", "2026-12-01 10:00:50",
            "2026-12-01 10:05:00", "2026-12-01 10:00:10", "2026-12-01 10:00:05",
            "2026-12-01 10:00:30"]),
    })
    flagged = flag_near_duplicates(dup)
    print(flagged.assign(ts=flagged.ts.dt.strftime("%H:%M:%S")).to_string(index=False))
    print(f"flagged {int(flagged.near_duplicate.sum())} of {len(flagged)} rows")
    try:
        import duckdb
        transactions = dup  # noqa: F841  (DuckDB reads the local DataFrame by name)
        sql = duckdb.sql(DUPLICATES_SQL).df()
        expected = flagged.loc[flagged.near_duplicate, ["account_id", "merchant_id", "amount", "ts"]]
        assert len(sql) == len(expected)
        assert (sql.ts.to_numpy() == expected.ts.to_numpy()).all()
        print("SQL version (DuckDB): returns the same rows")
    except ImportError:
        print("SQL version: DuckDB not installed, skipped")

    heading("Exercise 7. Amounts that are unusual for the account")
    amounts = [20.0, 25.0, 22.0, 30.0, 24.0, 21.0, 900.0, 26.0, 850.0]
    acct = pd.DataFrame({
        "account_id": "C",
        "ts": pd.date_range("2026-12-01 09:00", periods=len(amounts), freq="D"),
        "amount": amounts,
    })
    out = flag_unusual_amounts(acct)
    logs = np.log1p(out["amount"])
    out["plain_z"] = (logs - logs.expanding(5).mean().shift()) / logs.expanding(5).std().shift()
    print(out[["amount", "robust_z", "unusual", "plain_z"]].round(2).to_string(index=False))
    for i in range(len(out)):  # brute force: recompute each score from the earlier rows only
        prior = logs.iloc[:i].to_numpy()
        if len(prior) < 5:
            assert np.isnan(out.robust_z.iloc[i])
            continue
        med = np.median(prior)
        mad = max(np.median(np.abs(prior - med)), 0.05)
        assert math.isclose(out.robust_z.iloc[i], 0.6745 * (logs.iloc[i] - med) / mad)
    print("brute-force recomputation from earlier rows only: OK")

    heading("Exercise 8. Warm-ups")
    print("two_sum([40, 15, 25, 60], 85) =", two_sum([40, 15, 25, 60], 85))
    print("group_anagrams =", group_anagrams(["listen", "silent", "enlist", "night", "thing", "cat"]))
    print("max_subarray([3, -4, 5, -1, 2, -6, 4]) =", max_subarray([3, -4, 5, -1, 2, -6, 4]))
    print("max_profit([10, 7, 12, 9, 15, 14]) =", max_profit([10, 7, 12, 9, 15, 14]))
    print("min_abs_difference([31, 8, 20, 14]) =", min_abs_difference([31, 8, 20, 14]))
    print('longest_unique_substring("fraudfree") =', longest_unique_substring("fraudfree"))
    assert two_sum([1, 2], 9) is None and max_subarray([-3, -1, -2]) == -1
    assert max_profit([5, 4, 3]) == 0 and longest_unique_substring("") == 0
    print("edge cases (no pair, all negative, falling prices, empty string): OK")


if __name__ == "__main__":
    drills()
    exercises()

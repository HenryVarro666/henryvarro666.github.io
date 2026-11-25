"""Machine learning and statistics, the next layer: lab.

Prints every number quoted in the note "Machine learning and statistics, the next layer" and
runs its code.

    python3 next-layer-lab.py

Needs Python 3.10 or later, NumPy and pandas. If scikit-learn or DuckDB is installed, the
logistic regression, ridge, average precision, PCA and the SQL patterns are cross-checked.
There are no data files: every input is defined or generated here with a fixed seed.
"""

import math
import random
from fractions import Fraction
from itertools import product

import numpy as np
import pandas as pd


def heading(text):
    print(f"\n== {text} ==")


# --------------------------------------------------------------------------------------
# Derivations
# --------------------------------------------------------------------------------------
def leaf_weight(G, H, lam=1.0):
    """Best constant for a leaf under the second-order objective."""
    return -G / (H + lam)


def split_gain(G_L, H_L, G_R, H_R, lam=1.0, gamma=0.0):
    """XGBoost's gain from splitting a leaf into a left and a right child."""
    def score(G, H):
        return G * G / (H + lam)
    return 0.5 * (score(G_L, H_L) + score(G_R, H_R) - score(G_L + G_R, H_L + H_R)) - gamma


def fit_logistic_newton(X, y, tol=1e-10, max_iter=50):
    """Newton's method (iteratively reweighted least squares). X must include a column of ones."""
    w = np.zeros(X.shape[1])
    for step in range(1, max_iter + 1):
        p = 1 / (1 + np.exp(-X @ w))
        gradient = X.T @ (p - y)
        hessian = X.T @ (X * (p * (1 - p))[:, None])
        delta = np.linalg.solve(hessian, gradient)
        w -= delta
        if np.abs(delta).max() < tol:
            break
    return w, step


def ridge(X, y, lam):
    """Closed form: (X'X + lam I)^-1 X'y."""
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def em_two_gaussians(x, n_iter=200, seed=0):
    """EM for a one-dimensional mixture of two Gaussians. Returns parameters and the log-likelihoods."""
    rng = np.random.default_rng(seed)
    mu = rng.choice(x, 2, replace=False)
    sigma, weight, history = np.array([x.std(), x.std()]), np.array([0.5, 0.5]), []
    for _ in range(n_iter):
        dens = weight * np.exp(-0.5 * ((x[:, None] - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
        history.append(float(np.log(dens.sum(axis=1)).sum()))
        resp = dens / dens.sum(axis=1, keepdims=True)              # E-step: responsibilities
        total = resp.sum(axis=0)                                    # M-step: weighted estimates
        mu = (resp * x[:, None]).sum(axis=0) / total
        sigma = np.sqrt((resp * (x[:, None] - mu) ** 2).sum(axis=0) / total)
        weight = total / len(x)
    return mu, sigma, weight, history


def softmax(z):
    z = np.asarray(z, float)
    z = z - z.max()                      # subtracting the maximum changes nothing and avoids overflow
    e = np.exp(z)
    return e / e.sum()


def pca_svd(X, k):
    """Principal components from the SVD of the centered data."""
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    explained = S ** 2 / (S ** 2).sum()
    return Xc @ Vt[:k].T, Vt[:k], explained[:k]


# --------------------------------------------------------------------------------------
# Risk-modeling staples
# --------------------------------------------------------------------------------------
def woe_iv(good, bad):
    """Weight of evidence per bin and the information value of the feature."""
    good, bad = np.asarray(good, float), np.asarray(bad, float)
    pg, pb = good / good.sum(), bad / bad.sum()
    woe = np.log(pg / pb)
    return woe, float(((pg - pb) * woe).sum())


def precision_at_prevalence(tpr, fpr, prevalence):
    return tpr * prevalence / (tpr * prevalence + fpr * (1 - prevalence))


def expected_calibration_error(pred, observed, n):
    n = np.asarray(n, float)
    return float((n / n.sum() * np.abs(np.asarray(pred) - np.asarray(observed))).sum())


# --------------------------------------------------------------------------------------
# Survival
# --------------------------------------------------------------------------------------
def kaplan_meier(durations, events):
    """List of (time, at risk, events, survival) at each event time."""
    durations, events = np.asarray(durations), np.asarray(events)
    survival, rows = 1.0, []
    for t in np.unique(durations[events == 1]):
        at_risk = int((durations >= t).sum())
        d = int(((durations == t) & (events == 1)).sum())
        survival *= 1 - d / at_risk
        rows.append((int(t), at_risk, d, survival))
    return rows


# --------------------------------------------------------------------------------------
# Graphs
# --------------------------------------------------------------------------------------
def connected_components(edges):
    """Union-find with path compression. Returns a list of sets."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]        # path halving
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    groups = {}
    for node in parent:
        groups.setdefault(find(node), set()).add(node)
    return list(groups.values())


def personalized_pagerank(edges, seeds, restart=0.15, n_iter=100):
    """Random walk with restart on an undirected graph; the walk restarts at the seed nodes."""
    nodes = sorted({n for e in edges for n in e})
    index = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))
    for a, b in edges:
        A[index[a], index[b]] = A[index[b], index[a]] = 1
    P = A / A.sum(axis=0, keepdims=True)         # column-stochastic transition matrix
    s = np.zeros(len(nodes))
    s[[index[n] for n in seeds]] = 1 / len(seeds)
    r = s.copy()
    for _ in range(n_iter):
        r = (1 - restart) * P @ r + restart * s
    return dict(zip(nodes, r))


# --------------------------------------------------------------------------------------
# Reinforcement learning
# --------------------------------------------------------------------------------------
def ips_estimates(logs, policy):
    """Inverse propensity scoring and its self-normalized version.

    logs: (context, action, propensity of that action under the logging policy, reward)."""
    weights = np.array([(policy(c) == a) / p for c, a, p, _ in logs])
    rewards = np.array([r for *_, r in logs])
    return float((weights * rewards).mean()), float((weights * rewards).sum() / weights.sum())


# --------------------------------------------------------------------------------------
# More code
# --------------------------------------------------------------------------------------
def reservoir_sample(stream, k, seed=0):
    """Uniform sample of k items from a stream of unknown length, in one pass."""
    rng = random.Random(seed)
    sample = []
    for i, item in enumerate(stream):
        if i < k:
            sample.append(item)
        else:
            j = rng.randint(0, i)                # item i is kept with probability k / (i + 1)
            if j < k:
                sample[j] = item
    return sample


def average_precision(y, score):
    """Area under the precision-recall curve as the mean precision at each positive."""
    order = np.argsort(-np.asarray(score), kind="stable")
    y = np.asarray(y)[order]
    hits = np.cumsum(y)
    precision = hits / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


TOP_N_SQL = """
SELECT account_id, ts, amount
FROM (
  SELECT t.*, ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY amount DESC, ts) AS rn
  FROM transactions AS t
) AS x
WHERE rn <= 2
ORDER BY account_id, amount DESC
"""

STREAK_SQL = """
SELECT account_id, MIN(day) AS first_day, COUNT(*) AS days
FROM (
  SELECT account_id, day,
         day - CAST(ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY day) AS INTEGER) AS grp
  FROM (SELECT DISTINCT account_id, CAST(ts AS DATE) AS day FROM transactions) AS d
) AS x
GROUP BY account_id, grp
ORDER BY account_id, first_day
"""

SESSION_SQL = """
SELECT account_id, ts,
       SUM(new_session) OVER (PARTITION BY account_id ORDER BY ts) AS session_id
FROM (
  SELECT t.*,
         CASE WHEN ts - LAG(ts) OVER (PARTITION BY account_id ORDER BY ts) <= INTERVAL 30 MINUTE
              THEN 0 ELSE 1 END AS new_session
  FROM transactions AS t
) AS x
ORDER BY account_id, ts
"""

ASOF_SQL = """
SELECT t.account_id, t.ts, t.amount, r.risk_tier
FROM transactions AS t
ASOF LEFT JOIN risk_tiers AS r
  ON t.account_id = r.account_id AND t.ts >= r.valid_from
ORDER BY t.account_id, t.ts
"""


# --------------------------------------------------------------------------------------
# Printed numbers
# --------------------------------------------------------------------------------------
def derivations():
    heading("Derivations")
    p0 = 0.5                                       # current prediction for every row in the node
    left_y, right_y = [1, 1, 0], [0, 0, 0]
    G_L, H_L = sum(p0 - y for y in left_y), len(left_y) * p0 * (1 - p0)
    G_R, H_R = sum(p0 - y for y in right_y), len(right_y) * p0 * (1 - p0)
    print(f"left: G={G_L}, H={H_L}; right: G={G_R}, H={H_R}")
    print(f"leaf weights: left {leaf_weight(G_L, H_L):.4f}, right {leaf_weight(G_R, H_R):.4f}; "
          f"gain {split_gain(G_L, H_L, G_R, H_R):.4f}")

    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, (4000, 2))
    y = (rng.random(4000) < 1 / (1 + np.exp(-(X @ np.array([1.5, -1.0]) - 0.5)))).astype(float)
    Xb = np.column_stack([np.ones(len(X)), X])
    w, steps = fit_logistic_newton(Xb, y)
    print(f"Newton's method: intercept and weights {np.round(w, 3)} after {steps} iterations")
    try:
        from sklearn.linear_model import LogisticRegression, Ridge
        ref = LogisticRegression(C=1e9, max_iter=1000).fit(X, y)
        assert np.allclose(w[1:], ref.coef_[0], atol=1e-3) and math.isclose(w[0], ref.intercept_[0], abs_tol=1e-3)
        target = X @ np.array([2.0, -1.0]) + rng.normal(0, 1, len(X))
        assert np.allclose(ridge(X, target, 10.0), Ridge(alpha=10.0, fit_intercept=False).fit(X, target).coef_)
        print("scikit-learn agrees on the logistic fit and on ridge")
    except ImportError:
        print("scikit-learn not installed, cross-check skipped")
    for d in (10.0, 1.0):
        print(f"ridge, lambda 1: a direction with singular value {d:g} keeps {d * d / (d * d + 1):.3f} of its coefficient")

    x = np.concatenate([rng.normal(0, 1, 600), rng.normal(5, 1.5, 400)])
    mu, sigma, weight, history = em_two_gaussians(x)
    assert all(b >= a - 1e-9 for a, b in zip(history, history[1:])), "the log-likelihood must not fall"
    order = np.argsort(mu)
    print(f"EM on a two-component mixture: means {np.round(mu[order], 2)}, sds {np.round(sigma[order], 2)}, "
          f"weights {np.round(weight[order], 2)}; log-likelihood never fell")

    logits = np.array([1000.0, 1001.0, 1002.0])
    with np.errstate(over="ignore", invalid="ignore"):
        naive = np.exp(logits) / np.exp(logits).sum()
    print(f"softmax of {logits.tolist()}: naive {naive}, stable {np.round(softmax(logits), 4)}")

    data = rng.normal(0, 1, (500, 3)) @ np.array([[3, 0, 0], [1, 1, 0], [0, 0, 0.2]])
    _, _, explained = pca_svd(data, 2)
    print(f"PCA by SVD: first two components explain {np.round(explained, 3)}")
    try:
        from sklearn.decomposition import PCA
        assert np.allclose(explained, PCA(2).fit(data).explained_variance_ratio_)
        print("scikit-learn agrees on the explained variance")
    except ImportError:
        pass

    d, r = 4096, 16
    print(f"LoRA on a {d}x{d} matrix with rank {r}: {2 * d * r:,} trainable of {d * d:,} "
          f"({2 * d * r / (d * d):.2%})")
    g, lr, b1, b2 = 250.0, 0.001, 0.9, 0.999
    m, v = (1 - b1) * g, (1 - b2) * g * g
    print(f"Adam, first step with gradient {g:g}: step = {lr * (m / (1 - b1)) / math.sqrt(v / (1 - b2)):.6f} "
          f"(the learning rate, whatever the gradient's size)")


def staples():
    heading("Risk-modeling staples")
    good, bad = [6000, 3000, 800, 200], [45, 30, 17, 8]
    woe, iv = woe_iv(good, bad)
    for label, g, b, w in zip(("0-1", "2-3", "4-6", "7+"), good, bad, woe):
        print(f"bin {label:>3}: good {g:>5,} ({g / sum(good):.0%}), bad {b:>2} ({b / sum(bad):.0%}), WoE {w:+.3f}")
    print(f"information value = {iv:.3f}")

    pdo, base_score, base_odds = 20, 600, 50
    factor = pdo / math.log(2)
    offset = base_score - factor * math.log(base_odds)
    print(f"scorecard: factor {factor:.2f}, offset {offset:.1f}; odds 100:1 -> {offset + factor * math.log(100):.0f}, "
          f"odds 25:1 -> {offset + factor * math.log(25):.0f}")

    for prev in (0.01, 0.001):
        print(f"TPR 60%, FPR 0.5%, prevalence {prev:.1%}: precision {precision_at_prevalence(0.6, 0.005, prev):.1%}")
    beta = 0.01
    print(f"keeping {beta:.0%} of negatives shifts the logistic intercept by ln(beta) = {math.log(beta):.3f}")

    both, swap_in, swap_out = (700, 140), (300, 30), (300, 9)
    print(f"swap set: flagged by both {both[1]}/{both[0]}, new only {swap_in[1]}/{swap_in[0]} "
          f"({swap_in[1] / swap_in[0]:.0%}), old only {swap_out[1]}/{swap_out[0]} ({swap_out[1] / swap_out[0]:.0%}); "
          f"net {swap_in[1] - swap_out[1]:+d} frauds at the same {both[0] + swap_in[0]:,} alerts")
    print(f"expected calibration error: {expected_calibration_error([0.1, 0.5, 0.9], [0.08, 0.6, 0.7], [800, 150, 50]):.3f}")
    print(f"approval rates 60% and 45%: ratio {0.45 / 0.60:.2f} (the four-fifths rule asks for 0.80 or more)")


def statistics():
    heading("More statistics")
    shape, rate, events, days = 2, 1, 30, 10
    print(f"Gamma({shape}, {rate}) prior, {events} events in {days} days: posterior Gamma({shape + events}, {rate + days}), "
          f"mean {(shape + events) / (rate + days):.3f} per day")
    w, mean, sd = np.array([0.9, 0.1]), np.array([50.0, 300.0]), np.array([20.0, 100.0])
    overall = (w * mean).sum()
    within, between = (w * sd ** 2).sum(), (w * (mean - overall) ** 2).sum()
    print(f"two segments: mean {overall:.0f}, within-group variance {within:.0f}, between-group {between:.0f}, "
          f"total sd {math.sqrt(within + between):.1f}")
    print(f"three sd or more from the mean: Chebyshev bound {1 / 9:.3f}, normal {math.erfc(3 / math.sqrt(2)):.4f}")
    p1, n1, p2, n2 = 0.12, 5000, 0.10, 5000
    se = math.sqrt((1 - p1) / (n1 * p1) + (1 - p2) / (n2 * p2))
    lift = p1 / p2
    print(f"lift {lift:.2f}: se of its log {se:.4f}, 95% interval {lift * math.exp(-1.96 * se):.3f} to {lift * math.exp(1.96 * se):.3f}")
    b, c = 40, 20
    chi2 = (b - c) ** 2 / (b + c)
    chi2_cc = (abs(b - c) - 1) ** 2 / (b + c)
    print(f"McNemar with {b} and {c} discordant pairs: chi-square {chi2:.2f} (p = {math.erfc(math.sqrt(chi2 / 2)):.4f}), "
          f"with continuity correction {chi2_cc:.2f} (p = {math.erfc(math.sqrt(chi2_cc / 2)):.4f})")

    rng = np.random.default_rng(5)
    pre = rng.normal(0, 1, 200_000)
    post = 0.6 * pre + math.sqrt(1 - 0.36) * rng.normal(0, 1, 200_000)
    theta = np.cov(pre, post)[0, 1] / pre.var()
    adjusted = post - theta * (pre - pre.mean())
    print(f"CUPED with correlation 0.6: variance ratio {adjusted.var() / post.var():.3f} (theory {1 - 0.36:.2f})")

    phi = 0.5
    print(f"AR(1) with coefficient {phi}: the variance of a mean is inflated by {(1 + phi) / (1 - phi):.0f}, "
          f"so 300 points carry the information of {300 * (1 - phi) / (1 + phi):.0f}")
    truth = rng.normal(0, 1, 200_000)
    first = truth + rng.normal(0, 1, 200_000)
    second = truth + rng.normal(0, 1, 200_000)
    top = first >= np.quantile(first, 0.9)
    print(f"regression to the mean: the top 10% in period one average {first[top].mean():.2f}, "
          f"then {second[top].mean():.2f} in period two with no intervention")


def puzzles():
    heading("Probability puzzles")
    print(f"Monty Hall: switching wins {Fraction(2, 3)}")
    families = [f for f in product("BG", repeat=2) if "B" in f]
    print(f"two children, at least one boy: both boys {Fraction(sum(f == ('B', 'B') for f in families), len(families))}")
    print("expected flips until the first head: 2; until two heads in a row: 6")
    n = 6
    print(f"coupon collector, {n} types: {float(sum(Fraction(n, i) for i in range(1, n + 1))):.1f} draws")
    no_match = 1.0
    for i in range(23):
        no_match *= (365 - i) / 365
    print(f"birthday problem, 23 people: {1 - no_match:.4f}")
    rolls = list(product(range(1, 7), repeat=2))
    print(f"expected maximum of two dice: {Fraction(sum(max(r) for r in rolls), 36)} = {sum(max(r) for r in rolls) / 36:.3f}")
    print(f"at least one six in four rolls: {1 - (5 / 6) ** 4:.4f}")
    rng = np.random.default_rng(6)
    hh, trials = 0, 100_000
    for _ in range(trials):
        prev, flips = False, 0
        while True:
            flips += 1
            head = rng.random() < 0.5
            if head and prev:
                break
            prev = head
        hh += flips
    fixed = np.mean([(rng.permutation(10) == np.arange(10)).sum() for _ in range(20_000)])
    print(f"simulation: flips until two heads {hh / trials:.2f}; fixed points of a random permutation {fixed:.2f}")


def graphs():
    heading("Graphs")
    edges = [("a1", "d1"), ("a2", "d1"), ("a2", "d2"), ("a3", "d2"), ("a4", "d3"), ("a5", "d3")]
    comps = sorted((sorted(c) for c in connected_components(edges)), key=len, reverse=True)
    print("connected components:", comps, "(a6 has no shared device and is its own component)")
    scores = personalized_pagerank(edges, seeds=["a1"])
    print("personalized PageRank from a1:", {k: round(float(v), 3) for k, v in scores.items() if k.startswith("a")})
    x = np.array([[2.0, 2.0], [2.0, -2.0]])
    inv = np.linalg.inv(np.array([[1.0, 0.9], [0.9, 1.0]]))
    for point in x:
        print(f"point {point.tolist()}: Euclidean {np.linalg.norm(point):.2f}, "
              f"Mahalanobis {math.sqrt(point @ inv @ point):.2f}")


def survival():
    heading("Survival")
    durations = [5, 8, 12, 12, 20, 25, 30, 30, 40, 45]
    events = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    for t, at_risk, d, s in kaplan_meier(durations, events):
        print(f"day {t:>2}: at risk {at_risk:>2}, events {d}, survival {s:.3f}")
    print(f"share with an event, ignoring censoring: {sum(events) / len(events):.0%}; "
          f"Kaplan-Meier by day 40: {1 - kaplan_meier(durations, events)[-1][3]:.0%}")
    for age, share, seen in ((7, 0.5, 10), (30, 0.9, 27), (90, 0.99, 40)):
        print(f"cohort aged {age} days: {seen} frauds seen, {share:.0%} reported by then, expected final {seen / share:.1f}")


def reinforcement():
    heading("Reinforcement learning")
    rewards, gamma = [1, 0, 2], 0.9
    print(f"discounted return of {rewards} at gamma {gamma}: {sum(r * gamma ** i for i, r in enumerate(rewards)):.2f}")
    q, alpha, r, best_next = 2.0, 0.5, 1.0, 3.0
    print(f"Q-learning update: {q} -> {q + alpha * (r + gamma * best_next - q):.2f}")
    logs = [("R", "S", 0.2, 1.0), ("R", "A", 0.8, -5.0), ("N", "A", 0.8, 0.5), ("N", "A", 0.8, 0.5),
            ("N", "S", 0.2, 0.2), ("R", "S", 0.2, 1.0), ("N", "A", 0.8, 0.5), ("R", "A", 0.8, -5.0)]
    ips, snips = ips_estimates(logs, lambda context: "S" if context == "R" else "A")
    print(f"logged average reward {np.mean([row[3] for row in logs]):.3f}; new policy: IPS {ips:.3f}, "
          f"self-normalized {snips:.3f}, true value 0.750")


def code_checks():
    heading("More code")
    counts = np.zeros(10)
    for seed in range(20_000):
        for item in reservoir_sample(range(10), 3, seed=seed):
            counts[item] += 1
    print(f"reservoir sampling, 3 of 10: each item kept {counts.min() / 20_000:.3f} to {counts.max() / 20_000:.3f} "
          f"of the time (target 0.300)")

    rng = np.random.default_rng(7)
    y = (rng.random(5000) < 0.02).astype(int)
    score = rng.normal(0, 1, 5000) + 2.0 * y
    ap = average_precision(y, score)
    print(f"average precision from scratch: {ap:.4f} (prevalence {y.mean():.3f})")
    try:
        from sklearn.metrics import average_precision_score
        assert math.isclose(ap, average_precision_score(y, score), rel_tol=1e-9)
        print("scikit-learn agrees on average precision")
    except ImportError:
        pass

    transactions = pd.DataFrame({
        "account_id": ["A"] * 6 + ["B"] * 3,
        "ts": pd.to_datetime(["2026-12-01 09:00", "2026-12-01 09:10", "2026-12-01 10:30", "2026-12-02 08:00",
                              "2026-12-03 08:00", "2026-12-05 12:00",
                              "2026-12-01 11:00", "2026-12-02 11:00", "2026-12-02 11:20"]),
        "amount": [20.0, 300.0, 40.0, 75.0, 60.0, 500.0, 10.0, 90.0, 30.0],
    })
    risk_tiers = pd.DataFrame({
        "account_id": ["A", "A", "B"],
        "valid_from": pd.to_datetime(["2026-11-01 00:00", "2026-12-02 00:00", "2026-12-02 11:10"]),
        "risk_tier": ["low", "high", "medium"],
    })
    asof = pd.merge_asof(transactions.sort_values("ts"), risk_tiers.sort_values("valid_from"),
                         left_on="ts", right_on="valid_from", by="account_id").sort_values(["account_id", "ts"])
    print("point-in-time join (pandas):", asof["risk_tier"].fillna("none").tolist())
    gap = transactions.groupby("account_id")["ts"].diff() > pd.Timedelta(minutes=30)
    first = transactions.groupby("account_id").cumcount() == 0
    sessions = (gap | first).groupby(transactions["account_id"]).cumsum()
    print("sessions (pandas):", sessions.tolist())
    try:
        import duckdb
        top = duckdb.sql(TOP_N_SQL).df()
        print("top two amounts per account (SQL):", top.groupby("account_id")["amount"].apply(list).to_dict())
        streak = duckdb.sql(STREAK_SQL).df()
        print("streaks of consecutive active days (SQL):",
              [(a, str(d)[:10], int(n)) for a, d, n in streak.itertuples(index=False)])
        sql_sessions = duckdb.sql(SESSION_SQL).df()
        assert sql_sessions["session_id"].astype(int).tolist() == sessions.tolist()
        sql_asof = duckdb.sql(ASOF_SQL).df()
        assert sql_asof["risk_tier"].fillna("none").tolist() == asof["risk_tier"].fillna("none").tolist()
        print("DuckDB agrees with pandas on the sessions and on the point-in-time join")
    except ImportError:
        print("DuckDB not installed, SQL patterns skipped")


if __name__ == "__main__":
    derivations()
    staples()
    statistics()
    puzzles()
    graphs()
    survival()
    reinforcement()
    code_checks()

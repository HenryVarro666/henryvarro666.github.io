"""Machine learning and statistics fundamentals: lab.

Prints every number quoted in the note "Machine learning and statistics fundamentals: interview
questions with short answers" and runs its from-scratch code.

    python3 ml-stats-lab.py

Needs Python 3.10 or later and NumPy. If SciPy or scikit-learn is installed, the KS statistic,
the logistic regression and k-means are cross-checked against them. There are no data files:
every input is generated here with a fixed seed.
"""

import math

import numpy as np

Z975, Z80 = 1.959964, 0.841621   # normal quantiles for 97.5% and 80%


# --------------------------------------------------------------------------------------
# From scratch 1. Wilson interval for a proportion
# --------------------------------------------------------------------------------------
def wilson_interval(k, n, z=Z975):
    """Score interval for a binomial proportion; behaves well when k is small."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


# --------------------------------------------------------------------------------------
# From scratch 2. Bootstrap interval for any metric
# --------------------------------------------------------------------------------------
def bootstrap_interval(values, metric=np.mean, n_boot=2000, level=0.95, seed=0):
    """Percentile interval: resample rows with replacement and recompute the metric."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    stats = [metric(values[rng.integers(0, len(values), len(values))]) for _ in range(n_boot)]
    lo, hi = np.quantile(stats, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


# --------------------------------------------------------------------------------------
# From scratch 3. Permutation test for a difference in means
# --------------------------------------------------------------------------------------
def permutation_test(a, b, n_perm=10_000, seed=0):
    """Two-sided p-value: how often does a random relabeling give a gap at least this large?"""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    observed = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        hits += abs(pooled[:len(a)].mean() - pooled[len(a):].mean()) >= observed
    return (hits + 1) / (n_perm + 1)


# --------------------------------------------------------------------------------------
# From scratch 4. Benjamini-Hochberg
# --------------------------------------------------------------------------------------
def benjamini_hochberg(p_values, q=0.05):
    """Boolean mask of rejected hypotheses at false discovery rate q."""
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, len(p) + 1) / len(p)
    k = passed.nonzero()[0].max() + 1 if passed.any() else 0   # largest rank that passes
    reject = np.zeros(len(p), dtype=bool)
    reject[order[:k]] = True
    return reject


# --------------------------------------------------------------------------------------
# From scratch 5. KS statistic between two samples
# --------------------------------------------------------------------------------------
def ks_statistic(a, b):
    """Largest vertical gap between the two empirical distribution functions."""
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.abs(cdf_a - cdf_b).max())


# --------------------------------------------------------------------------------------
# From scratch 6. The best split of one feature by Gini impurity
# --------------------------------------------------------------------------------------
def gini(y):
    p = np.mean(y)
    return 2 * p * (1 - p)             # 1 - p^2 - (1 - p)^2 for two classes


def best_split(x, y):
    """Threshold on x that lowers the weighted Gini impurity the most."""
    order = np.argsort(x)
    x, y = np.asarray(x)[order], np.asarray(y)[order]
    parent, best = gini(y), (None, 0.0)
    for i in range(1, len(x)):
        if x[i] == x[i - 1]:
            continue                    # no threshold separates equal values
        child = (i * gini(y[:i]) + (len(x) - i) * gini(y[i:])) / len(x)
        if parent - child > best[1]:
            best = ((x[i] + x[i - 1]) / 2, parent - child)
    return best


# --------------------------------------------------------------------------------------
# From scratch 7. Logistic regression by gradient descent
# --------------------------------------------------------------------------------------
def fit_logistic(X, y, lr=0.5, steps=5000, l2=0.0):
    """Minimizes mean log loss (+ l2/2 * |w|^2). Returns (weights, intercept)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        w -= lr * (X.T @ (p - y) / len(y) + l2 * w)   # gradient of the mean log loss
        b -= lr * np.mean(p - y)
    return w, b


# --------------------------------------------------------------------------------------
# From scratch 8. k-means (Lloyd's algorithm)
# --------------------------------------------------------------------------------------
def kmeans(X, k, n_iter=100, seed=0):
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), k, replace=False)]
    for _ in range(n_iter):
        labels = np.argmin(((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
        new = np.array([X[labels == j].mean(axis=0) if (labels == j).any() else centers[j]
                        for j in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    inertia = float(((X - centers[labels]) ** 2).sum())
    return centers, labels, inertia


# --------------------------------------------------------------------------------------
# Printed numbers
# --------------------------------------------------------------------------------------
def heading(text):
    print(f"\n== {text} ==")


def poisson_tail(lam, k):
    """P(X >= k) for a Poisson variable."""
    return 1 - sum(math.exp(-lam) * lam ** i / math.factorial(i) for i in range(k))


def normal_cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2))


def probability():
    heading("Probability")
    n, p = 10_000, 0.001
    print(f"binomial n={n:,}, p={p}: mean {n * p:.0f}, sd {math.sqrt(n * p * (1 - p)):.2f}")
    print(f"P(no fraud): exact {(1 - p) ** n:.6f}, Poisson {math.exp(-n * p):.6f}")
    print(f"P(20 or more frauds | mean 10) = {poisson_tail(10, 20):.4f}")
    print(f"rule of three, 0 frauds in 1,000: 3/n = {3 / 1000:.4f}, exact = {1 - 0.05 ** (1 / 1000):.6f}")
    sd, m = 80, 400
    print(f"standard error of a mean, sd ${sd}, n={m}: ${sd / math.sqrt(m):.0f}")
    print(f"Poisson alert count with mean 100: sd {math.sqrt(100):.0f}, "
          f"P(130 or more) = {poisson_tail(100, 130):.4f}")
    monitors = 200
    print(f"{monitors} independent monitors at a 0.27% false-alarm rate each: "
          f"P(at least one false alarm) = {1 - (1 - 0.0027) ** monitors:.2f}")


def estimation():
    heading("Estimation and intervals")
    k, n = 3, 2000
    p = k / n
    se = math.sqrt(p * (1 - p) / n)
    lo, hi = wilson_interval(k, n)
    print(f"{k} frauds in {n:,}: rate {p:.4%}, Wald {p - Z975 * se:.4%} to {p + Z975 * se:.4%}, "
          f"Wilson {lo:.4%} to {hi:.4%}")
    a, b = 1, 999                         # Beta prior with mean 0.1%
    for frauds, total in ((2, 50), (200, 5000)):
        print(f"merchant with {frauds} frauds in {total:,}: raw {frauds / total:.2%}, "
              f"posterior mean {(a + frauds) / (a + b + total):.2%}")
    caught = np.zeros(400)
    caught[:320] = 1                      # 320 of 400 frauds caught
    lo, hi = bootstrap_interval(caught)
    se = math.sqrt(0.8 * 0.2 / 400)
    print(f"recall 320/400: bootstrap {lo:.3f} to {hi:.3f}, formula {0.8 - Z975 * se:.3f} to {0.8 + Z975 * se:.3f}")
    observed, reported_by_day_7 = 0.0005, 0.5
    print(f"fraud rate seen after 7 days {observed:.2%}, half reported by then: "
          f"expected final rate {observed / reported_by_day_7:.2%}")


def testing():
    heading("Hypothesis testing")
    p1, p2 = 0.10, 0.12
    n = (Z975 + Z80) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p1 - p2) ** 2
    print(f"sample size per arm for {p1:.0%} against {p2:.0%}, alpha 5%, power 80%: {math.ceil(n):,}")
    print(f"20 tests at 5%: P(at least one false positive) = {1 - 0.95 ** 20:.2f}; "
          f"Bonferroni threshold = {0.05 / 20}")
    pv = [0.001, 0.008, 0.012, 0.03, 0.04, 0.2, 0.5, 0.6, 0.8, 0.9]
    print(f"ten p-values: Bonferroni rejects {sum(x <= 0.05 / len(pv) for x in pv)}, "
          f"Benjamini-Hochberg rejects {int(benjamini_hochberg(pv).sum())}")
    prior, power, alpha = 0.10, 0.80, 0.05
    true_pos, false_pos = prior * power, (1 - prior) * alpha
    print(f"10% of tested ideas are real, power 80%, alpha 5%: "
          f"share of significant results that are real = {true_pos / (true_pos + false_pos):.2f}")
    rng = np.random.default_rng(2)
    a, b = rng.normal(100, 20, 60), rng.normal(110, 20, 60)
    print(f"permutation test, means {a.mean():.1f} and {b.mean():.1f}: p = {permutation_test(a, b):.4f}")


def regression_and_bias():
    heading("Regression and bias")
    beta, base = 0.7, 0.01
    odds = base / (1 - base) * math.exp(beta)
    print(f"coefficient {beta}: odds ratio {math.exp(beta):.2f}; "
          f"a {base:.0%} base probability becomes {odds / (1 + odds):.2%}")
    rules = {"A": {"web": (900, 1000), "mobile": (30, 100)},
             "B": {"web": (95, 100), "mobile": (400, 1000)}}
    for name, seg in rules.items():
        hit = sum(v[0] for v in seg.values())
        total = sum(v[1] for v in seg.values())
        print(f"rule {name}: " + ", ".join(f"{s} {v[0]}/{v[1]} = {v[0] / v[1]:.0%}" for s, v in seg.items())
              + f", overall {hit}/{total} = {hit / total:.1%}")


def evaluation():
    heading("Model evaluation")
    d = 1.0                               # scores: good ~ N(0, 1), bad ~ N(1, 1)
    auc = normal_cdf(d / math.sqrt(2))
    print(f"scores N(0,1) and N(1,1): AUC {auc:.3f}, Gini {2 * auc - 1:.3f}, "
          f"KS {2 * normal_cdf(d / 2) - 1:.3f}")
    rng = np.random.default_rng(3)
    good, bad = rng.normal(0, 1, 50_000), rng.normal(1, 1, 5_000)
    ks = ks_statistic(good, bad)
    print(f"simulated KS = {ks:.3f}")
    try:
        from scipy.stats import ks_2samp
        assert math.isclose(ks, ks_2samp(good, bad).statistic, rel_tol=1e-9)
        print("SciPy agrees on the KS statistic")
    except ImportError:
        print("SciPy not installed, cross-check skipped")
    print(f"top 1% of scores holds 50% of the fraud: lift = {0.50 / 0.01:.0f}")
    y, p_good, p_flat = np.array([1, 0, 0, 0]), np.array([0.7, 0.2, 0.1, 0.1]), np.full(4, 0.25)
    for name, q in (("sharp model", p_good), ("always 0.25", p_flat)):
        print(f"{name}: Brier {np.mean((q - y) ** 2):.4f}, "
              f"log loss {-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)):.4f}")


def algorithms():
    heading("Algorithms")
    parent = np.array([1] * 4 + [0] * 6)
    left, right = np.array([1, 1, 1, 0]), np.array([1, 0, 0, 0, 0, 0])
    child = (len(left) * gini(left) + len(right) * gini(right)) / len(parent)
    print(f"Gini: parent {gini(parent):.4f}, left {gini(left):.4f}, right {gini(right):.4f}, "
          f"weighted {child:.4f}, gain {gini(parent) - child:.4f}")
    x = np.array([5, 8, 12, 20, 25, 40, 60, 90, 150, 300])
    y = np.array([0, 0, 0, 0, 1, 0, 0, 1, 1, 1])
    threshold, gain = best_split(x, y)
    print(f"best split of the toy feature: threshold {threshold}, gain {gain:.4f}")

    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, (4000, 2))
    true_w, true_b = np.array([1.5, -1.0]), -0.5
    y = (rng.random(4000) < 1 / (1 + np.exp(-(X @ true_w + true_b)))).astype(float)
    w, b = fit_logistic(X, y)
    print(f"logistic regression from scratch: w = {np.round(w, 3)}, b = {b:.3f} (truth {true_w}, {true_b})")
    try:
        from sklearn.linear_model import LogisticRegression
        ref = LogisticRegression(C=1e9, max_iter=1000).fit(X, y)   # effectively no penalty
        assert np.allclose(w, ref.coef_[0], atol=1e-3) and math.isclose(b, ref.intercept_[0], abs_tol=1e-3)
        print("scikit-learn agrees to three decimals")
    except ImportError:
        print("scikit-learn not installed, cross-check skipped")

    blobs = np.vstack([rng.normal(c, 0.5, (200, 2)) for c in ((0, 0), (4, 0), (2, 3))])
    centers, _, inertia = kmeans(blobs, 3)
    print("k-means centers:", np.round(centers[np.argsort(centers[:, 0])], 1).tolist(), f"inertia {inertia:.1f}")
    try:
        from sklearn.cluster import KMeans
        ref = KMeans(3, n_init=10, random_state=0).fit(blobs)
        assert math.isclose(inertia, ref.inertia_, rel_tol=1e-6)
        print("scikit-learn reaches the same inertia")
    except ImportError:
        print("scikit-learn not installed, cross-check skipped")


if __name__ == "__main__":
    probability()
    estimation()
    testing()
    regression_and_bias()
    evaluation()
    algorithms()

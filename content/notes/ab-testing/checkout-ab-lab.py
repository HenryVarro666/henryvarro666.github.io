"""Reproduce the planning and readout numbers from the A/B testing note.

Requires Python 3.10+ and uses only its standard library.
All experiment counts are hypothetical.
"""

from hashlib import sha256
from json import dumps
from math import ceil, erfc, sqrt
from statistics import NormalDist


def assign_variant(experiment_id: str, user_id: str, salt: str) -> str:
    """Illustrate stable 50/50 assignment with a fixed experiment salt."""
    key = dumps([experiment_id, salt, user_id], ensure_ascii=False, separators=(",", ":"))
    bucket = int.from_bytes(sha256(key.encode("utf-8")).digest()[:8], "big")
    return "A" if bucket < 2**63 else "B"


def planned_sample_per_arm(
    baseline: float, lift: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Normal approximation for two independent proportions and equal allocation."""
    alternative = baseline + lift
    pooled = (baseline + alternative) / 2
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_power = NormalDist().inv_cdf(power)
    variance_null = 2 * pooled * (1 - pooled)
    variance_alt = baseline * (1 - baseline) + alternative * (1 - alternative)
    return ceil(((z_alpha * sqrt(variance_null) + z_power * sqrt(variance_alt)) / lift) ** 2)


def readout(k_a: int, n_a: int, k_b: int, n_b: int) -> tuple[float, float, float]:
    """Return B minus A, unpooled standard error, and two-sided pooled z-test p."""
    rate_a = k_a / n_a
    rate_b = k_b / n_b
    estimate = rate_b - rate_a
    se = sqrt(rate_a * (1 - rate_a) / n_a + rate_b * (1 - rate_b) / n_b)
    pooled = (k_a + k_b) / (n_a + n_b)
    se_null = sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    p_value = erfc(abs(estimate) / (se_null * sqrt(2)))
    return estimate, se, p_value


def srm_pvalue(n_a: int, n_b: int, share_a: float = 0.5) -> float:
    """Pearson goodness-of-fit p-value for two arms and a fixed allocation."""
    expected_a = (n_a + n_b) * share_a
    expected_b = (n_a + n_b) * (1 - share_a)
    chi_square = (n_a - expected_a) ** 2 / expected_a + (n_b - expected_b) ** 2 / expected_b
    # The chi-squared distribution with one degree of freedom is squared N(0, 1).
    return erfc(sqrt(chi_square / 2))


def main() -> None:
    print("Planned sample per arm (10% baseline, 80% power, two-sided alpha = 0.05)")
    for lift in (0.005, 0.01, 0.02):
        print(f"  MDE {lift * 100:.1f} pp: {planned_sample_per_arm(0.10, lift):,}")

    print("\nHypothetical fixed-horizon readout")
    n_a = n_b = 20_000
    k_a, k_b = 2_000, 2_200
    lift, se, p_value = readout(k_a, n_a, k_b, n_b)
    margin = NormalDist().inv_cdf(0.975) * se
    print(f"  Conversion: A {k_a / n_a:.2%}, B {k_b / n_b:.2%}")
    print(f"  Absolute lift: {lift * 100:+.2f} pp; relative lift: {lift / (k_a / n_a):+.1%}")
    print(f"  95% CI: [{(lift - margin) * 100:+.2f}, {(lift + margin) * 100:+.2f}] pp")
    print(f"  Two-sided p-value: {p_value:.6f}")
    print(f"  50/50 SRM p-value: {srm_pvalue(n_a, n_b):.2g}")

    opt_out, opt_out_se, _ = readout(100, n_a, 120, n_b)
    guardrail_limit = 0.002
    upper_bound = opt_out + NormalDist().inv_cdf(0.95) * opt_out_se
    guardrail_passed = upper_bound < guardrail_limit
    print(f"  Opt-out increase: {opt_out * 100:+.2f} pp")
    print(f"  One-sided 95% upper bound: {upper_bound * 100:+.3f} pp")
    print(f"  Guardrail upper bound < +{guardrail_limit * 100:.2f} pp: {guardrail_passed}")

    print("\nDiagnostics")
    print(f"  21,000 / 19,000 under fixed 50/50: SRM p={srm_pvalue(21_000, 19_000):.3g}")
    arm = assign_variant("checkout-reminder", "user-42", "demo-v1")
    print(f"  Same account on two devices: {arm}, {assign_variant('checkout-reminder', 'user-42', 'demo-v1')}")


if __name__ == "__main__":
    main()

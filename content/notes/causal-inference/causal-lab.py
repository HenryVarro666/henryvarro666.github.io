"""Reproduce the checkout reminder example with the Python 3 standard library."""

from dataclasses import dataclass
from math import isclose, sqrt


@dataclass(frozen=True)
class IntentStratum:
    label: str
    offered: int
    offered_purchases: int
    withheld: int
    withheld_purchases: int

    @property
    def count(self) -> int:
        return self.offered + self.withheld


def standardize(rows):
    """Estimate both arm means over the same target intent distribution."""
    if any(row.offered == 0 or row.withheld == 0 for row in rows):
        raise ValueError("The target includes an intent stratum without both arms.")
    total = sum(row.count for row in rows)
    offer_risk = sum(
        row.count / total * row.offered_purchases / row.offered
        for row in rows
    )
    withhold_risk = sum(
        row.count / total * row.withheld_purchases / row.withheld
        for row in rows
    )
    return offer_risk, withhold_risk


def weighted_effect(rows, propensity):
    """Horvitz–Thompson ATE, computed directly from the cell counts."""
    total = sum(row.count for row in rows)
    return sum(
        row.offered_purchases / propensity(row)
        - row.withheld_purchases / (1 - propensity(row))
        for row in rows
    ) / total


def augmented_effect(rows, propensity, offer_risk, withhold_risk):
    """AIPW ATE: predictions plus observed-minus-expected event corrections."""
    total = sum(row.count for row in rows)
    score_sum = 0.0
    for row in rows:
        e = propensity(row)
        m1 = offer_risk(row)
        m0 = withhold_risk(row)
        score_sum += (
            row.count * (m1 - m0)
            + (row.offered_purchases - row.offered * m1) / e
            - (row.withheld_purchases - row.withheld * m0) / (1 - e)
        )
    return score_sum / total


def empirical_propensity(row):
    return row.offered / row.count


def empirical_offer_risk(row):
    return row.offered_purchases / row.offered


def empirical_withhold_risk(row):
    return row.withheld_purchases / row.withheld


def main():
    rows = [
        IntentStratum("Low intent", 160, 32, 40, 4),
        IntentStratum("High intent", 40, 26, 160, 96),
    ]
    crude_offer = sum(row.offered_purchases for row in rows) / sum(
        row.offered for row in rows
    )
    crude_withhold = sum(row.withheld_purchases for row in rows) / sum(
        row.withheld for row in rows
    )
    crude = crude_offer - crude_withhold
    print(f"Crude offer / withhold: {crude_offer:.1%} / {crude_withhold:.1%}")
    print(f"Crude difference: {100 * crude:+.1f} pp")

    for row in rows:
        within_stratum = empirical_offer_risk(row) - empirical_withhold_risk(row)
        print(
            f"{row.label}: propensity {empirical_propensity(row):.0%}, "
            f"within-stratum difference {100 * within_stratum:+.1f} pp"
        )

    offer_risk, withhold_risk = standardize(rows)
    adjusted = offer_risk - withhold_risk
    print(f"Standardized offer / withhold: {offer_risk:.1%} / {withhold_risk:.1%}")
    print(f"Standardized difference: {100 * adjusted:+.1f} pp")

    ipw = weighted_effect(rows, empirical_propensity)
    both_fit = augmented_effect(
        rows, empirical_propensity, empirical_offer_risk, empirical_withhold_risk
    )
    propensity_only = augmented_effect(
        rows, empirical_propensity, lambda row: 0.40, lambda row: 0.35
    )
    outcome_only = augmented_effect(
        rows, lambda row: 0.50, empirical_offer_risk, empirical_withhold_risk
    )
    neither_fit = augmented_effect(
        rows, lambda row: 0.50, lambda row: 0.40, lambda row: 0.35
    )
    print(f"IPW: {100 * ipw:+.1f} pp")
    for label, estimate in (
        ("both inputs fitted", both_fit),
        ("only propensity fitted", propensity_only),
        ("only outcome risks fitted", outcome_only),
        ("neither input fitted", neither_fit),
    ):
        print(f"AIPW ({label}): {100 * estimate:+.1f} pp")

    # A teaching interval with fixed target shares and independent binomial cells.
    total = sum(row.count for row in rows)
    variance = 0.0
    for row in rows:
        p1, p0 = empirical_offer_risk(row), empirical_withhold_risk(row)
        variance += (row.count / total) ** 2 * (
            p1 * (1 - p1) / row.offered + p0 * (1 - p0) / row.withheld
        )
    margin = 1.96 * sqrt(variance)
    print(f"Illustrative 95% CI: [{100 * (adjusted - margin):+.1f}, "
          f"{100 * (adjusted + margin):+.1f}] pp")

    assert isclose(crude, -0.21)
    assert all(isclose(value, 0.075) for value in (
        adjusted, ipw, both_fit, propensity_only, outcome_only
    ))
    assert isclose(neither_fit, crude)

    no_overlap = [IntentStratum("Low intent", 200, 40, 0, 0), rows[1]]
    try:
        standardize(no_overlap)
    except ValueError as error:
        print(f"Deterministic offer policy: {error}")
    else:
        raise AssertionError("Standardization accepted a stratum with no controls.")


if __name__ == "__main__":
    main()

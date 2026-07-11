"""Canonical EQ tier scheme, mirroring src/tiers.py's structure.

Boundaries are derived from the real proxy-EQ-score distribution on
data/train_set.csv (computed via src.eq_data.proxy_labels.compute_overall_eq_proxy
over all 16,047 rows: p10=26.15, p25=33.7, p45=41.15, p65=52.1, p85=59.8),
rounded to clean integer cutoffs -- not arbitrary.
"""

EQ_TIER_BINS = [
    (0, 26, 1, "Low EQ"),
    (27, 34, 2, "Below Average EQ"),
    (35, 41, 3, "Balanced EQ (Developing)"),
    (42, 52, 4, "Balanced EQ (Established)"),
    (53, 60, 5, "Above Average EQ"),
    (61, 99, 6, "High EQ"),
]


def assign_eq_tier(score):
    if not (0 <= score <= 99):
        raise ValueError(f"score must be within 0-99, got {score}")
    for low, high, tier_num, label in EQ_TIER_BINS:
        if low <= score <= high:
            return tier_num, label
    raise ValueError(f"score {score} did not match any tier bin")

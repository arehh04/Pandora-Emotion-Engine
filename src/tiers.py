"""Canonical Extraversion tier scheme shared by every agent tool and the orchestrator."""

TIER_BINS = [
    (0, 10, 1, "Reserved"),
    (11, 25, 2, "Reflective"),
    (26, 45, 3, "Balanced (Introspective)"),
    (46, 65, 4, "Balanced (Sociable)"),
    (66, 85, 5, "Outgoing"),
    (86, 99, 6, "Highly Extraverted"),
]


def assign_tier(score):
    """Map a 0-99 continuous Extraversion score to (tier_number, tier_label).

    Raises ValueError if score falls outside the valid 0-99 range.
    """
    rounded = round(score)
    for low, high, tier_num, label in TIER_BINS:
        if low <= rounded <= high:
            return tier_num, label
    raise ValueError(f"Score {score} is outside the expected 0-99 range")

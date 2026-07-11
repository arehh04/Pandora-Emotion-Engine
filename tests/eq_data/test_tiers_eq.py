import pytest

from src.eq_data.tiers_eq import EQ_TIER_BINS, assign_eq_tier


def test_eq_tier_bins_has_six_tiers_covering_0_to_99():
    assert len(EQ_TIER_BINS) == 6
    assert EQ_TIER_BINS[0][0] == 0
    assert EQ_TIER_BINS[-1][1] == 99
    # Bins are contiguous with no gaps or overlaps.
    for i in range(len(EQ_TIER_BINS) - 1):
        assert EQ_TIER_BINS[i][1] + 1 == EQ_TIER_BINS[i + 1][0]


def test_assign_eq_tier_matches_percentile_derived_boundaries():
    # Boundaries derived from the real proxy-score distribution on
    # data/train_set.csv (p10=26.15, p25=33.7, p45=41.15, p65=52.1, p85=59.8),
    # rounded to clean integer cutoffs.
    assert assign_eq_tier(10) == (1, "Low EQ")
    assert assign_eq_tier(26) == (1, "Low EQ")
    assert assign_eq_tier(27) == (2, "Below Average EQ")
    assert assign_eq_tier(34) == (2, "Below Average EQ")
    assert assign_eq_tier(35) == (3, "Balanced EQ (Developing)")
    assert assign_eq_tier(41) == (3, "Balanced EQ (Developing)")
    assert assign_eq_tier(42) == (4, "Balanced EQ (Established)")
    assert assign_eq_tier(52) == (4, "Balanced EQ (Established)")
    assert assign_eq_tier(53) == (5, "Above Average EQ")
    assert assign_eq_tier(60) == (5, "Above Average EQ")
    assert assign_eq_tier(61) == (6, "High EQ")
    assert assign_eq_tier(99) == (6, "High EQ")


def test_assign_eq_tier_rejects_out_of_range_score():
    with pytest.raises(ValueError):
        assign_eq_tier(100)
    with pytest.raises(ValueError):
        assign_eq_tier(-1)

import pytest

from src.tiers import assign_tier


def test_assign_tier_boundaries():
    assert assign_tier(0) == (1, "Reserved")
    assert assign_tier(10) == (1, "Reserved")
    assert assign_tier(11) == (2, "Reflective")
    assert assign_tier(25) == (2, "Reflective")
    assert assign_tier(26) == (3, "Balanced (Introspective)")
    assert assign_tier(45) == (3, "Balanced (Introspective)")
    assert assign_tier(46) == (4, "Balanced (Sociable)")
    assert assign_tier(65) == (4, "Balanced (Sociable)")
    assert assign_tier(66) == (5, "Outgoing")
    assert assign_tier(85) == (5, "Outgoing")
    assert assign_tier(86) == (6, "Highly Extraverted")
    assert assign_tier(99) == (6, "Highly Extraverted")


def test_assign_tier_accepts_float_scores():
    assert assign_tier(10.4) == (1, "Reserved")
    assert assign_tier(45.6) == (4, "Balanced (Sociable)")


def test_assign_tier_out_of_range_raises():
    with pytest.raises(ValueError):
        assign_tier(-1)
    with pytest.raises(ValueError):
        assign_tier(100)

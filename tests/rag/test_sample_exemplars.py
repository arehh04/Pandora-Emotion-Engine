import pandas as pd

from src.rag.sample_exemplars import sample_balanced_exemplars


def _make_df(n_low, n_mid, n_high):
    rows = []
    for i in range(n_low):
        rows.append({"bert_text": f"low {i}", "extraversion": 5})  # Tier 1: Reserved
    for i in range(n_mid):
        rows.append({"bert_text": f"mid {i}", "extraversion": 50})  # Tier 4: Balanced (Sociable)
    for i in range(n_high):
        rows.append({"bert_text": f"high {i}", "extraversion": 90})  # Tier 6: Highly Extraverted
    return pd.DataFrame(rows)


def test_sample_balanced_exemplars_caps_at_n_per_tier():
    df = _make_df(n_low=100, n_mid=100, n_high=100)

    result = sample_balanced_exemplars(df, n_per_tier=10, seed=1)

    assert list(result.columns) == ["bert_text", "extraversion", "tier", "tier_label"]
    counts = result["tier"].value_counts().to_dict()
    assert counts == {1: 10, 4: 10, 6: 10}
    assert set(result[result["tier"] == 1]["tier_label"]) == {"Reserved"}
    assert set(result[result["tier"] == 4]["tier_label"]) == {"Balanced (Sociable)"}
    assert set(result[result["tier"] == 6]["tier_label"]) == {"Highly Extraverted"}


def test_sample_balanced_exemplars_keeps_all_when_fewer_than_n_per_tier():
    df = _make_df(n_low=3, n_mid=100, n_high=100)

    result = sample_balanced_exemplars(df, n_per_tier=10, seed=1)

    counts = result["tier"].value_counts().to_dict()
    assert counts[1] == 3  # only 3 available, keep all of them
    assert counts[4] == 10
    assert counts[6] == 10


def test_sample_balanced_exemplars_is_deterministic_given_seed():
    df = _make_df(n_low=50, n_mid=0, n_high=0)

    result_a = sample_balanced_exemplars(df, n_per_tier=10, seed=7)
    result_b = sample_balanced_exemplars(df, n_per_tier=10, seed=7)

    assert result_a["bert_text"].tolist() == result_b["bert_text"].tolist()

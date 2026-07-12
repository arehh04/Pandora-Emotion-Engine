import pytest

from src.eq_data.nrc_enrichment import (
    NRC_BLEND_WEIGHT,
    compute_enriched_branch_eq_proxies,
    compute_enriched_overall_eq_proxy,
)
from src.eq_data.proxy_labels import compute_branch_eq_proxies, compute_overall_eq_proxy

ROW = {"extraversion": 80, "openness": 90, "agreeableness": 70, "conscientiousness": 60, "neuroticism": 20}
LEXICON = {"happy": {"positive", "joy"}, "wonderful": {"positive", "joy"}}
TEXT = "happy wonderful day"


def test_nrc_blend_weight_is_a_minority_share():
    assert 0.0 < NRC_BLEND_WEIGHT < 0.5


def test_compute_enriched_overall_eq_proxy_blends_bigfive_and_text_score():
    bigfive_score = compute_overall_eq_proxy(ROW)
    from src.eq_data.nrc_features import compute_nrc_text_score
    text_score = compute_nrc_text_score(TEXT, LEXICON)
    expected = (1 - NRC_BLEND_WEIGHT) * bigfive_score + NRC_BLEND_WEIGHT * text_score

    result = compute_enriched_overall_eq_proxy(ROW, TEXT, LEXICON)

    assert result == pytest.approx(expected)


def test_compute_enriched_overall_eq_proxy_stays_within_0_to_99():
    all_max = {"extraversion": 99, "openness": 99, "agreeableness": 99, "conscientiousness": 99, "neuroticism": 0}
    dense_positive_lexicon = {"joy": {"positive", "joy"}}

    result = compute_enriched_overall_eq_proxy(all_max, "joy joy joy joy joy", dense_positive_lexicon)

    assert 0.0 <= result <= 99.0


def test_compute_enriched_branch_eq_proxies_blends_each_branch_independently():
    bigfive_branches = compute_branch_eq_proxies(ROW)
    from src.eq_data.nrc_features import compute_nrc_text_score
    text_score = compute_nrc_text_score(TEXT, LEXICON)

    result = compute_enriched_branch_eq_proxies(ROW, TEXT, LEXICON)

    assert set(result.keys()) == set(bigfive_branches.keys())
    for branch, bigfive_value in bigfive_branches.items():
        expected = (1 - NRC_BLEND_WEIGHT) * bigfive_value + NRC_BLEND_WEIGHT * text_score
        assert result[branch] == pytest.approx(expected)


def test_compute_enriched_overall_eq_proxy_falls_back_to_bigfive_only_when_no_emotion_words():
    result = compute_enriched_overall_eq_proxy(ROW, "the quick brown fox", {})
    bigfive_only = compute_overall_eq_proxy(ROW)

    # text_score is 0.0 with an empty lexicon match, so the blend pulls
    # slightly toward 0, not equal to the pure Big Five score.
    assert result == pytest.approx((1 - NRC_BLEND_WEIGHT) * bigfive_only)

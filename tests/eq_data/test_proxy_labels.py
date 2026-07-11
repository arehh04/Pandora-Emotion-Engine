import pytest

from src.eq_data.proxy_labels import compute_branch_eq_proxies, compute_overall_eq_proxy


def test_compute_overall_eq_proxy_matches_hand_computed_value():
    row = {"extraversion": 80, "openness": 90, "agreeableness": 70, "conscientiousness": 60, "neuroticism": 20}

    result = compute_overall_eq_proxy(row)

    # 80*0.25 + 90*0.25 + 70*0.20 + 60*0.20 + (99-20)*0.10 = 20 + 22.5 + 14 + 12 + 7.9 = 76.4
    assert result == pytest.approx(76.4)


def test_compute_branch_eq_proxies_matches_hand_computed_values():
    row = {"extraversion": 80, "openness": 90, "agreeableness": 70, "conscientiousness": 60, "neuroticism": 20}

    result = compute_branch_eq_proxies(row)

    # perceiving = 90*0.6 + (99-20)*0.4 = 54 + 31.6 = 85.6
    assert result["perceiving"] == pytest.approx(85.6)
    # using = 90*0.5 + 80*0.5 = 45 + 40 = 85.0
    assert result["using"] == pytest.approx(85.0)
    # understanding = 90*0.5 + 70*0.5 = 45 + 35 = 80.0
    assert result["understanding"] == pytest.approx(80.0)
    # managing = (99-20)*0.6 + 60*0.4 = 47.4 + 24 = 71.4
    assert result["managing"] == pytest.approx(71.4)


def test_compute_overall_eq_proxy_stays_within_0_to_99_at_the_extremes():
    all_max = {"extraversion": 99, "openness": 99, "agreeableness": 99, "conscientiousness": 99, "neuroticism": 0}
    all_min = {"extraversion": 0, "openness": 0, "agreeableness": 0, "conscientiousness": 0, "neuroticism": 99}

    assert 0.0 <= compute_overall_eq_proxy(all_max) <= 99.0
    assert 0.0 <= compute_overall_eq_proxy(all_min) <= 99.0


def test_compute_branch_eq_proxies_stays_within_0_to_99_at_the_extremes():
    all_max = {"extraversion": 99, "openness": 99, "agreeableness": 99, "conscientiousness": 99, "neuroticism": 0}

    result = compute_branch_eq_proxies(all_max)

    assert all(0.0 <= v <= 99.0 for v in result.values())

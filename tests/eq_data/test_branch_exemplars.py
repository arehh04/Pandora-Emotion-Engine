import pandas as pd

from src.eq_data.branch_exemplars import sample_branch_balanced_exemplars


def test_sample_branch_balanced_exemplars_produces_one_row_per_branch_per_text():
    # 6 texts, each with distinct Big Five traits so branch/tier assignment is
    # deterministic and varied across the tier range.
    df = pd.DataFrame({
        "text": [f"sample text {i}" for i in range(6)],
        "extraversion": [5, 20, 35, 55, 75, 95],
        "openness": [10, 30, 50, 60, 80, 95],
        "agreeableness": [5, 20, 40, 60, 75, 90],
        "conscientiousness": [5, 25, 45, 60, 80, 95],
        "neuroticism": [95, 75, 55, 35, 20, 5],
    })

    result = sample_branch_balanced_exemplars(df, n_per_tier=10, seed=1)

    assert set(result["branch"].unique()) == {"perceiving", "using", "understanding", "managing"}
    assert set(result.columns) == {"text", "branch", "eq_proxy_score", "tier", "tier_label"}
    # Every (text, branch) pair should appear at most once, and the sampler
    # shouldn't duplicate rows within a branch.
    assert not result.duplicated(subset=["text", "branch"]).any()
    # With only 6 tiny source rows spread across tiers and n_per_tier=10 (far
    # above the available pool), every source row should survive per branch.
    for branch in ["perceiving", "using", "understanding", "managing"]:
        assert len(result[result["branch"] == branch]) == 6


def test_sample_branch_balanced_exemplars_is_deterministic_given_a_seed():
    df = pd.DataFrame({
        "text": [f"sample text {i}" for i in range(12)],
        "extraversion": [5, 20, 35, 55, 75, 95] * 2,
        "openness": [10, 30, 50, 60, 80, 95] * 2,
        "agreeableness": [5, 20, 40, 60, 75, 90] * 2,
        "conscientiousness": [5, 25, 45, 60, 80, 95] * 2,
        "neuroticism": [95, 75, 55, 35, 20, 5] * 2,
    })

    result_a = sample_branch_balanced_exemplars(df, n_per_tier=1, seed=7)
    result_b = sample_branch_balanced_exemplars(df, n_per_tier=1, seed=7)

    pd.testing.assert_frame_equal(
        result_a.reset_index(drop=True), result_b.reset_index(drop=True)
    )

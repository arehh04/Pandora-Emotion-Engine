"""Samples tier-balanced, branch-tagged calibration exemplars from the
Pandora dataset's proxy-labeled text, for all 4 MSC branches. Each branch is
tier-balanced independently since the same text can land in a different
tier per branch (branch proxy scores are computed from different Big Five
trait weightings, see src.eq_data.proxy_labels).
"""
import pandas as pd

from src.eq_data.proxy_labels import compute_branch_eq_proxies
from src.eq_data.tiers_eq import assign_eq_tier

BRANCHES = ["perceiving", "using", "understanding", "managing"]


def _sample_one_branch(df, branch, text_col, n_per_tier, seed):
    branch_df = df.copy()
    branch_df["eq_proxy_score"] = branch_df.apply(
        lambda row: compute_branch_eq_proxies(row)[branch], axis=1
    )
    tiers = branch_df["eq_proxy_score"].apply(assign_eq_tier)
    branch_df["tier"] = tiers.apply(lambda t: t[0])
    branch_df["tier_label"] = tiers.apply(lambda t: t[1])

    sampled_parts = []
    for tier_num in sorted(branch_df["tier"].unique()):
        tier_df = branch_df[branch_df["tier"] == tier_num]
        n = min(n_per_tier, len(tier_df))
        sampled_parts.append(tier_df.sample(n=n, random_state=seed))

    result = pd.concat(sampled_parts, ignore_index=True)
    result["branch"] = branch
    return result[[text_col, "branch", "eq_proxy_score", "tier", "tier_label"]].rename(
        columns={text_col: "text"}
    )


def sample_branch_balanced_exemplars(df, text_col="text", n_per_tier=60, seed=42):
    per_branch = [_sample_one_branch(df, branch, text_col, n_per_tier, seed) for branch in BRANCHES]
    return pd.concat(per_branch, ignore_index=True)

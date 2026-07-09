"""Samples a tier-balanced exemplar set for the RAG exemplar corpus."""
import pandas as pd

from src.tiers import assign_tier


def sample_balanced_exemplars(df, text_col="bert_text", score_col="extraversion", n_per_tier=60, seed=42):
    df = df.copy()
    tiers = df[score_col].apply(assign_tier)
    df["tier"] = tiers.apply(lambda t: t[0])
    df["tier_label"] = tiers.apply(lambda t: t[1])

    sampled_parts = []
    for tier_num in sorted(df["tier"].unique()):
        tier_df = df[df["tier"] == tier_num]
        n = min(n_per_tier, len(tier_df))
        sampled_parts.append(tier_df.sample(n=n, random_state=seed))

    result = pd.concat(sampled_parts, ignore_index=True)
    return result[[text_col, score_col, "tier", "tier_label"]]

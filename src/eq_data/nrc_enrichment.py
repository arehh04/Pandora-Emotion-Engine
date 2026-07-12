"""Blends the existing Big-Five-trait-based proxy EQ label
(src.eq_data.proxy_labels) with a text-derived NRC lexicon score
(src.eq_data.nrc_features), grounding the proxy in what the text itself
expresses, not just the person's separately-measured personality traits.

NRC_BLEND_WEIGHT is a defensible starting point (a clear minority share, so
the more-established Big Five correlation remains primary) pending
literature citation -- same citation_needed convention used throughout
src.eq_data. Does not modify src.eq_data.proxy_labels; consumes it as-is.
"""
from src.eq_data.nrc_features import compute_nrc_text_score
from src.eq_data.proxy_labels import compute_branch_eq_proxies, compute_overall_eq_proxy

NRC_BLEND_WEIGHT = 0.2


def compute_enriched_overall_eq_proxy(row, text, nrc_lexicon):
    bigfive_score = compute_overall_eq_proxy(row)
    text_score = compute_nrc_text_score(text, nrc_lexicon)
    return (1 - NRC_BLEND_WEIGHT) * bigfive_score + NRC_BLEND_WEIGHT * text_score


def compute_enriched_branch_eq_proxies(row, text, nrc_lexicon):
    bigfive_scores = compute_branch_eq_proxies(row)
    text_score = compute_nrc_text_score(text, nrc_lexicon)
    return {
        branch: (1 - NRC_BLEND_WEIGHT) * bigfive_value + NRC_BLEND_WEIGHT * text_score
        for branch, bigfive_value in bigfive_scores.items()
    }

"""Domain-agnostic information-retrieval metrics operating on a ranked list
of relevance values (not document IDs) -- the caller determines relevance
per its own domain (src.evaluation.eq_retrieval_evaluation's caller uses
tier-match). Formulas verified against sklearn.metrics.ndcg_score before
this module was written (see this plan's Task 1 docstring for the worked
example); no external dependency is added since the formulas are this
simple and self-contained.
"""
import math


def recall_at_k(relevances, total_relevant, k):
    if not total_relevant:
        return 0.0
    hits = sum(relevances[:k])
    return hits / total_relevant


def reciprocal_rank(relevances):
    for i, rel in enumerate(relevances, start=1):
        if rel > 0:
            return 1.0 / i
    return 0.0


def ndcg_at_k(relevances, k):
    top_k = relevances[:k]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(top_k))
    ideal = sorted(relevances, reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0

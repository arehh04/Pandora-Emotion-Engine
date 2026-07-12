import math

from src.evaluation.retrieval_metrics import ndcg_at_k, recall_at_k, reciprocal_rank


def test_recall_at_k_counts_hits_within_top_k_over_total_relevant():
    relevances = [1, 0, 1]  # rank 1 and rank 3 are relevant, rank 2 is not

    assert recall_at_k(relevances, total_relevant=2, k=2) == 0.5
    assert recall_at_k(relevances, total_relevant=2, k=3) == 1.0


def test_recall_at_k_returns_zero_when_nothing_is_relevant():
    assert recall_at_k([0, 0, 0], total_relevant=0, k=2) == 0.0


def test_reciprocal_rank_returns_inverse_of_first_relevant_position():
    assert reciprocal_rank([0, 1, 1]) == 0.5
    assert reciprocal_rank([1, 0, 0]) == 1.0


def test_reciprocal_rank_returns_zero_when_nothing_relevant():
    assert reciprocal_rank([0, 0]) == 0.0


def test_ndcg_at_k_matches_verified_worked_example():
    relevances = [1, 0, 1]

    result = ndcg_at_k(relevances, k=3)

    assert math.isclose(result, 0.9197207891481876, rel_tol=1e-9)


def test_ndcg_at_k_returns_zero_when_nothing_relevant():
    assert ndcg_at_k([0, 0], k=2) == 0.0


def test_ndcg_at_k_returns_one_for_perfectly_ordered_relevance():
    assert ndcg_at_k([1, 1, 0], k=3) == 1.0

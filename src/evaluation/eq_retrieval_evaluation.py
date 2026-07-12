"""Reranking ablation over an EQ RAG exemplar table: retrieves the same
fetch_k candidate pool once via LanceDB's native hybrid search, then
compares Recall@k/MRR/nDCG@k between the plain hybrid order and a
cross-encoder-reranked order of that same pool -- isolating what reranking
changes, independent of absolute recall over the whole table.

Relevance is defined per-query as "the candidate's own tier metadata
matches the query's own known tier" -- the same tier-similarity signal
src.eq_agent.branch_config's retrieve_similar_eq_exemplars tool already
surfaces to specialist LLMs for calibration. No labeled query/relevant-doc
test set exists for this corpus (see this plan's Global Constraints), so
this evaluates reranking's effect on ordering a real candidate pool rather
than absolute recall against a ground-truth corpus-wide relevant set.
"""
from src.evaluation.retrieval_metrics import ndcg_at_k, recall_at_k, reciprocal_rank


def _hybrid_candidate_pool(table, query_text, fetch_k):
    if table.count_rows() == 0:
        return []
    df = table.search(query_text, query_type="hybrid").limit(fetch_k).to_pandas()
    return df.to_dict("records")


def _reranked_candidate_pool(table, query_text, fetch_k, reranker):
    if table.count_rows() == 0:
        return []
    df = (
        table.search(query_text, query_type="hybrid")
        .limit(fetch_k)
        .rerank(reranker=reranker)
        .limit(fetch_k)
        .to_pandas()
    )
    return df.to_dict("records")


def _relevances_for_query(candidates, query_tier):
    return [1 if c.get("tier") == query_tier else 0 for c in candidates]


def _average_metrics(rows, k_values):
    if not rows:
        return {}
    n = len(rows)
    result = {"mrr": sum(r["mrr"] for r in rows) / n}
    for k in k_values:
        result[f"recall@{k}"] = sum(r[f"recall@{k}"] for r in rows) / n
        result[f"ndcg@{k}"] = sum(r[f"ndcg@{k}"] for r in rows) / n
    return result


def evaluate_branch_reranking_ablation(exemplar_table, queries, reranker, fetch_k=50, k_values=(3, 5)):
    per_condition = {"hybrid_only": [], "hybrid_reranked": []}

    for query_text, query_tier in queries:
        hybrid_candidates = _hybrid_candidate_pool(exemplar_table, query_text, fetch_k)
        if not hybrid_candidates:
            continue
        reranked_candidates = _reranked_candidate_pool(exemplar_table, query_text, fetch_k, reranker)

        hybrid_relevances = _relevances_for_query(hybrid_candidates, query_tier)
        reranked_relevances = _relevances_for_query(reranked_candidates, query_tier)
        total_relevant = sum(hybrid_relevances)  # same underlying candidate pool for both conditions

        for condition, relevances in [("hybrid_only", hybrid_relevances), ("hybrid_reranked", reranked_relevances)]:
            row = {}
            for k in k_values:
                row[f"recall@{k}"] = recall_at_k(relevances, total_relevant, k)
                row[f"ndcg@{k}"] = ndcg_at_k(relevances, k)
            row["mrr"] = reciprocal_rank(relevances)
            per_condition[condition].append(row)

    return {condition: _average_metrics(rows, k_values) for condition, rows in per_condition.items()}

import lancedb
from lancedb.rerankers import CrossEncoderReranker

from src.eq_data.build_eq_corpus import _build_table, _make_exemplar_schema
from src.evaluation.eq_retrieval_evaluation import evaluate_branch_reranking_ablation
from tests.eq_data.lancedb_test_helpers import make_fake_embedding_func


def _make_table(tmp_path):
    vector_by_text = {
        "I felt completely overwhelmed and could not calm down": [1.0, 0.0],
        "I noticed my coworker was upset before they said anything": [0.0, 1.0],
        "The weather was sunny today": [1.0, 1.0],
        "I could tell my friend was feeling sad even though they said nothing": [0.0, 1.0],
    }
    embedding_func = make_fake_embedding_func(vector_by_text)
    db = lancedb.connect(str(tmp_path / "lancedb"))
    schema = _make_exemplar_schema(embedding_func)
    return _build_table(db, "eq_exemplars_perceiving", schema, [
        {"text": "I felt completely overwhelmed and could not calm down", "tier": 2},
        {"text": "I noticed my coworker was upset before they said anything", "tier": 5},
        {"text": "The weather was sunny today", "tier": 3},
    ])


def test_evaluate_branch_reranking_ablation_returns_metrics_for_both_conditions(tmp_path):
    table = _make_table(tmp_path)
    queries = [("I could tell my friend was feeling sad even though they said nothing", 5)]

    result = evaluate_branch_reranking_ablation(table, queries, CrossEncoderReranker(), fetch_k=3, k_values=(2, 3))

    assert set(result.keys()) == {"hybrid_only", "hybrid_reranked"}
    for condition in result.values():
        assert set(condition.keys()) == {"mrr", "recall@2", "recall@3", "ndcg@2", "ndcg@3"}


def test_evaluate_branch_reranking_ablation_handles_no_relevant_candidates(tmp_path):
    table = _make_table(tmp_path)
    queries = [("The weather was sunny today", 99)]  # tier 99 matches nothing in the table

    result = evaluate_branch_reranking_ablation(table, queries, CrossEncoderReranker(), fetch_k=3, k_values=(2,))

    assert result["hybrid_only"]["recall@2"] == 0.0
    assert result["hybrid_only"]["mrr"] == 0.0


def test_evaluate_branch_reranking_ablation_returns_empty_dicts_for_empty_table(tmp_path):
    embedding_func = make_fake_embedding_func({"anything": [1.0, 0.0]})
    db = lancedb.connect(str(tmp_path / "lancedb"))
    empty_table = _build_table(db, "eq_exemplars_using", _make_exemplar_schema(embedding_func), [])

    result = evaluate_branch_reranking_ablation(empty_table, [("anything", 1)], CrossEncoderReranker(), fetch_k=3)

    assert result == {"hybrid_only": {}, "hybrid_reranked": {}}

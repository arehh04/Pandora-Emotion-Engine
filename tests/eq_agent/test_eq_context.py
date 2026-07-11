import lancedb

from src.eq_agent.eq_context import load_eq_rag_context
from src.eq_data.branch_exemplars import BRANCHES
from src.eq_data.build_eq_corpus import (
    _build_table,
    _make_exemplar_schema,
    _make_theory_schema,
    exemplar_table_name,
    theory_table_name,
)
from tests.eq_data.lancedb_test_helpers import make_fake_embedding_func


def test_load_eq_rag_context_returns_none_when_persist_dir_missing(tmp_path):
    result = load_eq_rag_context(str(tmp_path / "does_not_exist"))

    assert result is None


def test_load_eq_rag_context_returns_none_when_a_table_is_missing(tmp_path):
    persist_dir = str(tmp_path / "lancedb")
    embedding_func = make_fake_embedding_func({"x": [1.0, 0.0]})
    db = lancedb.connect(persist_dir)
    # Only populate perceiving's tables -- the other 3 branches are missing entirely.
    _build_table(db, theory_table_name("perceiving"), _make_theory_schema(embedding_func),
                 [{"text": "x", "id": "a", "topic": "t", "citation_needed": "yes"}])
    _build_table(db, exemplar_table_name("perceiving"), _make_exemplar_schema(embedding_func),
                 [{"text": "x", "tier": 4, "tier_label": "l", "eq_proxy_score": 40.0}])

    result = load_eq_rag_context(persist_dir, embedding_func=embedding_func)

    assert result is None


def test_load_eq_rag_context_returns_none_when_a_table_is_empty(tmp_path):
    persist_dir = str(tmp_path / "lancedb")
    embedding_func = make_fake_embedding_func({"x": [1.0, 0.0]})
    db = lancedb.connect(persist_dir)
    for branch in BRANCHES:
        rows = [{"text": "x", "id": "a", "topic": "t", "citation_needed": "yes"}] if branch != "managing" else []
        _build_table(db, theory_table_name(branch), _make_theory_schema(embedding_func), rows)
        _build_table(db, exemplar_table_name(branch), _make_exemplar_schema(embedding_func),
                     [{"text": "x", "tier": 4, "tier_label": "l", "eq_proxy_score": 40.0}])

    result = load_eq_rag_context(persist_dir, embedding_func=embedding_func)

    assert result is None  # managing's theory table is empty (create_table with no rows, no FTS index)


def test_load_eq_rag_context_loads_all_eight_populated_tables(tmp_path):
    persist_dir = str(tmp_path / "lancedb")
    embedding_func = make_fake_embedding_func({"x": [1.0, 0.0]})
    db = lancedb.connect(persist_dir)
    for branch in BRANCHES:
        _build_table(db, theory_table_name(branch), _make_theory_schema(embedding_func),
                     [{"text": "x", "id": "a", "topic": "t", "citation_needed": "yes"}])
        _build_table(db, exemplar_table_name(branch), _make_exemplar_schema(embedding_func),
                     [{"text": "x", "tier": 4, "tier_label": "l", "eq_proxy_score": 40.0}])

    result = load_eq_rag_context(persist_dir, embedding_func=embedding_func, reranker=None)

    assert result is not None
    assert set(result["theory_tables"].keys()) == set(BRANCHES)
    assert set(result["exemplar_tables"].keys()) == set(BRANCHES)
    for branch in BRANCHES:
        assert result["theory_tables"][branch].count_rows() == 1
        assert result["exemplar_tables"][branch].count_rows() == 1
    assert result["reranker"] is None

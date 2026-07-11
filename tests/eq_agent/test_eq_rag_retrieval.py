from src.eq_agent.eq_rag_retrieval import retrieve_relevant_msc_theory, retrieve_similar_eq_exemplars
from src.eq_data.build_eq_corpus import _build_table, _make_exemplar_schema, _make_theory_schema
from tests.eq_data.lancedb_test_helpers import make_fake_embedding_func


def test_retrieve_similar_eq_exemplars_scopes_to_the_given_branch(tmp_path):
    import lancedb

    vector_by_text = {
        "I noticed my friend seemed upset": [1.0, 0.0],
        "I stayed calm under pressure": [0.0, 1.0],
        "query about noticing emotions": [1.0, 0.0],
    }
    embedding_func = make_fake_embedding_func(vector_by_text)
    db = lancedb.connect(str(tmp_path / "lancedb"))
    exemplar_schema = _make_exemplar_schema(embedding_func)

    perceiving_table = _build_table(db, "eq_exemplars_perceiving", exemplar_schema, [
        {"text": "I noticed my friend seemed upset", "tier": 5, "tier_label": "Above Average EQ"},
    ])
    managing_table = _build_table(db, "eq_exemplars_managing", exemplar_schema, [
        {"text": "I stayed calm under pressure", "tier": 6, "tier_label": "High EQ"},
    ])

    rag_ctx = {
        "exemplar_tables": {"perceiving": perceiving_table, "managing": managing_table},
        "theory_tables": {}, "reranker": None,
    }

    hits = retrieve_similar_eq_exemplars("query about noticing emotions", "perceiving", rag_ctx, k=1)

    assert len(hits) == 1
    assert hits[0]["text"] == "I noticed my friend seemed upset"
    assert hits[0]["tier_label"] == "Above Average EQ"


def test_retrieve_relevant_msc_theory_scopes_to_the_given_branch(tmp_path):
    import lancedb

    vector_by_text = {
        "perceiving emotions theory chunk": [1.0, 0.0],
        "managing emotions theory chunk": [0.0, 1.0],
        "query about theory": [1.0, 0.0],
    }
    embedding_func = make_fake_embedding_func(vector_by_text)
    db = lancedb.connect(str(tmp_path / "lancedb"))
    theory_schema = _make_theory_schema(embedding_func)

    perceiving_table = _build_table(db, "eq_theory_perceiving", theory_schema, [
        {"text": "perceiving emotions theory chunk", "id": "p1", "topic": "t", "citation_needed": "yes"},
    ])
    managing_table = _build_table(db, "eq_theory_managing", theory_schema, [
        {"text": "managing emotions theory chunk", "id": "m1", "topic": "t", "citation_needed": "yes"},
    ])

    rag_ctx = {
        "theory_tables": {"perceiving": perceiving_table, "managing": managing_table},
        "exemplar_tables": {}, "reranker": None,
    }

    hits = retrieve_relevant_msc_theory("query about theory", "perceiving", rag_ctx, k=1)

    assert len(hits) == 1
    assert hits[0]["id"] == "p1"


def test_retrieve_returns_empty_list_for_empty_table(tmp_path):
    import lancedb

    embedding_func = make_fake_embedding_func({"anything": [1.0, 0.0]})
    db = lancedb.connect(str(tmp_path / "lancedb"))
    empty_table = _build_table(db, "eq_exemplars_using", _make_exemplar_schema(embedding_func), [])

    rag_ctx = {"exemplar_tables": {"using": empty_table}, "theory_tables": {}, "reranker": None}

    hits = retrieve_similar_eq_exemplars("anything", "using", rag_ctx, k=5)

    assert hits == []

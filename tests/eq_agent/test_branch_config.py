import lancedb

from src.eq_agent.branch_config import build_branch_configs
from src.eq_agent.graph import BRANCHES
from src.eq_data.build_eq_corpus import _build_table, _make_exemplar_schema, _make_theory_schema
from tests.eq_data.lancedb_test_helpers import make_fake_embedding_func


def _build_rag_ctx(tmp_path):
    embedding_func = make_fake_embedding_func({
        f"{branch} exemplar text": [1.0, 0.0] for branch in BRANCHES
    } | {f"{branch} theory chunk": [1.0, 0.0] for branch in BRANCHES} | {"query": [1.0, 0.0]})
    db = lancedb.connect(str(tmp_path / "lancedb"))
    theory_schema = _make_theory_schema(embedding_func)
    exemplar_schema = _make_exemplar_schema(embedding_func)

    theory_tables, exemplar_tables = {}, {}
    for branch in BRANCHES:
        theory_tables[branch] = _build_table(db, f"eq_theory_{branch}", theory_schema, [
            {"text": f"{branch} theory chunk", "id": f"{branch}-1", "topic": "t", "citation_needed": "yes"},
        ])
        exemplar_tables[branch] = _build_table(db, f"eq_exemplars_{branch}", exemplar_schema, [
            {"text": f"{branch} exemplar text", "tier": 4, "tier_label": "Balanced EQ (Established)", "eq_proxy_score": 45.0},
        ])
    return {"theory_tables": theory_tables, "exemplar_tables": exemplar_tables, "reranker": None}


def test_build_branch_configs_covers_all_four_branches_with_two_tools_each():
    configs = build_branch_configs()

    assert set(configs.keys()) == set(BRANCHES)
    for branch, config in configs.items():
        assert {"tool_schemas", "dispatch_fn", "system_prompt"} <= config.keys()
        tool_names = {s["function"]["name"] for s in config["tool_schemas"]}
        assert tool_names == {"retrieve_similar_eq_exemplars", "retrieve_relevant_msc_theory"}
        assert branch in config["system_prompt"].lower()


def test_dispatch_fn_returns_error_when_eq_rag_absent():
    configs = build_branch_configs()
    ctx = {"eq_rag": None}

    result = configs["perceiving"]["dispatch_fn"]("retrieve_similar_eq_exemplars", {"text": "hello"}, ctx)

    assert "error" in result


def test_dispatch_fn_returns_results_scoped_to_its_own_branch(tmp_path):
    configs = build_branch_configs()
    ctx = {"eq_rag": _build_rag_ctx(tmp_path)}

    result = configs["perceiving"]["dispatch_fn"]("retrieve_similar_eq_exemplars", {"text": "query", "k": 1}, ctx)

    assert "results" in result
    assert result["results"][0]["text"] == "perceiving exemplar text"


def test_dispatch_fn_never_raises_on_unknown_tool():
    configs = build_branch_configs()
    ctx = {"eq_rag": None}

    result = configs["managing"]["dispatch_fn"]("not_a_real_tool", {}, ctx)

    assert "error" in result


def test_dispatch_fn_never_raises_on_bad_arguments(tmp_path):
    configs = build_branch_configs()
    ctx = {"eq_rag": _build_rag_ctx(tmp_path)}

    result = configs["using"]["dispatch_fn"]("retrieve_relevant_msc_theory", {}, ctx)  # missing required "text"

    assert "error" in result

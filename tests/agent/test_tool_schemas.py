import numpy as np

from src.agent.tool_schemas import TOOL_SCHEMAS, dispatch_tool_call
from src.rag.build_corpus import EXEMPLARS_COLLECTION, THEORY_COLLECTION
from src.rag.hybrid_store import build_hybrid_collection


class FakeEmbedder:
    def encode(self, texts):
        return np.array([[1.0, 0.0] for _ in texts])


def _build_rag_ctx(tmp_path):
    embedder = FakeEmbedder()
    persist_dir = str(tmp_path / "chroma")
    exemplars_collection = build_hybrid_collection(
        persist_dir, EXEMPLARS_COLLECTION,
        [{"text": "I love parties", "metadata": {"tier": 6, "tier_label": "Highly Extraverted", "extraversion": 90.0}}],
        embedder,
    )
    theory_collection = build_hybrid_collection(
        persist_dir, THEORY_COLLECTION,
        [{"text": "gregariousness", "metadata": {"id": "a", "topic": "t", "citation_needed": "n/a"}}],
        embedder,
    )
    return {"exemplars_collection": exemplars_collection, "theory_collection": theory_collection, "embedder": embedder}


def test_tool_schemas_have_three_entries_with_required_names():
    names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
    assert names == {"retrieve_similar_exemplars", "retrieve_relevant_theory", "submit_assessment"}
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]


def test_dispatch_rag_tools_return_error_when_corpus_absent():
    ctx = {"rag": None}

    exemplar_result = dispatch_tool_call("retrieve_similar_exemplars", {"text": "hello"}, ctx)
    theory_result = dispatch_tool_call("retrieve_relevant_theory", {"text": "hello"}, ctx)

    assert "error" in exemplar_result
    assert "error" in theory_result


def test_dispatch_rag_tools_return_results_when_corpus_present(tmp_path):
    ctx = {"rag": _build_rag_ctx(tmp_path)}

    result = dispatch_tool_call("retrieve_similar_exemplars", {"text": "party time", "k": 1}, ctx)

    assert "results" in result
    assert result["results"][0]["bert_text"] == "I love parties"


def test_dispatch_unknown_tool_returns_error():
    ctx = {"rag": None}

    result = dispatch_tool_call("not_a_real_tool", {}, ctx)

    assert "error" in result


def test_dispatch_never_raises_on_bad_arguments():
    ctx = {"rag": None}

    result = dispatch_tool_call("retrieve_similar_exemplars", {}, ctx)  # missing required "text"

    assert "error" in result

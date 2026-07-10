import numpy as np

from src.agent.context import load_rag_context
from src.rag.build_corpus import EXEMPLARS_COLLECTION, THEORY_COLLECTION
from src.rag.hybrid_store import build_hybrid_collection


class FakeEmbedder:
    def encode(self, texts):
        return np.array([[float(len(t)), 0.0] for t in texts])


def test_load_rag_context_returns_none_when_persist_dir_missing(tmp_path):
    result = load_rag_context(str(tmp_path / "does_not_exist"), embedder=FakeEmbedder())

    assert result is None


def test_load_rag_context_returns_none_when_collections_are_empty(tmp_path):
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()

    result = load_rag_context(str(persist_dir), embedder=FakeEmbedder())

    assert result is None


def test_load_rag_context_loads_populated_collections(tmp_path):
    persist_dir = str(tmp_path / "chroma")
    embedder = FakeEmbedder()
    build_hybrid_collection(
        persist_dir, EXEMPLARS_COLLECTION,
        [{"text": "I love parties", "metadata": {"tier": 6, "tier_label": "Highly Extraverted", "extraversion": 90.0}}],
        embedder,
    )
    build_hybrid_collection(
        persist_dir, THEORY_COLLECTION,
        [{"text": "gregariousness", "metadata": {"id": "a", "topic": "t", "citation_needed": "n/a"}}],
        embedder,
    )

    result = load_rag_context(persist_dir, embedder=embedder)

    assert result is not None
    assert result["exemplars_collection"].count() == 1
    assert result["theory_collection"].count() == 1
    assert result["embedder"] is embedder

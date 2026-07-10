import numpy as np

from src.agent.tools.rag_retrieval import retrieve_relevant_theory, retrieve_similar_exemplars
from src.rag.hybrid_store import build_hybrid_collection


class FakeEmbedder:
    """Deterministic stand-in for SentenceTransformer.encode()."""

    def __init__(self, vector_by_text):
        self.vector_by_text = vector_by_text

    def encode(self, texts):
        return np.array([self.vector_by_text[t] for t in texts])


def test_retrieve_similar_exemplars_returns_nearest_with_metadata(tmp_path):
    vector_by_text = {
        "I love parties": [1.0, 0.0],
        "I stayed home reading alone": [0.0, 1.0],
        "We had a huge group gathering": [0.9, 0.1],
        "I'm at a party!": [1.0, 0.0],
    }
    embedder = FakeEmbedder(vector_by_text)
    records = [
        {"text": "I love parties", "metadata": {"extraversion": 90.0, "tier": 6, "tier_label": "Highly Extraverted"}},
        {"text": "I stayed home reading alone", "metadata": {"extraversion": 5.0, "tier": 1, "tier_label": "Reserved"}},
        {"text": "We had a huge group gathering", "metadata": {"extraversion": 80.0, "tier": 5, "tier_label": "Outgoing"}},
    ]
    collection = build_hybrid_collection(str(tmp_path / "chroma"), "test_exemplars", records, embedder)
    rag_ctx = {"exemplars_collection": collection, "theory_collection": collection, "embedder": embedder}

    hits = retrieve_similar_exemplars("I'm at a party!", rag_ctx, k=1)

    assert len(hits) == 1
    assert hits[0]["bert_text"] == "I love parties"
    assert hits[0]["tier_label"] == "Highly Extraverted"
    assert "score" in hits[0]


def test_retrieve_relevant_theory_returns_nearest_entry(tmp_path):
    vector_by_text = {
        "gregariousness theory": [1.0, 0.0],
        "introversion theory": [0.0, 1.0],
        "why am I so quiet": [0.0, 1.0],
    }
    embedder = FakeEmbedder(vector_by_text)
    records = [
        {"text": "gregariousness theory", "metadata": {"id": "a", "topic": "t1", "citation_needed": "n/a"}},
        {"text": "introversion theory", "metadata": {"id": "b", "topic": "t2", "citation_needed": "n/a"}},
    ]
    collection = build_hybrid_collection(str(tmp_path / "chroma"), "test_theory", records, embedder)
    rag_ctx = {"exemplars_collection": collection, "theory_collection": collection, "embedder": embedder}

    hits = retrieve_relevant_theory("why am I so quiet", rag_ctx, k=1)

    assert len(hits) == 1
    assert hits[0]["id"] == "b"
    assert "score" in hits[0]

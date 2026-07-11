import numpy as np

from src.rag.hybrid_store import build_hybrid_collection, load_hybrid_collection, hybrid_search


class FakeEmbedder:
    def __init__(self, vector_by_text):
        self.vector_by_text = vector_by_text

    def encode(self, texts):
        return np.array([self.vector_by_text[t] for t in texts])


class ConstantEmbedder:
    """Returns the same vector for any text -- used where only chunk count
    and metadata matter, not embedding-driven ranking."""

    def encode(self, texts):
        return np.array([[1.0, 0.0] for _ in texts])


def test_build_hybrid_collection_chunks_and_stores_each_record(tmp_path):
    records = [{"text": "x" * 1000, "metadata": {"label": "long"}}]
    embedder = ConstantEmbedder()

    collection = build_hybrid_collection(
        str(tmp_path / "chroma"), "test_chunking", records, embedder, chunk_size=300, overlap=50
    )

    assert collection.count() == 4
    stored = collection.get()
    assert all(item["label"] == "long" for item in stored["metadatas"])
    assert {item["chunk_index"] for item in stored["metadatas"]} == {0, 1, 2, 3}


def test_load_hybrid_collection_reopens_a_previously_built_collection(tmp_path):
    persist_dir = str(tmp_path / "chroma")
    embedder = FakeEmbedder({"hello world": [1.0, 0.0]})
    build_hybrid_collection(persist_dir, "test_reload", [{"text": "hello world", "metadata": {}}], embedder)

    reopened = load_hybrid_collection(persist_dir, "test_reload")

    assert reopened.count() == 1


def test_load_hybrid_collection_returns_empty_collection_when_never_built(tmp_path):
    collection = load_hybrid_collection(str(tmp_path / "chroma"), "never_built")

    assert collection.count() == 0


def test_hybrid_search_returns_empty_list_for_empty_collection(tmp_path):
    embedder = FakeEmbedder({"query": [1.0, 0.0]})
    collection = load_hybrid_collection(str(tmp_path / "chroma"), "empty")

    assert hybrid_search("query", collection, embedder, k=5) == []


def test_hybrid_search_fuses_dense_and_bm25_rankings_via_rrf(tmp_path):
    # Verified ground-truth fixture: three docs where one is the best match on
    # both dense and BM25 signals, one is dense-only, one is BM25-only. RRF
    # (rrf_k=60) over full dense+BM25 rankings of all 3 docs produces the
    # order [both, dense_only, bm25_only] with the last two tied and broken
    # by dense-list insertion order.
    vector_by_text = {
        "apple banana cherry": [1.0, 0.0],
        "unrelated words here now": [0.99, 0.01],
        "apple banana extra padding words": [-1.0, 0.0],
        "apple banana": [1.0, 0.0],
    }
    embedder = FakeEmbedder(vector_by_text)
    records = [
        {"text": "apple banana cherry", "metadata": {"label": "both"}},
        {"text": "unrelated words here now", "metadata": {"label": "dense_only"}},
        {"text": "apple banana extra padding words", "metadata": {"label": "bm25_only"}},
    ]
    collection = build_hybrid_collection(str(tmp_path / "chroma"), "test_rrf", records, embedder)

    hits = hybrid_search("apple banana", collection, embedder, k=3)

    assert [h["metadata"]["label"] for h in hits] == ["both", "dense_only", "bm25_only"]
    assert hits[0]["score"] > hits[1]["score"] == hits[2]["score"]


def test_hybrid_search_respects_k(tmp_path):
    vector_by_text = {
        "apple banana cherry": [1.0, 0.0],
        "unrelated words here now": [0.99, 0.01],
        "apple banana extra padding words": [-1.0, 0.0],
        "apple banana": [1.0, 0.0],
    }
    embedder = FakeEmbedder(vector_by_text)
    records = [
        {"text": "apple banana cherry", "metadata": {"label": "both"}},
        {"text": "unrelated words here now", "metadata": {"label": "dense_only"}},
        {"text": "apple banana extra padding words", "metadata": {"label": "bm25_only"}},
    ]
    collection = build_hybrid_collection(str(tmp_path / "chroma"), "test_rrf_k", records, embedder)

    hits = hybrid_search("apple banana", collection, embedder, k=1)

    assert len(hits) == 1
    assert hits[0]["metadata"]["label"] == "both"

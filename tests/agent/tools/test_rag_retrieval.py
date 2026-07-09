import json

import numpy as np
import pandas as pd

from src.agent.tools.rag_retrieval import (
    cosine_similarity_topk,
    load_rag_corpus,
    retrieve_similar_exemplars,
    retrieve_relevant_theory,
)


class FakeEmbedder:
    """Deterministic stand-in for SentenceTransformer.encode()."""

    def __init__(self, vector_by_text):
        self.vector_by_text = vector_by_text

    def encode(self, texts):
        return np.array([self.vector_by_text[t] for t in texts])


def _write_fixture_corpus(rag_dir):
    exemplars_df = pd.DataFrame({
        "bert_text": ["I love parties", "I stayed home reading alone", "We had a huge group gathering"],
        "extraversion": [90, 5, 80],
        "tier": [6, 1, 5],
        "tier_label": ["Highly Extraverted", "Reserved", "Outgoing"],
    })
    exemplars_df.to_csv(rag_dir / "exemplars_meta.csv", index=False)

    # Orthogonal-ish 2D vectors so nearest-neighbor is unambiguous.
    exemplar_embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    np.save(rag_dir / "exemplars_embeddings.npy", exemplar_embeddings)

    theory_entries = [
        {"id": "a", "topic": "t1", "text": "gregariousness theory", "citation_needed": "n/a"},
        {"id": "b", "topic": "t2", "text": "introversion theory", "citation_needed": "n/a"},
    ]
    (rag_dir / "theory_meta.json").write_text(json.dumps(theory_entries), encoding="utf-8")
    theory_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    np.save(rag_dir / "theory_embeddings.npy", theory_embeddings)


def test_cosine_similarity_topk_ranks_closest_vector_first():
    query = np.array([1.0, 0.0])
    corpus = np.array([[0.0, 1.0], [1.0, 0.0], [0.9, 0.1]])

    hits = cosine_similarity_topk(query, corpus, k=2)

    assert hits[0][0] == 1  # exact match, index 1
    assert hits[1][0] == 2  # near match, index 2
    assert hits[0][1] > hits[1][1] > 0


def test_load_rag_corpus_reads_all_four_artifacts(tmp_path):
    _write_fixture_corpus(tmp_path)

    corpus = load_rag_corpus(str(tmp_path))

    assert len(corpus["exemplars_df"]) == 3
    assert corpus["exemplar_embeddings"].shape == (3, 2)
    assert len(corpus["theory_entries"]) == 2
    assert corpus["theory_embeddings"].shape == (2, 2)


def test_retrieve_similar_exemplars_returns_nearest_with_metadata(tmp_path):
    _write_fixture_corpus(tmp_path)
    corpus = load_rag_corpus(str(tmp_path))
    embedder = FakeEmbedder({"I'm at a party!": np.array([1.0, 0.0])})

    hits = retrieve_similar_exemplars("I'm at a party!", corpus, embedder, k=1)

    assert len(hits) == 1
    assert hits[0]["bert_text"] == "I love parties"
    assert hits[0]["tier_label"] == "Highly Extraverted"
    assert "similarity" in hits[0]


def test_retrieve_relevant_theory_returns_nearest_entry(tmp_path):
    _write_fixture_corpus(tmp_path)
    corpus = load_rag_corpus(str(tmp_path))
    embedder = FakeEmbedder({"why am I so quiet": np.array([0.0, 1.0])})

    hits = retrieve_relevant_theory("why am I so quiet", corpus, embedder, k=1)

    assert len(hits) == 1
    assert hits[0]["id"] == "b"
    assert "similarity" in hits[0]

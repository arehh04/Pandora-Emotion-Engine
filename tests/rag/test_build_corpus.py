import json

import numpy as np
import pandas as pd

from src.rag.build_corpus import build_rag_corpus, embed_corpus


class FakeEmbedder:
    """Deterministic stand-in for SentenceTransformer.encode() — avoids downloading
    a real model in the fast unit test suite."""

    def encode(self, texts):
        return np.array([[float(len(t)), float(i)] for i, t in enumerate(texts)])


def test_embed_corpus_returns_array_matching_input_length():
    embedder = FakeEmbedder()

    result = embed_corpus(["a", "bb", "ccc"], embedder)

    assert result.shape == (3, 2)


def test_embed_corpus_handles_empty_list():
    embedder = FakeEmbedder()

    result = embed_corpus([], embedder)

    assert result.shape == (0, 0)


def test_build_rag_corpus_produces_aligned_exemplars_and_theory(tmp_path):
    data_dir = tmp_path
    rag_dir = data_dir / "rag"
    rag_dir.mkdir()

    # Minimal train_clean.csv covering all 6 tiers so the sampler has something in each.
    scores = [5, 20, 35, 55, 75, 95] * 5
    pd.DataFrame({
        "bert_text": [f"sample text {i}" for i in range(len(scores))],
        "extraversion": scores,
    }).to_csv(data_dir / "train_clean.csv", index=False)

    theory_entries = [
        {"id": "a", "topic": "t1", "text": "theory chunk one", "citation_needed": "n/a"},
        {"id": "b", "topic": "t2", "text": "theory chunk two", "citation_needed": "n/a"},
    ]
    (rag_dir / "theory_corpus.json").write_text(json.dumps(theory_entries), encoding="utf-8")

    embedder = FakeEmbedder()

    exemplars_df, exemplar_embeddings, loaded_theory, theory_embeddings = build_rag_corpus(
        str(data_dir), embedder, n_per_tier=3, seed=1
    )

    assert len(exemplars_df) == exemplar_embeddings.shape[0]
    assert len(loaded_theory) == 2
    assert theory_embeddings.shape == (2, 2)
    assert set(exemplars_df["tier"].unique()) <= {1, 2, 3, 4, 5, 6}

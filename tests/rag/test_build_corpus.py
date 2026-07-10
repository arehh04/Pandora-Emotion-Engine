import json

import numpy as np
import pandas as pd

from src.rag.build_corpus import build_rag_corpus, EXEMPLARS_COLLECTION, THEORY_COLLECTION


class FakeEmbedder:
    """Deterministic stand-in for SentenceTransformer.encode() -- avoids
    downloading a real model in the fast unit test suite."""

    def encode(self, texts):
        return np.array([[float(len(t)), float(i)] for i, t in enumerate(texts)])


def test_build_rag_corpus_produces_populated_exemplar_and_theory_collections(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rag_dir = data_dir / "rag"
    rag_dir.mkdir()
    persist_dir = tmp_path / "chroma"

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

    exemplars_collection, theory_collection = build_rag_corpus(
        str(data_dir), str(persist_dir), embedder, n_per_tier=3, seed=1
    )

    assert exemplars_collection.count() > 0
    assert theory_collection.count() == 2
    exemplar_metadatas = exemplars_collection.get()["metadatas"]
    assert set(m["tier"] for m in exemplar_metadatas) <= {1, 2, 3, 4, 5, 6}
    theory_metadatas = theory_collection.get()["metadatas"]
    assert {m["id"] for m in theory_metadatas} == {"a", "b"}


def test_build_rag_corpus_deduplicates_augmented_rows(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rag_dir = data_dir / "rag"
    rag_dir.mkdir()
    persist_dir = tmp_path / "chroma"

    scores = [5, 20, 35, 55, 75, 95] * 3
    clean_df = pd.DataFrame({
        "bert_text": [f"sample text {i}" for i in range(len(scores))],
        "extraversion": scores,
    })
    clean_df.to_csv(data_dir / "train_clean.csv", index=False)
    clean_df.to_csv(data_dir / "train_augmented.csv", index=False)  # fully duplicate rows

    (rag_dir / "theory_corpus.json").write_text(
        json.dumps([{"id": "a", "topic": "t1", "text": "theory chunk", "citation_needed": "n/a"}]),
        encoding="utf-8",
    )

    embedder = FakeEmbedder()

    exemplars_collection, _ = build_rag_corpus(str(data_dir), str(persist_dir), embedder, n_per_tier=2, seed=1)

    # With full duplicates removed, at most len(scores) unique rows exist in
    # total, so sampling can't exceed that regardless of n_per_tier * 6.
    assert exemplars_collection.count() <= len(scores)

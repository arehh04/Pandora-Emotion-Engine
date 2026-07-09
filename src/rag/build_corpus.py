"""Builds the RAG knowledge base: tier-balanced exemplar embeddings + theory-corpus embeddings.

Run directly to (re)generate the on-disk artifacts under data/rag/:
    python -m src.rag.build_corpus
"""
import json
import os

import numpy as np
import pandas as pd

from src.rag.sample_exemplars import sample_balanced_exemplars
from src.rag.theory_corpus import load_theory_corpus


def embed_corpus(texts, embedder):
    """Embed a list of strings using any object exposing .encode(list[str])."""
    if not texts:
        return np.zeros((0, 0))
    return np.asarray(embedder.encode(texts))


def build_rag_corpus(data_dir, embedder, n_per_tier=60, seed=42):
    clean_df = pd.read_csv(os.path.join(data_dir, "train_clean.csv"))

    augmented_path = os.path.join(data_dir, "train_augmented.csv")
    if os.path.exists(augmented_path):
        aug_df = pd.read_csv(augmented_path)
        combined = pd.concat(
            [clean_df[["bert_text", "extraversion"]], aug_df[["bert_text", "extraversion"]]],
            ignore_index=True,
        )
    else:
        combined = clean_df[["bert_text", "extraversion"]]

    exemplars_df = sample_balanced_exemplars(combined, n_per_tier=n_per_tier, seed=seed)
    exemplar_embeddings = embed_corpus(exemplars_df["bert_text"].tolist(), embedder)

    theory_path = os.path.join(data_dir, "rag", "theory_corpus.json")
    theory_entries = load_theory_corpus(theory_path)
    theory_texts = [entry["text"] for entry in theory_entries]
    theory_embeddings = embed_corpus(theory_texts, embedder)

    return exemplars_df, exemplar_embeddings, theory_entries, theory_embeddings


def main():
    from sentence_transformers import SentenceTransformer

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    rag_dir = os.path.join(data_dir, "rag")
    os.makedirs(rag_dir, exist_ok=True)

    print("Loading sentence-transformers/all-MiniLM-L6-v2...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    exemplars_df, exemplar_embeddings, theory_entries, theory_embeddings = build_rag_corpus(data_dir, embedder)

    exemplars_df.to_csv(os.path.join(rag_dir, "exemplars_meta.csv"), index=False)
    np.save(os.path.join(rag_dir, "exemplars_embeddings.npy"), exemplar_embeddings)

    with open(os.path.join(rag_dir, "theory_meta.json"), "w", encoding="utf-8") as f:
        json.dump(theory_entries, f, indent=2)
    np.save(os.path.join(rag_dir, "theory_embeddings.npy"), theory_embeddings)

    print(f"Saved {len(exemplars_df)} exemplars -> exemplars_meta.csv / exemplars_embeddings.npy")
    print(f"Saved {len(theory_entries)} theory chunks -> theory_meta.json / theory_embeddings.npy")


if __name__ == "__main__":
    main()

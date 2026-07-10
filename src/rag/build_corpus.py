"""Builds the RAG knowledge base: a tier-balanced exemplar collection and a
theory-corpus collection, both chunked (900/200 by default) and stored as
hybrid (dense + BM25) ChromaDB collections under data/rag/chroma/.

Run directly to (re)generate them:
    python -m src.rag.build_corpus
"""
import os

import pandas as pd

from src.rag.hybrid_store import build_hybrid_collection
from src.rag.sample_exemplars import sample_balanced_exemplars
from src.rag.theory_corpus import load_theory_corpus

EXEMPLARS_COLLECTION = "extraversion_exemplars"
THEORY_COLLECTION = "extraversion_theory"


def _build_exemplar_records(data_dir, n_per_tier=60, seed=42):
    clean_df = pd.read_csv(os.path.join(data_dir, "train_clean.csv"))
    augmented_path = os.path.join(data_dir, "train_augmented.csv")
    if os.path.exists(augmented_path):
        aug_df = pd.read_csv(augmented_path)
        combined = pd.concat(
            [clean_df[["bert_text", "extraversion"]], aug_df[["bert_text", "extraversion"]]], ignore_index=True
        )
        combined = combined.drop_duplicates(subset=["bert_text"]).reset_index(drop=True)
    else:
        combined = clean_df[["bert_text", "extraversion"]]

    exemplars_df = sample_balanced_exemplars(combined, n_per_tier=n_per_tier, seed=seed)
    return [
        {
            "text": row["bert_text"],
            "metadata": {
                "extraversion": float(row["extraversion"]),
                "tier": int(row["tier"]),
                "tier_label": row["tier_label"],
            },
        }
        for _, row in exemplars_df.iterrows()
    ]


def _build_theory_records(data_dir):
    theory_entries = load_theory_corpus(os.path.join(data_dir, "rag", "theory_corpus.json"))
    return [
        {
            "text": entry["text"],
            "metadata": {"id": entry["id"], "topic": entry["topic"], "citation_needed": entry["citation_needed"]},
        }
        for entry in theory_entries
    ]


def build_rag_corpus(data_dir, persist_dir, embedder, n_per_tier=60, seed=42, chunk_size=900, overlap=200):
    exemplar_records = _build_exemplar_records(data_dir, n_per_tier=n_per_tier, seed=seed)
    theory_records = _build_theory_records(data_dir)

    exemplars_collection = build_hybrid_collection(
        persist_dir, EXEMPLARS_COLLECTION, exemplar_records, embedder, chunk_size=chunk_size, overlap=overlap
    )
    theory_collection = build_hybrid_collection(
        persist_dir, THEORY_COLLECTION, theory_records, embedder, chunk_size=chunk_size, overlap=overlap
    )
    return exemplars_collection, theory_collection


def main():
    from sentence_transformers import SentenceTransformer

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data")
    persist_dir = os.path.join(data_dir, "rag", "chroma")
    os.makedirs(persist_dir, exist_ok=True)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    exemplars_collection, theory_collection = build_rag_corpus(data_dir, persist_dir, embedder)
    print(f"Exemplars collection: {exemplars_collection.count()} chunks")
    print(f"Theory collection: {theory_collection.count()} chunks")


if __name__ == "__main__":
    main()

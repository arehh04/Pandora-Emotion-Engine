"""In-memory cosine-similarity retrieval over Plan 1's RAG corpus artifacts.

No FAISS/Chroma — the corpus is a few hundred exemplars plus ~17 theory
chunks, small enough that a plain NumPy top-k is simple and fast enough.
"""
import json
import os

import numpy as np
import pandas as pd


def cosine_similarity_topk(query_vec, corpus_vecs, k=5):
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    corpus_norms = corpus_vecs / (np.linalg.norm(corpus_vecs, axis=1, keepdims=True) + 1e-10)
    scores = corpus_norms @ query_norm
    top_idx = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in top_idx]


def load_rag_corpus(rag_dir):
    exemplars_df = pd.read_csv(os.path.join(rag_dir, "exemplars_meta.csv"))
    exemplar_embeddings = np.load(os.path.join(rag_dir, "exemplars_embeddings.npy"))
    with open(os.path.join(rag_dir, "theory_meta.json"), "r", encoding="utf-8") as f:
        theory_entries = json.load(f)
    theory_embeddings = np.load(os.path.join(rag_dir, "theory_embeddings.npy"))
    return {
        "exemplars_df": exemplars_df,
        "exemplar_embeddings": exemplar_embeddings,
        "theory_entries": theory_entries,
        "theory_embeddings": theory_embeddings,
    }


def retrieve_similar_exemplars(query_text, corpus, embedder, k=5):
    query_vec = np.asarray(embedder.encode([query_text])[0])
    hits = cosine_similarity_topk(query_vec, corpus["exemplar_embeddings"], k=k)
    return [
        {**corpus["exemplars_df"].iloc[idx].to_dict(), "similarity": score}
        for idx, score in hits
    ]


def retrieve_relevant_theory(query_text, corpus, embedder, k=3):
    query_vec = np.asarray(embedder.encode([query_text])[0])
    hits = cosine_similarity_topk(query_vec, corpus["theory_embeddings"], k=k)
    return [
        {**corpus["theory_entries"][idx], "similarity": score}
        for idx, score in hits
    ]

"""Builds the shared resources (a trained ML-prior model, an optional RAG
corpus) the orchestrator needs, once per process — not per request.
"""
import os

import pandas as pd

from src.agent.tools.classical_features import extract_features_for_text
from src.agent.tools.ml_prior import train_ml_prior
from src.agent.tools.rag_retrieval import load_rag_corpus

REQUIRED_RAG_FILES = [
    "exemplars_meta.csv", "exemplars_embeddings.npy", "theory_meta.json", "theory_embeddings.npy",
]


def train_ml_prior_from_data(data_path, nlp, nrc_dict, sample_size=300, seed=42):
    df = pd.read_csv(data_path)
    sample = df.sample(n=min(sample_size, len(df)), random_state=seed)
    feature_rows = [extract_features_for_text(str(text), nlp, nrc_dict) for text in sample["bert_text"]]
    scores = sample["extraversion"].tolist()
    return train_ml_prior(feature_rows, scores)


def load_rag_context(rag_dir, embedder=None):
    if not all(os.path.exists(os.path.join(rag_dir, f)) for f in REQUIRED_RAG_FILES):
        return None

    corpus = load_rag_corpus(rag_dir)

    if embedder is None:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")

    return {"corpus": corpus, "embedder": embedder}

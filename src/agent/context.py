"""Builds the shared RAG resources the orchestrator needs, once per process
-- not per request.
"""
import os

from src.rag.build_corpus import EXEMPLARS_COLLECTION, THEORY_COLLECTION
from src.rag.hybrid_store import load_hybrid_collection


def load_rag_context(persist_dir, embedder=None):
    if not os.path.isdir(persist_dir):
        return None

    exemplars_collection = load_hybrid_collection(persist_dir, EXEMPLARS_COLLECTION)
    theory_collection = load_hybrid_collection(persist_dir, THEORY_COLLECTION)
    if exemplars_collection.count() == 0 or theory_collection.count() == 0:
        return None

    if embedder is None:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")

    return {"exemplars_collection": exemplars_collection, "theory_collection": theory_collection, "embedder": embedder}

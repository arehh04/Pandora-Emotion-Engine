"""Hybrid (dense + BM25, RRF-fused) retrieval over the RAG corpus stored in
the ChromaDB collections built by src.rag.build_corpus.
"""
from src.rag.hybrid_store import hybrid_search


def retrieve_similar_exemplars(query_text, rag_ctx, k=5):
    hits = hybrid_search(query_text, rag_ctx["exemplars_collection"], rag_ctx["embedder"], k=k)
    return [{**hit["metadata"], "bert_text": hit["document"], "score": hit["score"]} for hit in hits]


def retrieve_relevant_theory(query_text, rag_ctx, k=3):
    hits = hybrid_search(query_text, rag_ctx["theory_collection"], rag_ctx["embedder"], k=k)
    return [{**hit["metadata"], "text": hit["document"], "score": hit["score"]} for hit in hits]

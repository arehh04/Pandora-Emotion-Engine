"""Hybrid (dense + BM25) retrieval over a ChromaDB collection, fused with
Reciprocal Rank Fusion. Records are chunked before storage so long source
texts don't get truncated or averaged away by the embedder.
"""
import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

from src.rag.chunking import chunk_text


def _tokenize(text):
    return text.lower().split()


def build_hybrid_collection(persist_dir, collection_name, records, embedder, chunk_size=900, overlap=200):
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    ids, documents, metadatas = [], [], []
    for i, record in enumerate(records):
        chunks = chunk_text(record["text"], chunk_size=chunk_size, overlap=overlap)
        for j, chunk in enumerate(chunks):
            ids.append(f"{i}_{j}")
            documents.append(chunk)
            metadatas.append({**record.get("metadata", {}), "source_index": i, "chunk_index": j})

    if documents:
        embeddings = embedder.encode(documents)
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=[list(map(float, e)) for e in embeddings],
            metadatas=metadatas,
        )
    return collection


def load_hybrid_collection(persist_dir, collection_name):
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})


def hybrid_search(query_text, collection, embedder, k=5, rrf_k=60):
    count = collection.count()
    if count == 0:
        return []

    query_embedding = embedder.encode([query_text])[0]
    dense_results = collection.query(query_embeddings=[list(map(float, query_embedding))], n_results=count)
    dense_ids = dense_results["ids"][0]

    all_items = collection.get()
    all_ids = all_items["ids"]
    doc_by_id = dict(zip(all_ids, all_items["documents"]))
    meta_by_id = dict(zip(all_ids, all_items["metadatas"]))

    tokenized_corpus = [_tokenize(doc) for doc in all_items["documents"]]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(_tokenize(query_text))
    bm25_ranked = [all_ids[i] for i in np.argsort(-bm25_scores)]

    rrf_scores = {}
    for rank, doc_id in enumerate(dense_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, doc_id in enumerate(bm25_ranked):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)

    fused_ids = sorted(rrf_scores.keys(), key=lambda doc_id: -rrf_scores[doc_id])[:k]
    return [
        {"id": doc_id, "document": doc_by_id[doc_id], "metadata": meta_by_id[doc_id], "score": rrf_scores[doc_id]}
        for doc_id in fused_ids
    ]

# Hybrid RAG (ChromaDB + BM25) + LLM-Only Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the numpy cosine-similarity RAG backend with a real hybrid (dense + BM25, Reciprocal-Rank-Fusion) retrieval system backed by ChromaDB, with 900/200 chunking — and remove the fuzzy-logic and ML-prior tools from the agent entirely, leaving the agent with exactly two capabilities: LLM reasoning and RAG retrieval.

**Architecture:** A new `src/rag/chunking.py` provides a plain sliding-window `chunk_text`. A new `src/rag/hybrid_store.py` builds/loads ChromaDB `PersistentClient` collections (one for exemplars, one for theory), chunking each record's text before embedding and storing it, and performs hybrid search by combining a full dense ranking (ChromaDB's cosine `.query()`) with a full BM25 ranking (`rank_bm25.BM25Okapi` over every document in the collection) via Reciprocal Rank Fusion. `src/rag/build_corpus.py` and `src/agent/tools/rag_retrieval.py` are rewired onto this new backend. `src/agent/context.py`, `src/agent/tool_schemas.py`, `src/agent/orchestrator.py`, `backend/agent_router.py`, and `src/evaluation/run_comparison.py` are then simplified to drop every trace of the fuzzy-logic and ML-prior tools, since the agent's context no longer needs spaCy, the NRC lexicon, or a trained Ridge model at all — only a RAG context (two ChromaDB collections + an embedder) or `None`.

**Tech Stack:** `chromadb` 1.5.9 (`PersistentClient`, cosine `hnsw:space`), `rank-bm25` 0.2.2 (`BM25Okapi`), existing `sentence-transformers`/`pandas`/`pytest` stack.

## Global Constraints

- Chunking uses `chunk_size=900, overlap=200` (exact values requested) as the default for all corpus building.
- Hybrid retrieval = a full dense ranking (ChromaDB cosine distance) fused with a full BM25 ranking (`rank_bm25.BM25Okapi`) via Reciprocal Rank Fusion: `score[doc] += 1 / (rrf_k + rank + 1)` per ranking list the doc appears in, `rrf_k=60` default. Both rankings cover the entire collection (no partial-k dense over-fetch heuristics) — simple and exactly matches this plan's verified test fixtures.
- ChromaDB collections are `PersistentClient(path=...)` stores under `data/rag/chroma/`, created with `metadata={"hnsw:space": "cosine"}`.
- The agent's active toolset after this plan is exactly `retrieve_similar_exemplars`, `retrieve_relevant_theory`, and the terminal `submit_assessment` — `fuzzy_logic_assessment` and `ml_prior_assessment` are removed, not merely disabled.
- `src/agent/tools/fuzzy_engine.py`, `src/agent/tools/ml_prior.py`, and `src/agent/tools/classical_features.py` (the agent's private copy — NOT `src/extract_classical_features.py`, which is a separate, still-used module for the old classical-ML prediction pipeline in `backend/main.py`) are deleted outright, along with their test files, once nothing in the agent module references them.
- `HISTORICAL_BASELINES` in `src/evaluation/run_comparison.py` (ridge/xgboost/random_forest RMSE/R2 numbers) stays unchanged — those are recorded comparison figures for the evaluation harness, not a live traditional-ML tool the agent calls.
- No new test may make a real network call, real LLM call, or download a real embedding model — inject fake embedders (an object with an `.encode(texts) -> array-like`) exactly as the existing test suite already does.
- After every task, run the full suite with `./.venv/Scripts/python.exe -m pytest -q` and confirm no regressions outside the files that task intentionally changed.

---

### Task 1: Chunking utility

**Files:**
- Create: `src/rag/chunking.py`
- Test: `tests/rag/test_chunking.py`

**Interfaces:**
- Produces: `chunk_text(text, chunk_size=900, overlap=200) -> list[str]` — used by Task 2's `build_hybrid_collection`.

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from src.rag.chunking import chunk_text


def test_chunk_text_splits_long_text_into_overlapping_windows():
    text = "x" * 1000

    chunks = chunk_text(text, chunk_size=300, overlap=50)

    assert [len(c) for c in chunks] == [300, 300, 300, 250]


def test_chunk_text_returns_single_chunk_when_text_shorter_than_chunk_size():
    text = "short text"

    chunks = chunk_text(text, chunk_size=900, overlap=200)

    assert chunks == ["short text"]


def test_chunk_text_returns_empty_list_for_empty_text():
    assert chunk_text("", chunk_size=900, overlap=200) == []


def test_chunk_text_raises_when_overlap_is_not_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=100, overlap=100)


def test_chunk_text_default_arguments_are_900_and_200():
    text = "y" * 1000

    chunks = chunk_text(text)

    assert len(chunks) == 2
    assert len(chunks[0]) == 900
    assert len(chunks[1]) == 300
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/rag/test_chunking.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.rag.chunking'`

- [ ] **Step 3: Write the implementation**

```python
"""Plain sliding-window text chunking for the RAG corpus builder."""


def chunk_text(text, chunk_size=900, overlap=200):
    if not text:
        return []

    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        chunks.append(text[start:start + chunk_size])
        if start + chunk_size >= n:
            break
        start += step
    return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/rag/test_chunking.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/rag/chunking.py tests/rag/test_chunking.py
git commit -m "feat: add sliding-window chunk_text utility for the RAG corpus"
```

---

### Task 2: Hybrid ChromaDB + BM25 store with RRF fusion

**Files:**
- Create: `src/rag/hybrid_store.py`
- Modify: `requirements.txt` (add `chromadb==1.5.9` and `rank-bm25==0.2.2`)
- Test: `tests/rag/test_hybrid_store.py`

**Interfaces:**
- Consumes: `chunk_text(text, chunk_size, overlap)` from Task 1.
- Produces:
  - `build_hybrid_collection(persist_dir, collection_name, records, embedder, chunk_size=900, overlap=200) -> Collection` — `records` is `list[{"text": str, "metadata": dict}]`.
  - `load_hybrid_collection(persist_dir, collection_name) -> Collection`
  - `hybrid_search(query_text, collection, embedder, k=5, rrf_k=60) -> list[{"id": str, "document": str, "metadata": dict, "score": float}]`, ranked best-first.
  These three are consumed by Task 3's rewire of `rag_retrieval.py` and `build_corpus.py`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from src.rag.hybrid_store import build_hybrid_collection, load_hybrid_collection, hybrid_search


class FakeEmbedder:
    def __init__(self, vector_by_text):
        self.vector_by_text = vector_by_text

    def encode(self, texts):
        return np.array([self.vector_by_text[t] for t in texts])


class ConstantEmbedder:
    """Returns the same vector for any text -- used where only chunk count
    and metadata matter, not embedding-driven ranking."""

    def encode(self, texts):
        return np.array([[1.0, 0.0] for _ in texts])


def test_build_hybrid_collection_chunks_and_stores_each_record(tmp_path):
    records = [{"text": "x" * 1000, "metadata": {"label": "long"}}]
    embedder = ConstantEmbedder()

    collection = build_hybrid_collection(
        str(tmp_path / "chroma"), "test_chunking", records, embedder, chunk_size=300, overlap=50
    )

    assert collection.count() == 4
    stored = collection.get()
    assert all(item["label"] == "long" for item in stored["metadatas"])
    assert {item["chunk_index"] for item in stored["metadatas"]} == {0, 1, 2, 3}


def test_load_hybrid_collection_reopens_a_previously_built_collection(tmp_path):
    persist_dir = str(tmp_path / "chroma")
    embedder = FakeEmbedder({"hello world": [1.0, 0.0]})
    build_hybrid_collection(persist_dir, "test_reload", [{"text": "hello world", "metadata": {}}], embedder)

    reopened = load_hybrid_collection(persist_dir, "test_reload")

    assert reopened.count() == 1


def test_load_hybrid_collection_returns_empty_collection_when_never_built(tmp_path):
    collection = load_hybrid_collection(str(tmp_path / "chroma"), "never_built")

    assert collection.count() == 0


def test_hybrid_search_returns_empty_list_for_empty_collection(tmp_path):
    embedder = FakeEmbedder({"query": [1.0, 0.0]})
    collection = load_hybrid_collection(str(tmp_path / "chroma"), "empty")

    assert hybrid_search("query", collection, embedder, k=5) == []


def test_hybrid_search_fuses_dense_and_bm25_rankings_via_rrf(tmp_path):
    # Verified ground-truth fixture: three docs where one is the best match on
    # both dense and BM25 signals, one is dense-only, one is BM25-only. RRF
    # (rrf_k=60) over full dense+BM25 rankings of all 3 docs produces the
    # order [both, dense_only, bm25_only] with the last two tied and broken
    # by dense-list insertion order.
    vector_by_text = {
        "apple banana cherry": [1.0, 0.0],
        "unrelated words here now": [0.99, 0.01],
        "apple banana extra padding words": [-1.0, 0.0],
        "apple banana": [1.0, 0.0],
    }
    embedder = FakeEmbedder(vector_by_text)
    records = [
        {"text": "apple banana cherry", "metadata": {"label": "both"}},
        {"text": "unrelated words here now", "metadata": {"label": "dense_only"}},
        {"text": "apple banana extra padding words", "metadata": {"label": "bm25_only"}},
    ]
    collection = build_hybrid_collection(str(tmp_path / "chroma"), "test_rrf", records, embedder)

    hits = hybrid_search("apple banana", collection, embedder, k=3)

    assert [h["metadata"]["label"] for h in hits] == ["both", "dense_only", "bm25_only"]
    assert hits[0]["score"] > hits[1]["score"] == hits[2]["score"]


def test_hybrid_search_respects_k(tmp_path):
    vector_by_text = {
        "apple banana cherry": [1.0, 0.0],
        "unrelated words here now": [0.99, 0.01],
        "apple banana extra padding words": [-1.0, 0.0],
        "apple banana": [1.0, 0.0],
    }
    embedder = FakeEmbedder(vector_by_text)
    records = [
        {"text": "apple banana cherry", "metadata": {"label": "both"}},
        {"text": "unrelated words here now", "metadata": {"label": "dense_only"}},
        {"text": "apple banana extra padding words", "metadata": {"label": "bm25_only"}},
    ]
    collection = build_hybrid_collection(str(tmp_path / "chroma"), "test_rrf_k", records, embedder)

    hits = hybrid_search("apple banana", collection, embedder, k=1)

    assert len(hits) == 1
    assert hits[0]["metadata"]["label"] == "both"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/rag/test_hybrid_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.rag.hybrid_store'`

- [ ] **Step 3: Add dependencies to requirements.txt**

Add these two lines at the end of `requirements.txt`:

```
chromadb==1.5.9
rank-bm25==0.2.2
```

(Both are already installed in `.venv` from earlier verification; this just records them for reproducible installs.)

- [ ] **Step 4: Write the implementation**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/rag/test_hybrid_store.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add src/rag/hybrid_store.py requirements.txt tests/rag/test_hybrid_store.py
git commit -m "feat: add ChromaDB+BM25 hybrid store with RRF fusion"
```

---

### Task 3: Rewire the corpus builder and RAG tool onto the hybrid backend

**Files:**
- Modify: `src/rag/build_corpus.py` (full rewrite)
- Modify: `src/agent/tools/rag_retrieval.py` (full rewrite)
- Modify: `src/agent/context.py` (full rewrite)
- Test: `tests/rag/test_build_corpus.py` (full rewrite)
- Test: `tests/agent/tools/test_rag_retrieval.py` (full rewrite)
- Test: `tests/agent/test_context.py` (full rewrite)

**Interfaces:**
- Consumes: `build_hybrid_collection`, `load_hybrid_collection`, `hybrid_search` from Task 2; `sample_balanced_exemplars(df, text_col, score_col, n_per_tier, seed)` from `src/rag/sample_exemplars.py` (unchanged); `load_theory_corpus(path)` from `src/rag/theory_corpus.py` (unchanged).
- Produces:
  - `src.rag.build_corpus.EXEMPLARS_COLLECTION = "extraversion_exemplars"`, `THEORY_COLLECTION = "extraversion_theory"` — collection names other tasks/modules reference.
  - `build_rag_corpus(data_dir, persist_dir, embedder, n_per_tier=60, seed=42, chunk_size=900, overlap=200) -> (exemplars_collection, theory_collection)`
  - `src.agent.tools.rag_retrieval.retrieve_similar_exemplars(query_text, rag_ctx, k=5) -> list[dict]` and `retrieve_relevant_theory(query_text, rag_ctx, k=3) -> list[dict]`, where `rag_ctx = {"exemplars_collection": Collection, "theory_collection": Collection, "embedder": embedder}` — consumed by Task 4's `tool_schemas.py`.
  - `src.agent.context.load_rag_context(persist_dir, embedder=None) -> rag_ctx | None` — consumed by Task 4's `backend/agent_router.py`. Returns `None` if `persist_dir` doesn't exist yet, or if either collection is empty (corpus not built).

- [ ] **Step 1: Write the failing tests for `build_corpus.py`**

Replace the entire contents of `tests/rag/test_build_corpus.py`:

```python
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
```

- [ ] **Step 2: Write the failing tests for `rag_retrieval.py`**

Replace the entire contents of `tests/agent/tools/test_rag_retrieval.py`:

```python
import numpy as np

from src.agent.tools.rag_retrieval import retrieve_relevant_theory, retrieve_similar_exemplars
from src.rag.hybrid_store import build_hybrid_collection


class FakeEmbedder:
    """Deterministic stand-in for SentenceTransformer.encode()."""

    def __init__(self, vector_by_text):
        self.vector_by_text = vector_by_text

    def encode(self, texts):
        return np.array([self.vector_by_text[t] for t in texts])


def test_retrieve_similar_exemplars_returns_nearest_with_metadata(tmp_path):
    vector_by_text = {
        "I love parties": [1.0, 0.0],
        "I stayed home reading alone": [0.0, 1.0],
        "We had a huge group gathering": [0.9, 0.1],
        "I'm at a party!": [1.0, 0.0],
    }
    embedder = FakeEmbedder(vector_by_text)
    records = [
        {"text": "I love parties", "metadata": {"extraversion": 90.0, "tier": 6, "tier_label": "Highly Extraverted"}},
        {"text": "I stayed home reading alone", "metadata": {"extraversion": 5.0, "tier": 1, "tier_label": "Reserved"}},
        {"text": "We had a huge group gathering", "metadata": {"extraversion": 80.0, "tier": 5, "tier_label": "Outgoing"}},
    ]
    collection = build_hybrid_collection(str(tmp_path / "chroma"), "test_exemplars", records, embedder)
    rag_ctx = {"exemplars_collection": collection, "theory_collection": collection, "embedder": embedder}

    hits = retrieve_similar_exemplars("I'm at a party!", rag_ctx, k=1)

    assert len(hits) == 1
    assert hits[0]["bert_text"] == "I love parties"
    assert hits[0]["tier_label"] == "Highly Extraverted"
    assert "score" in hits[0]


def test_retrieve_relevant_theory_returns_nearest_entry(tmp_path):
    vector_by_text = {
        "gregariousness theory": [1.0, 0.0],
        "introversion theory": [0.0, 1.0],
        "why am I so quiet": [0.0, 1.0],
    }
    embedder = FakeEmbedder(vector_by_text)
    records = [
        {"text": "gregariousness theory", "metadata": {"id": "a", "topic": "t1", "citation_needed": "n/a"}},
        {"text": "introversion theory", "metadata": {"id": "b", "topic": "t2", "citation_needed": "n/a"}},
    ]
    collection = build_hybrid_collection(str(tmp_path / "chroma"), "test_theory", records, embedder)
    rag_ctx = {"exemplars_collection": collection, "theory_collection": collection, "embedder": embedder}

    hits = retrieve_relevant_theory("why am I so quiet", rag_ctx, k=1)

    assert len(hits) == 1
    assert hits[0]["id"] == "b"
    assert "score" in hits[0]
```

- [ ] **Step 3: Write the failing tests for `context.py`**

Replace the entire contents of `tests/agent/test_context.py`:

```python
import numpy as np

from src.agent.context import load_rag_context
from src.rag.build_corpus import EXEMPLARS_COLLECTION, THEORY_COLLECTION
from src.rag.hybrid_store import build_hybrid_collection


class FakeEmbedder:
    def encode(self, texts):
        return np.array([[float(len(t)), 0.0] for t in texts])


def test_load_rag_context_returns_none_when_persist_dir_missing(tmp_path):
    result = load_rag_context(str(tmp_path / "does_not_exist"), embedder=FakeEmbedder())

    assert result is None


def test_load_rag_context_returns_none_when_collections_are_empty(tmp_path):
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()

    result = load_rag_context(str(persist_dir), embedder=FakeEmbedder())

    assert result is None


def test_load_rag_context_loads_populated_collections(tmp_path):
    persist_dir = str(tmp_path / "chroma")
    embedder = FakeEmbedder()
    build_hybrid_collection(
        persist_dir, EXEMPLARS_COLLECTION,
        [{"text": "I love parties", "metadata": {"tier": 6, "tier_label": "Highly Extraverted", "extraversion": 90.0}}],
        embedder,
    )
    build_hybrid_collection(
        persist_dir, THEORY_COLLECTION,
        [{"text": "gregariousness", "metadata": {"id": "a", "topic": "t", "citation_needed": "n/a"}}],
        embedder,
    )

    result = load_rag_context(persist_dir, embedder=embedder)

    assert result is not None
    assert result["exemplars_collection"].count() == 1
    assert result["theory_collection"].count() == 1
    assert result["embedder"] is embedder
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/rag/test_build_corpus.py tests/agent/tools/test_rag_retrieval.py tests/agent/test_context.py -v`
Expected: FAIL — `build_rag_corpus`/`retrieve_similar_exemplars`/`retrieve_relevant_theory`/`load_rag_context` still have the old numpy-based signatures.

- [ ] **Step 5: Rewrite `src/rag/build_corpus.py`**

Replace the entire file:

```python
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
```

- [ ] **Step 6: Rewrite `src/agent/tools/rag_retrieval.py`**

Replace the entire file:

```python
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
```

- [ ] **Step 7: Rewrite `src/agent/context.py`**

Replace the entire file:

```python
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
```

- [ ] **Step 8: Delete the obsolete numpy-based RAG artifact files (if present)**

```bash
rm -f data/rag/exemplars_meta.csv data/rag/exemplars_embeddings.npy data/rag/theory_meta.json data/rag/theory_embeddings.npy
```

(These were the old on-disk format; the new corpus lives in `data/rag/chroma/`, rebuilt via `python -m src.rag.build_corpus` after this plan merges. If none of these files exist yet, this step is a no-op.)

- [ ] **Step 9: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/rag/test_build_corpus.py tests/agent/tools/test_rag_retrieval.py tests/agent/test_context.py -v`
Expected: all pass

- [ ] **Step 10: Commit**

```bash
git add src/rag/build_corpus.py src/agent/tools/rag_retrieval.py src/agent/context.py \
        tests/rag/test_build_corpus.py tests/agent/tools/test_rag_retrieval.py tests/agent/test_context.py
git commit -m "feat: rewire RAG corpus builder and retrieval onto the hybrid ChromaDB+BM25 backend"
```

---

### Task 4: Remove fuzzy-logic and ML-prior tools from the agent

**Files:**
- Modify: `src/agent/tool_schemas.py` (full rewrite)
- Modify: `src/agent/orchestrator.py` (full rewrite)
- Modify: `backend/agent_router.py` (full rewrite)
- Modify: `src/evaluation/run_comparison.py:15-20` (update `ABLATION_VARIANTS`)
- Modify: `src/evaluation/run_real_evaluation.py` (drop ml/spacy setup)
- Modify: `frontend/src/routes/+page.svelte:32-55` (drop fuzzy/ml-prior UI labels)
- Delete: `src/agent/tools/fuzzy_engine.py`, `src/agent/tools/ml_prior.py`, `src/agent/tools/classical_features.py`
- Delete: `tests/agent/tools/test_fuzzy_engine.py`, `tests/agent/tools/test_ml_prior.py`, `tests/agent/tools/test_classical_features.py`
- Test: `tests/agent/test_tool_schemas.py` (full rewrite)
- Test: `tests/agent/test_orchestrator.py` (full rewrite)
- Test: `tests/backend/test_agent_router.py` (full rewrite)
- Test: `tests/evaluation/test_run_comparison.py` (update fixtures/assertions)

**Interfaces:**
- Consumes: `retrieve_similar_exemplars`, `retrieve_relevant_theory` from Task 3's `rag_retrieval.py`; `load_rag_context` from Task 3's `context.py`.
- Produces: agent `ctx` is now always exactly `{"rag": rag_ctx_or_None}` — no `nlp`, `nrc_dict`, or `ml_model` keys anywhere in the agent/backend/evaluation code paths.

- [ ] **Step 1: Confirm nothing else references the modules being deleted**

Run: `./.venv/Scripts/python.exe -c "import subprocess; print(subprocess.run(['git', 'grep', '-l', '-E', 'fuzzy_engine|ml_prior|agent.tools.classical_features'], capture_output=True, text=True).stdout)"`

Expected output: only the files this task is already modifying or deleting (`tool_schemas.py`, `orchestrator.py`, `agent_router.py`, the three test files being deleted, `test_orchestrator.py`, `test_agent_router.py`, `test_tool_schemas.py`, and this plan document). If any other file appears, stop and report it before proceeding — do not delete a module something else still needs.

- [ ] **Step 2: Rewrite `src/agent/tool_schemas.py`**

Replace the entire file:

```python
"""OpenAI-style function-calling tool schemas for the agent, and a resilient
dispatcher that routes tool calls to the RAG retrieval functions.

submit_assessment is intentionally NOT dispatched here -- it's the agent's
terminal action, handled directly by the orchestrator loop.
"""
from src.agent.tools.rag_retrieval import retrieve_relevant_theory, retrieve_similar_exemplars

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_similar_exemplars",
            "description": "Retrieve the most similar labeled example texts from the training corpus, for calibration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to find similar examples for."},
                    "k": {"type": "integer", "description": "How many examples to retrieve.", "default": 5},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_relevant_theory",
            "description": "Retrieve relevant Extraversion/Big Five psychology theory chunks to ground the reasoning.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to find relevant theory for."},
                    "k": {"type": "integer", "description": "How many theory chunks to retrieve.", "default": 3},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_assessment",
            "description": "Submit the final Extraversion assessment. Call this exactly once, when you are done reasoning.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tier": {"type": "integer", "description": "Extraversion tier, 1 (most reserved) to 6 (most extraverted)."},
                    "continuous_score_estimate": {"type": "number", "description": "Estimated score, 0-99."},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                    "rationale": {"type": "string", "description": "Brief explanation citing the tool evidence used."},
                },
                "required": ["tier", "continuous_score_estimate", "confidence", "rationale"],
            },
        },
    },
]


def dispatch_tool_call(name, arguments, ctx):
    try:
        if name == "retrieve_similar_exemplars":
            if not ctx.get("rag"):
                return {"error": "RAG corpus is not available (not built yet)."}
            k = arguments.get("k", 5)
            return {"results": retrieve_similar_exemplars(arguments["text"], ctx["rag"], k=k)}

        if name == "retrieve_relevant_theory":
            if not ctx.get("rag"):
                return {"error": "RAG corpus is not available (not built yet)."}
            k = arguments.get("k", 3)
            return {"results": retrieve_relevant_theory(arguments["text"], ctx["rag"], k=k)}

        return {"error": f"Unknown tool: {name}"}
    except Exception as e:
        return {"error": str(e)}
```

- [ ] **Step 3: Rewrite `tests/agent/test_tool_schemas.py`**

Replace the entire file:

```python
import numpy as np

from src.agent.tool_schemas import TOOL_SCHEMAS, dispatch_tool_call
from src.rag.build_corpus import EXEMPLARS_COLLECTION, THEORY_COLLECTION
from src.rag.hybrid_store import build_hybrid_collection


class FakeEmbedder:
    def encode(self, texts):
        return np.array([[1.0, 0.0] for _ in texts])


def _build_rag_ctx(tmp_path):
    embedder = FakeEmbedder()
    persist_dir = str(tmp_path / "chroma")
    exemplars_collection = build_hybrid_collection(
        persist_dir, EXEMPLARS_COLLECTION,
        [{"text": "I love parties", "metadata": {"tier": 6, "tier_label": "Highly Extraverted", "extraversion": 90.0}}],
        embedder,
    )
    theory_collection = build_hybrid_collection(
        persist_dir, THEORY_COLLECTION,
        [{"text": "gregariousness", "metadata": {"id": "a", "topic": "t", "citation_needed": "n/a"}}],
        embedder,
    )
    return {"exemplars_collection": exemplars_collection, "theory_collection": theory_collection, "embedder": embedder}


def test_tool_schemas_have_three_entries_with_required_names():
    names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
    assert names == {"retrieve_similar_exemplars", "retrieve_relevant_theory", "submit_assessment"}
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]


def test_dispatch_rag_tools_return_error_when_corpus_absent():
    ctx = {"rag": None}

    exemplar_result = dispatch_tool_call("retrieve_similar_exemplars", {"text": "hello"}, ctx)
    theory_result = dispatch_tool_call("retrieve_relevant_theory", {"text": "hello"}, ctx)

    assert "error" in exemplar_result
    assert "error" in theory_result


def test_dispatch_rag_tools_return_results_when_corpus_present(tmp_path):
    ctx = {"rag": _build_rag_ctx(tmp_path)}

    result = dispatch_tool_call("retrieve_similar_exemplars", {"text": "party time", "k": 1}, ctx)

    assert "results" in result
    assert result["results"][0]["bert_text"] == "I love parties"


def test_dispatch_unknown_tool_returns_error():
    ctx = {"rag": None}

    result = dispatch_tool_call("not_a_real_tool", {}, ctx)

    assert "error" in result


def test_dispatch_never_raises_on_bad_arguments():
    ctx = {"rag": None}

    result = dispatch_tool_call("retrieve_similar_exemplars", {}, ctx)  # missing required "text"

    assert "error" in result
```

- [ ] **Step 4: Rewrite `src/agent/orchestrator.py`**

Replace the entire file:

```python
"""The agent's ReAct-style orchestration loop: calls an LLM with tool-calling
enabled, dispatches whichever RAG tools it chooses, and terminates when it
calls submit_assessment. Degrades to a neutral default if the LLM call
fails, the response is malformed, or max_iterations is exhausted without a
submit_assessment call.
"""
import json

from src.agent.openrouter_client import call_with_fallback
from src.agent.tool_schemas import TOOL_SCHEMAS, dispatch_tool_call
from src.tiers import TIER_BINS

SYSTEM_PROMPT = (
    "You are an assessment agent estimating the Extraversion of a piece of text "
    "on a 1 (most reserved) to 6 (most extraverted) tier scale. You have tools "
    "available to gather evidence: retrieval of similar labeled examples and "
    "retrieval of relevant Extraversion/Big Five psychology theory from a "
    "knowledge base. Use as many tool calls as you find useful, then call "
    "submit_assessment exactly once with your final tier, a 0-99 continuous "
    "score estimate, your confidence, and a brief rationale citing the "
    "evidence you gathered."
)


def label_for_tier(tier_num):
    for _low, _high, num, label in TIER_BINS:
        if num == tier_num:
            return label
    raise ValueError(f"invalid tier {tier_num}")


def _degraded_result(text, ctx, error):
    return {
        "tier": 4,
        "tier_label": label_for_tier(4),
        "continuous_score_estimate": 50.0,
        "confidence": "low",
        "rationale": "The agent could not complete the assessment; returning a neutral default.",
        "trace": [],
        "degraded": True,
        "error": error,
    }


def run_agent(client, models, ctx, text, max_iterations=6, extra_params=None, enabled_tools=None):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Assess the Extraversion of this text:\n\n{text}"},
    ]
    trace = []

    if enabled_tools is None:
        available_schemas = TOOL_SCHEMAS
    else:
        allowed_names = set(enabled_tools) | {"submit_assessment"}
        available_schemas = [s for s in TOOL_SCHEMAS if s["function"]["name"] in allowed_names]

    for _ in range(max_iterations):
        try:
            response = call_with_fallback(client, models, messages, tools=available_schemas, extra_params=extra_params)
        except Exception as e:
            return _degraded_result(text, ctx, str(e))

        try:
            message = response["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                return _degraded_result(text, ctx, "Agent responded without calling a tool.")

            messages.append(message)

            for tool_call in tool_calls:
                name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])

                if name == "submit_assessment":
                    score = min(99.0, max(0.0, float(arguments["continuous_score_estimate"])))
                    return {
                        "tier": arguments["tier"],
                        "tier_label": label_for_tier(arguments["tier"]),
                        "continuous_score_estimate": score,
                        "confidence": arguments["confidence"],
                        "rationale": arguments["rationale"],
                        "trace": trace,
                        "degraded": False,
                        "error": None,
                    }

                if enabled_tools is not None and name != "submit_assessment" and name not in enabled_tools:
                    result = {"error": f"Tool '{name}' is disabled for this evaluation variant."}
                else:
                    result = dispatch_tool_call(name, arguments, ctx)
                trace.append({"tool": name, "arguments": arguments, "result": result})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(result),
                })
        except (KeyError, IndexError, TypeError, ValueError) as e:
            return _degraded_result(text, ctx, f"Malformed agent response: {e}")

    return _degraded_result(text, ctx, f"Max iterations ({max_iterations}) reached without submit_assessment.")
```

- [ ] **Step 5: Rewrite `tests/agent/test_orchestrator.py`**

Replace the entire file:

```python
import json

import httpx

from src.agent.openrouter_client import build_client
from src.agent.orchestrator import label_for_tier, run_agent


def _build_test_context():
    return {"rag": None}


def _assistant_tool_call_response(call_id, name, arguments):
    return httpx.Response(200, json={
        "choices": [{"message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": call_id, "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments),
            }}],
        }}],
    })


def test_label_for_tier_matches_tiers_module():
    assert label_for_tier(1) == "Reserved"
    assert label_for_tier(6) == "Highly Extraverted"


def test_run_agent_completes_via_tool_call_then_submit():
    turns = {"n": 0}

    def handler(request):
        turns["n"] += 1
        if turns["n"] == 1:
            return _assistant_tool_call_response("call_1", "retrieve_similar_exemplars", {"text": "I love parties!"})
        return _assistant_tool_call_response("call_2", "submit_assessment", {
            "tier": 6, "continuous_score_estimate": 92.0, "confidence": "high",
            "rationale": "Strongly positive, high-energy language.",
        })

    client = build_client("fake-key", transport=httpx.MockTransport(handler))
    ctx = _build_test_context()

    result = run_agent(client, ["fake-model"], ctx, "I love parties!")

    assert result["degraded"] is False
    assert result["tier"] == 6
    assert result["tier_label"] == "Highly Extraverted"
    assert len(result["trace"]) == 1
    assert result["trace"][0]["tool"] == "retrieve_similar_exemplars"


def test_run_agent_stops_after_max_iterations_without_submit():
    def handler(request):
        return _assistant_tool_call_response("call_x", "retrieve_relevant_theory", {"text": "hi"})

    client = build_client("fake-key", transport=httpx.MockTransport(handler))
    ctx = _build_test_context()

    result = run_agent(client, ["fake-model"], ctx, "hi", max_iterations=3)

    assert result["degraded"] is True
    assert "max iterations" in result["error"].lower()


def test_run_agent_degrades_gracefully_when_api_fails():
    def handler(request):
        return httpx.Response(500, json={"error": "down"})

    client = build_client("fake-key", transport=httpx.MockTransport(handler))
    ctx = _build_test_context()

    result = run_agent(client, ["fake-model"], ctx, "I love parties and talking to everyone!")

    assert result["degraded"] is True
    assert result["error"] is not None
    assert result["continuous_score_estimate"] == 50.0
    assert result["tier"] == 4


def test_run_agent_degrades_gracefully_on_malformed_response():
    def handler(request):
        return httpx.Response(200, json={"not_choices": "this response is malformed"})

    client = build_client("fake-key", transport=httpx.MockTransport(handler))
    ctx = _build_test_context()

    result = run_agent(client, ["fake-model"], ctx, "I love parties and talking to everyone!")

    assert result["degraded"] is True
    assert result["error"] is not None
    assert result["continuous_score_estimate"] == 50.0
    assert result["tier"] == 4


def test_run_agent_clamps_out_of_range_submitted_score():
    def handler(request):
        return _assistant_tool_call_response("call_1", "submit_assessment", {
            "tier": 6, "continuous_score_estimate": 150.0, "confidence": "high",
            "rationale": "Overconfident.",
        })

    client = build_client("fake-key", transport=httpx.MockTransport(handler))
    ctx = _build_test_context()

    result = run_agent(client, ["fake-model"], ctx, "I love parties!")

    assert result["continuous_score_estimate"] == 99.0


def test_degraded_result_returns_a_neutral_default():
    from src.agent.orchestrator import _degraded_result

    result = _degraded_result("some text", {"rag": None}, "original error")

    assert result["degraded"] is True
    assert result["tier"] == 4
    assert result["tier_label"] == label_for_tier(4)
    assert result["continuous_score_estimate"] == 50.0
    assert result["error"] == "original error"


def test_run_agent_restricts_tool_schemas_sent_to_the_api():
    sent_tool_names = []

    def handler(request):
        body = json.loads(request.content)
        sent_tool_names.append([t["function"]["name"] for t in body.get("tools", [])])
        return _assistant_tool_call_response("call_1", "submit_assessment", {
            "tier": 3, "continuous_score_estimate": 40.0, "confidence": "medium",
            "rationale": "Ambiguous signal.",
        })

    client = build_client("fake-key", transport=httpx.MockTransport(handler))
    ctx = _build_test_context()

    run_agent(client, ["fake-model"], ctx, "some text", enabled_tools={"retrieve_similar_exemplars"})

    assert set(sent_tool_names[0]) == {"retrieve_similar_exemplars", "submit_assessment"}


def test_run_agent_gracefully_refuses_a_disabled_tool_call():
    turns = {"n": 0}

    def handler(request):
        turns["n"] += 1
        if turns["n"] == 1:
            return _assistant_tool_call_response("call_1", "retrieve_relevant_theory", {"text": "hi"})
        return _assistant_tool_call_response("call_2", "submit_assessment", {
            "tier": 3, "continuous_score_estimate": 40.0, "confidence": "medium",
            "rationale": "Ambiguous signal.",
        })

    client = build_client("fake-key", transport=httpx.MockTransport(handler))
    ctx = _build_test_context()

    result = run_agent(client, ["fake-model"], ctx, "some text", enabled_tools={"retrieve_similar_exemplars"})

    assert result["degraded"] is False
    assert result["trace"][0]["tool"] == "retrieve_relevant_theory"
    assert "disabled" in result["trace"][0]["result"]["error"]


def test_run_agent_enabled_tools_none_still_exposes_all_tools():
    sent_tool_names = []

    def handler(request):
        body = json.loads(request.content)
        sent_tool_names.append([t["function"]["name"] for t in body.get("tools", [])])
        return _assistant_tool_call_response("call_1", "submit_assessment", {
            "tier": 3, "continuous_score_estimate": 40.0, "confidence": "medium",
            "rationale": "Ambiguous signal.",
        })

    client = build_client("fake-key", transport=httpx.MockTransport(handler))
    ctx = _build_test_context()

    run_agent(client, ["fake-model"], ctx, "some text")

    assert len(sent_tool_names[0]) == 3
```

- [ ] **Step 6: Rewrite `backend/agent_router.py`**

Replace the entire file:

```python
"""The /predict-agent FastAPI route. Deliberately isolated from backend/main.py
-- the LLM agent pipeline runs alongside the classical-ML pipeline there, not
in place of it.
"""
import os

from dotenv import load_dotenv
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.agent.context import load_rag_context
from src.agent.openrouter_client import DEEPSEEK_BASE_URL, build_client
from src.agent.orchestrator import run_agent

load_dotenv()
router = APIRouter()
_cached_ctx = None
_cached_client_and_models = None


def _build_deepseek_config(api_key):
    client = build_client(api_key, base_url=os.environ.get("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL))
    models_env = os.environ.get("DEEPSEEK_MODELS", "deepseek-v4-flash")
    models = [m.strip() for m in models_env.split(",") if m.strip()]
    extra_params = {"reasoning_effort": "high", "thinking": {"type": "enabled"}}
    return client, models, extra_params


def _build_openrouter_config(api_key):
    client = build_client(api_key)
    models_env = os.environ.get("OPENROUTER_MODELS", "")
    models = [m.strip() for m in models_env.split(",") if m.strip()]
    return client, models, {"reasoning": {"enabled": True}}


def get_agent_context():
    global _cached_ctx
    if _cached_ctx is None:
        rag = load_rag_context(os.path.join("data", "rag", "chroma"))
        _cached_ctx = {"rag": rag}
    return _cached_ctx


def get_agent_client_and_models():
    global _cached_client_and_models
    if _cached_client_and_models is None:
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if deepseek_key:
            _cached_client_and_models = _build_deepseek_config(deepseek_key)
        else:
            _cached_client_and_models = _build_openrouter_config(os.environ.get("OPENROUTER_API_KEY", ""))
    return _cached_client_and_models


class PredictAgentRequest(BaseModel):
    text: str


@router.post("/predict-agent")
def predict_agent(req: PredictAgentRequest, ctx=Depends(get_agent_context), client_and_models=Depends(get_agent_client_and_models)):
    if not req.text.strip():
        return {"error": "Empty text"}
    client, models, extra_params = client_and_models
    if not models:
        return {"error": "No models configured. Set DEEPSEEK_API_KEY (preferred) or OPENROUTER_API_KEY/OPENROUTER_MODELS."}
    return run_agent(client, models, ctx, req.text, extra_params=extra_params)
```

- [ ] **Step 7: Rewrite `tests/backend/test_agent_router.py`**

Replace the entire file:

```python
import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.agent_router as agent_router_module
from backend.agent_router import (
    _build_deepseek_config,
    _build_openrouter_config,
    get_agent_client_and_models,
    get_agent_context,
    router,
)
from src.agent.openrouter_client import DEEPSEEK_BASE_URL, OPENROUTER_BASE_URL, build_client

_TEST_CTX = {"rag": None}


def _make_test_app():
    app = FastAPI()
    app.include_router(router)
    return app


def test_predict_agent_returns_error_on_empty_text():
    app = _make_test_app()
    app.dependency_overrides[get_agent_context] = lambda: _TEST_CTX
    app.dependency_overrides[get_agent_client_and_models] = lambda: (None, ["fake-model"], None)
    client = TestClient(app)

    response = client.post("/predict-agent", json={"text": "   "})

    assert response.status_code == 200
    assert response.json() == {"error": "Empty text"}


def test_predict_agent_returns_error_when_no_models_configured():
    app = _make_test_app()
    app.dependency_overrides[get_agent_context] = lambda: _TEST_CTX
    app.dependency_overrides[get_agent_client_and_models] = lambda: (None, [], None)
    client = TestClient(app)

    response = client.post("/predict-agent", json={"text": "hello"})

    assert response.status_code == 200
    assert "OPENROUTER_MODELS" in response.json()["error"]


def test_predict_agent_returns_agent_result_on_success():
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {
                    "name": "submit_assessment",
                    "arguments": (
                        '{"tier": 6, "continuous_score_estimate": 90.0, '
                        '"confidence": "high", "rationale": "Very outgoing language."}'
                    ),
                }}],
            }}],
        })

    fake_client = build_client("fake-key", transport=httpx.MockTransport(handler))
    app = _make_test_app()
    app.dependency_overrides[get_agent_context] = lambda: _TEST_CTX
    app.dependency_overrides[get_agent_client_and_models] = lambda: (fake_client, ["fake-model"], None)
    client = TestClient(app)

    response = client.post("/predict-agent", json={"text": "I love parties!"})

    assert response.status_code == 200
    body = response.json()
    assert body["tier"] == 6
    assert body["degraded"] is False


def test_build_deepseek_config_uses_v4_flash_with_high_reasoning_effort():
    client, models, extra_params = _build_deepseek_config("fake-deepseek-key")

    assert str(client.base_url).rstrip("/") == DEEPSEEK_BASE_URL
    assert models == ["deepseek-v4-flash"]
    assert extra_params == {"reasoning_effort": "high", "thinking": {"type": "enabled"}}


def test_build_openrouter_config_reads_models_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODELS", "model-a,model-b")

    client, models, extra_params = _build_openrouter_config("fake-openrouter-key")

    assert str(client.base_url).rstrip("/") == OPENROUTER_BASE_URL
    assert models == ["model-a", "model-b"]
    assert extra_params == {"reasoning": {"enabled": True}}


def test_get_agent_client_and_models_prefers_deepseek(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-deepseek-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-openrouter-key")
    monkeypatch.setattr(agent_router_module, "_cached_client_and_models", None)

    client, models, extra_params = get_agent_client_and_models()

    assert models == ["deepseek-v4-flash"]
    assert "thinking" in extra_params

    monkeypatch.setattr(agent_router_module, "_cached_client_and_models", None)


def test_predict_agent_degrades_when_api_fails():
    def handler(request):
        return httpx.Response(500, json={"error": "down"})

    fake_client = build_client("fake-key", transport=httpx.MockTransport(handler))
    app = _make_test_app()
    app.dependency_overrides[get_agent_context] = lambda: _TEST_CTX
    app.dependency_overrides[get_agent_client_and_models] = lambda: (fake_client, ["fake-model"], None)
    client = TestClient(app)

    response = client.post("/predict-agent", json={"text": "I love parties and talking to everyone!"})

    assert response.status_code == 200
    body = response.json()
    assert body["degraded"] is True
```

- [ ] **Step 8: Update `ABLATION_VARIANTS` in `src/evaluation/run_comparison.py`**

Replace lines 15-20 (the `ABLATION_VARIANTS` dict) with:

```python
ABLATION_VARIANTS = {
    "llm_only": set(),
    "llm_rag": None,
}
```

- [ ] **Step 9: Update `tests/evaluation/test_run_comparison.py`**

Replace the entire file:

```python
from src.evaluation.run_comparison import (
    ABLATION_VARIANTS,
    HISTORICAL_BASELINES,
    run_evaluation,
    summarize_evaluation,
)


def _fake_predict_fn(text, enabled_tools):
    # Deterministic: score derives from text length, tier from a simple mapping.
    score = min(99.0, len(text) * 2.0)
    if score <= 10:
        tier = 1
    elif score <= 25:
        tier = 2
    elif score <= 45:
        tier = 3
    elif score <= 65:
        tier = 4
    elif score <= 85:
        tier = 5
    else:
        tier = 6
    trace = [] if enabled_tools == set() else [
        {"tool": "retrieve_similar_exemplars", "arguments": {"text": text},
         "result": {"results": [{"bert_text": text, "score": 0.5}]}},
    ]
    return {
        "tier": tier, "tier_label": "x", "continuous_score_estimate": score,
        "confidence": "medium", "rationale": f"retrieve_similar_exemplars grounded this at tier {tier}.",
        "trace": trace, "degraded": False, "error": None,
    }


def test_ablation_variants_cover_the_two_expected_configurations():
    assert set(ABLATION_VARIANTS.keys()) == {"llm_only", "llm_rag"}
    assert ABLATION_VARIANTS["llm_only"] == set()
    assert ABLATION_VARIANTS["llm_rag"] is None


def test_historical_baselines_include_all_three_classical_models():
    assert set(HISTORICAL_BASELINES.keys()) == {"ridge", "xgboost", "random_forest"}
    for entry in HISTORICAL_BASELINES.values():
        assert "rmse" in entry and "r2" in entry


def test_run_evaluation_produces_one_row_per_sample_per_variant():
    samples = [("short", 5.0), ("a much longer piece of text here", 90.0)]
    variants = {"llm_only": set(), "llm_rag": None}

    results = run_evaluation(_fake_predict_fn, samples, variants)

    assert set(results.keys()) == {"llm_only", "llm_rag"}
    assert len(results["llm_only"]) == 2
    assert len(results["llm_rag"]) == 2
    assert results["llm_only"][0]["text"] == "short"
    assert results["llm_only"][0]["true_score"] == 5.0
    assert "predicted_tier" in results["llm_only"][0]
    assert "faithful" in results["llm_rag"][0]


def test_summarize_evaluation_includes_metrics_and_historical_baselines():
    samples = [("short", 5.0), ("a much longer piece of text here", 90.0)]
    variants = {"llm_only": set(), "llm_rag": None}
    results = run_evaluation(_fake_predict_fn, samples, variants)

    summary = summarize_evaluation(results)

    assert "llm_only" in summary
    assert "rmse" in summary["llm_only"]
    assert "accuracy" in summary["llm_only"]
    assert "faithfulness_rate" in summary["llm_only"]
    assert summary["_historical_baselines"] == HISTORICAL_BASELINES
```

- [ ] **Step 10: Update `src/evaluation/run_real_evaluation.py`**

Replace `build_real_context` (drop spaCy/NRC/ML-prior setup) and its imports:

```python
import argparse
import json
import os
import time

import pandas as pd
from dotenv import load_dotenv

from src.agent.context import load_rag_context
from src.agent.openrouter_client import DEEPSEEK_BASE_URL, build_client
from src.evaluation.run_comparison import (
    ABLATION_VARIANTS,
    HISTORICAL_BASELINES,
    make_run_agent_predict_fn,
    run_evaluation,
    summarize_evaluation,
)

load_dotenv()


def build_real_context():
    rag = load_rag_context(os.path.join("data", "rag", "chroma"))
    if rag is None:
        print("Note: RAG corpus not built yet -- run `python -m src.rag.build_corpus` first.")
    return {"rag": rag}
```

(Leave `build_real_client_and_models`, `load_samples`, `print_comparison_table`, and `main` unchanged -- they don't reference `nlp`/`nrc_dict`/`ml_model`.)

- [ ] **Step 11: Update the frontend tool labels in `frontend/src/routes/+page.svelte`**

Replace lines 32-55 (the `toolLabel` and `toolSummary` functions):

```javascript
    function toolLabel(tool: string) {
        const labels: Record<string, string> = {
            retrieve_similar_exemplars: "RAG: Similar Examples",
            retrieve_relevant_theory: "RAG: Psychology Theory",
        };
        return labels[tool] || tool;
    }

    function toolSummary(step: any) {
        const r = step.result;
        if (r?.error) return r.error;
        if (r?.results) {
            return `${r.results.length} result(s) retrieved`;
        }
        return JSON.stringify(r);
    }
```

- [ ] **Step 12: Delete the obsolete fuzzy/ML-prior/classical-features modules and their tests**

```bash
rm -f src/agent/tools/fuzzy_engine.py src/agent/tools/ml_prior.py src/agent/tools/classical_features.py \
      tests/agent/tools/test_fuzzy_engine.py tests/agent/tools/test_ml_prior.py tests/agent/tools/test_classical_features.py
```

- [ ] **Step 13: Run the full suite**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: all pass, 0 references to `fuzzy_engine`, `ml_prior`, or the agent's `classical_features` remain (confirmed by Step 1's grep already, this is the executable confirmation).

- [ ] **Step 14: Commit**

```bash
git add src/agent/tool_schemas.py src/agent/orchestrator.py backend/agent_router.py \
        src/evaluation/run_comparison.py src/evaluation/run_real_evaluation.py \
        frontend/src/routes/+page.svelte \
        tests/agent/test_tool_schemas.py tests/agent/test_orchestrator.py tests/backend/test_agent_router.py \
        tests/evaluation/test_run_comparison.py
git rm src/agent/tools/fuzzy_engine.py src/agent/tools/ml_prior.py src/agent/tools/classical_features.py \
       tests/agent/tools/test_fuzzy_engine.py tests/agent/tools/test_ml_prior.py tests/agent/tools/test_classical_features.py
git commit -m "refactor: remove fuzzy-logic and ML-prior tools; agent is now LLM+RAG only"
```

---

## After This Plan Merges

1. Rebuild the real corpus: `./.venv/Scripts/python.exe -m src.rag.build_corpus` (downloads `all-MiniLM-L6-v2` if not cached, populates `data/rag/chroma/`).
2. Re-run the real evaluation with the new two-variant ablation: `./.venv/Scripts/python.exe -m src.evaluation.run_real_evaluation --n-samples 20`.
3. Compare `llm_only` vs `llm_rag` against the historical `ridge`/`xgboost`/`random_forest` baselines to see whether hybrid RAG grounding closes the gap this plan's predecessor evaluation found.

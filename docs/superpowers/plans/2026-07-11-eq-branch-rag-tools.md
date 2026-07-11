# EQ Per-Branch RAG Tools Implementation Plan (Revision: LanceDB)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the real per-branch RAG tools that ground each MSC specialist (Perceiving/Using/Understanding/Managing), replacing Plan 2's test-only empty `tool_schemas`/`dispatch_fn` with actual retrieval grounded in Plan 1's MSC theory corpus, branch-tagged Pandora exemplars, and (for Perceiving/Understanding) real external emotion-labeled datasets — using **LanceDB** (not ChromaDB) for native hybrid search and cross-encoder reranking — then assemble the real `branch_configs` dict that Plan 2's `run_eq_assessment` consumes.

**Architecture:** A corpus builder (`src/eq_data/build_eq_corpus.py`) creates 8 branch-tagged LanceDB tables (4 theory + 4 exemplar, one pair per MSC branch), using LanceDB's native hybrid search (vector + full-text, fused server-side) instead of the hand-rolled ChromaDB+BM25+RRF approach from Plan 6/the original Plan 3. A retrieval layer (`src/eq_agent/eq_rag_retrieval.py`) does branch-scoped hybrid search with optional cross-encoder reranking (top-50 → rerank → top-k). A config-assembly layer (`src/eq_agent/branch_config.py`) — unchanged in shape from the ChromaDB version — defines the 2 RAG tool schemas, a branch-scoped exception-safe dispatcher, real per-branch system prompts, and `build_branch_configs()`. A context loader (`src/eq_agent/eq_context.py`) loads the pre-built LanceDB tables and a `CrossEncoderReranker` at process startup.

**Tech Stack:** `lancedb` 0.34.0 (verified installed and API-tested before writing this plan — see Global Constraints for exact findings), `sentence-transformers` (already present, used both for the embedding model and the cross-encoder reranker), `pandas`, `pytest`.

## Global Constraints

- **This plan supersedes the original ChromaDB-based Plan 3** (never executed — no wasted work). `src/rag/hybrid_store.py` (ChromaDB+BM25+RRF, Plan 6) is **not** used by the EQ pivot past this point; it stays as-is solely for the separate, already-shipped Extraversion-era `/predict-agent` pipeline.
- **Verified LanceDB facts** (computed directly against the installed `lancedb==0.34.0` before writing this plan, not assumed):
  - `lancedb.connect(path)` is the embedded/serverless entry point — no server process, same "no infrastructure to run" property ChromaDB had.
  - Native hybrid search requires the table's vector column to be tied to a **registered embedding function** (`lancedb.embeddings.EmbeddingFunctionRegistry`), not a duck-typed `embedder.encode()` object passed per-call like the ChromaDB-era code. A custom registered embedding function's `compute_source_embeddings`/`compute_query_embeddings` receive `pyarrow.StringScalar` objects internally, **not plain Python strings** — omitting `str(t)` conversion causes silent failures wrapped in an exponential-backoff retry loop that takes 60+ seconds to surface. Always convert with `str(t)` before any dict lookup.
  - The built-in `registry.get("sentence-transformers").create(name="all-MiniLM-L6-v2")` embedding function is used for real/production corpus building — no custom wrapper needed for the real embedder, only for test fixtures (see the shared test helper in Task 1).
  - `Optional[T] = None` fields in a `LanceModel` schema accept `None` naturally (verified: stored as null/NaN, no error) — unlike ChromaDB's flat rejection of `None` metadata values. This means, unlike the original ChromaDB-based plan, no "omit None-valued keys" workaround is needed here.
  - `table.create_index(column, config=FTS(), replace=True)` builds the full-text index (the non-deprecated API; `create_fts_index` is deprecated as of lancedb 0.25.0).
  - `table.search(query_text, query_type="hybrid").limit(k).to_pandas()` performs native hybrid (vector+FTS) search, internally RRF-fused — this is the method that eliminates the hand-rolled BM25/RRF code from the ChromaDB-era `hybrid_store.py`.
  - Metadata filtering: `table.search(...).where("branch = 'perceiving'")` (SQL-like filter string) — verified working.
  - Reranking chain order, verified end-to-end: `table.search(query, query_type="hybrid").limit(fetch_k).rerank(reranker=reranker_instance).limit(k).to_pandas()` — fetch a wide candidate set, rerank it, then take the final top-k.
  - `lancedb.rerankers.CrossEncoderReranker(model_name="cross-encoder/ms-marco-TinyBERT-L-6")` is the built-in cross-encoder reranker (real model, downloaded lazily, requires `torch`) — used for real/production reranking. `lancedb.rerankers.RRFReranker()` needs no model download and was used purely to verify the `.rerank()` chaining mechanics for this plan; it is not part of the shipped design.
  - `db.open_table(name)` **raises `ValueError`** if the table doesn't exist (unlike ChromaDB's `get_or_create_collection`, which silently creates an empty one) — the context loader must check `db.list_tables()` membership first, not rely on open-then-catch or an auto-create fallback.
- Real network/dataset-fetching calls (`fetch_goemotions`, `fetch_isear`, `fetch_emobank`, `fetch_empathetic_dialogues` from Plan 1) are injected as a parameter (`external_fetchers`), not called directly — the automated test suite injects fake in-memory fetchers. No test in this plan makes a real network call, and no test downloads a real cross-encoder model (the reranker is optional/injectable — see Task 2).
- `run_specialist`'s (Plan 2) requirement that `dispatch_fn` never raises is honored: the dispatcher wraps its entire body in `try/except Exception`.
- Purely additive except for reading (never modifying) Plan 1's (`src/eq_data/`) and Plan 2's (`src/eq_agent/`) already-merged code. No file under `src/agent/`, `src/rag/`, `backend/`, or `frontend/` is touched. New files get no `__init__.py` (implicit namespace packages, established convention).
- New dependency: `lancedb==0.34.0` (add to `requirements.txt`).

---

### Task 1: EQ RAG corpus builder (LanceDB)

**Files:**
- Create: `src/eq_data/build_eq_corpus.py`
- Create: `tests/eq_data/lancedb_test_helpers.py` (shared test-only fixture helper, not a test file itself — pytest won't collect it, since it doesn't match `test_*.py`)
- Modify: `requirements.txt` (add `lancedb==0.34.0`)
- Test: `tests/eq_data/test_build_eq_corpus.py`

**Interfaces:**
- Consumes: `sample_branch_balanced_exemplars(df, text_col, n_per_tier, seed)` and `BRANCHES` from `src/eq_data/branch_exemplars.py` (Plan 1); `load_msc_theory_corpus(path)` from `src/eq_data/msc_theory_corpus.py` (Plan 1); `fetch_goemotions`/`fetch_isear`/`fetch_emobank`/`fetch_empathetic_dialogues` from `src/eq_data/external_datasets.py` (Plan 1, injected).
- Produces: `theory_table_name(branch) -> str`, `exemplar_table_name(branch) -> str`, `build_eq_corpus(pandora_df, data_dir, persist_dir, embedding_func, external_fetchers=None, n_per_tier=60, seed=42, n_external_samples=200) -> (dict[branch, Table], dict[branch, Table])` — consumed by Task 4's context loader (naming functions) and this plan's own `main()`. `embedding_func` is a LanceDB registered-embedding-function **instance** (e.g. `registry.get("sentence-transformers").create(name=...)`), not a duck-typed embedder object.

- [ ] **Step 1: Write the shared test helper**

```python
"""Shared test-only helper for registering fake LanceDB embedding functions
across the EQ RAG test suite -- avoids downloading a real model in the fast
unit test suite. Each call registers a uniquely-named embedding function
(via an incrementing counter) so multiple test files/fixtures never collide
on the same registry key.
"""
import itertools

from lancedb.embeddings import EmbeddingFunction, EmbeddingFunctionRegistry

_registry = EmbeddingFunctionRegistry.get_instance()
_counter = itertools.count()


def make_fake_embedding_func(vector_by_text, ndims=2):
    name = f"fake_eq_test_{next(_counter)}"

    @_registry.register(name)
    class _FakeEmbedder(EmbeddingFunction):
        def ndims(self):
            return ndims

        def compute_query_embeddings(self, query, *args, **kwargs):
            return [vector_by_text[str(query)]]

        def compute_source_embeddings(self, texts, *args, **kwargs):
            return [vector_by_text[str(t)] for t in texts]

    return _registry.get(name).create()
```

- [ ] **Step 2: Write the failing tests**

```python
import json

import pandas as pd

from src.eq_data.build_eq_corpus import build_eq_corpus, exemplar_table_name, theory_table_name
from tests.eq_data.lancedb_test_helpers import make_fake_embedding_func


def _write_fixture_theory_corpus(data_dir):
    eq_dir = data_dir / "eq"
    eq_dir.mkdir()
    entries = [
        {"id": "p1", "branch": "perceiving", "topic": "t", "text": "perceiving theory chunk", "citation_needed": "yes"},
        {"id": "u1", "branch": "using", "topic": "t", "text": "using theory chunk", "citation_needed": "yes"},
        {"id": "n1", "branch": "understanding", "topic": "t", "text": "understanding theory chunk", "citation_needed": "yes"},
        {"id": "m1", "branch": "managing", "topic": "t", "text": "managing theory chunk", "citation_needed": "yes"},
    ]
    (eq_dir / "msc_theory_corpus.json").write_text(json.dumps(entries), encoding="utf-8")


def _fake_pandora_df():
    scores = [5, 20, 35, 55, 75, 95] * 3
    return pd.DataFrame({
        "text": [f"sample pandora text {i}" for i in range(len(scores))],
        "extraversion": scores, "openness": scores, "agreeableness": scores,
        "conscientiousness": scores, "neuroticism": scores,
    })


def _fake_external_fetcher(n_rows=5):
    def fetcher():
        return pd.DataFrame({
            "text": [f"external text {i}" for i in range(n_rows)],
            "source": ["fake_source"] * n_rows,
            "emotion_labels": [["joy"] for _ in range(n_rows)],
            "valence": [None] * n_rows, "arousal": [None] * n_rows, "dominance": [None] * n_rows,
        })
    return fetcher


def _fake_embedding_func():
    vector_by_text = {}
    for i in range(18):
        vector_by_text[f"sample pandora text {i}"] = [float(i), 0.0]
    for i in range(10):
        vector_by_text[f"external text {i}"] = [0.0, float(i)]
    for text in ["perceiving theory chunk", "using theory chunk", "understanding theory chunk", "managing theory chunk"]:
        vector_by_text[text] = [1.0, 1.0]
    return make_fake_embedding_func(vector_by_text)


def test_theory_and_exemplar_table_names_are_branch_specific():
    assert theory_table_name("perceiving") == "eq_theory_perceiving"
    assert exemplar_table_name("managing") == "eq_exemplars_managing"


def test_build_eq_corpus_produces_eight_populated_tables(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    persist_dir = tmp_path / "lancedb"
    _write_fixture_theory_corpus(data_dir)

    theory_tables, exemplar_tables = build_eq_corpus(
        _fake_pandora_df(), str(data_dir), str(persist_dir), _fake_embedding_func(),
        external_fetchers={"perceiving": [], "using": [], "understanding": [], "managing": []},
        n_per_tier=2, seed=1,
    )

    assert set(theory_tables.keys()) == {"perceiving", "using", "understanding", "managing"}
    assert set(exemplar_tables.keys()) == {"perceiving", "using", "understanding", "managing"}
    for branch in theory_tables:
        assert theory_tables[branch].count_rows() > 0
    for branch in exemplar_tables:
        assert exemplar_tables[branch].count_rows() > 0


def test_build_eq_corpus_enriches_perceiving_and_understanding_with_external_records(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    persist_dir = tmp_path / "lancedb"
    _write_fixture_theory_corpus(data_dir)

    _, exemplar_tables = build_eq_corpus(
        _fake_pandora_df(), str(data_dir), str(persist_dir), _fake_embedding_func(),
        external_fetchers={
            "perceiving": [_fake_external_fetcher(5)], "using": [],
            "understanding": [_fake_external_fetcher(3)], "managing": [],
        },
        n_per_tier=2, seed=1, n_external_samples=200,
    )

    perceiving_texts = exemplar_tables["perceiving"].to_pandas()["text"].tolist()
    understanding_texts = exemplar_tables["understanding"].to_pandas()["text"].tolist()
    using_texts = exemplar_tables["using"].to_pandas()["text"].tolist()

    assert any("external text" in t for t in perceiving_texts)
    assert any("external text" in t for t in understanding_texts)
    assert not any("external text" in t for t in using_texts)


def test_build_eq_corpus_stores_none_for_inapplicable_optional_fields(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    persist_dir = tmp_path / "lancedb"
    _write_fixture_theory_corpus(data_dir)

    _, exemplar_tables = build_eq_corpus(
        _fake_pandora_df(), str(data_dir), str(persist_dir), _fake_embedding_func(),
        external_fetchers={"perceiving": [_fake_external_fetcher(2)], "using": [], "understanding": [], "managing": []},
        n_per_tier=2, seed=1,
    )

    df = exemplar_tables["perceiving"].to_pandas()
    external_rows = df[df["source"] == "fake_source"]
    pandora_rows = df[df["source"].isna()]

    assert len(external_rows) > 0
    assert external_rows["tier"].isna().all()  # externally-sourced rows have no proxy tier
    assert (external_rows["emotion_labels"] == "joy").all()
    assert len(pandora_rows) > 0
    assert pandora_rows["tier"].notna().all()  # Pandora-derived rows always have a tier
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_data/test_build_eq_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eq_data.build_eq_corpus'`

- [ ] **Step 4: Add the dependency to requirements.txt**

Add this line at the end of `requirements.txt`:

```
lancedb==0.34.0
```

- [ ] **Step 5: Write the implementation**

```python
"""Builds the EQ RAG knowledge base in LanceDB: 4 branch-tagged theory
tables and 4 branch-tagged exemplar tables (one pair per MSC branch).
Perceiving and Understanding exemplar tables are additionally enriched with
a sample of real external emotion-labeled text. Uses LanceDB's native
hybrid search (vector + full-text, fused server-side) instead of a
hand-rolled BM25/RRF layer -- see this plan's Global Constraints for the
exact API facts this implementation relies on.
"""
import os
from typing import Optional

import lancedb
from lancedb.index import FTS
from lancedb.pydantic import LanceModel, Vector

from src.eq_data.branch_exemplars import BRANCHES, sample_branch_balanced_exemplars
from src.eq_data.external_datasets import (
    fetch_emobank,
    fetch_empathetic_dialogues,
    fetch_goemotions,
    fetch_isear,
)
from src.eq_data.msc_theory_corpus import load_msc_theory_corpus

DEFAULT_EXTERNAL_FETCHERS = {
    "perceiving": [fetch_goemotions, fetch_isear, fetch_emobank],
    "using": [],
    "understanding": [fetch_empathetic_dialogues],
    "managing": [],
}


def theory_table_name(branch):
    return f"eq_theory_{branch}"


def exemplar_table_name(branch):
    return f"eq_exemplars_{branch}"


def _make_theory_schema(embedding_func):
    class TheoryRecord(LanceModel):
        text: str = embedding_func.SourceField()
        vector: Vector(embedding_func.ndims()) = embedding_func.VectorField()
        id: str
        topic: str
        citation_needed: str
    return TheoryRecord


def _make_exemplar_schema(embedding_func):
    class ExemplarRecord(LanceModel):
        text: str = embedding_func.SourceField()
        vector: Vector(embedding_func.ndims()) = embedding_func.VectorField()
        tier: Optional[int] = None
        tier_label: Optional[str] = None
        eq_proxy_score: Optional[float] = None
        source: Optional[str] = None
        emotion_labels: Optional[str] = None
        valence: Optional[float] = None
        arousal: Optional[float] = None
        dominance: Optional[float] = None
    return ExemplarRecord


def _theory_rows_for_branch(theory_entries, branch):
    return [
        {"text": e["text"], "id": e["id"], "topic": e["topic"], "citation_needed": e["citation_needed"]}
        for e in theory_entries if e["branch"] == branch
    ]


def _exemplar_rows_for_branch(exemplars_df, branch):
    branch_df = exemplars_df[exemplars_df["branch"] == branch]
    return [
        {"text": row["text"], "tier": int(row["tier"]), "tier_label": row["tier_label"],
         "eq_proxy_score": float(row["eq_proxy_score"])}
        for _, row in branch_df.iterrows()
    ]


def _external_rows(df, n_samples, seed):
    sample = df.sample(n=min(n_samples, len(df)), random_state=seed)
    rows = []
    for _, row in sample.iterrows():
        rows.append({
            "text": row["text"], "source": row["source"],
            "emotion_labels": ",".join(row["emotion_labels"]) if row["emotion_labels"] else None,
            "valence": row["valence"], "arousal": row["arousal"], "dominance": row["dominance"],
        })
    return rows


def _build_table(db, name, schema, rows):
    table = db.create_table(name, schema=schema, mode="overwrite")
    if rows:
        table.add(rows)
        table.create_index("text", config=FTS(), replace=True)
    return table


def build_eq_corpus(
    pandora_df, data_dir, persist_dir, embedding_func, external_fetchers=None,
    n_per_tier=60, seed=42, n_external_samples=200,
):
    if external_fetchers is None:
        external_fetchers = DEFAULT_EXTERNAL_FETCHERS

    theory_entries = load_msc_theory_corpus(os.path.join(data_dir, "eq", "msc_theory_corpus.json"))
    exemplars_df = sample_branch_balanced_exemplars(pandora_df, n_per_tier=n_per_tier, seed=seed)

    db = lancedb.connect(persist_dir)
    theory_schema = _make_theory_schema(embedding_func)
    exemplar_schema = _make_exemplar_schema(embedding_func)

    theory_tables = {}
    exemplar_tables = {}

    for branch in BRANCHES:
        theory_rows = _theory_rows_for_branch(theory_entries, branch)
        theory_tables[branch] = _build_table(db, theory_table_name(branch), theory_schema, theory_rows)

        exemplar_rows = _exemplar_rows_for_branch(exemplars_df, branch)
        for fetcher in external_fetchers.get(branch, []):
            exemplar_rows += _external_rows(fetcher(), n_external_samples, seed)
        exemplar_tables[branch] = _build_table(db, exemplar_table_name(branch), exemplar_schema, exemplar_rows)

    return theory_tables, exemplar_tables


def main():
    import pandas as pd
    from lancedb.embeddings import EmbeddingFunctionRegistry

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data")
    persist_dir = os.path.join(data_dir, "eq", "lancedb")
    os.makedirs(persist_dir, exist_ok=True)

    pandora_df = pd.read_csv(os.path.join(data_dir, "train_set.csv"))
    registry = EmbeddingFunctionRegistry.get_instance()
    embedding_func = registry.get("sentence-transformers").create(name="all-MiniLM-L6-v2")

    theory_tables, exemplar_tables = build_eq_corpus(pandora_df, data_dir, persist_dir, embedding_func)

    for branch in theory_tables:
        print(f"{branch}: theory={theory_tables[branch].count_rows()} rows, "
              f"exemplars={exemplar_tables[branch].count_rows()} rows")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_data/test_build_eq_corpus.py -v`
Expected: 4 passed

- [ ] **Step 7: Commit**

```bash
git add src/eq_data/build_eq_corpus.py tests/eq_data/lancedb_test_helpers.py tests/eq_data/test_build_eq_corpus.py requirements.txt
git commit -m "feat: add EQ RAG corpus builder on LanceDB (8 branch-tagged tables, native hybrid search)"
```

---

### Task 2: Branch-scoped hybrid retrieval with reranking

**Files:**
- Create: `src/eq_agent/eq_rag_retrieval.py`
- Test: `tests/eq_agent/test_eq_rag_retrieval.py`

**Interfaces:**
- Consumes: nothing beyond the LanceDB `Table` objects passed in via `rag_ctx` (no dependency on `src/rag/hybrid_store.py`).
- Produces: `retrieve_similar_eq_exemplars(query_text, branch, rag_ctx, k=5, fetch_k=50) -> list[dict]` and `retrieve_relevant_msc_theory(query_text, branch, rag_ctx, k=3, fetch_k=50) -> list[dict]`, where `rag_ctx = {"theory_tables": dict[branch, Table], "exemplar_tables": dict[branch, Table], "reranker": reranker_or_None}` — consumed by Task 3's dispatcher. Reranking is applied only when `rag_ctx["reranker"]` is not `None` (allows tests to exercise the no-reranker path without a real cross-encoder model).

- [ ] **Step 1: Write the failing tests**

```python
from src.eq_agent.eq_rag_retrieval import retrieve_relevant_msc_theory, retrieve_similar_eq_exemplars
from src.eq_data.build_eq_corpus import _build_table, _make_exemplar_schema, _make_theory_schema
from tests.eq_data.lancedb_test_helpers import make_fake_embedding_func


def test_retrieve_similar_eq_exemplars_scopes_to_the_given_branch(tmp_path):
    import lancedb

    vector_by_text = {
        "I noticed my friend seemed upset": [1.0, 0.0],
        "I stayed calm under pressure": [0.0, 1.0],
        "query about noticing emotions": [1.0, 0.0],
    }
    embedding_func = make_fake_embedding_func(vector_by_text)
    db = lancedb.connect(str(tmp_path / "lancedb"))
    exemplar_schema = _make_exemplar_schema(embedding_func)

    perceiving_table = _build_table(db, "eq_exemplars_perceiving", exemplar_schema, [
        {"text": "I noticed my friend seemed upset", "tier": 5, "tier_label": "Above Average EQ"},
    ])
    managing_table = _build_table(db, "eq_exemplars_managing", exemplar_schema, [
        {"text": "I stayed calm under pressure", "tier": 6, "tier_label": "High EQ"},
    ])

    rag_ctx = {
        "exemplar_tables": {"perceiving": perceiving_table, "managing": managing_table},
        "theory_tables": {}, "reranker": None,
    }

    hits = retrieve_similar_eq_exemplars("query about noticing emotions", "perceiving", rag_ctx, k=1)

    assert len(hits) == 1
    assert hits[0]["text"] == "I noticed my friend seemed upset"
    assert hits[0]["tier_label"] == "Above Average EQ"


def test_retrieve_relevant_msc_theory_scopes_to_the_given_branch(tmp_path):
    import lancedb

    vector_by_text = {
        "perceiving emotions theory chunk": [1.0, 0.0],
        "managing emotions theory chunk": [0.0, 1.0],
        "query about theory": [1.0, 0.0],
    }
    embedding_func = make_fake_embedding_func(vector_by_text)
    db = lancedb.connect(str(tmp_path / "lancedb"))
    theory_schema = _make_theory_schema(embedding_func)

    perceiving_table = _build_table(db, "eq_theory_perceiving", theory_schema, [
        {"text": "perceiving emotions theory chunk", "id": "p1", "topic": "t", "citation_needed": "yes"},
    ])
    managing_table = _build_table(db, "eq_theory_managing", theory_schema, [
        {"text": "managing emotions theory chunk", "id": "m1", "topic": "t", "citation_needed": "yes"},
    ])

    rag_ctx = {
        "theory_tables": {"perceiving": perceiving_table, "managing": managing_table},
        "exemplar_tables": {}, "reranker": None,
    }

    hits = retrieve_relevant_msc_theory("query about theory", "perceiving", rag_ctx, k=1)

    assert len(hits) == 1
    assert hits[0]["id"] == "p1"


def test_retrieve_returns_empty_list_for_empty_table(tmp_path):
    import lancedb

    embedding_func = make_fake_embedding_func({"anything": [1.0, 0.0]})
    db = lancedb.connect(str(tmp_path / "lancedb"))
    empty_table = _build_table(db, "eq_exemplars_using", _make_exemplar_schema(embedding_func), [])

    rag_ctx = {"exemplar_tables": {"using": empty_table}, "theory_tables": {}, "reranker": None}

    hits = retrieve_similar_eq_exemplars("anything", "using", rag_ctx, k=5)

    assert hits == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_agent/test_eq_rag_retrieval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eq_agent.eq_rag_retrieval'`

- [ ] **Step 3: Write the implementation**

```python
"""Branch-scoped native-hybrid retrieval (LanceDB) with optional
cross-encoder reranking over the EQ RAG corpus built by
src.eq_data.build_eq_corpus.
"""

EXEMPLAR_METADATA_COLUMNS = ["tier", "tier_label", "eq_proxy_score", "source", "emotion_labels", "valence", "arousal", "dominance"]
THEORY_METADATA_COLUMNS = ["id", "topic", "citation_needed"]


def _search(table, query_text, k, fetch_k, reranker, metadata_columns):
    if table.count_rows() == 0:
        return []

    query = table.search(query_text, query_type="hybrid").limit(fetch_k)
    if reranker is not None:
        query = query.rerank(reranker=reranker)
    df = query.limit(k).to_pandas()

    hits = []
    for _, row in df.iterrows():
        hit = {"text": row["text"]}
        for col in metadata_columns:
            if col in df.columns and row[col] == row[col]:  # excludes NaN (NaN != NaN)
                hit[col] = row[col]
        hits.append(hit)
    return hits


def retrieve_similar_eq_exemplars(query_text, branch, rag_ctx, k=5, fetch_k=50):
    table = rag_ctx["exemplar_tables"][branch]
    return _search(table, query_text, k, fetch_k, rag_ctx.get("reranker"), EXEMPLAR_METADATA_COLUMNS)


def retrieve_relevant_msc_theory(query_text, branch, rag_ctx, k=3, fetch_k=50):
    table = rag_ctx["theory_tables"][branch]
    return _search(table, query_text, k, fetch_k, rag_ctx.get("reranker"), THEORY_METADATA_COLUMNS)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_agent/test_eq_rag_retrieval.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/eq_agent/eq_rag_retrieval.py tests/eq_agent/test_eq_rag_retrieval.py
git commit -m "feat: add branch-scoped LanceDB hybrid retrieval with optional reranking"
```

---

### Task 3: Real branch_configs (tool schemas, dispatcher, system prompts)

**Files:**
- Create: `src/eq_agent/branch_config.py`
- Test: `tests/eq_agent/test_branch_config.py`

**Interfaces:**
- Consumes: `retrieve_similar_eq_exemplars`, `retrieve_relevant_msc_theory` (Task 2); `BRANCHES` from `src/eq_agent/graph.py` (Plan 2).
- Produces: `RETRIEVE_EXEMPLARS_SCHEMA`, `RETRIEVE_THEORY_SCHEMA`; `build_branch_configs() -> dict[branch, {"tool_schemas": list, "dispatch_fn": callable, "system_prompt": str}]` — the exact shape Plan 2's `run_eq_assessment` expects. `ctx["eq_rag"]` is the `rag_ctx` shape Task 2 expects (`{"theory_tables", "exemplar_tables", "reranker"}`).

This task is functionally unchanged from the ChromaDB-era design — only the underlying retrieval functions it calls changed (LanceDB tables instead of ChromaDB collections, an added `reranker` key in `rag_ctx`). The tool schemas, dispatcher shape, and system prompts are identical.

- [ ] **Step 1: Write the failing tests**

```python
import lancedb

from src.eq_agent.branch_config import build_branch_configs
from src.eq_agent.graph import BRANCHES
from src.eq_data.build_eq_corpus import _build_table, _make_exemplar_schema, _make_theory_schema
from tests.eq_data.lancedb_test_helpers import make_fake_embedding_func


def _build_rag_ctx(tmp_path):
    embedding_func = make_fake_embedding_func({
        f"{branch} exemplar text": [1.0, 0.0] for branch in BRANCHES
    } | {f"{branch} theory chunk": [1.0, 0.0] for branch in BRANCHES} | {"query": [1.0, 0.0]})
    db = lancedb.connect(str(tmp_path / "lancedb"))
    theory_schema = _make_theory_schema(embedding_func)
    exemplar_schema = _make_exemplar_schema(embedding_func)

    theory_tables, exemplar_tables = {}, {}
    for branch in BRANCHES:
        theory_tables[branch] = _build_table(db, f"eq_theory_{branch}", theory_schema, [
            {"text": f"{branch} theory chunk", "id": f"{branch}-1", "topic": "t", "citation_needed": "yes"},
        ])
        exemplar_tables[branch] = _build_table(db, f"eq_exemplars_{branch}", exemplar_schema, [
            {"text": f"{branch} exemplar text", "tier": 4, "tier_label": "Balanced EQ (Established)", "eq_proxy_score": 45.0},
        ])
    return {"theory_tables": theory_tables, "exemplar_tables": exemplar_tables, "reranker": None}


def test_build_branch_configs_covers_all_four_branches_with_two_tools_each():
    configs = build_branch_configs()

    assert set(configs.keys()) == set(BRANCHES)
    for branch, config in configs.items():
        assert {"tool_schemas", "dispatch_fn", "system_prompt"} <= config.keys()
        tool_names = {s["function"]["name"] for s in config["tool_schemas"]}
        assert tool_names == {"retrieve_similar_eq_exemplars", "retrieve_relevant_msc_theory"}
        assert branch in config["system_prompt"].lower()


def test_dispatch_fn_returns_error_when_eq_rag_absent():
    configs = build_branch_configs()
    ctx = {"eq_rag": None}

    result = configs["perceiving"]["dispatch_fn"]("retrieve_similar_eq_exemplars", {"text": "hello"}, ctx)

    assert "error" in result


def test_dispatch_fn_returns_results_scoped_to_its_own_branch(tmp_path):
    configs = build_branch_configs()
    ctx = {"eq_rag": _build_rag_ctx(tmp_path)}

    result = configs["perceiving"]["dispatch_fn"]("retrieve_similar_eq_exemplars", {"text": "query", "k": 1}, ctx)

    assert "results" in result
    assert result["results"][0]["text"] == "perceiving exemplar text"


def test_dispatch_fn_never_raises_on_unknown_tool():
    configs = build_branch_configs()
    ctx = {"eq_rag": None}

    result = configs["managing"]["dispatch_fn"]("not_a_real_tool", {}, ctx)

    assert "error" in result


def test_dispatch_fn_never_raises_on_bad_arguments(tmp_path):
    configs = build_branch_configs()
    ctx = {"eq_rag": _build_rag_ctx(tmp_path)}

    result = configs["using"]["dispatch_fn"]("retrieve_relevant_msc_theory", {}, ctx)  # missing required "text"

    assert "error" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_agent/test_branch_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eq_agent.branch_config'`

- [ ] **Step 3: Write the implementation**

```python
"""Assembles the real per-branch tool schemas, exception-safe dispatcher,
and system prompt that src.eq_agent.graph.run_eq_assessment's branch_configs
parameter expects -- grounding each MSC specialist in its own branch-scoped
LanceDB hybrid retrieval (src.eq_data.build_eq_corpus's theory/exemplar
tables), with optional cross-encoder reranking.

The dispatcher wraps its entire body in try/except, honoring the
exception-safety expectation src.eq_agent.specialist.run_specialist relies
on (matching the existing src.agent.tool_schemas.dispatch_tool_call
convention).
"""
from src.eq_agent.eq_rag_retrieval import retrieve_relevant_msc_theory, retrieve_similar_eq_exemplars
from src.eq_agent.graph import BRANCHES

RETRIEVE_EXEMPLARS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "retrieve_similar_eq_exemplars",
        "description": "Retrieve similar labeled example texts for this MSC branch, for calibration.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to find similar examples for."},
                "k": {"type": "integer", "description": "How many examples to retrieve.", "default": 5},
            },
            "required": ["text"],
        },
    },
}

RETRIEVE_THEORY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "retrieve_relevant_msc_theory",
        "description": "Retrieve relevant Mayer-Salovey-Caruso emotional-intelligence theory for this branch.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to find relevant theory for."},
                "k": {"type": "integer", "description": "How many theory chunks to retrieve.", "default": 3},
            },
            "required": ["text"],
        },
    },
}

BRANCH_DESCRIPTIONS = {
    "perceiving": "identifying and labeling emotions expressed or implied in text",
    "using": "using emotion to guide thought, decisions, and creativity",
    "understanding": "comprehending the causes, blends, and transitions of emotions",
    "managing": "regulating one's own emotions and, where relevant, helping manage others' emotions",
}


def _make_system_prompt(branch):
    return (
        f"You are an Emotional Intelligence specialist assessing the '{branch}' branch of the "
        f"Mayer-Salovey-Caruso model: {BRANCH_DESCRIPTIONS[branch]}. Use retrieve_similar_eq_exemplars "
        f"and retrieve_relevant_msc_theory as needed to ground your assessment, then call "
        f"submit_branch_assessment exactly once with your score (0-99), confidence, and a rationale "
        f"citing the evidence you gathered."
    )


def _make_dispatch_fn(branch):
    def dispatch_fn(name, arguments, ctx):
        try:
            rag_ctx = ctx.get("eq_rag")
            if not rag_ctx:
                return {"error": "EQ RAG corpus is not available (not built yet)."}

            if name == "retrieve_similar_eq_exemplars":
                k = arguments.get("k", 5)
                return {"results": retrieve_similar_eq_exemplars(arguments["text"], branch, rag_ctx, k=k)}

            if name == "retrieve_relevant_msc_theory":
                k = arguments.get("k", 3)
                return {"results": retrieve_relevant_msc_theory(arguments["text"], branch, rag_ctx, k=k)}

            return {"error": f"Unknown tool: {name}"}
        except Exception as e:
            return {"error": str(e)}
    return dispatch_fn


def build_branch_configs():
    return {
        branch: {
            "tool_schemas": [RETRIEVE_EXEMPLARS_SCHEMA, RETRIEVE_THEORY_SCHEMA],
            "dispatch_fn": _make_dispatch_fn(branch),
            "system_prompt": _make_system_prompt(branch),
        }
        for branch in BRANCHES
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_agent/test_branch_config.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/eq_agent/branch_config.py tests/eq_agent/test_branch_config.py
git commit -m "feat: assemble real per-branch tool schemas, dispatcher, and prompts (LanceDB)"
```

---

### Task 4: EQ RAG context loader

**Files:**
- Create: `src/eq_agent/eq_context.py`
- Test: `tests/eq_agent/test_eq_context.py`

**Interfaces:**
- Consumes: `theory_table_name`, `exemplar_table_name` (Task 1); `BRANCHES` from `src/eq_data/branch_exemplars.py` (Plan 1).
- Produces: `load_eq_rag_context(persist_dir, embedding_func=None, reranker=None) -> rag_ctx | None` where `rag_ctx = {"theory_tables", "exemplar_tables", "reranker"}` — the exact shape Task 2/3 expect. Returns `None` if `persist_dir` doesn't exist, or if any of the 8 expected tables is missing or empty.

- [ ] **Step 1: Write the failing tests**

```python
import lancedb

from src.eq_agent.eq_context import load_eq_rag_context
from src.eq_data.branch_exemplars import BRANCHES
from src.eq_data.build_eq_corpus import (
    _build_table,
    _make_exemplar_schema,
    _make_theory_schema,
    exemplar_table_name,
    theory_table_name,
)
from tests.eq_data.lancedb_test_helpers import make_fake_embedding_func


def test_load_eq_rag_context_returns_none_when_persist_dir_missing(tmp_path):
    result = load_eq_rag_context(str(tmp_path / "does_not_exist"))

    assert result is None


def test_load_eq_rag_context_returns_none_when_a_table_is_missing(tmp_path):
    persist_dir = str(tmp_path / "lancedb")
    embedding_func = make_fake_embedding_func({"x": [1.0, 0.0]})
    db = lancedb.connect(persist_dir)
    # Only populate perceiving's tables -- the other 3 branches are missing entirely.
    _build_table(db, theory_table_name("perceiving"), _make_theory_schema(embedding_func),
                 [{"text": "x", "id": "a", "topic": "t", "citation_needed": "yes"}])
    _build_table(db, exemplar_table_name("perceiving"), _make_exemplar_schema(embedding_func),
                 [{"text": "x", "tier": 4, "tier_label": "l", "eq_proxy_score": 40.0}])

    result = load_eq_rag_context(persist_dir, embedding_func=embedding_func)

    assert result is None


def test_load_eq_rag_context_returns_none_when_a_table_is_empty(tmp_path):
    persist_dir = str(tmp_path / "lancedb")
    embedding_func = make_fake_embedding_func({"x": [1.0, 0.0]})
    db = lancedb.connect(persist_dir)
    for branch in BRANCHES:
        rows = [{"text": "x", "id": "a", "topic": "t", "citation_needed": "yes"}] if branch != "managing" else []
        _build_table(db, theory_table_name(branch), _make_theory_schema(embedding_func), rows)
        _build_table(db, exemplar_table_name(branch), _make_exemplar_schema(embedding_func),
                     [{"text": "x", "tier": 4, "tier_label": "l", "eq_proxy_score": 40.0}])

    result = load_eq_rag_context(persist_dir, embedding_func=embedding_func)

    assert result is None  # managing's theory table is empty (create_table with no rows, no FTS index)


def test_load_eq_rag_context_loads_all_eight_populated_tables(tmp_path):
    persist_dir = str(tmp_path / "lancedb")
    embedding_func = make_fake_embedding_func({"x": [1.0, 0.0]})
    db = lancedb.connect(persist_dir)
    for branch in BRANCHES:
        _build_table(db, theory_table_name(branch), _make_theory_schema(embedding_func),
                     [{"text": "x", "id": "a", "topic": "t", "citation_needed": "yes"}])
        _build_table(db, exemplar_table_name(branch), _make_exemplar_schema(embedding_func),
                     [{"text": "x", "tier": 4, "tier_label": "l", "eq_proxy_score": 40.0}])

    result = load_eq_rag_context(persist_dir, embedding_func=embedding_func, reranker=None)

    assert result is not None
    assert set(result["theory_tables"].keys()) == set(BRANCHES)
    assert set(result["exemplar_tables"].keys()) == set(BRANCHES)
    for branch in BRANCHES:
        assert result["theory_tables"][branch].count_rows() == 1
        assert result["exemplar_tables"][branch].count_rows() == 1
    assert result["reranker"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_agent/test_eq_context.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eq_agent.eq_context'`

- [ ] **Step 3: Write the implementation**

```python
"""Loads the pre-built EQ RAG corpus (8 branch-tagged LanceDB tables) and a
cross-encoder reranker into the shape src.eq_agent.eq_rag_retrieval's
functions and src.eq_agent.branch_config's dispatcher expect.

db.open_table() raises ValueError for a nonexistent table (verified against
lancedb==0.34.0) -- unlike the ChromaDB-era get_or_create_collection, so
table existence is checked via db.list_tables() before opening, not by
catching an open failure.
"""
import os

import lancedb

from src.eq_data.branch_exemplars import BRANCHES
from src.eq_data.build_eq_corpus import exemplar_table_name, theory_table_name


def load_eq_rag_context(persist_dir, embedding_func=None, reranker=None):
    if not os.path.isdir(persist_dir):
        return None

    db = lancedb.connect(persist_dir)
    existing_tables = set(db.list_tables())

    expected_names = [theory_table_name(b) for b in BRANCHES] + [exemplar_table_name(b) for b in BRANCHES]
    if not all(name in existing_tables for name in expected_names):
        return None

    theory_tables = {b: db.open_table(theory_table_name(b)) for b in BRANCHES}
    exemplar_tables = {b: db.open_table(exemplar_table_name(b)) for b in BRANCHES}

    all_tables = list(theory_tables.values()) + list(exemplar_tables.values())
    if any(t.count_rows() == 0 for t in all_tables):
        return None

    if reranker is None:
        from lancedb.rerankers import CrossEncoderReranker
        reranker = CrossEncoderReranker()

    return {"theory_tables": theory_tables, "exemplar_tables": exemplar_tables, "reranker": reranker}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_agent/test_eq_context.py -v`
Expected: 4 passed

- [ ] **Step 5: Run the full project test suite**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: all pass, no regressions (this plan only adds new files under `src/eq_data/`/`src/eq_agent/`/`tests/eq_data/`/`tests/eq_agent/`).

- [ ] **Step 6: Commit**

```bash
git add src/eq_agent/eq_context.py tests/eq_agent/test_eq_context.py
git commit -m "feat: add EQ RAG context loader for the 8 branch-tagged LanceDB tables"
```

---

## After This Plan

Per the approved plan sequence: Plan 4 (NRC-lexicon proxy label enrichment + formalized 3-source ingestion pipeline), Plan 5 (Neo4j knowledge graph for MSC concept relationships), Plan 6 (LangSmith observability), Plan 7 (Backend Integration — a `/predict-eq` endpoint calling `run_eq_assessment(client, models, {"eq_rag": load_eq_rag_context(...)}, text, build_branch_configs())`, caching both once per process), Plan 8 (Evaluation Harness, including Recall@k/MRR/nDCG retrieval metrics and a reranking ablation). Also pending: actually running `python -m src.eq_data.build_eq_corpus` for real to populate `data/eq/lancedb/`.

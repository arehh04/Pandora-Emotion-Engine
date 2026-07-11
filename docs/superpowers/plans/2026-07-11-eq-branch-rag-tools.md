# EQ Per-Branch RAG Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the real per-branch RAG tools that ground each MSC specialist (Perceiving/Using/Understanding/Managing), replacing Plan 2's test-only empty `tool_schemas`/`dispatch_fn` with actual retrieval grounded in Plan 1's MSC theory corpus, branch-tagged Pandora exemplars, and (for Perceiving/Understanding) real external emotion-labeled datasets — then assemble the real `branch_configs` dict that Plan 2's `run_eq_assessment` consumes.

**Architecture:** A corpus builder (`src/eq_data/build_eq_corpus.py`) creates 8 branch-tagged ChromaDB collections (4 theory + 4 exemplar, one pair per MSC branch) using Plan 6's `build_hybrid_collection`/`load_hybrid_collection`/`hybrid_search` **completely unchanged** — no modification to any already-merged file. A retrieval layer (`src/eq_agent/eq_rag_retrieval.py`) wraps `hybrid_search` per branch. A config-assembly layer (`src/eq_agent/branch_config.py`) defines the 2 RAG tool schemas, a branch-scoped exception-safe dispatcher (honoring the exception-safety expectation flagged in Plan 2's Task 1 review), real per-branch system prompts, and `build_branch_configs()` — the function a later Backend Integration plan will call to get real `branch_configs` for `run_eq_assessment`. A context loader (`src/eq_agent/eq_context.py`) loads the pre-built corpus at process startup, mirroring `src/agent/context.py::load_rag_context`'s pattern.

**Tech Stack:** Existing `chromadb`/`rank-bm25` (via Plan 6's `src/rag/hybrid_store.py`, unchanged), `pandas`, `pytest`.

## Global Constraints

- **Deliberate deviation from the design spec's phrasing**: the design spec (`docs/superpowers/specs/2026-07-11-eq-multiagent-langgraph-pivot-design.md`) describes "one shared RAG collection per corpus type... filter via ChromaDB's `where=` at query time." This plan instead builds **4 separate collections per corpus type** (8 total: `eq_theory_{branch}`, `eq_exemplars_{branch}`) so that Plan 6's `build_hybrid_collection`/`load_hybrid_collection`/`hybrid_search` can be reused **literally unchanged** — no new `where=` parameter, no modification to already-merged, already-reviewed code from a prior branch. Simpler and lower-risk than extending shared infrastructure mid-pivot; revisit consolidation into fewer collections later only if collection count becomes unwieldy (it won't at this project's scale — 8 small collections).
- **Verified real constraint**: ChromaDB's `collection.add(metadatas=...)` rejects Python `None` values in a metadata dict with `TypeError: argument 'metadatas': Cannot convert Python object to MetadataValue` (confirmed by direct test against the installed `chromadb` before writing this plan). Any code building metadata dicts from the external datasets (which use `None` for inapplicable fields — e.g. GoEmotions rows have `valence=None`) must **omit** `None`-valued keys entirely rather than including them as `None`.
- Real network/dataset-fetching calls (`fetch_goemotions`, `fetch_isear`, `fetch_emobank`, `fetch_empathetic_dialogues` from Plan 1) are injected as a parameter (`external_fetchers`) into the corpus builder, not called directly inside it — so the automated test suite can inject fake in-memory fetchers instead, exactly like every other real-data-dependent module built in this pivot. No test in this plan makes a real network call.
- `run_specialist`'s (Plan 2) implicit requirement that `dispatch_fn` never raises — confirmed by Plan 2's Task 1 reviewer as this project's existing convention (matching `src/agent/tool_schemas.py::dispatch_tool_call`) — is honored here: this plan's dispatcher wraps its entire body in `try/except Exception`.
- Purely additive except for reading (never modifying) Plan 1's (`src/eq_data/`) and Plan 6's (`src/rag/hybrid_store.py`) already-merged code, and Plan 2's (`src/eq_agent/`) already-merged code. No file under `src/agent/`, `backend/`, or `frontend/` is touched. `src/eq_data/` and `src/eq_agent/` get new files but no `__init__.py` (implicit namespace packages, established convention).
- ChromaDB collection names use the pattern `eq_theory_<branch>` / `eq_exemplars_<branch>` (e.g. `eq_theory_perceiving`, 21 chars — all 8 names are well within the verified 3-512 char, `[a-zA-Z0-9._-]` naming constraint).

---

### Task 1: EQ RAG corpus builder

**Files:**
- Create: `src/eq_data/build_eq_corpus.py`
- Test: `tests/eq_data/test_build_eq_corpus.py`

**Interfaces:**
- Consumes: `sample_branch_balanced_exemplars(df, text_col, n_per_tier, seed)` and `BRANCHES` from `src/eq_data/branch_exemplars.py` (Plan 1); `load_msc_theory_corpus(path)` from `src/eq_data/msc_theory_corpus.py` (Plan 1); `fetch_goemotions`/`fetch_isear`/`fetch_emobank`/`fetch_empathetic_dialogues` from `src/eq_data/external_datasets.py` (Plan 1, injected not called directly); `build_hybrid_collection(persist_dir, collection_name, records, embedder, chunk_size, overlap)` from `src/rag/hybrid_store.py` (Plan 6, unchanged).
- Produces: `theory_collection_name(branch) -> str`, `exemplar_collection_name(branch) -> str`, `build_eq_corpus(pandora_df, data_dir, persist_dir, embedder, external_fetchers=None, n_per_tier=60, seed=42, n_external_samples=200, chunk_size=900, overlap=200) -> (dict[branch, Collection], dict[branch, Collection])` — consumed by Task 4's context loader (the naming functions) and by this plan's own `main()`.

- [ ] **Step 1: Write the failing tests**

```python
import json

import numpy as np
import pandas as pd

from src.eq_data.build_eq_corpus import build_eq_corpus, exemplar_collection_name, theory_collection_name


class FakeEmbedder:
    """Deterministic stand-in for SentenceTransformer.encode() -- avoids
    downloading a real model in the fast unit test suite."""

    def encode(self, texts):
        return np.array([[float(len(t)), float(i)] for i, t in enumerate(texts)])


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


def test_theory_and_exemplar_collection_names_are_branch_specific():
    assert theory_collection_name("perceiving") == "eq_theory_perceiving"
    assert exemplar_collection_name("managing") == "eq_exemplars_managing"


def test_build_eq_corpus_produces_eight_populated_collections(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    persist_dir = tmp_path / "chroma"
    _write_fixture_theory_corpus(data_dir)
    embedder = FakeEmbedder()

    theory_collections, exemplar_collections = build_eq_corpus(
        _fake_pandora_df(), str(data_dir), str(persist_dir), embedder,
        external_fetchers={"perceiving": [], "using": [], "understanding": [], "managing": []},
        n_per_tier=2, seed=1,
    )

    assert set(theory_collections.keys()) == {"perceiving", "using", "understanding", "managing"}
    assert set(exemplar_collections.keys()) == {"perceiving", "using", "understanding", "managing"}
    for branch in theory_collections:
        assert theory_collections[branch].count() > 0
    for branch in exemplar_collections:
        assert exemplar_collections[branch].count() > 0


def test_build_eq_corpus_enriches_perceiving_and_understanding_with_external_records(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    persist_dir = tmp_path / "chroma"
    _write_fixture_theory_corpus(data_dir)
    embedder = FakeEmbedder()

    _, exemplar_collections = build_eq_corpus(
        _fake_pandora_df(), str(data_dir), str(persist_dir), embedder,
        external_fetchers={
            "perceiving": [_fake_external_fetcher(5)], "using": [],
            "understanding": [_fake_external_fetcher(3)], "managing": [],
        },
        n_per_tier=2, seed=1, n_external_samples=200,
    )

    perceiving_docs = exemplar_collections["perceiving"].get()["documents"]
    understanding_docs = exemplar_collections["understanding"].get()["documents"]
    using_docs = exemplar_collections["using"].get()["documents"]

    assert any("external text" in d for d in perceiving_docs)
    assert any("external text" in d for d in understanding_docs)
    assert not any("external text" in d for d in using_docs)  # no fetcher injected for "using"


def test_build_eq_corpus_omits_none_metadata_fields_from_external_records(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    persist_dir = tmp_path / "chroma"
    _write_fixture_theory_corpus(data_dir)
    embedder = FakeEmbedder()

    # This must not raise -- ChromaDB rejects None metadata values, so the
    # builder must omit valence/arousal/dominance (all None for this fetcher).
    _, exemplar_collections = build_eq_corpus(
        _fake_pandora_df(), str(data_dir), str(persist_dir), embedder,
        external_fetchers={"perceiving": [_fake_external_fetcher(2)], "using": [], "understanding": [], "managing": []},
        n_per_tier=2, seed=1,
    )

    metadatas = exemplar_collections["perceiving"].get()["metadatas"]
    external_metas = [m for m in metadatas if m.get("source") == "fake_source"]
    assert len(external_metas) > 0
    for meta in external_metas:
        assert "valence" not in meta
        assert "arousal" not in meta
        assert "dominance" not in meta
        assert meta["emotion_labels"] == "joy"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_data/test_build_eq_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eq_data.build_eq_corpus'`

- [ ] **Step 3: Write the implementation**

```python
"""Builds the EQ RAG knowledge base: 4 branch-tagged theory collections and
4 branch-tagged exemplar collections. Perceiving and Understanding exemplar
collections are additionally enriched with a sample of real external
emotion-labeled text (see external_fetchers). All storage goes through
src.rag.hybrid_store's build_hybrid_collection, completely unchanged from
its Plan 6 form -- see this plan's Global Constraints for why this uses 8
separate collections instead of a single shared one with a metadata filter.
"""
import os

from src.eq_data.branch_exemplars import BRANCHES, sample_branch_balanced_exemplars
from src.eq_data.external_datasets import (
    fetch_emobank,
    fetch_empathetic_dialogues,
    fetch_goemotions,
    fetch_isear,
)
from src.eq_data.msc_theory_corpus import load_msc_theory_corpus
from src.rag.hybrid_store import build_hybrid_collection

DEFAULT_EXTERNAL_FETCHERS = {
    "perceiving": [fetch_goemotions, fetch_isear, fetch_emobank],
    "using": [],
    "understanding": [fetch_empathetic_dialogues],
    "managing": [],
}


def theory_collection_name(branch):
    return f"eq_theory_{branch}"


def exemplar_collection_name(branch):
    return f"eq_exemplars_{branch}"


def _theory_records_for_branch(theory_entries, branch):
    return [
        {"text": e["text"], "metadata": {"id": e["id"], "topic": e["topic"], "citation_needed": e["citation_needed"]}}
        for e in theory_entries if e["branch"] == branch
    ]


def _exemplar_records_for_branch(exemplars_df, branch):
    branch_df = exemplars_df[exemplars_df["branch"] == branch]
    return [
        {"text": row["text"], "metadata": {
            "eq_proxy_score": float(row["eq_proxy_score"]), "tier": int(row["tier"]), "tier_label": row["tier_label"],
        }}
        for _, row in branch_df.iterrows()
    ]


def _external_records(df, n_samples, seed):
    sample = df.sample(n=min(n_samples, len(df)), random_state=seed)
    records = []
    for _, row in sample.iterrows():
        metadata = {"source": row["source"]}
        if row["emotion_labels"]:
            metadata["emotion_labels"] = ",".join(row["emotion_labels"])
        if row["valence"] is not None:
            metadata["valence"] = float(row["valence"])
        if row["arousal"] is not None:
            metadata["arousal"] = float(row["arousal"])
        if row["dominance"] is not None:
            metadata["dominance"] = float(row["dominance"])
        records.append({"text": row["text"], "metadata": metadata})
    return records


def build_eq_corpus(
    pandora_df, data_dir, persist_dir, embedder, external_fetchers=None,
    n_per_tier=60, seed=42, n_external_samples=200, chunk_size=900, overlap=200,
):
    if external_fetchers is None:
        external_fetchers = DEFAULT_EXTERNAL_FETCHERS

    theory_entries = load_msc_theory_corpus(os.path.join(data_dir, "eq", "msc_theory_corpus.json"))
    exemplars_df = sample_branch_balanced_exemplars(pandora_df, n_per_tier=n_per_tier, seed=seed)

    theory_collections = {}
    exemplar_collections = {}

    for branch in BRANCHES:
        theory_records = _theory_records_for_branch(theory_entries, branch)
        theory_collections[branch] = build_hybrid_collection(
            persist_dir, theory_collection_name(branch), theory_records, embedder,
            chunk_size=chunk_size, overlap=overlap,
        )

        exemplar_records = _exemplar_records_for_branch(exemplars_df, branch)
        for fetcher in external_fetchers.get(branch, []):
            exemplar_records += _external_records(fetcher(), n_external_samples, seed)

        exemplar_collections[branch] = build_hybrid_collection(
            persist_dir, exemplar_collection_name(branch), exemplar_records, embedder,
            chunk_size=chunk_size, overlap=overlap,
        )

    return theory_collections, exemplar_collections


def main():
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data")
    persist_dir = os.path.join(data_dir, "eq", "chroma")
    os.makedirs(persist_dir, exist_ok=True)

    pandora_df = pd.read_csv(os.path.join(data_dir, "train_set.csv"))
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    theory_collections, exemplar_collections = build_eq_corpus(pandora_df, data_dir, persist_dir, embedder)

    for branch in theory_collections:
        print(f"{branch}: theory={theory_collections[branch].count()} chunks, "
              f"exemplars={exemplar_collections[branch].count()} chunks")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_data/test_build_eq_corpus.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/eq_data/build_eq_corpus.py tests/eq_data/test_build_eq_corpus.py
git commit -m "feat: add EQ RAG corpus builder (8 branch-tagged ChromaDB collections)"
```

---

### Task 2: Branch-scoped hybrid retrieval

**Files:**
- Create: `src/eq_agent/eq_rag_retrieval.py`
- Test: `tests/eq_agent/test_eq_rag_retrieval.py`

**Interfaces:**
- Consumes: `hybrid_search(query_text, collection, embedder, k, rrf_k)` from `src/rag/hybrid_store.py` (Plan 6, unchanged); `build_hybrid_collection` (Plan 6, for test fixtures only).
- Produces: `retrieve_similar_eq_exemplars(query_text, branch, rag_ctx, k=5) -> list[dict]` and `retrieve_relevant_msc_theory(query_text, branch, rag_ctx, k=3) -> list[dict]`, where `rag_ctx = {"theory_collections": dict[branch, Collection], "exemplar_collections": dict[branch, Collection], "embedder": embedder}` — consumed by Task 3's dispatcher.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from src.eq_agent.eq_rag_retrieval import retrieve_relevant_msc_theory, retrieve_similar_eq_exemplars
from src.rag.hybrid_store import build_hybrid_collection


class FakeEmbedder:
    def __init__(self, vector_by_text):
        self.vector_by_text = vector_by_text

    def encode(self, texts):
        return np.array([self.vector_by_text[t] for t in texts])


def test_retrieve_similar_eq_exemplars_scopes_to_the_given_branch(tmp_path):
    vector_by_text = {
        "I noticed my friend seemed upset": [1.0, 0.0],
        "I stayed calm under pressure": [0.0, 1.0],
        "query about noticing emotions": [1.0, 0.0],
    }
    embedder = FakeEmbedder(vector_by_text)
    perceiving_records = [{"text": "I noticed my friend seemed upset", "metadata": {"tier": 5, "tier_label": "Above Average EQ"}}]
    managing_records = [{"text": "I stayed calm under pressure", "metadata": {"tier": 6, "tier_label": "High EQ"}}]

    perceiving_collection = build_hybrid_collection(str(tmp_path / "chroma"), "eq_exemplars_perceiving", perceiving_records, embedder)
    managing_collection = build_hybrid_collection(str(tmp_path / "chroma"), "eq_exemplars_managing", managing_records, embedder)

    rag_ctx = {
        "exemplar_collections": {"perceiving": perceiving_collection, "managing": managing_collection},
        "theory_collections": {},
        "embedder": embedder,
    }

    hits = retrieve_similar_eq_exemplars("query about noticing emotions", "perceiving", rag_ctx, k=1)

    assert len(hits) == 1
    assert hits[0]["text"] == "I noticed my friend seemed upset"
    assert hits[0]["tier_label"] == "Above Average EQ"
    assert "score" in hits[0]


def test_retrieve_relevant_msc_theory_scopes_to_the_given_branch(tmp_path):
    vector_by_text = {
        "perceiving emotions theory chunk": [1.0, 0.0],
        "managing emotions theory chunk": [0.0, 1.0],
        "query about theory": [1.0, 0.0],
    }
    embedder = FakeEmbedder(vector_by_text)
    perceiving_records = [{"text": "perceiving emotions theory chunk", "metadata": {"id": "p1", "topic": "t"}}]
    managing_records = [{"text": "managing emotions theory chunk", "metadata": {"id": "m1", "topic": "t"}}]

    perceiving_collection = build_hybrid_collection(str(tmp_path / "chroma"), "eq_theory_perceiving", perceiving_records, embedder)
    managing_collection = build_hybrid_collection(str(tmp_path / "chroma"), "eq_theory_managing", managing_records, embedder)

    rag_ctx = {
        "theory_collections": {"perceiving": perceiving_collection, "managing": managing_collection},
        "exemplar_collections": {},
        "embedder": embedder,
    }

    hits = retrieve_relevant_msc_theory("query about theory", "perceiving", rag_ctx, k=1)

    assert len(hits) == 1
    assert hits[0]["id"] == "p1"
    assert "score" in hits[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_agent/test_eq_rag_retrieval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eq_agent.eq_rag_retrieval'`

- [ ] **Step 3: Write the implementation**

```python
"""Branch-scoped hybrid retrieval over the EQ RAG corpus built by
src.eq_data.build_eq_corpus. Each MSC branch has its own theory and
exemplar collection (see src.eq_data.build_eq_corpus's Global Constraints
note on why this is 8 separate collections, not one shared collection with
a metadata filter).
"""
from src.rag.hybrid_store import hybrid_search


def retrieve_similar_eq_exemplars(query_text, branch, rag_ctx, k=5):
    collection = rag_ctx["exemplar_collections"][branch]
    hits = hybrid_search(query_text, collection, rag_ctx["embedder"], k=k)
    return [{**hit["metadata"], "text": hit["document"], "score": hit["score"]} for hit in hits]


def retrieve_relevant_msc_theory(query_text, branch, rag_ctx, k=3):
    collection = rag_ctx["theory_collections"][branch]
    hits = hybrid_search(query_text, collection, rag_ctx["embedder"], k=k)
    return [{**hit["metadata"], "text": hit["document"], "score": hit["score"]} for hit in hits]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_agent/test_eq_rag_retrieval.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/eq_agent/eq_rag_retrieval.py tests/eq_agent/test_eq_rag_retrieval.py
git commit -m "feat: add branch-scoped hybrid retrieval for the EQ RAG corpus"
```

---

### Task 3: Real branch_configs (tool schemas, dispatcher, system prompts)

**Files:**
- Create: `src/eq_agent/branch_config.py`
- Test: `tests/eq_agent/test_branch_config.py`

**Interfaces:**
- Consumes: `retrieve_similar_eq_exemplars`, `retrieve_relevant_msc_theory` (Task 2); `BRANCHES` from `src/eq_agent/graph.py` (Plan 2).
- Produces: `RETRIEVE_EXEMPLARS_SCHEMA`, `RETRIEVE_THEORY_SCHEMA` (tool schema dicts); `build_branch_configs() -> dict[branch, {"tool_schemas": list, "dispatch_fn": callable, "system_prompt": str}]` — this is the exact shape Plan 2's `run_eq_assessment(client, models, ctx, text, branch_configs, ...)` expects, and `ctx["eq_rag"]` is the `rag_ctx` shape Task 2's retrieval functions expect (i.e. a later Backend Integration plan builds `ctx = {"eq_rag": load_eq_rag_context(...)}` and calls `run_eq_assessment(client, models, ctx, text, build_branch_configs())`).

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from src.eq_agent.branch_config import build_branch_configs
from src.eq_agent.graph import BRANCHES
from src.rag.hybrid_store import build_hybrid_collection


class FakeEmbedder:
    def encode(self, texts):
        return np.array([[1.0, 0.0] for _ in texts])


def _build_rag_ctx(tmp_path):
    embedder = FakeEmbedder()
    persist_dir = str(tmp_path / "chroma")
    theory_collections, exemplar_collections = {}, {}
    for branch in BRANCHES:
        theory_collections[branch] = build_hybrid_collection(
            persist_dir, f"eq_theory_{branch}",
            [{"text": f"{branch} theory chunk", "metadata": {"id": f"{branch}-1", "topic": "t"}}], embedder,
        )
        exemplar_collections[branch] = build_hybrid_collection(
            persist_dir, f"eq_exemplars_{branch}",
            [{"text": f"{branch} exemplar text", "metadata": {"tier": 4, "tier_label": "Balanced EQ (Established)"}}], embedder,
        )
    return {"theory_collections": theory_collections, "exemplar_collections": exemplar_collections, "embedder": embedder}


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
RAG retrieval (src.eq_data.build_eq_corpus's theory/exemplar collections).

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
git commit -m "feat: assemble real per-branch tool schemas, dispatcher, and prompts"
```

---

### Task 4: EQ RAG context loader

**Files:**
- Create: `src/eq_agent/eq_context.py`
- Test: `tests/eq_agent/test_eq_context.py`

**Interfaces:**
- Consumes: `theory_collection_name`, `exemplar_collection_name` (Task 1); `load_hybrid_collection` from `src/rag/hybrid_store.py` (Plan 6, unchanged); `BRANCHES` from `src/eq_data/branch_exemplars.py` (Plan 1).
- Produces: `load_eq_rag_context(persist_dir, embedder=None) -> rag_ctx | None` where `rag_ctx` is the exact shape Task 2's retrieval functions and Task 3's dispatcher expect (`{"theory_collections", "exemplar_collections", "embedder"}`). Returns `None` if `persist_dir` doesn't exist, or if any of the 8 collections is empty (corpus not built yet).

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from src.eq_agent.eq_context import load_eq_rag_context
from src.eq_data.build_eq_corpus import exemplar_collection_name, theory_collection_name
from src.eq_data.branch_exemplars import BRANCHES
from src.rag.hybrid_store import build_hybrid_collection


class FakeEmbedder:
    def encode(self, texts):
        return np.array([[float(len(t)), 0.0] for t in texts])


def test_load_eq_rag_context_returns_none_when_persist_dir_missing(tmp_path):
    result = load_eq_rag_context(str(tmp_path / "does_not_exist"), embedder=FakeEmbedder())

    assert result is None


def test_load_eq_rag_context_returns_none_when_any_collection_is_empty(tmp_path):
    persist_dir = str(tmp_path / "chroma")
    embedder = FakeEmbedder()
    # Only populate perceiving's collections -- the other 3 branches are missing/empty.
    build_hybrid_collection(persist_dir, theory_collection_name("perceiving"), [{"text": "x", "metadata": {}}], embedder)
    build_hybrid_collection(persist_dir, exemplar_collection_name("perceiving"), [{"text": "x", "metadata": {}}], embedder)

    result = load_eq_rag_context(persist_dir, embedder=embedder)

    assert result is None


def test_load_eq_rag_context_loads_all_eight_populated_collections(tmp_path):
    persist_dir = str(tmp_path / "chroma")
    embedder = FakeEmbedder()
    for branch in BRANCHES:
        build_hybrid_collection(persist_dir, theory_collection_name(branch), [{"text": f"{branch} theory", "metadata": {}}], embedder)
        build_hybrid_collection(persist_dir, exemplar_collection_name(branch), [{"text": f"{branch} exemplar", "metadata": {}}], embedder)

    result = load_eq_rag_context(persist_dir, embedder=embedder)

    assert result is not None
    assert set(result["theory_collections"].keys()) == set(BRANCHES)
    assert set(result["exemplar_collections"].keys()) == set(BRANCHES)
    for branch in BRANCHES:
        assert result["theory_collections"][branch].count() == 1
        assert result["exemplar_collections"][branch].count() == 1
    assert result["embedder"] is embedder
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_agent/test_eq_context.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eq_agent.eq_context'`

- [ ] **Step 3: Write the implementation**

```python
"""Loads the pre-built EQ RAG corpus (8 branch-tagged ChromaDB collections)
into the shape src.eq_agent.eq_rag_retrieval's functions and
src.eq_agent.branch_config's dispatcher expect, mirroring
src.agent.context.load_rag_context's pattern for the Extraversion-era corpus.
"""
import os

from src.eq_data.branch_exemplars import BRANCHES
from src.eq_data.build_eq_corpus import exemplar_collection_name, theory_collection_name
from src.rag.hybrid_store import load_hybrid_collection


def load_eq_rag_context(persist_dir, embedder=None):
    if not os.path.isdir(persist_dir):
        return None

    theory_collections = {b: load_hybrid_collection(persist_dir, theory_collection_name(b)) for b in BRANCHES}
    exemplar_collections = {b: load_hybrid_collection(persist_dir, exemplar_collection_name(b)) for b in BRANCHES}

    all_collections = list(theory_collections.values()) + list(exemplar_collections.values())
    if any(c.count() == 0 for c in all_collections):
        return None

    if embedder is None:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")

    return {"theory_collections": theory_collections, "exemplar_collections": exemplar_collections, "embedder": embedder}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_agent/test_eq_context.py -v`
Expected: 3 passed

- [ ] **Step 5: Run the full project test suite**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: all pass, no regressions (this plan only adds new files under `src/eq_data/`/`src/eq_agent/`/`tests/eq_data/`/`tests/eq_agent/`).

- [ ] **Step 6: Commit**

```bash
git add src/eq_agent/eq_context.py tests/eq_agent/test_eq_context.py
git commit -m "feat: add EQ RAG context loader for the 8 branch-tagged collections"
```

---

## After This Plan

The next plan is Backend Integration: a `/predict-eq` FastAPI endpoint (deciding how it coexists with or replaces the existing `/predict-agent`), caching `load_eq_rag_context(...)` and `build_branch_configs()` once per process (mirroring `backend/agent_router.py`'s existing caching pattern), and calling `run_eq_assessment(client, models, {"eq_rag": cached_rag_ctx}, text, cached_branch_configs)`. After that: the Evaluation Harness (per-branch metrics against GoEmotions/ISEAR/EmoBank/EmpatheticDialogues ground truth and the proxy label, ablations, LLM-judge routing to a different model) from the design spec, plus actually running `python -m src.eq_data.build_eq_corpus` for real to populate `data/eq/chroma/`.

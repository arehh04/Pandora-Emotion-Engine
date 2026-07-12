# EQ Evaluation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an evaluation harness for the EQ multi-agent pipeline (Plans 1-7): generic Recall@k/MRR/nDCG retrieval-quality metrics, a reranking ablation over the EQ RAG exemplar tables, and a score/tier/faithfulness comparison harness for the full multi-agent assessment — completing the approved 8-plan EQ pivot sequence.

**Architecture:** Three independent, composable modules under `src/evaluation/`: (1) domain-agnostic IR metric functions operating on ranked relevance-value lists; (2) a branch-scoped retrieval ablation that retrieves the same candidate pool via LanceDB's native hybrid search once, then compares Recall@k/MRR/nDCG@k between the plain hybrid order and a cross-encoder-reranked order of that same pool; (3) an EQ score/tier/faithfulness comparison harness mirroring `src/evaluation/run_comparison.py`'s shape (predict_fn + run + summarize) but for the EQ pipeline, using `src/eq_data/nrc_enrichment.py`'s NRC-blended proxy label as ground truth.

**Tech Stack:** `lancedb.rerankers.CrossEncoderReranker` (already an installed dependency via `sentence-transformers`), the existing `src/eq_agent/`, `src/eq_data/`, `src/evaluation/` packages.

## Global Constraints

- Purely additive: only new files under `src/evaluation/` and `tests/evaluation/`. Do not modify `src/evaluation/run_comparison.py`, `src/evaluation/metrics.py`, `src/evaluation/faithfulness.py`, or anything under `src/eq_agent/`/`src/eq_data/`/`backend/` — reuse only, via import.
- No labeled query/relevant-document-id test set exists for the EQ RAG corpus, and none is built by this plan. Retrieval relevance is defined per-query as "the candidate's own `tier` metadata matches the query's own known tier" — the same tier-similarity signal `src/eq_agent/branch_config.py`'s `retrieve_similar_eq_exemplars` tool already exists to surface to specialist LLMs for calibration. This makes the retrieval evaluation (Task 2) an ablation on how reranking reorders a real retrieved candidate pool, not an absolute corpus-wide recall benchmark — that distinction must stay documented in the module docstring, not silently implied.
- Task 2's tests use a real `lancedb.rerankers.CrossEncoderReranker()` (verified: loads its ~105-tensor model in well under a second locally, no network call needed once the HF cache is warm) rather than a hand-rolled fake — this is a deliberate, explicitly-flagged exception to this project's usual "fakes only in the fast unit suite" convention (see `tests/eq_data/lancedb_test_helpers.py`'s embedder fake), because a correct fake would have to duck-type `lancedb.rerankers.Reranker.rerank_hybrid`'s raw `pyarrow.Table` transform, and the whole point of this task is exercising real reranking behavior.
- The ground-truth EQ proxy label for Task 3 is the NRC-enriched proxy (`src/eq_data/nrc_enrichment.py::compute_enriched_overall_eq_proxy`), not the plain Big-Five-only proxy (`src/eq_data/proxy_labels.py::compute_overall_eq_proxy`) — per the user's earlier explicit approval of NRC blending as the improved proxy. This is independent of what the LanceDB exemplar tables were built with (they still use the plain proxy, per Plan 3/4 — rebuilding the corpus with the enriched proxy remains a separate deferred item); evaluating LLM predictions against the best available ground-truth label is orthogonal to what documents happen to be in a retrieval corpus.
- `HISTORICAL_BASELINES`-style classical-ML comparison is explicitly out of scope here, same decision preserved from the Extraversion pivot's evaluation harness: EQ has no retrained classical-ML baseline to compare against (Ridge/XGBoost/RandomForest were never retrained for EQ), so this harness only compares LLM-configuration vs LLM-configuration (and, within Task 2, hybrid-only vs hybrid-reranked).
- All new tests are part of the automated `pytest` suite (no `real_evaluation`-style manual script in this plan) — this plan only adds the reusable harness functions, not a manual real-API-call runner script (unlike `src/evaluation/run_real_evaluation.py`, which was a separate, later, explicitly-deferred addition for the Extraversion pipeline). A future manual run using these functions with a real corpus/API key is left to whoever executes it, exactly like the Extraversion pipeline's own deferred real-run step.

---

### Task 1: Generic retrieval metrics

**Files:**
- Create: `src/evaluation/retrieval_metrics.py`
- Test: `tests/evaluation/test_retrieval_metrics.py`

**Interfaces:**
- Produces: `recall_at_k(relevances, total_relevant, k) -> float`, `reciprocal_rank(relevances) -> float`, `ndcg_at_k(relevances, k) -> float`. All three take `relevances`: a list of graded relevance values (here always `0`/`1`) **already in ranked order** (rank 1 first) — not document IDs. `total_relevant` is the count of relevant items in the full candidate pool being evaluated (the caller computes this; Task 2 computes it as `sum()` of the full un-truncated relevance list for that query). Verified against `sklearn.metrics.ndcg_score` computationally before writing this task (own worked example: `relevances=[1,0,1]` → `ndcg_at_k(relevances, 3) == 0.9197207891481876`, matching sklearn to floating-point precision) — no need to re-derive that check, just implement the verified formula below.

- [ ] **Step 1: Write the failing tests**

```python
import math

from src.evaluation.retrieval_metrics import ndcg_at_k, recall_at_k, reciprocal_rank


def test_recall_at_k_counts_hits_within_top_k_over_total_relevant():
    relevances = [1, 0, 1]  # rank 1 and rank 3 are relevant, rank 2 is not

    assert recall_at_k(relevances, total_relevant=2, k=2) == 0.5
    assert recall_at_k(relevances, total_relevant=2, k=3) == 1.0


def test_recall_at_k_returns_zero_when_nothing_is_relevant():
    assert recall_at_k([0, 0, 0], total_relevant=0, k=2) == 0.0


def test_reciprocal_rank_returns_inverse_of_first_relevant_position():
    assert reciprocal_rank([0, 1, 1]) == 0.5
    assert reciprocal_rank([1, 0, 0]) == 1.0


def test_reciprocal_rank_returns_zero_when_nothing_relevant():
    assert reciprocal_rank([0, 0]) == 0.0


def test_ndcg_at_k_matches_verified_worked_example():
    relevances = [1, 0, 1]

    result = ndcg_at_k(relevances, k=3)

    assert math.isclose(result, 0.9197207891481876, rel_tol=1e-9)


def test_ndcg_at_k_returns_zero_when_nothing_relevant():
    assert ndcg_at_k([0, 0], k=2) == 0.0


def test_ndcg_at_k_returns_one_for_perfectly_ordered_relevance():
    assert ndcg_at_k([1, 1, 0], k=3) == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/evaluation/test_retrieval_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.evaluation.retrieval_metrics'`

- [ ] **Step 3: Write the implementation**

```python
"""Domain-agnostic information-retrieval metrics operating on a ranked list
of relevance values (not document IDs) -- the caller determines relevance
per its own domain (src.evaluation.eq_retrieval_evaluation's caller uses
tier-match). Formulas verified against sklearn.metrics.ndcg_score before
this module was written (see this plan's Task 1 docstring for the worked
example); no external dependency is added since the formulas are this
simple and self-contained.
"""
import math


def recall_at_k(relevances, total_relevant, k):
    if not total_relevant:
        return 0.0
    hits = sum(relevances[:k])
    return hits / total_relevant


def reciprocal_rank(relevances):
    for i, rel in enumerate(relevances, start=1):
        if rel > 0:
            return 1.0 / i
    return 0.0


def ndcg_at_k(relevances, k):
    top_k = relevances[:k]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(top_k))
    ideal = sorted(relevances, reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/evaluation/test_retrieval_metrics.py -v`
Expected: 7 passed

- [ ] **Step 5: Run the full project test suite**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: all pass, no regressions (this task only adds new files).

- [ ] **Step 6: Commit**

```bash
git add src/evaluation/retrieval_metrics.py tests/evaluation/test_retrieval_metrics.py
git commit -m "feat: add generic Recall@k/MRR/nDCG retrieval metrics"
```

---

### Task 2: Reranking ablation over the EQ exemplar tables

**Files:**
- Create: `src/evaluation/eq_retrieval_evaluation.py`
- Test: `tests/evaluation/test_eq_retrieval_evaluation.py`

**Interfaces:**
- Consumes: `src.evaluation.retrieval_metrics.recall_at_k/reciprocal_rank/ndcg_at_k` (Task 1). `src.eq_data.build_eq_corpus._build_table`/`_make_exemplar_schema` and `tests.eq_data.lancedb_test_helpers.make_fake_embedding_func` (both already exist, Plan 3) — for tests only. `lancedb.rerankers.CrossEncoderReranker` (already an installed dependency).
- Produces: `evaluate_branch_reranking_ablation(exemplar_table, queries, reranker, fetch_k=50, k_values=(3, 5)) -> dict` where `queries` is a list of `(query_text, query_tier)` tuples and the return shape is `{"hybrid_only": {...averaged metrics...}, "hybrid_reranked": {...averaged metrics...}}`, each averaged-metrics dict containing `"mrr"` plus `"recall@{k}"`/`"ndcg@{k}"` for each `k` in `k_values` (or `{}` if no query produced any candidates, e.g. an empty table).

- [ ] **Step 1: Write the failing tests**

```python
import lancedb
from lancedb.rerankers import CrossEncoderReranker

from src.eq_data.build_eq_corpus import _build_table, _make_exemplar_schema
from src.evaluation.eq_retrieval_evaluation import evaluate_branch_reranking_ablation
from tests.eq_data.lancedb_test_helpers import make_fake_embedding_func


def _make_table(tmp_path):
    vector_by_text = {
        "I felt completely overwhelmed and could not calm down": [1.0, 0.0],
        "I noticed my coworker was upset before they said anything": [0.0, 1.0],
        "The weather was sunny today": [1.0, 1.0],
        "I could tell my friend was feeling sad even though they said nothing": [0.0, 1.0],
    }
    embedding_func = make_fake_embedding_func(vector_by_text)
    db = lancedb.connect(str(tmp_path / "lancedb"))
    schema = _make_exemplar_schema(embedding_func)
    return _build_table(db, "eq_exemplars_perceiving", schema, [
        {"text": "I felt completely overwhelmed and could not calm down", "tier": 2},
        {"text": "I noticed my coworker was upset before they said anything", "tier": 5},
        {"text": "The weather was sunny today", "tier": 3},
    ])


def test_evaluate_branch_reranking_ablation_returns_metrics_for_both_conditions(tmp_path):
    table = _make_table(tmp_path)
    queries = [("I could tell my friend was feeling sad even though they said nothing", 5)]

    result = evaluate_branch_reranking_ablation(table, queries, CrossEncoderReranker(), fetch_k=3, k_values=(2, 3))

    assert set(result.keys()) == {"hybrid_only", "hybrid_reranked"}
    for condition in result.values():
        assert set(condition.keys()) == {"mrr", "recall@2", "recall@3", "ndcg@2", "ndcg@3"}


def test_evaluate_branch_reranking_ablation_handles_no_relevant_candidates(tmp_path):
    table = _make_table(tmp_path)
    queries = [("The weather was sunny today", 99)]  # tier 99 matches nothing in the table

    result = evaluate_branch_reranking_ablation(table, queries, CrossEncoderReranker(), fetch_k=3, k_values=(2,))

    assert result["hybrid_only"]["recall@2"] == 0.0
    assert result["hybrid_only"]["mrr"] == 0.0


def test_evaluate_branch_reranking_ablation_returns_empty_dicts_for_empty_table(tmp_path):
    embedding_func = make_fake_embedding_func({"anything": [1.0, 0.0]})
    db = lancedb.connect(str(tmp_path / "lancedb"))
    empty_table = _build_table(db, "eq_exemplars_using", _make_exemplar_schema(embedding_func), [])

    result = evaluate_branch_reranking_ablation(empty_table, [("anything", 1)], CrossEncoderReranker(), fetch_k=3)

    assert result == {"hybrid_only": {}, "hybrid_reranked": {}}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/evaluation/test_eq_retrieval_evaluation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.evaluation.eq_retrieval_evaluation'`

- [ ] **Step 3: Write the implementation**

```python
"""Reranking ablation over an EQ RAG exemplar table: retrieves the same
fetch_k candidate pool once via LanceDB's native hybrid search, then
compares Recall@k/MRR/nDCG@k between the plain hybrid order and a
cross-encoder-reranked order of that same pool -- isolating what reranking
changes, independent of absolute recall over the whole table.

Relevance is defined per-query as "the candidate's own tier metadata
matches the query's own known tier" -- the same tier-similarity signal
src.eq_agent.branch_config's retrieve_similar_eq_exemplars tool already
surfaces to specialist LLMs for calibration. No labeled query/relevant-doc
test set exists for this corpus (see this plan's Global Constraints), so
this evaluates reranking's effect on ordering a real candidate pool rather
than absolute recall against a ground-truth corpus-wide relevant set.
"""
from src.evaluation.retrieval_metrics import ndcg_at_k, recall_at_k, reciprocal_rank


def _hybrid_candidate_pool(table, query_text, fetch_k):
    if table.count_rows() == 0:
        return []
    df = table.search(query_text, query_type="hybrid").limit(fetch_k).to_pandas()
    return df.to_dict("records")


def _reranked_candidate_pool(table, query_text, fetch_k, reranker):
    if table.count_rows() == 0:
        return []
    df = (
        table.search(query_text, query_type="hybrid")
        .limit(fetch_k)
        .rerank(reranker=reranker)
        .limit(fetch_k)
        .to_pandas()
    )
    return df.to_dict("records")


def _relevances_for_query(candidates, query_tier):
    return [1 if c.get("tier") == query_tier else 0 for c in candidates]


def _average_metrics(rows, k_values):
    if not rows:
        return {}
    n = len(rows)
    result = {"mrr": sum(r["mrr"] for r in rows) / n}
    for k in k_values:
        result[f"recall@{k}"] = sum(r[f"recall@{k}"] for r in rows) / n
        result[f"ndcg@{k}"] = sum(r[f"ndcg@{k}"] for r in rows) / n
    return result


def evaluate_branch_reranking_ablation(exemplar_table, queries, reranker, fetch_k=50, k_values=(3, 5)):
    per_condition = {"hybrid_only": [], "hybrid_reranked": []}

    for query_text, query_tier in queries:
        hybrid_candidates = _hybrid_candidate_pool(exemplar_table, query_text, fetch_k)
        if not hybrid_candidates:
            continue
        reranked_candidates = _reranked_candidate_pool(exemplar_table, query_text, fetch_k, reranker)

        hybrid_relevances = _relevances_for_query(hybrid_candidates, query_tier)
        reranked_relevances = _relevances_for_query(reranked_candidates, query_tier)
        total_relevant = sum(hybrid_relevances)  # same underlying candidate pool for both conditions

        for condition, relevances in [("hybrid_only", hybrid_relevances), ("hybrid_reranked", reranked_relevances)]:
            row = {}
            for k in k_values:
                row[f"recall@{k}"] = recall_at_k(relevances, total_relevant, k)
                row[f"ndcg@{k}"] = ndcg_at_k(relevances, k)
            row["mrr"] = reciprocal_rank(relevances)
            per_condition[condition].append(row)

    return {condition: _average_metrics(rows, k_values) for condition, rows in per_condition.items()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/evaluation/test_eq_retrieval_evaluation.py -v`
Expected: 3 passed (the `CrossEncoderReranker()` model load may take a few seconds the first time; it is a small model already used elsewhere in this environment)

- [ ] **Step 5: Run the full project test suite**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: all pass, no regressions (this task only adds new files).

- [ ] **Step 6: Commit**

```bash
git add src/evaluation/eq_retrieval_evaluation.py tests/evaluation/test_eq_retrieval_evaluation.py
git commit -m "feat: add reranking ablation over the EQ RAG exemplar tables"
```

---

### Task 3: EQ score/tier/faithfulness comparison harness

**Files:**
- Create: `src/evaluation/eq_comparison.py`
- Test: `tests/evaluation/test_eq_comparison.py`

**Interfaces:**
- Consumes: `src.eq_agent.traced_assessment.traced_run_eq_assessment(client, models, ctx, text, branch_configs, extra_params=None, max_loop_rounds=2)` (Plan 6). `src.eq_data.nrc_enrichment.compute_enriched_overall_eq_proxy(row, text, nrc_lexicon)` (Plan 4) — `row` is a dict/Series with `extraversion`/`openness`/`agreeableness`/`conscientiousness`/`neuroticism` keys (0-99 scale, matching `data/train_set.csv`/`data/test_set.csv`'s real columns); `nrc_lexicon` is a `dict[str, set[str]]` (Plan 4's `load_nrc_lexicon` return shape — tests use a small fixture dict, per this project's established NRC-test convention, not the real lexicon file). `src.eq_data.tiers_eq.assign_eq_tier(score) -> (tier_num, label)` (Plan 1). `src.evaluation.faithfulness.check_rationale_faithfulness(agent_result) -> bool` (pre-existing, generic — works unmodified on each EQ branch-specialist result dict since those also carry `rationale`+`trace` keys). `src.evaluation.metrics.compute_regression_metrics(y_true, y_pred) -> dict` / `compute_tier_metrics(tier_true, tier_pred) -> dict` (pre-existing, generic).
- Produces: `make_run_eq_assessment_predict_fn(client, models, ctx, branch_configs, extra_params=None) -> predict_fn` where `predict_fn(text)` returns a `traced_run_eq_assessment` result dict. `run_eq_evaluation(predict_fn, samples, nrc_lexicon) -> list[dict]` where `samples` is a list of `(text, row)` tuples, one row dict per row. `summarize_eq_evaluation(rows) -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
import json

import httpx

from src.agent.openrouter_client import build_client
from src.eq_agent.graph import BRANCHES
from src.evaluation.eq_comparison import make_run_eq_assessment_predict_fn, run_eq_evaluation, summarize_eq_evaluation

_NRC_FIXTURE = {"happy": {"positive", "joy"}, "sad": {"negative", "sadness"}}


def _tool_call_response(call_id, name, arguments):
    return httpx.Response(200, json={
        "choices": [{"message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": call_id, "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments),
            }}],
        }}],
    })


def _happy_path_handler(request):
    body = json.loads(request.content)
    tool_names = {t["function"]["name"] for t in body.get("tools", [])}
    if "submit_critique" in tool_names:
        return _tool_call_response("c1", "submit_critique", {"consistent": True, "branches_to_recheck": [], "reason": ""})
    if "submit_overall_assessment" in tool_names:
        return _tool_call_response(
            "c1", "submit_overall_assessment",
            {"score": 70.0, "confidence": "high", "rationale": "Consistently positive language."},
        )
    return _tool_call_response(
        "c1", "submit_branch_assessment",
        {"score": 65.0, "confidence": "high", "rationale": "Positive language."},
    )


def _make_branch_configs():
    return {
        branch: {"tool_schemas": [], "dispatch_fn": lambda *a: {}, "system_prompt": f"Assess {branch}."}
        for branch in BRANCHES
    }


def test_run_eq_evaluation_produces_expected_row_shape():
    fake_client = build_client("fake-key", transport=httpx.MockTransport(_happy_path_handler))
    predict_fn = make_run_eq_assessment_predict_fn(fake_client, ["fake-model"], {}, _make_branch_configs())
    samples = [("I am so happy today", {
        "extraversion": 80.0, "openness": 70.0, "agreeableness": 60.0,
        "conscientiousness": 50.0, "neuroticism": 20.0,
    })]

    rows = run_eq_evaluation(predict_fn, samples, _NRC_FIXTURE)

    assert len(rows) == 1
    assert rows[0]["predicted_score"] == 70.0
    assert rows[0]["degraded"] is False
    assert isinstance(rows[0]["true_score"], float)


def test_summarize_eq_evaluation_returns_regression_and_tier_metrics():
    rows = [
        {"true_score": 70.0, "predicted_score": 70.0, "true_tier": 5, "predicted_tier": 5, "faithful": True, "degraded": False},
        {"true_score": 40.0, "predicted_score": 45.0, "true_tier": 3, "predicted_tier": 3, "faithful": False, "degraded": False},
    ]

    summary = summarize_eq_evaluation(rows)

    assert summary["rmse"] >= 0.0
    assert summary["faithfulness_rate"] == 0.5
    assert summary["degraded_rate"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/evaluation/test_eq_comparison.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.evaluation.eq_comparison'`

- [ ] **Step 3: Write the implementation**

```python
"""EQ evaluation harness: runs the EQ multi-agent pipeline over a text
sample and produces continuous + tiered metrics, compared against the
NRC-enriched proxy EQ label as the best available ground truth. Mirrors
src.evaluation.run_comparison's harness shape (predict_fn / run / summarize)
for the original Extraversion pipeline, but is a new, separate module --
see this plan's Global Constraints for why it does not modify that one.
"""
from src.eq_agent.traced_assessment import traced_run_eq_assessment
from src.eq_data.nrc_enrichment import compute_enriched_overall_eq_proxy
from src.eq_data.tiers_eq import assign_eq_tier
from src.evaluation.faithfulness import check_rationale_faithfulness
from src.evaluation.metrics import compute_regression_metrics, compute_tier_metrics


def make_run_eq_assessment_predict_fn(client, models, ctx, branch_configs, extra_params=None):
    def predict_fn(text):
        return traced_run_eq_assessment(client, models, ctx, text, branch_configs, extra_params=extra_params)
    return predict_fn


def run_eq_evaluation(predict_fn, samples, nrc_lexicon):
    rows = []
    for text, row in samples:
        eq_result = predict_fn(text)
        true_score = compute_enriched_overall_eq_proxy(row, text, nrc_lexicon)
        true_tier, _ = assign_eq_tier(true_score)
        branch_faithful = [
            check_rationale_faithfulness(branch_result)
            for branch_result in eq_result["branch_results"].values()
        ]
        rows.append({
            "text": text,
            "true_score": true_score,
            "true_tier": true_tier,
            "predicted_score": eq_result["score"],
            "predicted_tier": eq_result["tier"],
            "faithful": all(branch_faithful) if branch_faithful else True,
            "degraded": eq_result["degraded"],
            "degraded_branches": eq_result["degraded_branches"],
        })
    return rows


def summarize_eq_evaluation(rows):
    true_scores = [r["true_score"] for r in rows]
    pred_scores = [r["predicted_score"] for r in rows]
    true_tiers = [r["true_tier"] for r in rows]
    pred_tiers = [r["predicted_tier"] for r in rows]

    reg_metrics = compute_regression_metrics(true_scores, pred_scores)
    tier_metrics = compute_tier_metrics(true_tiers, pred_tiers)
    faithfulness_rate = sum(1 for r in rows if r["faithful"]) / len(rows) if rows else 0.0
    degraded_rate = sum(1 for r in rows if r["degraded"]) / len(rows) if rows else 0.0

    return {**reg_metrics, **tier_metrics, "faithfulness_rate": faithfulness_rate, "degraded_rate": degraded_rate}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/evaluation/test_eq_comparison.py -v`
Expected: 2 passed

- [ ] **Step 5: Run the full project test suite**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: all pass, no regressions (this task only adds new files).

- [ ] **Step 6: Commit**

```bash
git add src/evaluation/eq_comparison.py tests/evaluation/test_eq_comparison.py
git commit -m "feat: add EQ score/tier/faithfulness comparison harness"
```

---

## After This Plan

This completes the approved 8-plan EQ pivot sequence. Remaining, explicitly-deferred items (unchanged from prior plans' notes, not part of this sequence): building the real EQ RAG corpus for production use (`python -m src.eq_data.build_eq_corpus`, not yet run for real); a manual real-API-call evaluation script analogous to `src/evaluation/run_real_evaluation.py`, using this plan's `eq_comparison`/`eq_retrieval_evaluation` functions against a real corpus and real `DEEPSEEK_API_KEY`/`OPENROUTER_API_KEY`; wiring the Neo4j knowledge graph into the live agent's prompts/tools; frontend wiring for `/predict-eq`; and whether real LangSmith tracing appears correctly in a dashboard once a real `LANGSMITH_API_KEY` is configured.

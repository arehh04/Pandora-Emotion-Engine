# Evaluation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the ablation-study and comparison-metrics tooling described in the original design spec (Section 6): continuous (RMSE/MAE/R²) and tiered (accuracy/macro-F1/weighted kappa) metrics, a tool-restricted ablation runner (LLM-only → LLM+Fuzzy → LLM+Fuzzy+ML → full agent), a rationale-faithfulness heuristic, and a harness that ties them together against a sample of the real test set — producing the old-ML-vs-new-agent comparison table that's the thesis-facing deliverable of this whole pivot.

**Architecture:** Every piece is deterministic, self-contained, and unit-testable via dependency injection — the harness itself never makes a real LLM call in its test suite; it's driven through an injectable `predict_fn(text, enabled_tools) -> result_dict` closure, with a thin real implementation wrapping Plan 3's `run_agent`. The "old ML" comparison side uses the historical metrics already recorded in `docs/thesis/Chapter4_Results_Interpretability.md` as fixed reference values rather than re-running the project's other (volatile, out-of-scope) classical training pipeline — this is a deliberate boundary, consistent with every prior plan in this pivot.

**Tech Stack:** scikit-learn (`sklearn.metrics`, already a project dependency) for regression/classification metrics. Everything else from Plans 1-3 (`src.tiers`, `src.agent.orchestrator`, `src.agent.tool_schemas`).

## Global Constraints

- **Run every command with `"./.venv/Scripts/python.exe"`**, not a bare `python`.
- **No real LLM/API calls in automated tests.** The evaluation harness's own tests use an injected fake `predict_fn`; only a separate, manual, explicitly-deferred script run (not part of the automated suite) calls the real `run_agent`/OpenRouter or DeepSeek API over a real data sample — consistent with every prior "real external call" deferred throughout Plans 1-4.
- **The historical ML baselines (Ridge/XGBoost/RF) are hardcoded reference constants** sourced from `docs/thesis/Chapter4_Results_Interpretability.md`'s existing results table (RMSE/R² for each), not re-computed by this plan. Do not attempt to retrain or re-run the project's classical pipeline (`src/extract_classical_features.py`, `src/train_classical_models.py`, etc.) — those remain out of scope, per the same reasoning Plan 2 already established.
- No `__init__.py` files anywhere under `src/` (implicit namespace packages — established convention).
- The tiered metrics operate on tier numbers 1-6 from `src.tiers.assign_tier` — do not re-derive tier boundaries locally.
- `run_agent`'s new `enabled_tools` parameter must default to `None` (meaning "all tools available," i.e. today's existing behavior) so this is a purely additive, backward-compatible change to already-merged Plan 3 code — none of Plan 3's or Plan 4's existing tests should need to change.

---

### Task 1: Regression and tiered classification metrics

**Files:**
- Create: `src/evaluation/metrics.py`
- Test: `tests/evaluation/test_metrics.py`

**Interfaces:**
- Produces: `compute_regression_metrics(y_true: list[float], y_pred: list[float]) -> dict` returning `{"rmse": float, "mae": float, "r2": float}`, and `compute_tier_metrics(tier_true: list[int], tier_pred: list[int]) -> dict` returning `{"accuracy": float, "macro_f1": float, "weighted_kappa": float, "confusion_matrix": list[list[int]]}`. Consumed by Task 4 (evaluation harness).

- [ ] **Step 1: Write the failing test**

Create `tests/evaluation/test_metrics.py`:

```python
import math

from src.evaluation.metrics import compute_regression_metrics, compute_tier_metrics


def test_compute_regression_metrics_perfect_prediction():
    result = compute_regression_metrics([10.0, 50.0, 90.0], [10.0, 50.0, 90.0])

    assert result["rmse"] == 0.0
    assert result["mae"] == 0.0
    assert result["r2"] == 1.0


def test_compute_regression_metrics_known_error():
    # y_true=[0, 10], y_pred=[2, 8] -> errors [2, -2], MAE=2, RMSE=2
    result = compute_regression_metrics([0.0, 10.0], [2.0, 8.0])

    assert result["mae"] == 2.0
    assert math.isclose(result["rmse"], 2.0, rel_tol=1e-9)


def test_compute_tier_metrics_perfect_prediction():
    result = compute_tier_metrics([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6])

    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["weighted_kappa"] == 1.0
    assert len(result["confusion_matrix"]) == 6


def test_compute_tier_metrics_all_wrong_by_one_tier():
    # Every prediction is off by exactly one tier — weighted kappa should
    # still be well above 0 (near misses are penalized less than random),
    # while plain accuracy is 0.
    result = compute_tier_metrics([2, 3, 4, 5], [1, 2, 3, 4])

    assert result["accuracy"] == 0.0
    assert result["weighted_kappa"] > 0.0


def test_compute_tier_metrics_confusion_matrix_shape_matches_label_count():
    result = compute_tier_metrics([1, 1, 2], [1, 2, 2])

    assert len(result["confusion_matrix"]) == len(result["confusion_matrix"][0])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/evaluation/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.evaluation.metrics'`

- [ ] **Step 3: Write the minimal implementation**

Create `src/evaluation/metrics.py`:

```python
"""Regression and tiered-classification metrics for comparing the LLM agent
against the historical classical-ML pipeline (see Task 4's hardcoded
HISTORICAL_BASELINES for the classical side).
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def compute_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def compute_tier_metrics(tier_true, tier_pred):
    labels = sorted(set(tier_true) | set(tier_pred))
    accuracy = float(accuracy_score(tier_true, tier_pred))
    macro_f1 = float(f1_score(tier_true, tier_pred, average="macro", labels=labels, zero_division=0))
    weighted_kappa = float(cohen_kappa_score(tier_true, tier_pred, weights="linear", labels=labels))
    cm = confusion_matrix(tier_true, tier_pred, labels=labels).tolist()
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_kappa": weighted_kappa,
        "confusion_matrix": cm,
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/evaluation/test_metrics.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/metrics.py tests/evaluation/test_metrics.py
git commit -m "feat: add regression and tiered classification metrics for evaluation"
```

---

### Task 2: Tool-restricted ablation support in run_agent

**Files:**
- Modify: `src/agent/orchestrator.py`
- Modify: `tests/agent/test_orchestrator.py` (add new tests only — do not remove or alter existing ones)

**Interfaces:**
- Consumes: `TOOL_SCHEMAS` (`src.agent.tool_schemas`, already imported by this file).
- Produces: `run_agent(client, models, ctx, text, max_iterations=6, extra_params=None, enabled_tools=None)` — new optional parameter. `enabled_tools` is `None` (default, all tools available — today's behavior, unchanged) or a `set[str]` of tool names to expose to the LLM in addition to the always-available `submit_assessment`. If the LLM calls a tool outside this set anyway, `dispatch_tool_call` is not invoked for it — the loop records a graceful `{"error": "Tool '<name>' is disabled for this evaluation variant."}` in the trace instead. Consumed by Task 4 (evaluation harness) to run the LLM-only / LLM+Fuzzy / LLM+Fuzzy+ML / full-agent ablation variants.

- [ ] **Step 1: Write the failing test**

Append to `tests/agent/test_orchestrator.py`:

```python
def test_run_agent_restricts_tool_schemas_sent_to_the_api():
    sent_tool_names = []

    def handler(request):
        import json as _json
        body = _json.loads(request.content)
        sent_tool_names.append([t["function"]["name"] for t in body.get("tools", [])])
        return _assistant_tool_call_response("call_1", "submit_assessment", {
            "tier": 3, "continuous_score_estimate": 40.0, "confidence": "medium",
            "rationale": "Ambiguous signal.",
        })

    client = build_client("fake-key", transport=httpx.MockTransport(handler))
    ctx = _build_test_context()

    run_agent(client, ["fake-model"], ctx, "some text", enabled_tools={"fuzzy_logic_assessment"})

    assert set(sent_tool_names[0]) == {"fuzzy_logic_assessment", "submit_assessment"}


def test_run_agent_gracefully_refuses_a_disabled_tool_call():
    turns = {"n": 0}

    def handler(request):
        turns["n"] += 1
        if turns["n"] == 1:
            return _assistant_tool_call_response("call_1", "ml_prior_assessment", {"text": "hi"})
        return _assistant_tool_call_response("call_2", "submit_assessment", {
            "tier": 3, "continuous_score_estimate": 40.0, "confidence": "medium",
            "rationale": "Ambiguous signal.",
        })

    client = build_client("fake-key", transport=httpx.MockTransport(handler))
    ctx = _build_test_context()

    result = run_agent(client, ["fake-model"], ctx, "some text", enabled_tools={"fuzzy_logic_assessment"})

    assert result["degraded"] is False
    assert result["trace"][0]["tool"] == "ml_prior_assessment"
    assert "disabled" in result["trace"][0]["result"]["error"]


def test_run_agent_enabled_tools_none_still_exposes_all_tools():
    sent_tool_names = []

    def handler(request):
        import json as _json
        body = _json.loads(request.content)
        sent_tool_names.append([t["function"]["name"] for t in body.get("tools", [])])
        return _assistant_tool_call_response("call_1", "submit_assessment", {
            "tier": 3, "continuous_score_estimate": 40.0, "confidence": "medium",
            "rationale": "Ambiguous signal.",
        })

    client = build_client("fake-key", transport=httpx.MockTransport(handler))
    ctx = _build_test_context()

    run_agent(client, ["fake-model"], ctx, "some text")

    assert len(sent_tool_names[0]) == 5
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/agent/test_orchestrator.py -v -k "restricts_tool_schemas or disabled_tool_call or enabled_tools_none"`
Expected: FAIL — `run_agent() got an unexpected keyword argument 'enabled_tools'`

- [ ] **Step 3: Add enabled_tools support to run_agent**

In `src/agent/orchestrator.py`, change the `run_agent` signature and body. Replace:

```python
def run_agent(client, models, ctx, text, max_iterations=6, extra_params=None):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Assess the Extraversion of this text:\n\n{text}"},
    ]
    trace = []

    for _ in range(max_iterations):
        try:
            response = call_with_fallback(client, models, messages, tools=TOOL_SCHEMAS, extra_params=extra_params)
        except Exception as e:
```

with:

```python
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
```

Then, inside the tool-call processing loop, replace:

```python
                result = dispatch_tool_call(name, arguments, ctx)
                trace.append({"tool": name, "arguments": arguments, "result": result})
```

with:

```python
                if enabled_tools is not None and name != "submit_assessment" and name not in enabled_tools:
                    result = {"error": f"Tool '{name}' is disabled for this evaluation variant."}
                else:
                    result = dispatch_tool_call(name, arguments, ctx)
                trace.append({"tool": name, "arguments": arguments, "result": result})
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/agent/test_orchestrator.py -v`
Expected: PASS (10 tests — 7 existing + 3 new)

- [ ] **Step 5: Run the full project test suite**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/ -v`
Expected: PASS (all existing tests unaffected, plus this task's 3 new ones)

- [ ] **Step 6: Commit**

```bash
git add src/agent/orchestrator.py tests/agent/test_orchestrator.py
git commit -m "feat: add enabled_tools ablation support to run_agent"
```

---

### Task 3: Rationale faithfulness heuristic

**Files:**
- Create: `src/evaluation/faithfulness.py`
- Test: `tests/evaluation/test_faithfulness.py`

**Interfaces:**
- Produces: `check_rationale_faithfulness(agent_result: dict) -> bool` — a heuristic (explicitly not a proof) checking whether an agent result's `rationale` text references at least one concrete piece of evidence that actually appears in its own `trace` (a matching tier/score number, or a word from a called tool's name). Consumed by Task 4 (evaluation harness) to compute a faithfulness rate per ablation variant, operationalizing the design spec's "rationale faithfulness spot-check."

- [ ] **Step 1: Write the failing test**

Create `tests/evaluation/test_faithfulness.py`:

```python
from src.evaluation.faithfulness import check_rationale_faithfulness


def test_faithfulness_true_when_rationale_cites_a_trace_tier_number():
    agent_result = {
        "rationale": "The fuzzy logic engine returned tier 5, which strongly supports this assessment.",
        "trace": [
            {"tool": "fuzzy_logic_assessment", "arguments": {"text": "x"},
             "result": {"fuzzy_score": 70.0, "tier": 5, "tier_label": "Outgoing", "fired_rules": []}},
        ],
    }

    assert check_rationale_faithfulness(agent_result) is True


def test_faithfulness_true_when_rationale_mentions_tool_name():
    agent_result = {
        "rationale": "The ml prior model suggested a moderate score, but the direct content is clearer.",
        "trace": [
            {"tool": "ml_prior_assessment", "arguments": {"text": "x"},
             "result": {"score": 40.0, "tier": 3, "tier_label": "Balanced (Introspective)"}},
        ],
    }

    assert check_rationale_faithfulness(agent_result) is True


def test_faithfulness_false_when_rationale_is_unrelated_to_trace():
    agent_result = {
        "rationale": "Based on general writing style, this seems like a reserved individual.",
        "trace": [
            {"tool": "ml_prior_assessment", "arguments": {"text": "x"},
             "result": {"score": 90.0, "tier": 6, "tier_label": "Highly Extraverted"}},
        ],
    }

    assert check_rationale_faithfulness(agent_result) is False


def test_faithfulness_true_when_trace_is_empty():
    agent_result = {"rationale": "Purely based on the text itself.", "trace": []}

    assert check_rationale_faithfulness(agent_result) is True


def test_faithfulness_ignores_tool_calls_that_errored():
    agent_result = {
        "rationale": "No tool evidence was usable, so this is based on direct reading.",
        "trace": [
            {"tool": "retrieve_similar_exemplars", "arguments": {"text": "x"},
             "result": {"error": "RAG corpus is not available (not built yet)."}},
        ],
    }

    assert check_rationale_faithfulness(agent_result) is True
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/evaluation/test_faithfulness.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.evaluation.faithfulness'`

- [ ] **Step 3: Write the minimal implementation**

Create `src/evaluation/faithfulness.py`:

```python
"""A heuristic (not a proof) for whether an agent's rationale references
concrete evidence from its own tool-call trace, rather than being generic
or unsupported. Flagged results (returns False) are candidates for manual
spot-checking, per the design spec's rationale-faithfulness requirement —
this function does not replace that manual review, it triages it.
"""


def check_rationale_faithfulness(agent_result):
    rationale = (agent_result.get("rationale") or "").lower()
    trace = agent_result.get("trace") or []

    if not trace:
        return True  # nothing to be unfaithful to (e.g. the llm-only ablation variant)

    for step in trace:
        result = step.get("result") or {}
        if "error" in result:
            continue

        for key in ("tier", "score", "fuzzy_score"):
            if key in result and str(result[key]) in rationale:
                return True

        tool_words = step["tool"].replace("_", " ").split()
        if any(word in rationale for word in tool_words if len(word) > 4):
            return True

    return False
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/evaluation/test_faithfulness.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/faithfulness.py tests/evaluation/test_faithfulness.py
git commit -m "feat: add rationale faithfulness heuristic for evaluation"
```

---

### Task 4: Ablation/comparison evaluation harness

**Files:**
- Create: `src/evaluation/run_comparison.py`
- Test: `tests/evaluation/test_run_comparison.py`

**Interfaces:**
- Consumes: `compute_regression_metrics`, `compute_tier_metrics` (Task 1); `check_rationale_faithfulness` (Task 3); `assign_tier` (`src.tiers`, Plan 1); `run_agent` (`src.agent.orchestrator`, Plan 3 + Task 2's `enabled_tools`).
- Produces: `ABLATION_VARIANTS` (dict mapping variant name to an `enabled_tools` value: `"llm_only": set()`, `"llm_fuzzy": {"fuzzy_logic_assessment"}`, `"llm_fuzzy_ml": {"fuzzy_logic_assessment", "ml_prior_assessment"}`, `"llm_full": None`), `HISTORICAL_BASELINES` (dict of the Chapter 4 classical-model metrics), `make_run_agent_predict_fn(client, models, ctx, extra_params=None, max_iterations=6) -> callable` (returns a `predict_fn(text, enabled_tools) -> agent_result_dict` closure wrapping `run_agent`), `run_evaluation(predict_fn, samples: list[tuple[str, float]], variants: dict) -> dict[str, list[dict]]`, and `summarize_evaluation(results: dict) -> dict` (per-variant metrics + `"_historical_baselines"` key).

- [ ] **Step 1: Write the failing test**

Create `tests/evaluation/test_run_comparison.py`. This test drives the harness with a fully deterministic fake `predict_fn` — no real LLM/API calls:

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
        {"tool": "fuzzy_logic_assessment", "arguments": {"text": text},
         "result": {"fuzzy_score": score, "tier": tier, "tier_label": "x", "fired_rules": []}},
    ]
    return {
        "tier": tier, "tier_label": "x", "continuous_score_estimate": score,
        "confidence": "medium", "rationale": f"Fuzzy logic assessment gave tier {tier}.",
        "trace": trace, "degraded": False, "error": None,
    }


def test_ablation_variants_cover_the_four_expected_configurations():
    assert set(ABLATION_VARIANTS.keys()) == {"llm_only", "llm_fuzzy", "llm_fuzzy_ml", "llm_full"}
    assert ABLATION_VARIANTS["llm_only"] == set()
    assert ABLATION_VARIANTS["llm_full"] is None


def test_historical_baselines_include_all_three_classical_models():
    assert set(HISTORICAL_BASELINES.keys()) == {"ridge", "xgboost", "random_forest"}
    for entry in HISTORICAL_BASELINES.values():
        assert "rmse" in entry and "r2" in entry


def test_run_evaluation_produces_one_row_per_sample_per_variant():
    samples = [("short", 5.0), ("a much longer piece of text here", 90.0)]
    variants = {"llm_only": set(), "llm_full": None}

    results = run_evaluation(_fake_predict_fn, samples, variants)

    assert set(results.keys()) == {"llm_only", "llm_full"}
    assert len(results["llm_only"]) == 2
    assert len(results["llm_full"]) == 2
    assert results["llm_only"][0]["text"] == "short"
    assert results["llm_only"][0]["true_score"] == 5.0
    assert "predicted_tier" in results["llm_only"][0]
    assert "faithful" in results["llm_full"][0]


def test_summarize_evaluation_includes_metrics_and_historical_baselines():
    samples = [("short", 5.0), ("a much longer piece of text here", 90.0)]
    variants = {"llm_only": set(), "llm_full": None}
    results = run_evaluation(_fake_predict_fn, samples, variants)

    summary = summarize_evaluation(results)

    assert "llm_only" in summary
    assert "rmse" in summary["llm_only"]
    assert "accuracy" in summary["llm_only"]
    assert "faithfulness_rate" in summary["llm_only"]
    assert summary["_historical_baselines"] == HISTORICAL_BASELINES
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/evaluation/test_run_comparison.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.evaluation.run_comparison'`

- [ ] **Step 3: Write the minimal implementation**

Create `src/evaluation/run_comparison.py`:

```python
"""Ablation/comparison harness: runs the LLM-only -> LLM+Fuzzy ->
LLM+Fuzzy+ML -> full-agent variants over a text sample and produces
continuous + tiered metrics for each, alongside the historical classical-ML
baselines recorded in docs/thesis/Chapter4_Results_Interpretability.md.

The real (non-mocked) run is a separate, manual, explicitly deferred script
invocation -- see the module docstring in scripts using make_run_agent_predict_fn.
Nothing in this module's automated tests makes a real LLM/API call.
"""
from src.agent.orchestrator import run_agent
from src.evaluation.faithfulness import check_rationale_faithfulness
from src.evaluation.metrics import compute_regression_metrics, compute_tier_metrics
from src.tiers import assign_tier

ABLATION_VARIANTS = {
    "llm_only": set(),
    "llm_fuzzy": {"fuzzy_logic_assessment"},
    "llm_fuzzy_ml": {"fuzzy_logic_assessment", "ml_prior_assessment"},
    "llm_full": None,
}

# Sourced from docs/thesis/Chapter4_Results_Interpretability.md's existing
# results table -- not re-computed by this plan (see Global Constraints).
HISTORICAL_BASELINES = {
    "ridge": {"rmse": 29.18, "r2": 0.103},
    "xgboost": {"rmse": 28.29, "r2": 0.158},
    "random_forest": {"rmse": 28.22, "r2": 0.162},
}


def make_run_agent_predict_fn(client, models, ctx, extra_params=None, max_iterations=6):
    def predict_fn(text, enabled_tools):
        return run_agent(
            client, models, ctx, text,
            max_iterations=max_iterations, extra_params=extra_params, enabled_tools=enabled_tools,
        )
    return predict_fn


def run_evaluation(predict_fn, samples, variants):
    results = {}
    for variant_name, enabled_tools in variants.items():
        variant_results = []
        for text, true_score in samples:
            agent_result = predict_fn(text, enabled_tools)
            true_tier, _ = assign_tier(true_score)
            variant_results.append({
                "text": text,
                "true_score": true_score,
                "true_tier": true_tier,
                "predicted_score": agent_result["continuous_score_estimate"],
                "predicted_tier": agent_result["tier"],
                "faithful": check_rationale_faithfulness(agent_result),
                "degraded": agent_result["degraded"],
            })
        results[variant_name] = variant_results
    return results


def summarize_evaluation(results):
    summary = {}
    for variant_name, rows in results.items():
        true_scores = [r["true_score"] for r in rows]
        pred_scores = [r["predicted_score"] for r in rows]
        true_tiers = [r["true_tier"] for r in rows]
        pred_tiers = [r["predicted_tier"] for r in rows]

        reg_metrics = compute_regression_metrics(true_scores, pred_scores)
        tier_metrics = compute_tier_metrics(true_tiers, pred_tiers)
        faithfulness_rate = sum(1 for r in rows if r["faithful"]) / len(rows) if rows else 0.0

        summary[variant_name] = {**reg_metrics, **tier_metrics, "faithfulness_rate": faithfulness_rate}

    summary["_historical_baselines"] = HISTORICAL_BASELINES
    return summary
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/evaluation/test_run_comparison.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the full project test suite**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/ -v`
Expected: PASS (all tests from Plans 1-4 plus this plan's new tests)

- [ ] **Step 6: Commit**

```bash
git add src/evaluation/run_comparison.py tests/evaluation/test_run_comparison.py
git commit -m "feat: add ablation/comparison evaluation harness"
```

*(Note: actually producing the thesis-facing comparison table means writing a small, separate manual script — not part of this plan's automated tests — that: builds a real `ctx` via `src.agent.context`, a real `client` via `src.agent.openrouter_client.build_client` pointed at a configured provider, samples N rows from `data/test_set.csv` — a modest N, e.g. 30-50, given each real request can take 10-30+ seconds with reasoning enabled and free-tier rate limits apply — calls `run_evaluation`/`summarize_evaluation`, and prints/saves the resulting comparison table. This mirrors every other "real external call" deferred throughout Plans 1-4: the harness itself is fully tested; running it for real against live data and a live LLM is a follow-up the user performs once ready, given the cost/time/rate-limit considerations involved.)*

---

## Plan Self-Review Notes

- **Spec coverage:** Design spec Section 6 (Evaluation plan) → all four tasks: continuous metrics (Task 1), tiered metrics (Task 1), ablation study (Task 2 + Task 4's `ABLATION_VARIANTS`), rationale faithfulness (Task 3), historical baseline comparison (Task 4's `HISTORICAL_BASELINES`). Cost/latency measurement (also named in Section 6) is NOT covered by this plan — no task currently records token counts or wall-clock time per request; if that's wanted, it would need `run_agent` or the harness to capture response `usage` fields and timing, which is a reasonable small follow-up but is not included here since the design spec treated it as secondary to the metrics/ablation core.
- **No placeholders:** every task has complete, runnable code; no TBD/TODO markers. The deferred "real run against live data" step is explicitly named as a manual follow-up, not silently skipped.
- **Type/interface consistency:** `run_evaluation`'s `predict_fn(text, enabled_tools) -> dict` contract matches exactly what `make_run_agent_predict_fn`'s closure produces (both call sites use the same two positional arguments) and what the test's `_fake_predict_fn` implements. `ABLATION_VARIANTS`' values (`set()`, a populated `set`, or `None`) match exactly what Task 2's `run_agent(..., enabled_tools=...)` expects. `compute_regression_metrics`/`compute_tier_metrics`'s return dict keys are consumed identically in `summarize_evaluation`.
- **Scope note:** this plan does not modify the FastAPI backend or frontend — it's a standalone evaluation toolkit, run via manual script invocation once real data/API access is exercised, consistent with the design spec's own framing of Section 6 as a research/evaluation deliverable rather than a served feature.

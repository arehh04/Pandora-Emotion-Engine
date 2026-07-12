# LangSmith Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add LangSmith run tracing to the EQ multi-agent pipeline — a natural pairing since the orchestrator is already LangGraph-based — plus a small token-usage extraction utility, without modifying any already-PR'd Plan 2 code (`src/eq_agent/graph.py`).

**Architecture:** Unlike Neo4j (self-hostable, verified against a real local container), LangSmith is a hosted SaaS product — there is no way to spin up a local instance to test against. This plan's testable surface is therefore scoped to what's genuinely verifiable without real credentials: a config helper (`src/eq_agent/observability.py`) that wires `.env`-style settings into LangSmith's environment-variable-based activation, and a `@traceable`-wrapped entry point (`src/eq_agent/traced_assessment.py`) around Plan 2's `run_eq_assessment` — verified directly that `@traceable` safely no-ops (returns the wrapped function's normal result, no network call attempt) when tracing isn't configured, so it's safe to apply unconditionally. Whether real tracing actually appears in a LangSmith dashboard once a real API key is configured is a manual verification step for whoever has a real account, not part of the automated suite — matching this project's established pattern for anything requiring real external credentials (Plan 1's dataset fetchers, Plan 5's real Neo4j connection).

**Tech Stack:** `langsmith` 0.10.2 (already installed as a transitive dependency of `langgraph`/`langchain-core` from Plan 2 — no new dependency to add).

## Global Constraints

- **Verified real facts about `langsmith` 0.10.2** (checked directly against the installed package, not assumed): the `@traceable` decorator, when tracing environment variables aren't set, transparently returns the wrapped function's normal result with no observable delay or network attempt (confirmed: a decorated `int`-returning function still returns a plain `int`, not a wrapped/proxy object). The package recognizes both the newer `LANGSMITH_*` env var names (`LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, `LANGSMITH_ENDPOINT`) and the older `LANGCHAIN_*` equivalents for backward compatibility — this plan uses the newer `LANGSMITH_*` names. `@traceable(name=..., run_type=...)` accepts arbitrary keyword arguments and both were confirmed to work without error.
- **No real LangSmith account/API key is required to run this plan's automated test suite.** Tests verify (a) the config helper's env-var-setting logic in isolation, and (b) that `@traceable`-wrapped functions still behave correctly (same return value, no exception) whether or not tracing is configured — not that real tracing data actually reaches LangSmith's servers, which is unverifiable without real credentials and is explicitly out of scope for automated tests.
- **Env var test hygiene**: verified directly that `pytest`'s `monkeypatch.delenv`/`setenv` does **not** automatically clean up an environment variable that gets set by *code under test* via a raw `os.environ[...] = ...` assignment (only variables monkeypatch itself sets/deletes are tracked for teardown) — confirmed this experimentally (a var set directly during a test leaked into a subsequent test that used no `monkeypatch` at all). Any test in this plan that calls `configure_langsmith_tracing(...)` (which sets real env vars as its whole purpose) must manually clean up the exact keys it touched in a `try`/`finally` block, not rely on `monkeypatch` alone.
- Purely additive: `src/eq_agent/graph.py` (Plan 2, already-PR'd) is consumed, never modified. No file under `src/eq_data/`, `src/agent/`, `src/rag/`, `backend/`, or `frontend/` is touched. New files under `src/eq_agent/`/`tests/eq_agent/`, no `__init__.py` (implicit namespace packages, established convention).
- No new dependency to add to `requirements.txt` (`langsmith` is already present transitively; verified installed at 0.10.2).

---

### Task 1: LangSmith tracing configuration helper

**Files:**
- Create: `src/eq_agent/observability.py`
- Test: `tests/eq_agent/test_observability.py`

**Interfaces:**
- Produces: `DEFAULT_LANGSMITH_PROJECT = "pandora-eq"`, `configure_langsmith_tracing(api_key=None, project=None, endpoint=None) -> bool` (returns `True` if tracing was actually enabled, `False` if no API key was available) — consumed by a later Backend Integration plan at process startup, and by Task 2's tests to establish a known env state.

- [ ] **Step 1: Write the failing tests**

```python
import os

from src.eq_agent.observability import DEFAULT_LANGSMITH_PROJECT, configure_langsmith_tracing

_LANGSMITH_ENV_KEYS = ["LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT"]


def _clear_langsmith_env():
    for key in _LANGSMITH_ENV_KEYS:
        os.environ.pop(key, None)


def test_configure_langsmith_tracing_returns_false_when_no_api_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    result = configure_langsmith_tracing()

    assert result is False


def test_configure_langsmith_tracing_sets_env_vars_when_api_key_given():
    _clear_langsmith_env()
    try:
        result = configure_langsmith_tracing(api_key="fake-key", project="test-project")

        assert result is True
        assert os.environ["LANGSMITH_TRACING"] == "true"
        assert os.environ["LANGSMITH_API_KEY"] == "fake-key"
        assert os.environ["LANGSMITH_PROJECT"] == "test-project"
    finally:
        _clear_langsmith_env()


def test_configure_langsmith_tracing_defaults_project_name():
    _clear_langsmith_env()
    try:
        configure_langsmith_tracing(api_key="fake-key")

        assert os.environ["LANGSMITH_PROJECT"] == DEFAULT_LANGSMITH_PROJECT
    finally:
        _clear_langsmith_env()


def test_configure_langsmith_tracing_sets_endpoint_only_when_given():
    _clear_langsmith_env()
    try:
        configure_langsmith_tracing(api_key="fake-key")
        assert "LANGSMITH_ENDPOINT" not in os.environ

        configure_langsmith_tracing(api_key="fake-key", endpoint="https://example-langsmith.test")
        assert os.environ["LANGSMITH_ENDPOINT"] == "https://example-langsmith.test"
    finally:
        _clear_langsmith_env()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_agent/test_observability.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eq_agent.observability'`

- [ ] **Step 3: Write the implementation**

```python
"""Configures LangSmith run tracing for the EQ multi-agent pipeline via
environment variables -- the mechanism LangSmith's SDK uses to decide
whether @traceable-wrapped calls actually report anywhere. Safe to call
even without a real API key: returns False and leaves tracing disabled
rather than raising. Uses the newer LANGSMITH_* env var names (verified
these are recognized by the installed langsmith==0.10.2, alongside the
older LANGCHAIN_* equivalents kept for backward compatibility elsewhere).
"""
import os

DEFAULT_LANGSMITH_PROJECT = "pandora-eq"


def configure_langsmith_tracing(api_key=None, project=None, endpoint=None):
    api_key = api_key or os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = project or os.environ.get("LANGSMITH_PROJECT", DEFAULT_LANGSMITH_PROJECT)
    if endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = endpoint

    return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_agent/test_observability.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/eq_agent/observability.py tests/eq_agent/test_observability.py
git commit -m "feat: add LangSmith tracing configuration helper"
```

---

### Task 2: Traced entry point for the EQ assessment pipeline

**Files:**
- Create: `src/eq_agent/traced_assessment.py`
- Test: `tests/eq_agent/test_traced_assessment.py`

**Interfaces:**
- Consumes: `run_eq_assessment(client, models, ctx, text, branch_configs, extra_params=None, max_loop_rounds=2)` from `src/eq_agent/graph.py` (Plan 2, unmodified — same signature verified against the real committed file).
- Produces: `traced_run_eq_assessment(client, models, ctx, text, branch_configs, extra_params=None, max_loop_rounds=2) -> dict` — the same return shape as `run_eq_assessment`, wrapped with `@traceable` — consumed by a later Backend Integration plan in place of the untraced function, once real LangSmith credentials are configured.

- [ ] **Step 1: Write the failing tests**

```python
import json

import httpx

from src.agent.openrouter_client import build_client
from src.eq_agent.graph import BRANCHES
from src.eq_agent.traced_assessment import traced_run_eq_assessment


def _tool_call_response(call_id, name, arguments):
    return httpx.Response(200, json={
        "choices": [{"message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": call_id, "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments),
            }}],
        }}],
    })


def _make_branch_configs():
    return {
        branch: {"tool_schemas": [], "dispatch_fn": lambda *a: {}, "system_prompt": f"Assess {branch}."}
        for branch in BRANCHES
    }


def _happy_path_handler(request):
    body = json.loads(request.content)
    tool_names = {t["function"]["name"] for t in body.get("tools", [])}
    if "submit_critique" in tool_names:
        return _tool_call_response("c1", "submit_critique", {"consistent": True, "branches_to_recheck": [], "reason": ""})
    if "submit_overall_assessment" in tool_names:
        return _tool_call_response("c1", "submit_overall_assessment", {"score": 60.0, "confidence": "medium", "rationale": "x"})
    return _tool_call_response("c1", "submit_branch_assessment", {"score": 55.0, "confidence": "medium", "rationale": "x"})


def test_traced_run_eq_assessment_returns_the_expected_result_shape():
    client = build_client("fake-key", transport=httpx.MockTransport(_happy_path_handler))

    result = traced_run_eq_assessment(client, ["fake-model"], {}, "some text", _make_branch_configs())

    assert result["degraded_branches"] == []
    assert result["score"] == 60.0
    assert set(result["branch_results"].keys()) == set(BRANCHES)


def test_traced_run_eq_assessment_works_without_langsmith_tracing_configured(monkeypatch):
    # Must not raise even with tracing fully disabled -- @traceable's no-op
    # path must be genuinely transparent (verified directly against the
    # installed langsmith package before writing this plan).
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    client = build_client("fake-key", transport=httpx.MockTransport(_happy_path_handler))

    result = traced_run_eq_assessment(client, ["fake-model"], {}, "some text", _make_branch_configs())

    assert result["degraded"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_agent/test_traced_assessment.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eq_agent.traced_assessment'`

- [ ] **Step 3: Write the implementation**

```python
"""Traced entry point for the EQ multi-agent assessment pipeline, wrapping
src.eq_agent.graph.run_eq_assessment (Plan 2, unmodified) with LangSmith's
@traceable decorator. Verified directly against the installed
langsmith==0.10.2: @traceable safely no-ops (returns the wrapped function's
normal result, no network call attempt) when LANGSMITH_TRACING isn't set --
safe to apply unconditionally regardless of whether real tracing is
configured for this process.
"""
from langsmith import traceable

from src.eq_agent.graph import run_eq_assessment


@traceable(name="eq_assessment", run_type="chain")
def traced_run_eq_assessment(client, models, ctx, text, branch_configs, extra_params=None, max_loop_rounds=2):
    return run_eq_assessment(
        client, models, ctx, text, branch_configs,
        extra_params=extra_params, max_loop_rounds=max_loop_rounds,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_agent/test_traced_assessment.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/eq_agent/traced_assessment.py tests/eq_agent/test_traced_assessment.py
git commit -m "feat: add @traceable-wrapped entry point for LangSmith run tracing"
```

---

### Task 3: Token usage extraction utility

**Files:**
- Create: `src/eq_agent/token_usage.py`
- Test: `tests/eq_agent/test_token_usage.py`

**Interfaces:**
- Produces: `extract_token_usage(response) -> dict | None` — extracts `{"prompt_tokens", "completion_tokens", "total_tokens"}` from an OpenAI-compatible chat completion response dict (the shape returned by `src/agent/openrouter_client.py::call_with_fallback`), or `None` if no usage data is present. Not yet wired into any live code path — available for a later plan to surface per-call token cost (e.g. in the frontend Agent Trace panel or LangSmith run metadata), matching this plan's deferred-integration precedent.

- [ ] **Step 1: Write the failing tests**

```python
from src.eq_agent.token_usage import extract_token_usage


def test_extract_token_usage_returns_the_three_fields():
    response = {
        "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165},
    }

    result = extract_token_usage(response)

    assert result == {"prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165}


def test_extract_token_usage_returns_none_when_usage_absent():
    response = {"choices": []}

    result = extract_token_usage(response)

    assert result is None


def test_extract_token_usage_handles_partial_usage_dict():
    response = {"usage": {"prompt_tokens": 100}}

    result = extract_token_usage(response)

    assert result == {"prompt_tokens": 100, "completion_tokens": None, "total_tokens": None}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_agent/test_token_usage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eq_agent.token_usage'`

- [ ] **Step 3: Write the implementation**

```python
"""Extracts token usage from an OpenAI-compatible chat completion response
(the shape src.agent.openrouter_client.call_with_fallback returns), for
future cost/token observability surfacing. Standalone utility, not yet
wired into any live code path -- a later plan decides where to surface it
(e.g. the frontend Agent Trace panel or LangSmith run metadata).
"""


def extract_token_usage(response):
    usage = response.get("usage")
    if not usage:
        return None
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_agent/test_token_usage.py -v`
Expected: 3 passed

- [ ] **Step 5: Run the full project test suite**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: all pass, no regressions (this plan only adds new files under `src/eq_agent/`/`tests/eq_agent/`).

- [ ] **Step 6: Commit**

```bash
git add src/eq_agent/token_usage.py tests/eq_agent/test_token_usage.py
git commit -m "feat: add token usage extraction utility for future cost observability"
```

---

## After This Plan

Per the approved sequence: Plan 7 (Backend Integration — a `/predict-eq` endpoint, which is where `configure_langsmith_tracing(...)` gets called once at process startup with a real `LANGSMITH_API_KEY` from `.env`, and where `traced_run_eq_assessment` replaces the untraced `run_eq_assessment` call), Plan 8 (Evaluation Harness). Whether real tracing actually appears correctly in a LangSmith dashboard once real credentials are configured is a manual verification step for whoever has a real LangSmith account — this plan does not and cannot verify that without one. Similarly, wiring `extract_token_usage` into the frontend Agent Trace panel or LangSmith run metadata is deferred to Backend Integration.

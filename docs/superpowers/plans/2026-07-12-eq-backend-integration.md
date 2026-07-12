# EQ Backend Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the EQ multi-agent LangGraph pipeline (Plans 1-6) as a real `POST /predict-eq` FastAPI endpoint, wired into the running backend alongside the existing classical-ML pipeline (`backend/main.py`) and the Extraversion LLM-agent pipeline (`backend/agent_router.py`).

**Architecture:** A new `backend/eq_router.py`, structurally mirroring `backend/agent_router.py`'s exact pattern (cached FastAPI dependencies for context/client/models, a thin route handler, `dotenv`-loaded env config), calls `src.eq_agent.traced_assessment.traced_run_eq_assessment` (Plan 6's `@traceable`-wrapped entry point) instead of the untraced `run_eq_assessment`, so real requests report to LangSmith once `LANGSMITH_API_KEY` is set. `backend/main.py` gains a two-line wiring addition (import + `include_router`), identical in shape to how `agent_router` was wired in.

**Tech Stack:** FastAPI, Pydantic, python-dotenv, httpx (via existing `src.agent.openrouter_client`), the already-built `src/eq_agent/` and `src/eq_data/` packages.

## Global Constraints

- Purely additive except one 2-line wiring change to `backend/main.py` (import + `include_router`), mirroring the existing `agent_router` wiring at `backend/main.py:23,34` exactly — no other line of `backend/main.py` changes.
- Do not modify `backend/agent_router.py`, or any file under `src/eq_agent/` or `src/eq_data/` (Plans 1-6, already merged/PR'd) — reuse only, via import.
- Do not wire the Neo4j knowledge graph into the request path in this plan — the `ctx` dict passed to `traced_run_eq_assessment` is `{"eq_rag": rag_ctx}` only, matching `src/eq_agent/branch_config.py`'s existing `dispatch_fn` contract (`ctx.get("eq_rag")`). Neo4j-in-the-live-agent remains a deferred item beyond this plan.
- Do not modify the frontend in this plan — scope is backend-only, per the approved 8-plan sequence ("Plan 7: Backend Integration"; frontend wiring is not part of it).
- `configure_langsmith_tracing()` (Plan 6, `src/eq_agent/observability.py`) is called once at module import time in the new router module (process startup) — not per-request. It is safe to call with no real API key present (returns `False`, no-ops).
- The EQ RAG corpus persist directory defaults to `data/eq/lancedb`, matching `src/eq_data/build_eq_corpus.py`'s own `main()` default, overridable via an `EQ_RAG_PERSIST_DIR` env var. `src.eq_agent.eq_context.load_eq_rag_context` already returns `None` gracefully if the directory or its tables don't exist yet — the endpoint must keep working in that case (specialists will just receive `{"error": "EQ RAG corpus is not available (not built yet)."}` from their retrieval tools, per `branch_config.py`'s already-tested `dispatch_fn` contract — no new handling needed here).
- Reuse `backend.agent_router._build_deepseek_config` / `_build_openrouter_config` for LLM client construction instead of duplicating that logic — the same DeepSeek/OpenRouter provider configuration (`DEEPSEEK_API_KEY`/`DEEPSEEK_MODELS`, `OPENROUTER_API_KEY`/`OPENROUTER_MODELS` env vars) serves both the Extraversion and EQ agent pipelines.

---

### Task 1: `/predict-eq` FastAPI route

**Files:**
- Create: `backend/eq_router.py`
- Modify: `backend/main.py` (add one import line after line 23's `from backend.agent_router import router as agent_router`, and one `app.include_router(...)` line after line 34's `app.include_router(agent_router)`)
- Test: `tests/backend/test_eq_router.py`

**Interfaces:**
- Consumes: `src.eq_agent.eq_context.load_eq_rag_context(persist_dir)` (Plan 3) — returns a rag_ctx dict or `None`. `src.eq_agent.branch_config.build_branch_configs()` (Plan 3) — returns the `branch_configs` dict `run_eq_assessment`/`traced_run_eq_assessment` expect. `src.eq_agent.traced_assessment.traced_run_eq_assessment(client, models, ctx, text, branch_configs, extra_params=None, max_loop_rounds=2)` (Plan 6) — same signature and return shape as `src.eq_agent.graph.run_eq_assessment`: `{"score", "tier", "tier_label", "confidence", "rationale", "branch_results", "degraded_branches", "loop_count", "degraded", "error"}`. `src.eq_agent.observability.configure_langsmith_tracing(api_key=None, project=None, endpoint=None) -> bool` (Plan 6). `backend.agent_router._build_deepseek_config(api_key)` / `_build_openrouter_config(api_key)` — each returns `(client, models, extra_params)`.
- Produces: `router` (FastAPI `APIRouter` with the `POST /predict-eq` route), `get_eq_context()`, `get_eq_client_and_models()`, `get_eq_branch_configs()` — dependency-override hooks for tests, matching `backend/agent_router.py`'s exact naming/caching pattern (`get_agent_context`, `get_agent_client_and_models`).

- [ ] **Step 1: Write the failing tests**

```python
import json

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.eq_router import get_eq_branch_configs, get_eq_client_and_models, get_eq_context, router
from src.agent.openrouter_client import build_client
from src.eq_agent.graph import BRANCHES

_TEST_CTX = {"eq_rag": None}


def _make_test_branch_configs():
    return {
        branch: {"tool_schemas": [], "dispatch_fn": lambda *a: {}, "system_prompt": f"Assess {branch}."}
        for branch in BRANCHES
    }


def _make_test_app():
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_eq_context] = lambda: _TEST_CTX
    app.dependency_overrides[get_eq_branch_configs] = lambda: _make_test_branch_configs()
    return app


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
        return _tool_call_response("c1", "submit_overall_assessment", {"score": 62.0, "confidence": "medium", "rationale": "x"})
    return _tool_call_response("c1", "submit_branch_assessment", {"score": 55.0, "confidence": "medium", "rationale": "x"})


def test_predict_eq_returns_error_on_empty_text():
    app = _make_test_app()
    app.dependency_overrides[get_eq_client_and_models] = lambda: (None, ["fake-model"], None)
    client = TestClient(app)

    response = client.post("/predict-eq", json={"text": "   "})

    assert response.status_code == 200
    assert response.json() == {"error": "Empty text"}


def test_predict_eq_returns_error_when_no_models_configured():
    app = _make_test_app()
    app.dependency_overrides[get_eq_client_and_models] = lambda: (None, [], None)
    client = TestClient(app)

    response = client.post("/predict-eq", json={"text": "hello"})

    assert response.status_code == 200
    assert "OPENROUTER_MODELS" in response.json()["error"]


def test_predict_eq_returns_eq_result_on_success():
    fake_client = build_client("fake-key", transport=httpx.MockTransport(_happy_path_handler))
    app = _make_test_app()
    app.dependency_overrides[get_eq_client_and_models] = lambda: (fake_client, ["fake-model"], None)
    client = TestClient(app)

    response = client.post("/predict-eq", json={"text": "I could tell she was upset before she said anything."})

    assert response.status_code == 200
    body = response.json()
    assert body["score"] == 62.0
    assert body["degraded"] is False
    assert set(body["branch_results"].keys()) == set(BRANCHES)


def test_predict_eq_degrades_when_api_fails():
    def handler(request):
        return httpx.Response(500, json={"error": "down"})

    fake_client = build_client("fake-key", transport=httpx.MockTransport(handler))
    app = _make_test_app()
    app.dependency_overrides[get_eq_client_and_models] = lambda: (fake_client, ["fake-model"], None)
    client = TestClient(app)

    response = client.post("/predict-eq", json={"text": "I could tell she was upset before she said anything."})

    assert response.status_code == 200
    body = response.json()
    assert body["degraded"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/backend/test_eq_router.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.eq_router'`

- [ ] **Step 3: Write the implementation**

Create `backend/eq_router.py`:

```python
"""The /predict-eq FastAPI route: the EQ (Emotional Intelligence) multi-agent
LangGraph pipeline (src/eq_agent/), running alongside /predict-agent's
Extraversion pipeline and backend/main.py's classical-ML pipeline -- these
three run side by side, none replaces another.
"""
import os

from dotenv import load_dotenv
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.agent_router import _build_deepseek_config, _build_openrouter_config
from src.eq_agent.branch_config import build_branch_configs
from src.eq_agent.eq_context import load_eq_rag_context
from src.eq_agent.observability import configure_langsmith_tracing
from src.eq_agent.traced_assessment import traced_run_eq_assessment

load_dotenv()
configure_langsmith_tracing()

router = APIRouter()
_cached_ctx = None
_cached_client_and_models = None
_cached_branch_configs = None


def get_eq_context():
    global _cached_ctx
    if _cached_ctx is None:
        persist_dir = os.environ.get("EQ_RAG_PERSIST_DIR", os.path.join("data", "eq", "lancedb"))
        rag = load_eq_rag_context(persist_dir)
        _cached_ctx = {"eq_rag": rag}
    return _cached_ctx


def get_eq_client_and_models():
    global _cached_client_and_models
    if _cached_client_and_models is None:
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if deepseek_key:
            _cached_client_and_models = _build_deepseek_config(deepseek_key)
        else:
            _cached_client_and_models = _build_openrouter_config(os.environ.get("OPENROUTER_API_KEY", ""))
    return _cached_client_and_models


def get_eq_branch_configs():
    global _cached_branch_configs
    if _cached_branch_configs is None:
        _cached_branch_configs = build_branch_configs()
    return _cached_branch_configs


class PredictEQRequest(BaseModel):
    text: str


@router.post("/predict-eq")
def predict_eq(
    req: PredictEQRequest,
    ctx=Depends(get_eq_context),
    client_and_models=Depends(get_eq_client_and_models),
    branch_configs=Depends(get_eq_branch_configs),
):
    if not req.text.strip():
        return {"error": "Empty text"}
    client, models, extra_params = client_and_models
    if not models:
        return {"error": "No models configured. Set DEEPSEEK_API_KEY (preferred) or OPENROUTER_API_KEY/OPENROUTER_MODELS."}
    return traced_run_eq_assessment(client, models, ctx, req.text, branch_configs, extra_params=extra_params)
```

Modify `backend/main.py` — add the import immediately after the existing `agent_router` import (currently line 23):

```python
from backend.agent_router import router as agent_router
from backend.eq_router import router as eq_router
```

And add the include immediately after the existing `agent_router` include (currently line 34):

```python
app.include_router(agent_router)
app.include_router(eq_router)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/backend/test_eq_router.py -v`
Expected: 4 passed

- [ ] **Step 5: Run the full project test suite**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: all pass, no regressions (this task adds `backend/eq_router.py` + its test, and a strictly additive 2-line change to `backend/main.py`; no other file changes).

- [ ] **Step 6: Commit**

```bash
git add backend/eq_router.py backend/main.py tests/backend/test_eq_router.py
git commit -m "feat: add /predict-eq FastAPI endpoint for the EQ multi-agent pipeline"
```

---

## After This Plan

Per the approved sequence: Plan 8 (Evaluation Harness — Recall@k/MRR/nDCG retrieval metrics and a reranking ablation over the EQ RAG corpus). Whether real LangSmith tracing appears correctly in a dashboard once a real `LANGSMITH_API_KEY` is set in `.env` is a manual verification step for whoever has a real LangSmith account. Frontend wiring for `/predict-eq` (a UI route/selector analogous to how `/predict-agent` is surfaced), wiring the Neo4j knowledge graph into the live agent's prompts/tools, and building the real EQ RAG corpus for production use (`python -m src.eq_data.build_eq_corpus`, not yet run for real in this pivot) all remain explicitly deferred, unchanged from prior plans' notes.

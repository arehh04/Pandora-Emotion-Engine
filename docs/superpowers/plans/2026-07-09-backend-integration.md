# Backend Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose Plan 3's agent orchestrator as a real FastAPI endpoint, `/predict-agent`, with minimal, isolated impact on `backend/main.py` — a file under active, uncommitted, parallel revision outside this plan's scope, exactly as `src/extract_classical_features.py` was during Plans 1-3.

**Architecture:** A new, self-contained router module, `backend/agent_router.py`, owns everything: FastAPI dependency-injected builders for the agent context (spaCy, NRC lexicon, a freshly-trained ML-prior model, optional RAG corpus — all from Plan 3's `src.agent.context`) and the OpenRouter client (from Plan 3's `src.agent.openrouter_client`), plus the `/predict-agent` route itself, which calls Plan 3's `run_agent`. This router is tested in complete isolation — a throwaway `FastAPI()` test app with just this router mounted, using `TestClient` and dependency overrides — so its test suite never imports the volatile `backend/main.py` at all (which would otherwise pull in real BERT/Redis/spaCy startup work through a completely unrelated code path). Only the final task touches `backend/main.py`, and only by two lines: importing the router and mounting it.

**Tech Stack:** FastAPI (already a dependency), `fastapi.testclient.TestClient` (confirmed working with the installed FastAPI/httpx versions in `.venv` — note it currently emits a harmless `StarletteDeprecationWarning` about `httpx2`, not a real problem), everything from Plans 1-3.

## Global Constraints

- **Run every command with `"./.venv/Scripts/python.exe"`**, not a bare `python`.
- **`backend/main.py` is out of scope except for a single, minimal, explicitly-isolated edit in the final task.** Do not read it as a dependency to build against, do not refactor it, do not fix anything in it. If its content has changed again by the time this plan executes, that's expected — re-read it fresh immediately before the one edit and keep the edit to the same two lines regardless of what else is in the file.
- `backend/agent_router.py`'s own test suite must NOT import `backend.main` — it exercises a standalone test `FastAPI()` app with only this plan's router mounted, so it stays fast and fully decoupled from `backend/main.py`'s volatility and heavy real-model startup path.
- No `__init__.py` files anywhere under `src/` or `backend/` (implicit namespace packages — established convention; confirmed `backend/` currently has none either).
- The agent context (spaCy model, NRC lexicon, trained ML-prior model, optional RAG corpus) and the OpenRouter client are built once and cached at module level — not rebuilt per request.
- `OPENROUTER_API_KEY` and `OPENROUTER_MODELS` (comma-separated model slugs) are read from environment variables; if no models are configured, the endpoint returns a clear `{"error": ...}` rather than attempting a call with an empty model list.

---

### Task 1: `/predict-agent` router with dependency-injected context and client

**Files:**
- Create: `backend/agent_router.py`
- Test: `tests/backend/test_agent_router.py`

**Interfaces:**
- Consumes: `train_ml_prior_from_data`, `load_rag_context` (`src.agent.context`); `build_client` (`src.agent.openrouter_client`); `run_agent` (`src.agent.orchestrator`); `load_nrc_lexicon` (`src.agent.tools.classical_features`) — all from Plan 3, already merged into this branch's history.
- Produces: `router` (a `fastapi.APIRouter` with one route, `POST /predict-agent`), `get_agent_context()` and `get_agent_client_and_models()` (FastAPI dependency functions, each lazily building and caching its resource at module level — override these in tests via `app.dependency_overrides`). Consumed by Task 2 (the one-line mount into `backend/main.py`'s real app).

- [ ] **Step 1: Write the failing test**

Create `tests/backend/test_agent_router.py`. This builds a real (but fast) test context once at module level — real spaCy, real NRC lexicon, a Ridge model trained on two synthetic rows (same pattern as Plan 3's tests) — and mounts the router on a throwaway `FastAPI()` app, never importing `backend.main`:

```python
import os

import httpx
import spacy
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.agent_router import router, get_agent_context, get_agent_client_and_models
from src.agent.openrouter_client import build_client
from src.agent.tools.classical_features import load_nrc_lexicon
from src.agent.tools.ml_prior import train_ml_prior


def _build_test_ctx():
    nlp = spacy.load("en_core_web_sm")
    nrc_dict = load_nrc_lexicon(os.path.join("data", "NRC-Emotion-Lexicon-Senselevel-v0.92.txt"))
    feature_rows = [
        {"positive": 0.3, "negative": 0.0, "semantic_polarity": 0.9, "behav_exclamation_ratio": 0.2,
         "behav_question_ratio": 0.0, "behav_verb_ratio": 0.3, "behav_1st_sg_pronoun_ratio": 0.0,
         "behav_1st_pl_pronoun_ratio": 0.2},
        {"positive": 0.0, "negative": 0.3, "semantic_polarity": -0.9, "behav_exclamation_ratio": 0.0,
         "behav_question_ratio": 0.0, "behav_verb_ratio": 0.05, "behav_1st_sg_pronoun_ratio": 0.2,
         "behav_1st_pl_pronoun_ratio": 0.0},
    ]
    ml_model = train_ml_prior(feature_rows, [90.0, 10.0])
    return {"nlp": nlp, "nrc_dict": nrc_dict, "ml_model": ml_model, "rag": None}


_TEST_CTX = _build_test_ctx()


def _make_test_app():
    app = FastAPI()
    app.include_router(router)
    return app


def test_predict_agent_returns_error_on_empty_text():
    app = _make_test_app()
    app.dependency_overrides[get_agent_context] = lambda: _TEST_CTX
    app.dependency_overrides[get_agent_client_and_models] = lambda: (None, ["fake-model"])
    client = TestClient(app)

    response = client.post("/predict-agent", json={"text": "   "})

    assert response.status_code == 200
    assert response.json() == {"error": "Empty text"}


def test_predict_agent_returns_error_when_no_models_configured():
    app = _make_test_app()
    app.dependency_overrides[get_agent_context] = lambda: _TEST_CTX
    app.dependency_overrides[get_agent_client_and_models] = lambda: (None, [])
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
    app.dependency_overrides[get_agent_client_and_models] = lambda: (fake_client, ["fake-model"])
    client = TestClient(app)

    response = client.post("/predict-agent", json={"text": "I love parties!"})

    assert response.status_code == 200
    body = response.json()
    assert body["tier"] == 6
    assert body["degraded"] is False


def test_predict_agent_degrades_when_api_fails():
    def handler(request):
        return httpx.Response(500, json={"error": "down"})

    fake_client = build_client("fake-key", transport=httpx.MockTransport(handler))
    app = _make_test_app()
    app.dependency_overrides[get_agent_context] = lambda: _TEST_CTX
    app.dependency_overrides[get_agent_client_and_models] = lambda: (fake_client, ["fake-model"])
    client = TestClient(app)

    response = client.post("/predict-agent", json={"text": "I love parties and talking to everyone!"})

    assert response.status_code == 200
    body = response.json()
    assert body["degraded"] is True
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/backend/test_agent_router.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.agent_router'`

- [ ] **Step 3: Write the minimal implementation**

Create `backend/agent_router.py`:

```python
"""The /predict-agent FastAPI route. Deliberately isolated from backend/main.py
(which is under active, unrelated revision) — this module owns its own
context/client construction and is mounted onto the real app with a single
`include_router` call, kept separate in backend/main.py itself.
"""
import os

import spacy
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.agent.context import load_rag_context, train_ml_prior_from_data
from src.agent.openrouter_client import build_client
from src.agent.orchestrator import run_agent
from src.agent.tools.classical_features import load_nrc_lexicon

router = APIRouter()

_cached_ctx = None
_cached_client_and_models = None


def get_agent_context():
    global _cached_ctx
    if _cached_ctx is None:
        nlp = spacy.load("en_core_web_sm")
        nrc_dict = load_nrc_lexicon(os.path.join("data", "NRC-Emotion-Lexicon-Senselevel-v0.92.txt"))
        ml_model = train_ml_prior_from_data("data/train_clean.csv", nlp, nrc_dict)
        rag = load_rag_context("data/rag")
        _cached_ctx = {"nlp": nlp, "nrc_dict": nrc_dict, "ml_model": ml_model, "rag": rag}
    return _cached_ctx


def get_agent_client_and_models():
    global _cached_client_and_models
    if _cached_client_and_models is None:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        client = build_client(api_key)
        models_env = os.environ.get("OPENROUTER_MODELS", "")
        models = [m.strip() for m in models_env.split(",") if m.strip()]
        _cached_client_and_models = (client, models)
    return _cached_client_and_models


class PredictAgentRequest(BaseModel):
    text: str


@router.post("/predict-agent")
def predict_agent(
    req: PredictAgentRequest,
    ctx=Depends(get_agent_context),
    client_and_models=Depends(get_agent_client_and_models),
):
    if not req.text.strip():
        return {"error": "Empty text"}

    client, models = client_and_models
    if not models:
        return {"error": "No OPENROUTER_MODELS configured. Set the OPENROUTER_MODELS environment variable."}

    return run_agent(client, models, ctx, req.text)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/backend/test_agent_router.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the full project test suite**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/ -v`
Expected: PASS (all tests from Plans 1-3 plus this task's 4 new tests)

- [ ] **Step 6: Commit**

```bash
git add backend/agent_router.py tests/backend/test_agent_router.py
git commit -m "feat: add isolated /predict-agent FastAPI router"
```

---

### Task 2: Mount the router onto the real backend app

**Files:**
- Modify: `backend/main.py` (two lines only — see below)

**Interfaces:**
- Consumes: `router` from Task 1's `backend.agent_router`.
- Produces: the real, already-running FastAPI `app` in `backend/main.py` now also serves `POST /predict-agent`.

This task is intentionally not TDD'd against `backend/main.py` — that file is out of scope for this plan (see Global Constraints) and importing it in an automated test would trigger its own real startup path (BERT, Redis, spaCy) unrelated to this plan's work. Verification here is a one-time manual import check, not a permanent test.

- [ ] **Step 1: Re-read the current backend/main.py**

Open `backend/main.py` fresh right now (its content may have changed since this plan was written — that's expected, not a problem). Find the line `app = FastAPI(title="Pandora API")` (or whatever the current equivalent app-construction line is called) and the CORS middleware block that follows it.

- [ ] **Step 2: Add exactly two lines**

Immediately after the `app.add_middleware(CORSMiddleware, ...)` block (or immediately after the `app = FastAPI(...)` line if no CORS block exists — use whatever is actually present), add:

```python
from backend.agent_router import router as agent_router

app.include_router(agent_router)
```

Place the `from backend.agent_router import router as agent_router` line alongside the file's other imports at the top (not inline in the middle of the file), and the `app.include_router(agent_router)` call immediately after `app`'s construction/middleware setup, before the first `@app.get`/`@app.post` route decorator. Do not reformat, reorder, or otherwise touch any other line in the file.

- [ ] **Step 3: Verify the router is mounted (manual check, not a permanent test)**

Run this one-off command to confirm the route registers without crashing anything at import time:

```bash
"./.venv/Scripts/python.exe" -c "from backend.main import app; print('/predict-agent' in [r.path for r in app.routes])"
```

Expected output: `True`

If this command fails with an import error unrelated to your two-line change (e.g. a missing package, a Redis connection error not already handled by the existing try/except, or an error inside code you did not touch), that's a pre-existing issue with `backend/main.py`'s current state — report it back rather than trying to fix it, since that file is out of scope for this plan.

- [ ] **Step 4: Run this plan's own test suite once more to confirm nothing broke**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/ -v`
Expected: PASS (same count as Task 1's Step 5 — this task doesn't add or change any test)

- [ ] **Step 5: Commit**

```bash
git add backend/main.py
git commit -m "feat: mount /predict-agent router onto the main FastAPI app"
```

---

## Plan Self-Review Notes

- **Spec coverage:** the "Backend & Frontend Integration" item from the original 5-plan breakdown is split by explicit user decision this round: this plan covers only the backend `/predict-agent` endpoint (TDD-able via `TestClient`); the frontend Agent Trace panel is deferred to a separate, manually-verified follow-up (no frontend test runner — no vitest/playwright — is configured in this repo, so it can't go through the same automated review loop).
- **No placeholders:** both tasks have complete, runnable code (or, for Task 2, exact literal lines to add) — no TBD/TODO markers.
- **Type/interface consistency:** `get_agent_context()`'s returned dict shape (`nlp`, `nrc_dict`, `ml_model`, `rag`) matches exactly what `run_agent` (Plan 3) expects. `get_agent_client_and_models()`'s `(client, models)` tuple matches `run_agent(client, models, ctx, text)`'s first two positional arguments.
- **Isolation is the load-bearing design choice here:** Task 1's tests never import `backend.main`, and Task 2 makes no claim about testing `backend/main.py` itself beyond a manual smoke check — this is deliberate, not a coverage gap, given that file's documented volatility throughout Plans 1-3.

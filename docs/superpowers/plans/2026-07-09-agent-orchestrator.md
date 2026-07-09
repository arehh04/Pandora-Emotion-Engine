# Agent Orchestrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire Plan 2's four tools (classical features, fuzzy logic, ML-prior, RAG retrieval) into a working LLM agent — a custom ReAct-style loop that calls an OpenRouter-hosted model, lets it choose which tools to consult, and terminates via a structured "submit assessment" tool call.

**Architecture:** A thin `httpx`-based OpenRouter client (no `openai` SDK dependency — OpenRouter's endpoint is a plain OpenAI-compatible REST call, and `httpx` is already a project dependency) with model-fallback retry. Five OpenAI-style function-calling tool schemas: four wrap Plan 2's tools (`fuzzy_logic_assessment`, `ml_prior_assessment`, `retrieve_similar_exemplars`, `retrieve_relevant_theory`), and a fifth, `submit_assessment`, is the agent's only way to end the conversation — this is the standard, most reliable pattern for getting structured output from a tool-calling LLM (asking it to emit a free-text JSON blob is not). `tier_label` in the final result is derived server-side from `src.tiers.TIER_BINS`, not trusted from the LLM's own words. If the OpenRouter call fails outright (after retrying across the configured fallback models), the orchestrator degrades gracefully to calling the ML-prior tool directly and returns a `degraded: true` result rather than raising.

**Tech Stack:** Python (`.venv`), httpx (already installed), pandas (for the real-data ML-prior training helper), everything from Plan 1 (`src.tiers`, `src.rag`) and Plan 2 (`src.agent.tools.*`). No new dependencies.

## Global Constraints

- **Run every command with the project's virtual environment:** `"./.venv/Scripts/python.exe"`, not a bare `python` (this repo has two Python installs with different partial dependency sets; only `.venv` has spaCy/xgboost/scikit-learn, and it's also the one this plan's tests assume has `httpx`, which is already present there — confirmed via `.venv/Scripts/python.exe -m pip list`).
- **No `openai` SDK, no other LLM framework (LangChain/LangGraph/etc.)** — OpenRouter's API is called directly via `httpx` so the orchestration logic stays fully inspectable for the thesis defense.
- **No real network calls in automated tests.** Every test that exercises the orchestrator loop or the OpenRouter client uses `httpx.MockTransport` to simulate responses — no test may require a real `OPENROUTER_API_KEY` or produce a real API bill. A real, manual end-to-end run against the live OpenRouter API is a deferred follow-up step, called out explicitly where relevant.
- **The exact OpenRouter model slug(s) are a runtime configuration value, not a hardcoded fact.** This plan's code reads the model list from an environment variable (`OPENROUTER_MODELS`, comma-separated) with no baked-in assumption about which specific free-tier models are currently available — model rosters on OpenRouter change over time, and the person running this code must supply real, currently-valid model slugs before the real (non-mocked) integration step.
- Every tool call is wrapped so it returns a structured `{"error": str}` dict on failure rather than raising — the orchestrator loop must never crash because one tool call failed; it should continue reasoning with whatever else succeeded.
- Canonical tier scheme lives in `src.tiers` (`TIER_BINS`, `assign_tier`) — the final assessment's `tier_label` is looked up from `TIER_BINS` by `tier` number, never generated or trusted from the LLM's tool-call arguments.
- No `__init__.py` files anywhere under `src/` (implicit namespace packages — established convention from Plans 1 and 2).
- The RAG corpus artifacts (`data/rag/exemplars_meta.csv` etc.) may not exist on disk yet in this repo (Plan 1 built the builder script but the real, slow embedding run is still a deferred manual step) — any code touching them must handle their absence gracefully (return `None`/a clear error), not assume they exist.

---

### Task 1: OpenRouter HTTP client with model fallback

**Files:**
- Create: `src/agent/openrouter_client.py`
- Test: `tests/agent/test_openrouter_client.py`

**Interfaces:**
- Produces: `build_client(api_key: str, base_url: str = OPENROUTER_BASE_URL, transport=None, timeout: float = 30.0) -> httpx.Client`, `call_chat_completion(client: httpx.Client, model: str, messages: list[dict], tools: list[dict] | None = None) -> dict` (raises on HTTP error), and `call_with_fallback(client: httpx.Client, models: list[str], messages: list[dict], tools: list[dict] | None = None, max_retries_per_model: int = 2) -> dict` (tries each model in order, raises `RuntimeError` only if every model/attempt fails). Consumed by Task 4 (Orchestrator loop).

- [ ] **Step 1: Write the failing test**

Create `tests/agent/test_openrouter_client.py`. This uses `httpx.MockTransport` to simulate the OpenRouter API without any real network call:

```python
import json

import httpx

from src.agent.openrouter_client import build_client, call_chat_completion, call_with_fallback


def test_call_chat_completion_returns_parsed_json():
    def handler(request):
        assert request.headers["authorization"] == "Bearer fake-key"
        body = json.loads(request.content)
        assert body["model"] == "some-model"
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "hi"}}]
        })

    client = build_client("fake-key", transport=httpx.MockTransport(handler))

    result = call_chat_completion(client, "some-model", [{"role": "user", "content": "hello"}])

    assert result["choices"][0]["message"]["content"] == "hi"


def test_call_chat_completion_raises_on_http_error():
    def handler(request):
        return httpx.Response(500, json={"error": "server error"})

    client = build_client("fake-key", transport=httpx.MockTransport(handler))

    try:
        call_chat_completion(client, "some-model", [{"role": "user", "content": "hi"}])
        assert False, "expected an exception"
    except httpx.HTTPStatusError:
        pass


def test_call_with_fallback_tries_next_model_on_failure():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body["model"])
        if body["model"] == "model-a":
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "ok"}}]
        })

    client = build_client("fake-key", transport=httpx.MockTransport(handler))

    result = call_with_fallback(
        client, ["model-a", "model-b"], [{"role": "user", "content": "hi"}], max_retries_per_model=1
    )

    assert result["choices"][0]["message"]["content"] == "ok"
    assert calls == ["model-a", "model-b"]


def test_call_with_fallback_raises_when_every_model_fails():
    def handler(request):
        return httpx.Response(500, json={"error": "down"})

    client = build_client("fake-key", transport=httpx.MockTransport(handler))

    try:
        call_with_fallback(client, ["model-a", "model-b"], [{"role": "user", "content": "hi"}], max_retries_per_model=1)
        assert False, "expected an exception"
    except RuntimeError as e:
        assert "model-b" in str(e) or "All models failed" in str(e)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/agent/test_openrouter_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agent.openrouter_client'`

- [ ] **Step 3: Write the minimal implementation**

Create `src/agent/openrouter_client.py`:

```python
"""Thin httpx-based OpenRouter chat-completions client with model fallback.

Deliberately avoids the `openai` SDK: OpenRouter's endpoint is a plain
OpenAI-compatible REST call, and httpx is already a project dependency —
adding a whole SDK for one POST request isn't warranted, and a direct
implementation is easier to inspect end-to-end.
"""
import httpx

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def build_client(api_key, base_url=OPENROUTER_BASE_URL, transport=None, timeout=30.0):
    headers = {"Authorization": f"Bearer {api_key}"}
    return httpx.Client(base_url=base_url, headers=headers, timeout=timeout, transport=transport)


def call_chat_completion(client, model, messages, tools=None):
    payload = {"model": model, "messages": messages}
    if tools:
        payload["tools"] = tools
    response = client.post("/chat/completions", json=payload)
    response.raise_for_status()
    return response.json()


def call_with_fallback(client, models, messages, tools=None, max_retries_per_model=2):
    last_error = None
    for model in models:
        for _ in range(max_retries_per_model):
            try:
                return call_chat_completion(client, model, messages, tools)
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                last_error = e
    raise RuntimeError(f"All models failed ({models}). Last error: {last_error}")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/agent/test_openrouter_client.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/agent/openrouter_client.py tests/agent/test_openrouter_client.py
git commit -m "feat: add httpx-based OpenRouter client with model fallback"
```

---

### Task 2: Tool schemas and resilient dispatch

**Files:**
- Create: `src/agent/tool_schemas.py`
- Test: `tests/agent/test_tool_schemas.py`

**Interfaces:**
- Consumes: `extract_features_for_text` (`src.agent.tools.classical_features`), `run_fuzzy_inference` (`src.agent.tools.fuzzy_engine`), `predict_ml_prior` (`src.agent.tools.ml_prior`), `retrieve_similar_exemplars`/`retrieve_relevant_theory` (`src.agent.tools.rag_retrieval`) — all from Plan 2, already merged into this branch's history.
- Produces: `TOOL_SCHEMAS` (list of 5 OpenAI-style function-calling schema dicts, names: `fuzzy_logic_assessment`, `ml_prior_assessment`, `retrieve_similar_exemplars`, `retrieve_relevant_theory`, `submit_assessment`) and `dispatch_tool_call(name: str, arguments: dict, ctx: dict) -> dict` — routes a tool-call name + arguments to the right underlying function using a context dict (keys: `nlp`, `nrc_dict`, `ml_model`, and optionally `rag` — a dict with `corpus`/`embedder`, or absent/`None` if the RAG corpus isn't built yet). Every call is wrapped so failures return `{"error": str(e)}` instead of raising. `submit_assessment` is NOT dispatched through this function — it's the orchestrator's terminal signal, handled directly in Task 4's loop.

- [ ] **Step 1: Write the failing test**

Create `tests/agent/test_tool_schemas.py`. Uses the real `en_core_web_sm` spaCy model and NRC lexicon (fast, as established in Plan 2) plus a tiny real-data Ridge model for the ML-prior tool, and a fixture RAG corpus for the retrieval tools:

```python
import json
import os

import numpy as np
import pandas as pd
import spacy

from src.agent.tools.classical_features import load_nrc_lexicon
from src.agent.tools.ml_prior import train_ml_prior
from src.agent.tool_schemas import TOOL_SCHEMAS, dispatch_tool_call


def _build_test_context(tmp_path=None):
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
    scores = [90.0, 10.0]
    ml_model = train_ml_prior(feature_rows, scores)

    ctx = {"nlp": nlp, "nrc_dict": nrc_dict, "ml_model": ml_model, "rag": None}

    if tmp_path is not None:
        rag_dir = tmp_path / "rag"
        rag_dir.mkdir()
        pd.DataFrame({
            "bert_text": ["I love parties"], "extraversion": [90], "tier": [6], "tier_label": ["Highly Extraverted"],
        }).to_csv(rag_dir / "exemplars_meta.csv", index=False)
        np.save(rag_dir / "exemplars_embeddings.npy", np.array([[1.0, 0.0]]))
        (rag_dir / "theory_meta.json").write_text(
            json.dumps([{"id": "a", "topic": "t", "text": "gregariousness", "citation_needed": "n/a"}]),
            encoding="utf-8",
        )
        np.save(rag_dir / "theory_embeddings.npy", np.array([[1.0, 0.0]]))

        class FakeEmbedder:
            def encode(self, texts):
                return np.array([[1.0, 0.0] for _ in texts])

        from src.agent.tools.rag_retrieval import load_rag_corpus
        ctx["rag"] = {"corpus": load_rag_corpus(str(rag_dir)), "embedder": FakeEmbedder()}

    return ctx


def test_tool_schemas_have_five_entries_with_required_names():
    names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
    assert names == {
        "fuzzy_logic_assessment", "ml_prior_assessment",
        "retrieve_similar_exemplars", "retrieve_relevant_theory", "submit_assessment",
    }
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]


def test_dispatch_fuzzy_logic_assessment_returns_valid_result():
    ctx = _build_test_context()

    result = dispatch_tool_call("fuzzy_logic_assessment", {"text": "I love parties and meeting people!"}, ctx)

    assert set(result.keys()) == {"fuzzy_score", "tier", "tier_label", "fired_rules"}


def test_dispatch_ml_prior_assessment_returns_valid_result():
    ctx = _build_test_context()

    result = dispatch_tool_call("ml_prior_assessment", {"text": "I love parties!"}, ctx)

    assert set(result.keys()) == {"score", "tier", "tier_label"}


def test_dispatch_rag_tools_return_error_when_corpus_absent():
    ctx = _build_test_context()  # ctx["rag"] is None

    exemplar_result = dispatch_tool_call("retrieve_similar_exemplars", {"text": "hello"}, ctx)
    theory_result = dispatch_tool_call("retrieve_relevant_theory", {"text": "hello"}, ctx)

    assert "error" in exemplar_result
    assert "error" in theory_result


def test_dispatch_rag_tools_return_results_when_corpus_present(tmp_path):
    ctx = _build_test_context(tmp_path)

    result = dispatch_tool_call("retrieve_similar_exemplars", {"text": "party time", "k": 1}, ctx)

    assert "results" in result
    assert result["results"][0]["bert_text"] == "I love parties"


def test_dispatch_unknown_tool_returns_error():
    ctx = _build_test_context()

    result = dispatch_tool_call("not_a_real_tool", {}, ctx)

    assert "error" in result


def test_dispatch_never_raises_on_bad_arguments():
    ctx = _build_test_context()

    result = dispatch_tool_call("fuzzy_logic_assessment", {}, ctx)  # missing required "text"

    assert "error" in result
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/agent/test_tool_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agent.tool_schemas'`

- [ ] **Step 3: Write the minimal implementation**

Create `src/agent/tool_schemas.py`:

```python
"""OpenAI-style function-calling tool schemas for the agent, and a resilient
dispatcher that routes tool calls to Plan 2's underlying tool functions.

submit_assessment is intentionally NOT dispatched here — it's the agent's
terminal action, handled directly by the orchestrator loop (Task 4).
"""
from src.agent.tools.classical_features import extract_features_for_text
from src.agent.tools.fuzzy_engine import run_fuzzy_inference
from src.agent.tools.ml_prior import predict_ml_prior
from src.agent.tools.rag_retrieval import retrieve_relevant_theory, retrieve_similar_exemplars

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "fuzzy_logic_assessment",
            "description": (
                "Assess Extraversion using a hand-rolled fuzzy logic engine over "
                "linguistic/emotional signals extracted from the text. Returns a "
                "continuous score, a tier, and which fuzzy rules fired (for explainability)."
            ),
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "The text to assess."}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ml_prior_assessment",
            "description": "Assess Extraversion using a small trained Ridge regression model.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "The text to assess."}},
                "required": ["text"],
            },
        },
    },
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
        if name == "fuzzy_logic_assessment":
            features = extract_features_for_text(arguments["text"], ctx["nlp"], ctx["nrc_dict"])
            return run_fuzzy_inference(features)

        if name == "ml_prior_assessment":
            return predict_ml_prior(arguments["text"], ctx["nlp"], ctx["nrc_dict"], ctx["ml_model"])

        if name == "retrieve_similar_exemplars":
            if not ctx.get("rag"):
                return {"error": "RAG corpus is not available (not built yet)."}
            k = arguments.get("k", 5)
            results = retrieve_similar_exemplars(arguments["text"], ctx["rag"]["corpus"], ctx["rag"]["embedder"], k=k)
            return {"results": results}

        if name == "retrieve_relevant_theory":
            if not ctx.get("rag"):
                return {"error": "RAG corpus is not available (not built yet)."}
            k = arguments.get("k", 3)
            results = retrieve_relevant_theory(arguments["text"], ctx["rag"]["corpus"], ctx["rag"]["embedder"], k=k)
            return {"results": results}

        return {"error": f"Unknown tool: {name}"}
    except Exception as e:
        return {"error": str(e)}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/agent/test_tool_schemas.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/agent/tool_schemas.py tests/agent/test_tool_schemas.py
git commit -m "feat: add agent tool schemas and resilient tool-call dispatcher"
```

---

### Task 3: Context builders (real-data ML-prior training, RAG context loading)

**Files:**
- Create: `src/agent/context.py`
- Test: `tests/agent/test_context.py`

**Interfaces:**
- Consumes: `extract_features_for_text` and `train_ml_prior` (Plan 2), `load_rag_corpus` (Plan 1's `src.rag.build_corpus` module reuses this name from `src.agent.tools.rag_retrieval` — actually imports `load_rag_corpus` from `src.agent.tools.rag_retrieval`, Plan 2 Task 4).
- Produces: `train_ml_prior_from_data(data_path: str, nlp, nrc_dict: dict, sample_size: int = 300, seed: int = 42)` returning a fitted Ridge model, and `load_rag_context(rag_dir: str, embedder=None) -> dict | None` returning `{"corpus": ..., "embedder": ...}` or `None` if the four required RAG artifact files aren't present at `rag_dir`. Consumed by Task 4 (Orchestrator loop) to assemble its context dict.

- [ ] **Step 1: Write the failing test**

Create `tests/agent/test_context.py`. The ML-prior training test uses the project's real `data/train_clean.csv` (a small sample, so this stays fast) and the real spaCy/NRC pipeline, consistent with Plan 2's approach; the RAG context test uses fixture files so it doesn't require the real (not-yet-built) corpus artifacts:

```python
import json
import os

import numpy as np
import pandas as pd
import spacy

from src.agent.tools.classical_features import load_nrc_lexicon
from src.agent.context import train_ml_prior_from_data, load_rag_context


def test_train_ml_prior_from_data_returns_working_model():
    nlp = spacy.load("en_core_web_sm")
    nrc_dict = load_nrc_lexicon(os.path.join("data", "NRC-Emotion-Lexicon-Senselevel-v0.92.txt"))

    model = train_ml_prior_from_data("data/train_clean.csv", nlp, nrc_dict, sample_size=30, seed=1)

    prediction = model.predict([[0.1, 0.0, 0.5, 0.05, 0.0, 0.2, 0.05, 0.05]])
    assert prediction.shape == (1,)


def test_load_rag_context_returns_none_when_artifacts_missing(tmp_path):
    result = load_rag_context(str(tmp_path))

    assert result is None


def test_load_rag_context_loads_corpus_with_injected_embedder(tmp_path):
    pd.DataFrame({
        "bert_text": ["I love parties"], "extraversion": [90], "tier": [6], "tier_label": ["Highly Extraverted"],
    }).to_csv(tmp_path / "exemplars_meta.csv", index=False)
    np.save(tmp_path / "exemplars_embeddings.npy", np.array([[1.0, 0.0]]))
    (tmp_path / "theory_meta.json").write_text(
        json.dumps([{"id": "a", "topic": "t", "text": "gregariousness", "citation_needed": "n/a"}]),
        encoding="utf-8",
    )
    np.save(tmp_path / "theory_embeddings.npy", np.array([[1.0, 0.0]]))

    class FakeEmbedder:
        def encode(self, texts):
            return np.array([[1.0, 0.0] for _ in texts])

    result = load_rag_context(str(tmp_path), embedder=FakeEmbedder())

    assert result is not None
    assert len(result["corpus"]["exemplars_df"]) == 1
    assert result["embedder"] is not None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/agent/test_context.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agent.context'`

- [ ] **Step 3: Write the minimal implementation**

Create `src/agent/context.py`:

```python
"""Builds the shared resources (a trained ML-prior model, an optional RAG
corpus) the orchestrator needs, once per process — not per request.
"""
import os

import pandas as pd

from src.agent.tools.classical_features import extract_features_for_text
from src.agent.tools.ml_prior import train_ml_prior
from src.agent.tools.rag_retrieval import load_rag_corpus

REQUIRED_RAG_FILES = [
    "exemplars_meta.csv", "exemplars_embeddings.npy", "theory_meta.json", "theory_embeddings.npy",
]


def train_ml_prior_from_data(data_path, nlp, nrc_dict, sample_size=300, seed=42):
    df = pd.read_csv(data_path)
    sample = df.sample(n=min(sample_size, len(df)), random_state=seed)
    feature_rows = [extract_features_for_text(str(text), nlp, nrc_dict) for text in sample["bert_text"]]
    scores = sample["extraversion"].tolist()
    return train_ml_prior(feature_rows, scores)


def load_rag_context(rag_dir, embedder=None):
    if not all(os.path.exists(os.path.join(rag_dir, f)) for f in REQUIRED_RAG_FILES):
        return None

    corpus = load_rag_corpus(rag_dir)

    if embedder is None:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")

    return {"corpus": corpus, "embedder": embedder}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/agent/test_context.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/agent/context.py tests/agent/test_context.py
git commit -m "feat: add ML-prior real-data training and RAG context loading helpers"
```

*(Note: `load_rag_context`'s lazy `sentence_transformers` import is exercised for real only when `embedder=None` and the artifacts exist — neither is true yet in this repo, so this path stays untested by the automated suite until Plan 1's real corpus-building run has actually happened. This matches the same deferred-real-run pattern used throughout Plans 1 and 2.)*

---

### Task 4: Orchestrator ReAct loop

**Files:**
- Create: `src/agent/orchestrator.py`
- Test: `tests/agent/test_orchestrator.py`

**Interfaces:**
- Consumes: `call_with_fallback`, `build_client` (Task 1); `TOOL_SCHEMAS`, `dispatch_tool_call` (Task 2); `TIER_BINS` (`src.tiers`, Plan 1); `predict_ml_prior` (Plan 2, used directly for the degraded-fallback path).
- Produces: `SYSTEM_PROMPT` (str), `label_for_tier(tier_num: int) -> str`, and `run_agent(client, models: list[str], ctx: dict, text: str, max_iterations: int = 6) -> dict` returning `{"tier": int, "tier_label": str, "continuous_score_estimate": float, "confidence": str, "rationale": str, "trace": list[dict], "degraded": bool, "error": str | None}`.

- [ ] **Step 1: Write the failing test**

Create `tests/agent/test_orchestrator.py`. This uses `httpx.MockTransport` to script a multi-turn conversation with no real network access, and a minimal real context (fast to build, per Plan 2's established pattern) so tool dispatch genuinely executes:

```python
import json
import os

import httpx
import spacy

from src.agent.tools.classical_features import load_nrc_lexicon
from src.agent.tools.ml_prior import train_ml_prior
from src.agent.openrouter_client import build_client
from src.agent.orchestrator import run_agent, label_for_tier, SYSTEM_PROMPT


def _build_test_context():
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
    scores = [90.0, 10.0]
    ml_model = train_ml_prior(feature_rows, scores)
    return {"nlp": nlp, "nrc_dict": nrc_dict, "ml_model": ml_model, "rag": None}


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
            return _assistant_tool_call_response("call_1", "fuzzy_logic_assessment", {"text": "I love parties!"})
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
    assert result["trace"][0]["tool"] == "fuzzy_logic_assessment"


def test_run_agent_stops_after_max_iterations_without_submit():
    def handler(request):
        return _assistant_tool_call_response("call_x", "ml_prior_assessment", {"text": "hi"})

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
    assert 0.0 <= result["continuous_score_estimate"] <= 99.0
    assert 1 <= result["tier"] <= 6
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/agent/test_orchestrator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agent.orchestrator'`

- [ ] **Step 3: Write the minimal implementation**

Create `src/agent/orchestrator.py`:

```python
"""The agent's ReAct-style orchestration loop: calls an OpenRouter model with
tool-calling enabled, dispatches whichever tools it chooses, and terminates
when it calls submit_assessment. Falls back to the ML-prior tool directly
if the OpenRouter call fails outright after retrying across all configured
fallback models.
"""
import json

from src.agent.openrouter_client import call_with_fallback
from src.agent.tool_schemas import TOOL_SCHEMAS, dispatch_tool_call
from src.agent.tools.ml_prior import predict_ml_prior
from src.tiers import TIER_BINS

SYSTEM_PROMPT = (
    "You are an assessment agent estimating the Extraversion of a piece of text "
    "on a 1 (most reserved) to 6 (most extraverted) tier scale. You have tools "
    "available to gather evidence: a fuzzy-logic signal-fusion assessment, a "
    "small trained ML model, and retrieval of similar labeled examples and "
    "relevant psychology theory. Use as many tools as you find useful, then "
    "call submit_assessment exactly once with your final tier, a 0-99 "
    "continuous score estimate, your confidence, and a brief rationale citing "
    "the evidence you gathered."
)


def label_for_tier(tier_num):
    for _low, _high, num, label in TIER_BINS:
        if num == tier_num:
            return label
    raise ValueError(f"invalid tier {tier_num}")


def _degraded_result(text, ctx, error):
    prior = predict_ml_prior(text, ctx["nlp"], ctx["nrc_dict"], ctx["ml_model"])
    return {
        "tier": prior["tier"],
        "tier_label": prior["tier_label"],
        "continuous_score_estimate": prior["score"],
        "confidence": "low",
        "rationale": "Agent unavailable; falling back to the ML-prior tool directly.",
        "trace": [],
        "degraded": True,
        "error": error,
    }


def run_agent(client, models, ctx, text, max_iterations=6):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Assess the Extraversion of this text:\n\n{text}"},
    ]
    trace = []

    for _ in range(max_iterations):
        try:
            response = call_with_fallback(client, models, messages, tools=TOOL_SCHEMAS)
        except Exception as e:
            return _degraded_result(text, ctx, str(e))

        message = response["choices"][0]["message"]
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            return _degraded_result(text, ctx, "Agent responded without calling a tool.")

        messages.append(message)

        for tool_call in tool_calls:
            name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])

            if name == "submit_assessment":
                return {
                    "tier": arguments["tier"],
                    "tier_label": label_for_tier(arguments["tier"]),
                    "continuous_score_estimate": arguments["continuous_score_estimate"],
                    "confidence": arguments["confidence"],
                    "rationale": arguments["rationale"],
                    "trace": trace,
                    "degraded": False,
                    "error": None,
                }

            result = dispatch_tool_call(name, arguments, ctx)
            trace.append({"tool": name, "arguments": arguments, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(result),
            })

    return _degraded_result(text, ctx, f"Max iterations ({max_iterations}) reached without submit_assessment.")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/agent/test_orchestrator.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the full project test suite**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/ -v`
Expected: PASS (all tests from Plans 1, 2, and this plan)

- [ ] **Step 6: Commit**

```bash
git add src/agent/orchestrator.py tests/agent/test_orchestrator.py
git commit -m "feat: add agent orchestrator ReAct loop with degraded fallback"
```

*(Note: a real, non-mocked end-to-end run — setting a real `OPENROUTER_API_KEY` and real model slugs, then calling `run_agent` against the live OpenRouter API — is a manual follow-up step, not part of this plan's automated tests, consistent with every other "real external call" deferred elsewhere in Plans 1-3.)*

---

## Plan Self-Review Notes

- **Spec coverage:** Design spec Section 2 (architecture: custom ReAct loop, OpenRouter, tool-calling, resilience/fallback) → all four tasks. The "fallback to legacy `/predict`" idea from the original design spec no longer applies (Plan 2 decoupled every tool from `backend/main.py`); the degraded fallback here calls the self-contained `ml_prior` tool directly instead, which is the only reachable local, reliable signal now that the legacy backend is out of scope.
- **No placeholders:** every task has complete, runnable code; no TBD/TODO markers. The one deliberately-untested path (`load_rag_context`'s real `sentence_transformers` import) is explicitly called out as deferred, not silently skipped.
- **Type/interface consistency:** `dispatch_tool_call`'s `ctx` dict shape (`nlp`, `nrc_dict`, `ml_model`, `rag`) is identical across Task 2's tests, Task 3's `load_rag_context` output shape, and Task 4's orchestrator usage. `TOOL_SCHEMAS`' five function names exactly match the `if name == ...` branches in both `dispatch_tool_call` (four of them) and `run_agent` (the fifth, `submit_assessment`, handled as the terminal case). `label_for_tier` reads `src.tiers.TIER_BINS` (Plan 1) directly — no re-derived tier boundaries.
- **Scope note:** this plan does not build a `/predict-agent` FastAPI endpoint or a frontend panel — that's Plan 4 (Backend & Frontend Integration), which will import `run_agent` and `build_client`/`load_rag_context`/`train_ml_prior_from_data` from this plan's modules to assemble a real request-serving context.

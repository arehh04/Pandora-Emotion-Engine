# EQ LangGraph Multi-Agent Orchestrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the LangGraph multi-agent orchestrator for the EQ pivot: 4 parallel MSC-branch specialist agents, a coordinator that synthesizes their verdicts, and a critic that can trigger one bounded re-assessment loop — all wired into a compiled `StateGraph` with a single entry-point function.

**Architecture:** A new `src/eq_agent/` package (parallel to the existing `src/agent/` package, which stays untouched and keeps serving the current Extraversion single-agent `/predict-agent` endpoint until a later Backend Integration plan decides how to transition). Three generic, branch/config-parameterized LLM-calling functions (`run_specialist`, `run_coordinator`, `run_critic`) each reuse the proven `call_with_fallback` ReAct-loop pattern from `src/agent/orchestrator.py`, but take their tool schemas, dispatcher, and prompts as parameters instead of hardcoded module constants — so Plan 3 can plug in the real branch-specific RAG tools without touching this plan's code. `src/eq_agent/graph.py` wires these three functions into a `StateGraph` with a verified fan-out/fan-in/conditional-loop structure and exposes `run_eq_assessment(...)` as the single call site a later Backend Integration plan will use.

**Tech Stack:** `langgraph` 1.2.9 (`StateGraph`, `START`/`END`, `add_conditional_edges`) — deliberately used only for graph/state/parallelism mechanics, NOT for LLM calls (see Global Constraints). Existing `httpx`-based `call_with_fallback` client, `pytest`.

## Global Constraints

- `langgraph==1.2.9` is the verified installed version (installs `langgraph-checkpoint`, `langgraph-prebuilt`, `langchain-core` as transitive deps). Do **not** add `langchain-deepseek` or `langchain-openai` — `langchain-deepseek`'s `ChatDeepSeek` has an active, documented bug ([langchain-ai/langchain#37174](https://github.com/langchain-ai/langchain/issues/37174), [#34166](https://github.com/langchain-ai/langchain/issues/34166)) dropping `reasoning_content` on multi-turn tool-calling requests with DeepSeek thinking mode enabled — exactly this project's configuration. All LLM calls in this plan go through the existing `src/agent/openrouter_client.call_with_fallback`, called as a plain Python function from inside graph nodes.
- **Verified LangGraph facts** (computed directly against the installed library before writing this plan, not assumed): (a) a `TypedDict` state field can be written by multiple same-superstep parallel nodes without an `InvalidUpdateError`, *if* it's annotated with a merge-dict reducer (`Annotated[dict, merge_fn]`) where each node's partial update touches a different key; (b) `add_edge(START, node)` from multiple nodes, converging via `add_edge(node, "coordinator")` from each, correctly fans out and fans in — a node with several incoming edges runs once all of its *currently active* predecessors for that superstep have completed, not literally every node in the graph; (c) `add_conditional_edges(source, routing_fn, mapping)` accepts a routing function that returns a **list** of mapping keys, triggering all of them — used here so the same routing function always returns a list, whether 1+ branch names or a single custom `"__end__"` key mapped to `END`; (d) a compiled graph's plain synchronous `.invoke()` runs independent same-superstep nodes **concurrently** by default — a timing test with 4 nodes each sleeping 0.5s completed in ~0.51s wall-clock (not ~2.0s), confirming no manual `ThreadPoolExecutor`/async rewrite is needed to get the "parallel specialists" performance property. All 4 findings were confirmed by running real smoke-test graphs against the installed `langgraph==1.2.9`, not inferred from documentation.
- The critique loop is capped at `max_loop_rounds=2` (default). The critic node's `loop_count` increment must be computed **after** applying the cap override, not before — otherwise the reported `loop_count` overshoots by one at exactly the cap boundary (a bug caught and fixed during this plan's own verification, see Task 3).
- `run_specialist`/`run_coordinator`/`run_critic` must never raise — every LLM-call failure mode (API error, malformed response, max-iterations-without-submit) degrades to a documented neutral default, exactly like the existing `src/agent/orchestrator.py::_degraded_result` pattern. `run_eq_assessment` additionally wraps the whole graph invocation in a defensive try/except as a last-resort safety net.
- Partial specialist failure: a degraded branch's neutral-default result (score 50.0, confidence "low", an explicit "could not complete its assessment" rationale) is still passed to the coordinator alongside the successful branches, rather than omitted — the coordinator always sees exactly 4 branch entries. `degraded_branches` is surfaced separately in `run_eq_assessment`'s return value so a caller (or a later plan) can act on it explicitly. Confidence-lowering in response to a degraded branch is a soft signal (the coordinator LLM can see the degraded branch's own "low confidence" text in its prompt and factor it in) rather than a hard-coded rule — this plan does not force the coordinator's own confidence down programmatically.
- This plan is purely additive: no file under `src/agent/`, `backend/`, or `frontend/` is modified. `src/eq_agent/` has no `__init__.py` (implicit namespace packages, same convention as `src/eq_data/` and `src/rag/`).
- No test in this plan makes a real network or LLM call — all tests use `httpx.MockTransport`-backed fake clients, exactly like the existing `tests/agent/test_orchestrator.py` suite.

---

### Task 1: Generic MSC-branch specialist runner

**Files:**
- Create: `src/eq_agent/specialist.py`
- Test: `tests/eq_agent/test_specialist.py`

**Interfaces:**
- Consumes: `call_with_fallback(client, models, messages, tools=None, max_retries_per_model=2, extra_params=None)` from `src/agent/openrouter_client.py` (unchanged); `assign_eq_tier(score) -> (tier_num, label)` from `src/eq_data/tiers_eq.py` (Plan 1, already merged).
- Produces: `SUBMIT_BRANCH_ASSESSMENT_SCHEMA` (a tool schema dict) and `run_specialist(client, models, ctx, text, branch, tool_schemas, dispatch_fn, system_prompt, extra_params=None, max_iterations=6, critic_feedback=None) -> dict` returning `{score, tier, tier_label, confidence, rationale, trace, degraded, error}` — consumed by Task 4's graph node wrapper.

- [ ] **Step 1: Write the failing tests**

```python
import json

import httpx

from src.agent.openrouter_client import build_client
from src.eq_agent.specialist import run_specialist


def _tool_call_response(call_id, name, arguments):
    return httpx.Response(200, json={
        "choices": [{"message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": call_id, "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments),
            }}],
        }}],
    })


def test_run_specialist_submits_immediately_with_no_extra_tools():
    def handler(request):
        return _tool_call_response("call_1", "submit_branch_assessment", {
            "score": 70.0, "confidence": "high", "rationale": "Rich emotional vocabulary.",
        })

    client = build_client("fake-key", transport=httpx.MockTransport(handler))

    result = run_specialist(client, ["fake-model"], {}, "some text", "perceiving", [], lambda *a: {}, "system prompt")

    assert result["degraded"] is False
    assert result["score"] == 70.0
    assert result["confidence"] == "high"
    assert result["tier"] >= 1


def test_run_specialist_calls_a_tool_then_submits():
    turns = {"n": 0}

    def handler(request):
        turns["n"] += 1
        if turns["n"] == 1:
            return _tool_call_response("call_1", "lookup_evidence", {"query": "x"})
        return _tool_call_response("call_2", "submit_branch_assessment", {
            "score": 55.0, "confidence": "medium", "rationale": "Based on retrieved evidence.",
        })

    client = build_client("fake-key", transport=httpx.MockTransport(handler))
    dispatch_calls = []

    def fake_dispatch(name, arguments, ctx):
        dispatch_calls.append((name, arguments))
        return {"results": ["evidence found"]}

    tool_schema = {"type": "function", "function": {"name": "lookup_evidence", "description": "x",
                                                      "parameters": {"type": "object", "properties": {}}}}

    result = run_specialist(client, ["fake-model"], {}, "some text", "using", [tool_schema], fake_dispatch, "system prompt")

    assert result["degraded"] is False
    assert len(result["trace"]) == 1
    assert result["trace"][0]["tool"] == "lookup_evidence"
    assert dispatch_calls == [("lookup_evidence", {"query": "x"})]


def test_run_specialist_degrades_on_api_failure():
    def handler(request):
        return httpx.Response(500, json={"error": "down"})

    client = build_client("fake-key", transport=httpx.MockTransport(handler))

    result = run_specialist(client, ["fake-model"], {}, "text", "managing", [], lambda *a: {}, "prompt")

    assert result["degraded"] is True
    assert result["score"] == 50.0
    assert result["error"] is not None


def test_run_specialist_degrades_on_malformed_response():
    def handler(request):
        return httpx.Response(200, json={"not_choices": "bad"})

    client = build_client("fake-key", transport=httpx.MockTransport(handler))

    result = run_specialist(client, ["fake-model"], {}, "text", "understanding", [], lambda *a: {}, "prompt")

    assert result["degraded"] is True


def test_run_specialist_stops_after_max_iterations_without_submit():
    def handler(request):
        return _tool_call_response("call_x", "lookup_evidence", {})

    client = build_client("fake-key", transport=httpx.MockTransport(handler))
    tool_schema = {"type": "function", "function": {"name": "lookup_evidence", "description": "x",
                                                      "parameters": {"type": "object", "properties": {}}}}

    result = run_specialist(client, ["fake-model"], {}, "text", "perceiving", [tool_schema], lambda *a: {}, "prompt", max_iterations=2)

    assert result["degraded"] is True
    assert "max iterations" in result["error"].lower()


def test_run_specialist_clamps_out_of_range_score():
    def handler(request):
        return _tool_call_response("call_1", "submit_branch_assessment", {
            "score": 150.0, "confidence": "high", "rationale": "x",
        })

    client = build_client("fake-key", transport=httpx.MockTransport(handler))

    result = run_specialist(client, ["fake-model"], {}, "text", "perceiving", [], lambda *a: {}, "prompt")

    assert result["score"] == 99.0


def test_run_specialist_injects_critic_feedback_into_messages():
    sent_messages = []

    def handler(request):
        body = json.loads(request.content)
        sent_messages.append(body["messages"])
        return _tool_call_response("call_1", "submit_branch_assessment", {
            "score": 60.0, "confidence": "medium", "rationale": "Reconsidered.",
        })

    client = build_client("fake-key", transport=httpx.MockTransport(handler))

    run_specialist(
        client, ["fake-model"], {}, "text", "perceiving", [], lambda *a: {}, "prompt",
        critic_feedback="the score seemed too high",
    )

    all_message_contents = [m.get("content", "") for m in sent_messages[0]]
    assert any("the score seemed too high" in (c or "") for c in all_message_contents)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_agent/test_specialist.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eq_agent.specialist'`

- [ ] **Step 3: Write the implementation**

```python
"""A generic, branch-parameterized ReAct tool-calling loop for one MSC
specialist agent. Mirrors src/agent/orchestrator.py's proven run_agent
pattern, but takes its tool schemas, dispatcher, and system prompt as
parameters instead of hardcoded module constants, so the same runner works
for all 4 MSC branches (perceiving/using/understanding/managing) -- Plan 3
supplies each branch's real RAG-grounded tool schemas and dispatcher.
"""
import json

from src.agent.openrouter_client import call_with_fallback
from src.eq_data.tiers_eq import assign_eq_tier

SUBMIT_BRANCH_ASSESSMENT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_branch_assessment",
        "description": "Submit the final assessment for this MSC branch. Call this exactly once, when you are done reasoning.",
        "parameters": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "description": "Estimated score for this branch, 0-99."},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                "rationale": {"type": "string", "description": "Brief explanation citing the tool evidence used."},
            },
            "required": ["score", "confidence", "rationale"],
        },
    },
}


def _degraded_branch_result(error):
    tier, tier_label = assign_eq_tier(50.0)
    return {
        "score": 50.0, "tier": tier, "tier_label": tier_label, "confidence": "low",
        "rationale": "This specialist could not complete its assessment; returning a neutral default.",
        "trace": [], "degraded": True, "error": error,
    }


def run_specialist(
    client, models, ctx, text, branch, tool_schemas, dispatch_fn, system_prompt,
    extra_params=None, max_iterations=6, critic_feedback=None,
):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Assess the {branch} branch of Emotional Intelligence for this text:\n\n{text}"},
    ]
    if critic_feedback:
        messages.append({
            "role": "user",
            "content": f"A critique of your prior assessment flagged this concern: {critic_feedback}. Please reconsider.",
        })

    available_schemas = list(tool_schemas) + [SUBMIT_BRANCH_ASSESSMENT_SCHEMA]
    trace = []

    for _ in range(max_iterations):
        try:
            response = call_with_fallback(client, models, messages, tools=available_schemas, extra_params=extra_params)
        except Exception as e:
            return _degraded_branch_result(str(e))

        try:
            message = response["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                return _degraded_branch_result(f"{branch} specialist responded without calling a tool.")

            messages.append(message)

            for tool_call in tool_calls:
                name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])

                if name == "submit_branch_assessment":
                    score = min(99.0, max(0.0, float(arguments["score"])))
                    tier, tier_label = assign_eq_tier(score)
                    return {
                        "score": score, "tier": tier, "tier_label": tier_label,
                        "confidence": arguments["confidence"], "rationale": arguments["rationale"],
                        "trace": trace, "degraded": False, "error": None,
                    }

                result = dispatch_fn(name, arguments, ctx)
                trace.append({"tool": name, "arguments": arguments, "result": result})
                messages.append({
                    "role": "tool", "tool_call_id": tool_call["id"], "content": json.dumps(result),
                })
        except (KeyError, IndexError, TypeError, ValueError) as e:
            return _degraded_branch_result(f"Malformed specialist response: {e}")

    return _degraded_branch_result(f"Max iterations ({max_iterations}) reached without submit_branch_assessment.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_agent/test_specialist.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/eq_agent/specialist.py tests/eq_agent/test_specialist.py
git commit -m "feat: add generic MSC-branch specialist runner for the EQ multi-agent orchestrator"
```

---

### Task 2: Coordinator (branch verdict synthesis)

**Files:**
- Create: `src/eq_agent/coordinator.py`
- Test: `tests/eq_agent/test_coordinator.py`

**Interfaces:**
- Consumes: `call_with_fallback` (unchanged); `assign_eq_tier` (Plan 1); branch results shaped like Task 1's `run_specialist` output (`{score, confidence, rationale, ...}` per branch, keyed by branch name).
- Produces: `SUBMIT_OVERALL_ASSESSMENT_SCHEMA`, `run_coordinator(client, models, branch_results, extra_params=None) -> dict` returning `{score, tier, tier_label, confidence, rationale, degraded, error}` — consumed by Task 4's graph node wrapper.

- [ ] **Step 1: Write the failing tests**

```python
import json

import httpx
import pytest

from src.agent.openrouter_client import build_client
from src.eq_agent.coordinator import run_coordinator


def _tool_call_response(call_id, name, arguments):
    return httpx.Response(200, json={
        "choices": [{"message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": call_id, "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments),
            }}],
        }}],
    })


BRANCH_RESULTS = {
    "perceiving": {"score": 70.0, "confidence": "high", "rationale": "Clear emotional vocabulary."},
    "using": {"score": 60.0, "confidence": "medium", "rationale": "Some emotional reasoning."},
    "understanding": {"score": 65.0, "confidence": "medium", "rationale": "Recognizes emotion causes."},
    "managing": {"score": 55.0, "confidence": "low", "rationale": "Limited regulation evidence."},
}


def test_run_coordinator_synthesizes_from_branch_results():
    def handler(request):
        return _tool_call_response("call_1", "submit_overall_assessment", {
            "score": 63.0, "confidence": "medium", "rationale": "Balanced across all branches.",
        })

    client = build_client("fake-key", transport=httpx.MockTransport(handler))

    result = run_coordinator(client, ["fake-model"], BRANCH_RESULTS)

    assert result["degraded"] is False
    assert result["score"] == 63.0
    assert result["tier"] >= 1


def test_run_coordinator_degrades_to_average_on_api_failure():
    def handler(request):
        return httpx.Response(500, json={"error": "down"})

    client = build_client("fake-key", transport=httpx.MockTransport(handler))

    result = run_coordinator(client, ["fake-model"], BRANCH_RESULTS)

    assert result["degraded"] is True
    assert result["score"] == pytest.approx((70.0 + 60.0 + 65.0 + 55.0) / 4)


def test_run_coordinator_degrades_on_malformed_response():
    def handler(request):
        return httpx.Response(200, json={"not_choices": "bad"})

    client = build_client("fake-key", transport=httpx.MockTransport(handler))

    result = run_coordinator(client, ["fake-model"], BRANCH_RESULTS)

    assert result["degraded"] is True


def test_run_coordinator_clamps_out_of_range_score():
    def handler(request):
        return _tool_call_response("call_1", "submit_overall_assessment", {
            "score": 150.0, "confidence": "high", "rationale": "x",
        })

    client = build_client("fake-key", transport=httpx.MockTransport(handler))

    result = run_coordinator(client, ["fake-model"], BRANCH_RESULTS)

    assert result["score"] == 99.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_agent/test_coordinator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eq_agent.coordinator'`

- [ ] **Step 3: Write the implementation**

```python
"""Synthesizes the 4 MSC branch verdicts into one overall EQ assessment via
a single LLM tool-call (no ReAct loop -- the coordinator has nothing to
retrieve, only branch summaries to reason over).
"""
import json

from src.agent.openrouter_client import call_with_fallback
from src.eq_data.tiers_eq import assign_eq_tier

SUBMIT_OVERALL_ASSESSMENT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_overall_assessment",
        "description": "Submit the synthesized overall EQ assessment. Call this exactly once.",
        "parameters": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "description": "Overall EQ score, 0-99."},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                "rationale": {"type": "string", "description": "Brief explanation reconciling the 4 branch verdicts."},
            },
            "required": ["score", "confidence", "rationale"],
        },
    },
}

COORDINATOR_SYSTEM_PROMPT = (
    "You are the coordinator of a 4-specialist Emotional Intelligence assessment "
    "team, following the Mayer-Salovey-Caruso model: Perceiving, Using, "
    "Understanding, and Managing emotions. You will be given each specialist's "
    "branch score (0-99), confidence, and rationale. Synthesize these into one "
    "overall EQ score and rationale, reconciling any disagreement between "
    "branches rather than simply averaging them. Call submit_overall_assessment "
    "exactly once."
)


def _format_branch_summary(branch_results):
    lines = [
        f"- {branch}: score={r['score']}, confidence={r['confidence']}, rationale=\"{r['rationale']}\""
        for branch, r in branch_results.items()
    ]
    return "\n".join(lines)


def _degraded_overall_result(branch_results, error):
    scores = [r["score"] for r in branch_results.values()] or [50.0]
    score = sum(scores) / len(scores)
    tier, tier_label = assign_eq_tier(score)
    return {
        "score": score, "tier": tier, "tier_label": tier_label, "confidence": "low",
        "rationale": "Coordinator unavailable; falling back to a plain average of branch scores.",
        "degraded": True, "error": error,
    }


def run_coordinator(client, models, branch_results, extra_params=None):
    messages = [
        {"role": "system", "content": COORDINATOR_SYSTEM_PROMPT},
        {"role": "user", "content": f"Branch assessments:\n{_format_branch_summary(branch_results)}"},
    ]

    try:
        response = call_with_fallback(
            client, models, messages, tools=[SUBMIT_OVERALL_ASSESSMENT_SCHEMA], extra_params=extra_params
        )
        message = response["choices"][0]["message"]
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return _degraded_overall_result(branch_results, "Coordinator responded without calling a tool.")

        arguments = json.loads(tool_calls[0]["function"]["arguments"])
        score = min(99.0, max(0.0, float(arguments["score"])))
        tier, tier_label = assign_eq_tier(score)
        return {
            "score": score, "tier": tier, "tier_label": tier_label,
            "confidence": arguments["confidence"], "rationale": arguments["rationale"],
            "degraded": False, "error": None,
        }
    except (KeyError, IndexError, TypeError, ValueError) as e:
        return _degraded_overall_result(branch_results, f"Malformed coordinator response: {e}")
    except Exception as e:
        return _degraded_overall_result(branch_results, str(e))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_agent/test_coordinator.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/eq_agent/coordinator.py tests/eq_agent/test_coordinator.py
git commit -m "feat: add EQ coordinator to synthesize the 4 branch verdicts"
```

---

### Task 3: Critic (consistency check + bounded re-assessment signal)

**Files:**
- Create: `src/eq_agent/critic.py`
- Test: `tests/eq_agent/test_critic.py`

**Interfaces:**
- Consumes: `call_with_fallback` (unchanged); branch results (Task 1's shape) and an overall verdict (Task 2's shape).
- Produces: `SUBMIT_CRITIQUE_SCHEMA`, `run_critic(client, models, branch_results, overall_verdict, extra_params=None) -> {"consistent": bool, "branches_to_recheck": list[str], "reason": str|None}` — consumed by Task 4's graph node wrapper. Fails open (`consistent=True, branches_to_recheck=[]`) on any error, since the critique loop must never be a single point of failure.

- [ ] **Step 1: Write the failing tests**

```python
import json

import httpx

from src.agent.openrouter_client import build_client
from src.eq_agent.critic import run_critic


def _tool_call_response(call_id, name, arguments):
    return httpx.Response(200, json={
        "choices": [{"message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": call_id, "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments),
            }}],
        }}],
    })


BRANCH_RESULTS = {
    "perceiving": {"score": 90.0, "confidence": "high", "rationale": "Highly attuned to emotional cues."},
    "using": {"score": 60.0, "confidence": "medium", "rationale": "Some use of emotion in reasoning."},
    "understanding": {"score": 65.0, "confidence": "medium", "rationale": "Understands emotion causes."},
    "managing": {"score": 10.0, "confidence": "high", "rationale": "No evidence of self-regulation."},
}
OVERALL_VERDICT = {"score": 56.0, "confidence": "medium", "rationale": "Averaged across branches."}


def test_run_critic_reports_consistent_verdict():
    def handler(request):
        return _tool_call_response("call_1", "submit_critique", {
            "consistent": True, "branches_to_recheck": [], "reason": "",
        })

    client = build_client("fake-key", transport=httpx.MockTransport(handler))

    result = run_critic(client, ["fake-model"], BRANCH_RESULTS, OVERALL_VERDICT)

    assert result["consistent"] is True
    assert result["branches_to_recheck"] == []


def test_run_critic_flags_inconsistency_with_branches_to_recheck():
    def handler(request):
        return _tool_call_response("call_1", "submit_critique", {
            "consistent": False, "branches_to_recheck": ["managing"],
            "reason": "High perceiving score contradicts very low managing score for the same signal.",
        })

    client = build_client("fake-key", transport=httpx.MockTransport(handler))

    result = run_critic(client, ["fake-model"], BRANCH_RESULTS, OVERALL_VERDICT)

    assert result["consistent"] is False
    assert result["branches_to_recheck"] == ["managing"]
    assert "contradicts" in result["reason"]


def test_run_critic_fails_open_on_api_failure():
    def handler(request):
        return httpx.Response(500, json={"error": "down"})

    client = build_client("fake-key", transport=httpx.MockTransport(handler))

    result = run_critic(client, ["fake-model"], BRANCH_RESULTS, OVERALL_VERDICT)

    assert result["consistent"] is True
    assert result["branches_to_recheck"] == []


def test_run_critic_fails_open_on_malformed_response():
    def handler(request):
        return httpx.Response(200, json={"not_choices": "bad"})

    client = build_client("fake-key", transport=httpx.MockTransport(handler))

    result = run_critic(client, ["fake-model"], BRANCH_RESULTS, OVERALL_VERDICT)

    assert result["consistent"] is True


def test_run_critic_ignores_invalid_branch_names_in_recheck_list():
    def handler(request):
        return _tool_call_response("call_1", "submit_critique", {
            "consistent": False, "branches_to_recheck": ["managing", "not_a_real_branch"],
            "reason": "x",
        })

    client = build_client("fake-key", transport=httpx.MockTransport(handler))

    result = run_critic(client, ["fake-model"], BRANCH_RESULTS, OVERALL_VERDICT)

    assert result["branches_to_recheck"] == ["managing"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_agent/test_critic.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eq_agent.critic'`

- [ ] **Step 3: Write the implementation**

```python
"""Checks the coordinator's overall verdict and the 4 branch results for
internal inconsistency, via a single LLM tool-call. Fails open (treats the
input as consistent) if the LLM call fails -- the critique loop is an
enhancement, never a single point of failure for the whole assessment.
"""
import json

from src.agent.openrouter_client import call_with_fallback

SUBMIT_CRITIQUE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_critique",
        "description": "Submit your critique of the assessment's internal consistency. Call this exactly once.",
        "parameters": {
            "type": "object",
            "properties": {
                "consistent": {"type": "boolean", "description": "True if the branch verdicts and overall score are internally consistent."},
                "branches_to_recheck": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Branch names to re-assess, if not consistent. Empty if consistent.",
                },
                "reason": {"type": "string", "description": "Brief explanation of the inconsistency, if any."},
            },
            "required": ["consistent", "branches_to_recheck", "reason"],
        },
    },
}

CRITIC_SYSTEM_PROMPT = (
    "You are a critic reviewing an Emotional Intelligence assessment made up of "
    "4 branch scores (Perceiving, Using, Understanding, Managing) and one "
    "overall synthesized score. Check whether the branch verdicts are "
    "internally consistent with each other and with the overall score -- for "
    "example, a high Perceiving score paired with a contradictory low Managing "
    "rationale citing the same emotional signal. If everything is reasonably "
    "consistent, set consistent=true. Call submit_critique exactly once."
)

VALID_BRANCHES = {"perceiving", "using", "understanding", "managing"}


def _format_verdict_summary(branch_results, overall_verdict):
    lines = [f"- {branch}: score={r['score']}, rationale=\"{r['rationale']}\"" for branch, r in branch_results.items()]
    lines.append(f"Overall: score={overall_verdict['score']}, rationale=\"{overall_verdict['rationale']}\"")
    return "\n".join(lines)


def run_critic(client, models, branch_results, overall_verdict, extra_params=None):
    messages = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": _format_verdict_summary(branch_results, overall_verdict)},
    ]

    try:
        response = call_with_fallback(
            client, models, messages, tools=[SUBMIT_CRITIQUE_SCHEMA], extra_params=extra_params
        )
        message = response["choices"][0]["message"]
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return {"consistent": True, "branches_to_recheck": [], "reason": None}

        arguments = json.loads(tool_calls[0]["function"]["arguments"])
        branches_to_recheck = [b for b in arguments.get("branches_to_recheck", []) if b in VALID_BRANCHES]
        return {
            "consistent": bool(arguments["consistent"]) and not branches_to_recheck,
            "branches_to_recheck": branches_to_recheck,
            "reason": arguments.get("reason"),
        }
    except Exception:
        return {"consistent": True, "branches_to_recheck": [], "reason": None}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_agent/test_critic.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/eq_agent/critic.py tests/eq_agent/test_critic.py
git commit -m "feat: add EQ critic for consistency-checking and bounded re-assessment"
```

---

### Task 4: LangGraph assembly and entry point

**Files:**
- Create: `src/eq_agent/graph.py`
- Modify: `requirements.txt` (add `langgraph==1.2.9`)
- Test: `tests/eq_agent/test_graph.py`

**Interfaces:**
- Consumes: `run_specialist` (Task 1), `run_coordinator` (Task 2), `run_critic` (Task 3).
- Produces: `build_eq_graph(client, models, ctx, branch_configs, extra_params=None, max_loop_rounds=2) -> CompiledGraph` and `run_eq_assessment(client, models, ctx, text, branch_configs, extra_params=None, max_loop_rounds=2) -> dict` returning `{score, tier, tier_label, confidence, rationale, branch_results, degraded_branches, loop_count, degraded, error}`. `branch_configs` is `dict[branch_name, {"tool_schemas": list, "dispatch_fn": callable, "system_prompt": str}]` — this is what a later plan's real per-branch RAG tools plug into.

- [ ] **Step 1: Write the failing tests**

```python
import json

import httpx

from src.agent.openrouter_client import build_client
from src.eq_agent.graph import BRANCHES, run_eq_assessment


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


def test_run_eq_assessment_happy_path_all_branches_succeed():
    def handler(request):
        body = json.loads(request.content)
        tool_names = {t["function"]["name"] for t in body.get("tools", [])}
        if "submit_critique" in tool_names:
            return _tool_call_response("c1", "submit_critique", {
                "consistent": True, "branches_to_recheck": [], "reason": "",
            })
        if "submit_overall_assessment" in tool_names:
            return _tool_call_response("c1", "submit_overall_assessment", {
                "score": 60.0, "confidence": "medium", "rationale": "Balanced.",
            })
        return _tool_call_response("c1", "submit_branch_assessment", {
            "score": 55.0, "confidence": "medium", "rationale": "x",
        })

    client = build_client("fake-key", transport=httpx.MockTransport(handler))

    result = run_eq_assessment(client, ["fake-model"], {}, "some text", _make_branch_configs())

    assert result["degraded_branches"] == []
    assert result["loop_count"] == 0
    assert set(result["branch_results"].keys()) == set(BRANCHES)
    assert result["score"] == 60.0


def test_run_eq_assessment_critique_loop_runs_once_then_converges():
    critic_calls = {"n": 0}

    def handler(request):
        body = json.loads(request.content)
        tool_names = {t["function"]["name"] for t in body.get("tools", [])}
        if "submit_critique" in tool_names:
            critic_calls["n"] += 1
            if critic_calls["n"] == 1:
                return _tool_call_response("c1", "submit_critique", {
                    "consistent": False, "branches_to_recheck": ["managing"], "reason": "recheck managing",
                })
            return _tool_call_response("c1", "submit_critique", {
                "consistent": True, "branches_to_recheck": [], "reason": "",
            })
        if "submit_overall_assessment" in tool_names:
            return _tool_call_response("c1", "submit_overall_assessment", {
                "score": 60.0, "confidence": "medium", "rationale": "x",
            })
        return _tool_call_response("c1", "submit_branch_assessment", {
            "score": 55.0, "confidence": "medium", "rationale": "x",
        })

    client = build_client("fake-key", transport=httpx.MockTransport(handler))

    result = run_eq_assessment(client, ["fake-model"], {}, "some text", _make_branch_configs())

    assert result["loop_count"] == 1
    assert critic_calls["n"] == 2


def test_run_eq_assessment_caps_the_critique_loop():
    def handler(request):
        body = json.loads(request.content)
        tool_names = {t["function"]["name"] for t in body.get("tools", [])}
        if "submit_critique" in tool_names:
            return _tool_call_response("c1", "submit_critique", {
                "consistent": False, "branches_to_recheck": ["perceiving"], "reason": "always flag",
            })
        if "submit_overall_assessment" in tool_names:
            return _tool_call_response("c1", "submit_overall_assessment", {
                "score": 60.0, "confidence": "medium", "rationale": "x",
            })
        return _tool_call_response("c1", "submit_branch_assessment", {
            "score": 55.0, "confidence": "medium", "rationale": "x",
        })

    client = build_client("fake-key", transport=httpx.MockTransport(handler))

    result = run_eq_assessment(client, ["fake-model"], {}, "some text", _make_branch_configs(), max_loop_rounds=2)

    assert result["loop_count"] == 2  # capped, not runaway


def test_run_eq_assessment_tolerates_one_specialist_failure():
    def handler(request):
        body = json.loads(request.content)
        tool_names = {t["function"]["name"] for t in body.get("tools", [])}
        if "submit_critique" in tool_names:
            return _tool_call_response("c1", "submit_critique", {
                "consistent": True, "branches_to_recheck": [], "reason": "",
            })
        if "submit_overall_assessment" in tool_names:
            return _tool_call_response("c1", "submit_overall_assessment", {
                "score": 55.0, "confidence": "medium", "rationale": "x",
            })
        message_text = json.dumps(body["messages"])
        if "managing branch" in message_text:
            return httpx.Response(500, json={"error": "down"})
        return _tool_call_response("c1", "submit_branch_assessment", {
            "score": 55.0, "confidence": "medium", "rationale": "x",
        })

    client = build_client("fake-key", transport=httpx.MockTransport(handler))

    result = run_eq_assessment(client, ["fake-model"], {}, "some text", _make_branch_configs())

    assert result["degraded_branches"] == ["managing"]
    assert result["branch_results"]["managing"]["degraded"] is True
    assert result["branch_results"]["perceiving"]["degraded"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_agent/test_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eq_agent.graph'`

- [ ] **Step 3: Add the dependency to requirements.txt**

Add this line at the end of `requirements.txt`:

```
langgraph==1.2.9
```

(Already installed in `.venv` from this plan's own verification; this records it for reproducible installs. Deliberately does NOT add `langchain-deepseek` or `langchain-openai` — see Global Constraints.)

- [ ] **Step 4: Write the implementation**

```python
"""Assembles the EQ multi-agent LangGraph: 4 parallel MSC specialists ->
coordinator -> critic, with a critic-driven re-assessment loop capped at
max_loop_rounds. LangGraph's synchronous StateGraph.invoke() runs
same-superstep nodes concurrently by default (verified: 4 nodes each
sleeping 0.5s completed in ~0.51s wall-clock, not ~2.0s) -- no manual
threading is needed for the parallel fan-out.
"""
from typing import Annotated, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from src.eq_agent.coordinator import run_coordinator
from src.eq_agent.critic import run_critic
from src.eq_agent.specialist import run_specialist
from src.eq_data.tiers_eq import assign_eq_tier

BRANCHES = ["perceiving", "using", "understanding", "managing"]


def _merge_branch_results(existing, new):
    return {**existing, **new}


class EQAssessmentState(TypedDict):
    text: str
    branch_results: Annotated[dict, _merge_branch_results]
    overall_verdict: Optional[dict]
    critic_result: Optional[dict]
    loop_count: int


def _make_specialist_node(branch, client, models, ctx, config, extra_params):
    def node(state):
        critic_result = state.get("critic_result")
        feedback = None
        if critic_result and branch in (critic_result.get("branches_to_recheck") or []):
            feedback = critic_result.get("reason")
        result = run_specialist(
            client, models, ctx, state["text"], branch,
            config["tool_schemas"], config["dispatch_fn"], config["system_prompt"],
            extra_params=extra_params, critic_feedback=feedback,
        )
        return {"branch_results": {branch: result}}
    return node


def _make_coordinator_node(client, models, extra_params):
    def node(state):
        return {"overall_verdict": run_coordinator(client, models, state["branch_results"], extra_params=extra_params)}
    return node


def _make_critic_node(client, models, extra_params, max_loop_rounds):
    def node(state):
        result = run_critic(client, models, state["branch_results"], state["overall_verdict"], extra_params=extra_params)
        if result["branches_to_recheck"] and state["loop_count"] >= max_loop_rounds:
            result = {**result, "consistent": True, "branches_to_recheck": []}
        next_loop_count = state["loop_count"] + 1 if result["branches_to_recheck"] else state["loop_count"]
        return {"critic_result": result, "loop_count": next_loop_count}
    return node


def _route_after_critic(state):
    to_recheck = state["critic_result"]["branches_to_recheck"]
    return to_recheck if to_recheck else ["__end__"]


def build_eq_graph(client, models, ctx, branch_configs, extra_params=None, max_loop_rounds=2):
    builder = StateGraph(EQAssessmentState)

    for branch in BRANCHES:
        builder.add_node(branch, _make_specialist_node(branch, client, models, ctx, branch_configs[branch], extra_params))
        builder.add_edge(START, branch)
        builder.add_edge(branch, "coordinator")

    builder.add_node("coordinator", _make_coordinator_node(client, models, extra_params))
    builder.add_node("critic", _make_critic_node(client, models, extra_params, max_loop_rounds))
    builder.add_edge("coordinator", "critic")
    builder.add_conditional_edges(
        "critic", _route_after_critic, {branch: branch for branch in BRANCHES} | {"__end__": END}
    )

    return builder.compile()


def run_eq_assessment(client, models, ctx, text, branch_configs, extra_params=None, max_loop_rounds=2):
    graph = build_eq_graph(client, models, ctx, branch_configs, extra_params=extra_params, max_loop_rounds=max_loop_rounds)

    initial_state = {
        "text": text, "branch_results": {}, "overall_verdict": None,
        "critic_result": None, "loop_count": 0,
    }

    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        tier, tier_label = assign_eq_tier(50.0)
        return {
            "score": 50.0, "tier": tier, "tier_label": tier_label,
            "confidence": "low", "rationale": "The assessment graph failed to complete; returning a neutral default.",
            "branch_results": {}, "degraded_branches": list(BRANCHES), "loop_count": 0,
            "degraded": True, "error": str(e),
        }

    overall = final_state["overall_verdict"]
    branch_results = final_state["branch_results"]
    degraded_branches = [b for b, r in branch_results.items() if r.get("degraded")]

    return {
        "score": overall["score"], "tier": overall["tier"], "tier_label": overall["tier_label"],
        "confidence": overall["confidence"], "rationale": overall["rationale"],
        "branch_results": branch_results, "degraded_branches": degraded_branches,
        "loop_count": final_state["loop_count"], "degraded": overall.get("degraded", False),
        "error": overall.get("error"),
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_agent/test_graph.py -v`
Expected: 4 passed

- [ ] **Step 6: Run the full project test suite**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: all pass, no regressions in Plan 1's or any earlier branch's tests (this plan only adds new files under `src/eq_agent/`/`tests/eq_agent/`).

- [ ] **Step 7: Commit**

```bash
git add src/eq_agent/graph.py requirements.txt tests/eq_agent/test_graph.py
git commit -m "feat: assemble the EQ LangGraph multi-agent orchestrator with a bounded critique loop"
```

---

## After This Plan

The next plan builds the real per-branch RAG tools (branch-filtered hybrid ChromaDB+BM25 retrieval over Plan 1's MSC theory corpus and branch-tagged exemplars, plus Perceiving/Understanding grounding from the external GoEmotions/ISEAR/EmoBank/EmpatheticDialogues datasets fetched in Plan 1 Task 1) and wires them into `branch_configs` for `run_eq_assessment` — replacing this plan's test-only empty tool lists with the real specialist capabilities. After that: Backend Integration (a `/predict-eq` endpoint, deciding how it coexists with or replaces the existing `/predict-agent`) and the Evaluation Harness (per-branch metrics, ablations, LLM-judge routing) from the design spec.

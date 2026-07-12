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

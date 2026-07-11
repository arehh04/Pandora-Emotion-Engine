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

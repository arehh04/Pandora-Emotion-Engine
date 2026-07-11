import json

import httpx

from src.agent.openrouter_client import build_client
from src.agent.orchestrator import label_for_tier, run_agent


def _build_test_context():
    return {"rag": None}


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
            return _assistant_tool_call_response("call_1", "retrieve_similar_exemplars", {"text": "I love parties!"})
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
    assert result["trace"][0]["tool"] == "retrieve_similar_exemplars"


def test_run_agent_stops_after_max_iterations_without_submit():
    def handler(request):
        return _assistant_tool_call_response("call_x", "retrieve_relevant_theory", {"text": "hi"})

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
    assert result["continuous_score_estimate"] == 50.0
    assert result["tier"] == 4


def test_run_agent_degrades_gracefully_on_malformed_response():
    def handler(request):
        return httpx.Response(200, json={"not_choices": "this response is malformed"})

    client = build_client("fake-key", transport=httpx.MockTransport(handler))
    ctx = _build_test_context()

    result = run_agent(client, ["fake-model"], ctx, "I love parties and talking to everyone!")

    assert result["degraded"] is True
    assert result["error"] is not None
    assert result["continuous_score_estimate"] == 50.0
    assert result["tier"] == 4


def test_run_agent_clamps_out_of_range_submitted_score():
    def handler(request):
        return _assistant_tool_call_response("call_1", "submit_assessment", {
            "tier": 6, "continuous_score_estimate": 150.0, "confidence": "high",
            "rationale": "Overconfident.",
        })

    client = build_client("fake-key", transport=httpx.MockTransport(handler))
    ctx = _build_test_context()

    result = run_agent(client, ["fake-model"], ctx, "I love parties!")

    assert result["continuous_score_estimate"] == 99.0


def test_degraded_result_returns_a_neutral_default():
    from src.agent.orchestrator import _degraded_result

    result = _degraded_result("some text", {"rag": None}, "original error")

    assert result["degraded"] is True
    assert result["tier"] == 4
    assert result["tier_label"] == label_for_tier(4)
    assert result["continuous_score_estimate"] == 50.0
    assert result["error"] == "original error"


def test_run_agent_restricts_tool_schemas_sent_to_the_api():
    sent_tool_names = []

    def handler(request):
        body = json.loads(request.content)
        sent_tool_names.append([t["function"]["name"] for t in body.get("tools", [])])
        return _assistant_tool_call_response("call_1", "submit_assessment", {
            "tier": 3, "continuous_score_estimate": 40.0, "confidence": "medium",
            "rationale": "Ambiguous signal.",
        })

    client = build_client("fake-key", transport=httpx.MockTransport(handler))
    ctx = _build_test_context()

    run_agent(client, ["fake-model"], ctx, "some text", enabled_tools={"retrieve_similar_exemplars"})

    assert set(sent_tool_names[0]) == {"retrieve_similar_exemplars", "submit_assessment"}


def test_run_agent_gracefully_refuses_a_disabled_tool_call():
    turns = {"n": 0}

    def handler(request):
        turns["n"] += 1
        if turns["n"] == 1:
            return _assistant_tool_call_response("call_1", "retrieve_relevant_theory", {"text": "hi"})
        return _assistant_tool_call_response("call_2", "submit_assessment", {
            "tier": 3, "continuous_score_estimate": 40.0, "confidence": "medium",
            "rationale": "Ambiguous signal.",
        })

    client = build_client("fake-key", transport=httpx.MockTransport(handler))
    ctx = _build_test_context()

    result = run_agent(client, ["fake-model"], ctx, "some text", enabled_tools={"retrieve_similar_exemplars"})

    assert result["degraded"] is False
    assert result["trace"][0]["tool"] == "retrieve_relevant_theory"
    assert "disabled" in result["trace"][0]["result"]["error"]


def test_run_agent_enabled_tools_none_still_exposes_all_tools():
    sent_tool_names = []

    def handler(request):
        body = json.loads(request.content)
        sent_tool_names.append([t["function"]["name"] for t in body.get("tools", [])])
        return _assistant_tool_call_response("call_1", "submit_assessment", {
            "tier": 3, "continuous_score_estimate": 40.0, "confidence": "medium",
            "rationale": "Ambiguous signal.",
        })

    client = build_client("fake-key", transport=httpx.MockTransport(handler))
    ctx = _build_test_context()

    run_agent(client, ["fake-model"], ctx, "some text")

    assert len(sent_tool_names[0]) == 3

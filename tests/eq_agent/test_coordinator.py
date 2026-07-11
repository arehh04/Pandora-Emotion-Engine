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

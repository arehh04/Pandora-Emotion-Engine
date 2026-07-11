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

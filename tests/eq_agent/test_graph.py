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

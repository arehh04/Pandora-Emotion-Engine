import json

import httpx

from src.agent.openrouter_client import build_client
from src.eq_agent.graph import BRANCHES
from src.evaluation.eq_comparison import make_run_eq_assessment_predict_fn, run_eq_evaluation, summarize_eq_evaluation

_NRC_FIXTURE = {"happy": {"positive", "joy"}, "sad": {"negative", "sadness"}}


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
        return _tool_call_response(
            "c1", "submit_overall_assessment",
            {"score": 70.0, "confidence": "high", "rationale": "Consistently positive language."},
        )
    return _tool_call_response(
        "c1", "submit_branch_assessment",
        {"score": 65.0, "confidence": "high", "rationale": "Positive language."},
    )


def _make_branch_configs():
    return {
        branch: {"tool_schemas": [], "dispatch_fn": lambda *a: {}, "system_prompt": f"Assess {branch}."}
        for branch in BRANCHES
    }


def test_run_eq_evaluation_produces_expected_row_shape():
    fake_client = build_client("fake-key", transport=httpx.MockTransport(_happy_path_handler))
    predict_fn = make_run_eq_assessment_predict_fn(fake_client, ["fake-model"], {}, _make_branch_configs())
    samples = [("I am so happy today", {
        "extraversion": 80.0, "openness": 70.0, "agreeableness": 60.0,
        "conscientiousness": 50.0, "neuroticism": 20.0,
    })]

    rows = run_eq_evaluation(predict_fn, samples, _NRC_FIXTURE)

    assert len(rows) == 1
    assert rows[0]["predicted_score"] == 70.0
    assert rows[0]["degraded"] is False
    assert isinstance(rows[0]["true_score"], float)


def test_summarize_eq_evaluation_returns_regression_and_tier_metrics():
    rows = [
        {"true_score": 70.0, "predicted_score": 70.0, "true_tier": 5, "predicted_tier": 5, "faithful": True, "degraded": False},
        {"true_score": 40.0, "predicted_score": 45.0, "true_tier": 3, "predicted_tier": 3, "faithful": False, "degraded": False},
    ]

    summary = summarize_eq_evaluation(rows)

    assert summary["rmse"] >= 0.0
    assert summary["faithfulness_rate"] == 0.5
    assert summary["degraded_rate"] == 0.0

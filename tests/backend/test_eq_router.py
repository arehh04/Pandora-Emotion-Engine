import json

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.eq_router import get_eq_branch_configs, get_eq_client_and_models, get_eq_context, router
from src.agent.openrouter_client import build_client
from src.eq_agent.graph import BRANCHES

_TEST_CTX = {"eq_rag": None}


def _make_test_branch_configs():
    return {
        branch: {"tool_schemas": [], "dispatch_fn": lambda *a: {}, "system_prompt": f"Assess {branch}."}
        for branch in BRANCHES
    }


def _make_test_app():
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_eq_context] = lambda: _TEST_CTX
    app.dependency_overrides[get_eq_branch_configs] = lambda: _make_test_branch_configs()
    return app


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
        return _tool_call_response("c1", "submit_overall_assessment", {"score": 62.0, "confidence": "medium", "rationale": "x"})
    return _tool_call_response("c1", "submit_branch_assessment", {"score": 55.0, "confidence": "medium", "rationale": "x"})


def test_predict_eq_returns_error_on_empty_text():
    app = _make_test_app()
    app.dependency_overrides[get_eq_client_and_models] = lambda: (None, ["fake-model"], None)
    client = TestClient(app)

    response = client.post("/predict-eq", json={"text": "   "})

    assert response.status_code == 200
    assert response.json() == {"error": "Empty text"}


def test_predict_eq_returns_error_when_no_models_configured():
    app = _make_test_app()
    app.dependency_overrides[get_eq_client_and_models] = lambda: (None, [], None)
    client = TestClient(app)

    response = client.post("/predict-eq", json={"text": "hello"})

    assert response.status_code == 200
    assert "OPENROUTER_MODELS" in response.json()["error"]


def test_predict_eq_returns_eq_result_on_success():
    fake_client = build_client("fake-key", transport=httpx.MockTransport(_happy_path_handler))
    app = _make_test_app()
    app.dependency_overrides[get_eq_client_and_models] = lambda: (fake_client, ["fake-model"], None)
    client = TestClient(app)

    response = client.post("/predict-eq", json={"text": "I could tell she was upset before she said anything."})

    assert response.status_code == 200
    body = response.json()
    assert body["score"] == 62.0
    assert body["degraded"] is False
    assert set(body["branch_results"].keys()) == set(BRANCHES)


def test_predict_eq_degrades_when_api_fails():
    def handler(request):
        return httpx.Response(500, json={"error": "down"})

    fake_client = build_client("fake-key", transport=httpx.MockTransport(handler))
    app = _make_test_app()
    app.dependency_overrides[get_eq_client_and_models] = lambda: (fake_client, ["fake-model"], None)
    client = TestClient(app)

    response = client.post("/predict-eq", json={"text": "I could tell she was upset before she said anything."})

    assert response.status_code == 200
    body = response.json()
    assert body["degraded"] is True

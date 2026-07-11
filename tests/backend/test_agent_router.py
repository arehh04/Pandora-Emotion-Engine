import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.agent_router as agent_router_module
from backend.agent_router import (
    _build_deepseek_config,
    _build_openrouter_config,
    get_agent_client_and_models,
    get_agent_context,
    router,
)
from src.agent.openrouter_client import DEEPSEEK_BASE_URL, OPENROUTER_BASE_URL, build_client

_TEST_CTX = {"rag": None}


def _make_test_app():
    app = FastAPI()
    app.include_router(router)
    return app


def test_predict_agent_returns_error_on_empty_text():
    app = _make_test_app()
    app.dependency_overrides[get_agent_context] = lambda: _TEST_CTX
    app.dependency_overrides[get_agent_client_and_models] = lambda: (None, ["fake-model"], None)
    client = TestClient(app)

    response = client.post("/predict-agent", json={"text": "   "})

    assert response.status_code == 200
    assert response.json() == {"error": "Empty text"}


def test_predict_agent_returns_error_when_no_models_configured():
    app = _make_test_app()
    app.dependency_overrides[get_agent_context] = lambda: _TEST_CTX
    app.dependency_overrides[get_agent_client_and_models] = lambda: (None, [], None)
    client = TestClient(app)

    response = client.post("/predict-agent", json={"text": "hello"})

    assert response.status_code == 200
    assert "OPENROUTER_MODELS" in response.json()["error"]


def test_predict_agent_returns_agent_result_on_success():
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {
                    "name": "submit_assessment",
                    "arguments": (
                        '{"tier": 6, "continuous_score_estimate": 90.0, '
                        '"confidence": "high", "rationale": "Very outgoing language."}'
                    ),
                }}],
            }}],
        })

    fake_client = build_client("fake-key", transport=httpx.MockTransport(handler))
    app = _make_test_app()
    app.dependency_overrides[get_agent_context] = lambda: _TEST_CTX
    app.dependency_overrides[get_agent_client_and_models] = lambda: (fake_client, ["fake-model"], None)
    client = TestClient(app)

    response = client.post("/predict-agent", json={"text": "I love parties!"})

    assert response.status_code == 200
    body = response.json()
    assert body["tier"] == 6
    assert body["degraded"] is False


def test_build_deepseek_config_uses_v4_flash_with_high_reasoning_effort():
    client, models, extra_params = _build_deepseek_config("fake-deepseek-key")

    assert str(client.base_url).rstrip("/") == DEEPSEEK_BASE_URL
    assert models == ["deepseek-v4-flash"]
    assert extra_params == {"reasoning_effort": "high", "thinking": {"type": "enabled"}}


def test_build_openrouter_config_reads_models_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODELS", "model-a,model-b")

    client, models, extra_params = _build_openrouter_config("fake-openrouter-key")

    assert str(client.base_url).rstrip("/") == OPENROUTER_BASE_URL
    assert models == ["model-a", "model-b"]
    assert extra_params == {"reasoning": {"enabled": True}}


def test_get_agent_client_and_models_prefers_deepseek(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-deepseek-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-openrouter-key")
    monkeypatch.setattr(agent_router_module, "_cached_client_and_models", None)

    client, models, extra_params = get_agent_client_and_models()

    assert models == ["deepseek-v4-flash"]
    assert "thinking" in extra_params

    monkeypatch.setattr(agent_router_module, "_cached_client_and_models", None)


def test_predict_agent_degrades_when_api_fails():
    def handler(request):
        return httpx.Response(500, json={"error": "down"})

    fake_client = build_client("fake-key", transport=httpx.MockTransport(handler))
    app = _make_test_app()
    app.dependency_overrides[get_agent_context] = lambda: _TEST_CTX
    app.dependency_overrides[get_agent_client_and_models] = lambda: (fake_client, ["fake-model"], None)
    client = TestClient(app)

    response = client.post("/predict-agent", json={"text": "I love parties and talking to everyone!"})

    assert response.status_code == 200
    body = response.json()
    assert body["degraded"] is True

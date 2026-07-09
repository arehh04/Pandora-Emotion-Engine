import os

import httpx
import spacy
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.agent_router import router, get_agent_context, get_agent_client_and_models
from src.agent.openrouter_client import build_client
from src.agent.tools.classical_features import load_nrc_lexicon
from src.agent.tools.ml_prior import train_ml_prior


def _build_test_ctx():
    nlp = spacy.load("en_core_web_sm")
    nrc_dict = load_nrc_lexicon(os.path.join("data", "NRC-Emotion-Lexicon-Senselevel-v0.92.txt"))
    feature_rows = [
        {"positive": 0.3, "negative": 0.0, "semantic_polarity": 0.9, "behav_exclamation_ratio": 0.2,
         "behav_question_ratio": 0.0, "behav_verb_ratio": 0.3, "behav_1st_sg_pronoun_ratio": 0.0,
         "behav_1st_pl_pronoun_ratio": 0.2},
        {"positive": 0.0, "negative": 0.3, "semantic_polarity": -0.9, "behav_exclamation_ratio": 0.0,
         "behav_question_ratio": 0.0, "behav_verb_ratio": 0.05, "behav_1st_sg_pronoun_ratio": 0.2,
         "behav_1st_pl_pronoun_ratio": 0.0},
    ]
    ml_model = train_ml_prior(feature_rows, [90.0, 10.0])
    return {"nlp": nlp, "nrc_dict": nrc_dict, "ml_model": ml_model, "rag": None}


_TEST_CTX = _build_test_ctx()


def _make_test_app():
    app = FastAPI()
    app.include_router(router)
    return app


def test_predict_agent_returns_error_on_empty_text():
    app = _make_test_app()
    app.dependency_overrides[get_agent_context] = lambda: _TEST_CTX
    app.dependency_overrides[get_agent_client_and_models] = lambda: (None, ["fake-model"])
    client = TestClient(app)

    response = client.post("/predict-agent", json={"text": "   "})

    assert response.status_code == 200
    assert response.json() == {"error": "Empty text"}


def test_predict_agent_returns_error_when_no_models_configured():
    app = _make_test_app()
    app.dependency_overrides[get_agent_context] = lambda: _TEST_CTX
    app.dependency_overrides[get_agent_client_and_models] = lambda: (None, [])
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
    app.dependency_overrides[get_agent_client_and_models] = lambda: (fake_client, ["fake-model"])
    client = TestClient(app)

    response = client.post("/predict-agent", json={"text": "I love parties!"})

    assert response.status_code == 200
    body = response.json()
    assert body["tier"] == 6
    assert body["degraded"] is False


def test_predict_agent_degrades_when_api_fails():
    def handler(request):
        return httpx.Response(500, json={"error": "down"})

    fake_client = build_client("fake-key", transport=httpx.MockTransport(handler))
    app = _make_test_app()
    app.dependency_overrides[get_agent_context] = lambda: _TEST_CTX
    app.dependency_overrides[get_agent_client_and_models] = lambda: (fake_client, ["fake-model"])
    client = TestClient(app)

    response = client.post("/predict-agent", json={"text": "I love parties and talking to everyone!"})

    assert response.status_code == 200
    body = response.json()
    assert body["degraded"] is True

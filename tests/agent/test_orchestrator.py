import json
import os

import httpx
import spacy

from src.agent.tools.classical_features import load_nrc_lexicon
from src.agent.tools.ml_prior import train_ml_prior
from src.agent.openrouter_client import build_client
from src.agent.orchestrator import run_agent, label_for_tier, SYSTEM_PROMPT


def _build_test_context():
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
    scores = [90.0, 10.0]
    ml_model = train_ml_prior(feature_rows, scores)
    return {"nlp": nlp, "nrc_dict": nrc_dict, "ml_model": ml_model, "rag": None}


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
            return _assistant_tool_call_response("call_1", "fuzzy_logic_assessment", {"text": "I love parties!"})
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
    assert result["trace"][0]["tool"] == "fuzzy_logic_assessment"


def test_run_agent_stops_after_max_iterations_without_submit():
    def handler(request):
        return _assistant_tool_call_response("call_x", "ml_prior_assessment", {"text": "hi"})

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
    assert 0.0 <= result["continuous_score_estimate"] <= 99.0
    assert 1 <= result["tier"] <= 6


def test_run_agent_degrades_gracefully_on_malformed_response():
    def handler(request):
        return httpx.Response(200, json={"not_choices": "this response is malformed"})

    client = build_client("fake-key", transport=httpx.MockTransport(handler))
    ctx = _build_test_context()

    result = run_agent(client, ["fake-model"], ctx, "I love parties and talking to everyone!")

    assert result["degraded"] is True
    assert result["error"] is not None
    assert 0.0 <= result["continuous_score_estimate"] <= 99.0
    assert 1 <= result["tier"] <= 6


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


def test_degraded_result_survives_ml_prior_failure():
    from src.agent.orchestrator import _degraded_result

    broken_ctx = {"nlp": None, "nrc_dict": None, "ml_model": None}  # will make predict_ml_prior raise

    result = _degraded_result("some text", broken_ctx, "original error")

    assert result["degraded"] is True
    assert 0.0 <= result["continuous_score_estimate"] <= 99.0
    assert 1 <= result["tier"] <= 6

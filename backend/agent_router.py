"""The /predict-agent FastAPI route. Deliberately isolated from backend/main.py
(which is under active, unrelated revision) — this module owns its own
context/client construction and is mounted onto the real app with a single
`include_router` call, kept separate in backend/main.py itself.
"""
import os

import spacy
from dotenv import load_dotenv
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.agent.context import load_rag_context, train_ml_prior_from_data
from src.agent.openrouter_client import DEEPSEEK_BASE_URL, build_client
from src.agent.orchestrator import run_agent
from src.agent.tools.classical_features import load_nrc_lexicon

load_dotenv()

router = APIRouter()

_cached_ctx = None
_cached_client_and_models = None


def _build_deepseek_config(api_key):
    client = build_client(api_key, base_url=os.environ.get("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL))
    models_env = os.environ.get("DEEPSEEK_MODELS", "deepseek-v4-flash")
    models = [m.strip() for m in models_env.split(",") if m.strip()]
    extra_params = {"reasoning_effort": "high", "thinking": {"type": "enabled"}}
    return client, models, extra_params


def _build_openrouter_config(api_key):
    client = build_client(api_key)
    models_env = os.environ.get("OPENROUTER_MODELS", "")
    models = [m.strip() for m in models_env.split(",") if m.strip()]
    extra_params = {"reasoning": {"enabled": True}}
    return client, models, extra_params


def get_agent_context():
    global _cached_ctx
    if _cached_ctx is None:
        nlp = spacy.load("en_core_web_sm")
        nrc_dict = load_nrc_lexicon(os.path.join("data", "NRC-Emotion-Lexicon-Senselevel-v0.92.txt"))
        ml_model = train_ml_prior_from_data("data/train_clean.csv", nlp, nrc_dict)
        rag = load_rag_context("data/rag")
        _cached_ctx = {"nlp": nlp, "nrc_dict": nrc_dict, "ml_model": ml_model, "rag": rag}
    return _cached_ctx


def get_agent_client_and_models():
    """DeepSeek is the primary provider when DEEPSEEK_API_KEY is set (v4-flash,
    thinking mode enabled, high reasoning effort); falls back to whatever
    OPENROUTER_API_KEY/OPENROUTER_MODELS are configured otherwise.
    """
    global _cached_client_and_models
    if _cached_client_and_models is None:
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if deepseek_key:
            _cached_client_and_models = _build_deepseek_config(deepseek_key)
        else:
            _cached_client_and_models = _build_openrouter_config(os.environ.get("OPENROUTER_API_KEY", ""))
    return _cached_client_and_models


class PredictAgentRequest(BaseModel):
    text: str


@router.post("/predict-agent")
def predict_agent(
    req: PredictAgentRequest,
    ctx=Depends(get_agent_context),
    client_and_models=Depends(get_agent_client_and_models),
):
    if not req.text.strip():
        return {"error": "Empty text"}

    client, models, extra_params = client_and_models
    if not models:
        return {"error": "No models configured. Set DEEPSEEK_API_KEY (preferred) or OPENROUTER_API_KEY/OPENROUTER_MODELS."}

    return run_agent(client, models, ctx, req.text, extra_params=extra_params)

"""The /predict-agent FastAPI route. Deliberately isolated from backend/main.py
(which is under active, unrelated revision) — this module owns its own
context/client construction and is mounted onto the real app with a single
`include_router` call, kept separate in backend/main.py itself.
"""
import os

import spacy
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.agent.context import load_rag_context, train_ml_prior_from_data
from src.agent.openrouter_client import build_client
from src.agent.orchestrator import run_agent
from src.agent.tools.classical_features import load_nrc_lexicon

router = APIRouter()

_cached_ctx = None
_cached_client_and_models = None


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
    global _cached_client_and_models
    if _cached_client_and_models is None:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        client = build_client(api_key)
        models_env = os.environ.get("OPENROUTER_MODELS", "")
        models = [m.strip() for m in models_env.split(",") if m.strip()]
        _cached_client_and_models = (client, models)
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

    client, models = client_and_models
    if not models:
        return {"error": "No OPENROUTER_MODELS configured. Set the OPENROUTER_MODELS environment variable."}

    return run_agent(client, models, ctx, req.text)

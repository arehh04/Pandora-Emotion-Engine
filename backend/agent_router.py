"""The /predict-agent FastAPI route. Deliberately isolated from backend/main.py
-- the LLM agent pipeline runs alongside the classical-ML pipeline there, not
in place of it.
"""
import os

from dotenv import load_dotenv
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.agent.context import load_rag_context
from src.agent.openrouter_client import DEEPSEEK_BASE_URL, build_client
from src.agent.orchestrator import run_agent

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
    return client, models, {"reasoning": {"enabled": True}}


def get_agent_context():
    global _cached_ctx
    if _cached_ctx is None:
        rag = load_rag_context(os.path.join("data", "rag", "chroma"))
        _cached_ctx = {"rag": rag}
    return _cached_ctx


def get_agent_client_and_models():
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
def predict_agent(req: PredictAgentRequest, ctx=Depends(get_agent_context), client_and_models=Depends(get_agent_client_and_models)):
    if not req.text.strip():
        return {"error": "Empty text"}
    client, models, extra_params = client_and_models
    if not models:
        return {"error": "No models configured. Set DEEPSEEK_API_KEY (preferred) or OPENROUTER_API_KEY/OPENROUTER_MODELS."}
    return run_agent(client, models, ctx, req.text, extra_params=extra_params)

"""The /predict-eq FastAPI route: the EQ (Emotional Intelligence) multi-agent
LangGraph pipeline (src/eq_agent/), running alongside /predict-agent's
Extraversion pipeline and backend/main.py's classical-ML pipeline -- these
three run side by side, none replaces another.
"""
import os

from dotenv import load_dotenv
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.agent_router import _build_deepseek_config, _build_openrouter_config
from src.eq_agent.branch_config import build_branch_configs
from src.eq_agent.eq_context import load_eq_rag_context
from src.eq_agent.observability import configure_langsmith_tracing
from src.eq_agent.traced_assessment import traced_run_eq_assessment

load_dotenv()
configure_langsmith_tracing()

router = APIRouter()
_cached_ctx = None
_cached_client_and_models = None
_cached_branch_configs = None


def get_eq_context():
    global _cached_ctx
    if _cached_ctx is None:
        persist_dir = os.environ.get("EQ_RAG_PERSIST_DIR", os.path.join("data", "eq", "lancedb"))
        rag = load_eq_rag_context(persist_dir)
        _cached_ctx = {"eq_rag": rag}
    return _cached_ctx


def get_eq_client_and_models():
    global _cached_client_and_models
    if _cached_client_and_models is None:
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if deepseek_key:
            _cached_client_and_models = _build_deepseek_config(deepseek_key)
        else:
            _cached_client_and_models = _build_openrouter_config(os.environ.get("OPENROUTER_API_KEY", ""))
    return _cached_client_and_models


def get_eq_branch_configs():
    global _cached_branch_configs
    if _cached_branch_configs is None:
        _cached_branch_configs = build_branch_configs()
    return _cached_branch_configs


class PredictEQRequest(BaseModel):
    text: str


@router.post("/predict-eq")
def predict_eq(
    req: PredictEQRequest,
    ctx=Depends(get_eq_context),
    client_and_models=Depends(get_eq_client_and_models),
    branch_configs=Depends(get_eq_branch_configs),
):
    if not req.text.strip():
        return {"error": "Empty text"}
    client, models, extra_params = client_and_models
    if not models:
        return {"error": "No models configured. Set DEEPSEEK_API_KEY (preferred) or OPENROUTER_API_KEY/OPENROUTER_MODELS."}
    return traced_run_eq_assessment(client, models, ctx, req.text, branch_configs, extra_params=extra_params)

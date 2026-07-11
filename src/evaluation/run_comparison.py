"""Ablation/comparison harness: runs the LLM-only -> LLM+RAG variants over a
text sample and produces continuous + tiered metrics for each, alongside the
historical classical-ML baselines recorded in
docs/thesis/Chapter4_Results_Interpretability.md.

The real (non-mocked) run is a separate, manual, explicitly deferred script
invocation -- see the module docstring in scripts using make_run_agent_predict_fn.
Nothing in this module's automated tests makes a real LLM/API call.
"""
from src.agent.orchestrator import run_agent
from src.evaluation.faithfulness import check_rationale_faithfulness
from src.evaluation.metrics import compute_regression_metrics, compute_tier_metrics
from src.tiers import assign_tier

ABLATION_VARIANTS = {
    "llm_only": set(),
    "llm_rag": None,
}

# Sourced from docs/thesis/Chapter4_Results_Interpretability.md's existing
# results table -- not re-computed by this plan (see Global Constraints).
HISTORICAL_BASELINES = {
    "ridge": {"rmse": 29.18, "r2": 0.103},
    "xgboost": {"rmse": 28.29, "r2": 0.158},
    "random_forest": {"rmse": 28.22, "r2": 0.162},
}


def make_run_agent_predict_fn(client, models, ctx, extra_params=None, max_iterations=6):
    def predict_fn(text, enabled_tools):
        return run_agent(
            client, models, ctx, text,
            max_iterations=max_iterations, extra_params=extra_params, enabled_tools=enabled_tools,
        )
    return predict_fn


def run_evaluation(predict_fn, samples, variants):
    results = {}
    for variant_name, enabled_tools in variants.items():
        variant_results = []
        for text, true_score in samples:
            agent_result = predict_fn(text, enabled_tools)
            true_tier, _ = assign_tier(true_score)
            variant_results.append({
                "text": text,
                "true_score": true_score,
                "true_tier": true_tier,
                "predicted_score": agent_result["continuous_score_estimate"],
                "predicted_tier": agent_result["tier"],
                "faithful": check_rationale_faithfulness(agent_result),
                "degraded": agent_result["degraded"],
            })
        results[variant_name] = variant_results
    return results


def summarize_evaluation(results):
    summary = {}
    for variant_name, rows in results.items():
        true_scores = [r["true_score"] for r in rows]
        pred_scores = [r["predicted_score"] for r in rows]
        true_tiers = [r["true_tier"] for r in rows]
        pred_tiers = [r["predicted_tier"] for r in rows]

        reg_metrics = compute_regression_metrics(true_scores, pred_scores)
        tier_metrics = compute_tier_metrics(true_tiers, pred_tiers)
        faithfulness_rate = sum(1 for r in rows if r["faithful"]) / len(rows) if rows else 0.0

        summary[variant_name] = {**reg_metrics, **tier_metrics, "faithfulness_rate": faithfulness_rate}

    summary["_historical_baselines"] = HISTORICAL_BASELINES
    return summary

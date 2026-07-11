from src.evaluation.run_comparison import (
    ABLATION_VARIANTS,
    HISTORICAL_BASELINES,
    run_evaluation,
    summarize_evaluation,
)


def _fake_predict_fn(text, enabled_tools):
    # Deterministic: score derives from text length, tier from a simple mapping.
    score = min(99.0, len(text) * 2.0)
    if score <= 10:
        tier = 1
    elif score <= 25:
        tier = 2
    elif score <= 45:
        tier = 3
    elif score <= 65:
        tier = 4
    elif score <= 85:
        tier = 5
    else:
        tier = 6
    trace = [] if enabled_tools == set() else [
        {"tool": "retrieve_similar_exemplars", "arguments": {"text": text},
         "result": {"results": [{"bert_text": text, "score": 0.5}]}},
    ]
    return {
        "tier": tier, "tier_label": "x", "continuous_score_estimate": score,
        "confidence": "medium", "rationale": f"retrieve_similar_exemplars grounded this at tier {tier}.",
        "trace": trace, "degraded": False, "error": None,
    }


def test_ablation_variants_cover_the_two_expected_configurations():
    assert set(ABLATION_VARIANTS.keys()) == {"llm_only", "llm_rag"}
    assert ABLATION_VARIANTS["llm_only"] == set()
    assert ABLATION_VARIANTS["llm_rag"] is None


def test_historical_baselines_include_all_three_classical_models():
    assert set(HISTORICAL_BASELINES.keys()) == {"ridge", "xgboost", "random_forest"}
    for entry in HISTORICAL_BASELINES.values():
        assert "rmse" in entry and "r2" in entry


def test_run_evaluation_produces_one_row_per_sample_per_variant():
    samples = [("short", 5.0), ("a much longer piece of text here", 90.0)]
    variants = {"llm_only": set(), "llm_rag": None}

    results = run_evaluation(_fake_predict_fn, samples, variants)

    assert set(results.keys()) == {"llm_only", "llm_rag"}
    assert len(results["llm_only"]) == 2
    assert len(results["llm_rag"]) == 2
    assert results["llm_only"][0]["text"] == "short"
    assert results["llm_only"][0]["true_score"] == 5.0
    assert "predicted_tier" in results["llm_only"][0]
    assert "faithful" in results["llm_rag"][0]


def test_summarize_evaluation_includes_metrics_and_historical_baselines():
    samples = [("short", 5.0), ("a much longer piece of text here", 90.0)]
    variants = {"llm_only": set(), "llm_rag": None}
    results = run_evaluation(_fake_predict_fn, samples, variants)

    summary = summarize_evaluation(results)

    assert "llm_only" in summary
    assert "rmse" in summary["llm_only"]
    assert "accuracy" in summary["llm_only"]
    assert "faithfulness_rate" in summary["llm_only"]
    assert summary["_historical_baselines"] == HISTORICAL_BASELINES

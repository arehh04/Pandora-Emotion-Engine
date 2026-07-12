"""EQ evaluation harness: runs the EQ multi-agent pipeline over a text
sample and produces continuous + tiered metrics, compared against the
NRC-enriched proxy EQ label as the best available ground truth. Mirrors
src.evaluation.run_comparison's harness shape (predict_fn / run / summarize)
for the original Extraversion pipeline, but is a new, separate module --
see this plan's Global Constraints for why it does not modify that one.
"""
from src.eq_agent.traced_assessment import traced_run_eq_assessment
from src.eq_data.nrc_enrichment import compute_enriched_overall_eq_proxy
from src.eq_data.tiers_eq import assign_eq_tier
from src.evaluation.faithfulness import check_rationale_faithfulness
from src.evaluation.metrics import compute_regression_metrics, compute_tier_metrics


def make_run_eq_assessment_predict_fn(client, models, ctx, branch_configs, extra_params=None):
    def predict_fn(text):
        return traced_run_eq_assessment(client, models, ctx, text, branch_configs, extra_params=extra_params)
    return predict_fn


def run_eq_evaluation(predict_fn, samples, nrc_lexicon):
    rows = []
    for text, row in samples:
        eq_result = predict_fn(text)
        true_score = compute_enriched_overall_eq_proxy(row, text, nrc_lexicon)
        true_tier, _ = assign_eq_tier(true_score)
        branch_faithful = [
            check_rationale_faithfulness(branch_result)
            for branch_result in eq_result["branch_results"].values()
        ]
        rows.append({
            "text": text,
            "true_score": true_score,
            "true_tier": true_tier,
            "predicted_score": eq_result["score"],
            "predicted_tier": eq_result["tier"],
            "faithful": all(branch_faithful) if branch_faithful else True,
            "degraded": eq_result["degraded"],
            "degraded_branches": eq_result["degraded_branches"],
        })
    return rows


def summarize_eq_evaluation(rows):
    true_scores = [r["true_score"] for r in rows]
    pred_scores = [r["predicted_score"] for r in rows]
    true_tiers = [r["true_tier"] for r in rows]
    pred_tiers = [r["predicted_tier"] for r in rows]

    reg_metrics = compute_regression_metrics(true_scores, pred_scores)
    tier_metrics = compute_tier_metrics(true_tiers, pred_tiers)
    faithfulness_rate = sum(1 for r in rows if r["faithful"]) / len(rows) if rows else 0.0
    degraded_rate = sum(1 for r in rows if r["degraded"]) / len(rows) if rows else 0.0

    return {**reg_metrics, **tier_metrics, "faithfulness_rate": faithfulness_rate, "degraded_rate": degraded_rate}

"""Synthesizes the 4 MSC branch verdicts into one overall EQ assessment via
a single LLM tool-call (no ReAct loop -- the coordinator has nothing to
retrieve, only branch summaries to reason over).
"""
import json

from src.agent.openrouter_client import call_with_fallback
from src.eq_data.tiers_eq import assign_eq_tier

SUBMIT_OVERALL_ASSESSMENT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_overall_assessment",
        "description": "Submit the synthesized overall EQ assessment. Call this exactly once.",
        "parameters": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "description": "Overall EQ score, 0-99."},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                "rationale": {"type": "string", "description": "Brief explanation reconciling the 4 branch verdicts."},
            },
            "required": ["score", "confidence", "rationale"],
        },
    },
}

COORDINATOR_SYSTEM_PROMPT = (
    "You are the coordinator of a 4-specialist Emotional Intelligence assessment "
    "team, following the Mayer-Salovey-Caruso model: Perceiving, Using, "
    "Understanding, and Managing emotions. You will be given each specialist's "
    "branch score (0-99), confidence, and rationale. Synthesize these into one "
    "overall EQ score and rationale, reconciling any disagreement between "
    "branches rather than simply averaging them. Call submit_overall_assessment "
    "exactly once."
)


def _format_branch_summary(branch_results):
    lines = [
        f"- {branch}: score={r['score']}, confidence={r['confidence']}, rationale=\"{r['rationale']}\""
        for branch, r in branch_results.items()
    ]
    return "\n".join(lines)


def _degraded_overall_result(branch_results, error):
    scores = [r["score"] for r in branch_results.values()] or [50.0]
    score = sum(scores) / len(scores)
    tier, tier_label = assign_eq_tier(score)
    return {
        "score": score, "tier": tier, "tier_label": tier_label, "confidence": "low",
        "rationale": "Coordinator unavailable; falling back to a plain average of branch scores.",
        "degraded": True, "error": error,
    }


def run_coordinator(client, models, branch_results, extra_params=None):
    messages = [
        {"role": "system", "content": COORDINATOR_SYSTEM_PROMPT},
        {"role": "user", "content": f"Branch assessments:\n{_format_branch_summary(branch_results)}"},
    ]

    try:
        response = call_with_fallback(
            client, models, messages, tools=[SUBMIT_OVERALL_ASSESSMENT_SCHEMA], extra_params=extra_params
        )
        message = response["choices"][0]["message"]
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return _degraded_overall_result(branch_results, "Coordinator responded without calling a tool.")

        arguments = json.loads(tool_calls[0]["function"]["arguments"])
        score = min(99.0, max(0.0, float(arguments["score"])))
        tier, tier_label = assign_eq_tier(score)
        return {
            "score": score, "tier": tier, "tier_label": tier_label,
            "confidence": arguments["confidence"], "rationale": arguments["rationale"],
            "degraded": False, "error": None,
        }
    except (KeyError, IndexError, TypeError, ValueError) as e:
        return _degraded_overall_result(branch_results, f"Malformed coordinator response: {e}")
    except Exception as e:
        return _degraded_overall_result(branch_results, str(e))

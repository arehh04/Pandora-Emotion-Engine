"""Checks the coordinator's overall verdict and the 4 branch results for
internal inconsistency, via a single LLM tool-call. Fails open (treats the
input as consistent) if the LLM call fails -- the critique loop is an
enhancement, never a single point of failure for the whole assessment.
"""
import json

from src.agent.openrouter_client import call_with_fallback

SUBMIT_CRITIQUE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_critique",
        "description": "Submit your critique of the assessment's internal consistency. Call this exactly once.",
        "parameters": {
            "type": "object",
            "properties": {
                "consistent": {"type": "boolean", "description": "True if the branch verdicts and overall score are internally consistent."},
                "branches_to_recheck": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Branch names to re-assess, if not consistent. Empty if consistent.",
                },
                "reason": {"type": "string", "description": "Brief explanation of the inconsistency, if any."},
            },
            "required": ["consistent", "branches_to_recheck", "reason"],
        },
    },
}

CRITIC_SYSTEM_PROMPT = (
    "You are a critic reviewing an Emotional Intelligence assessment made up of "
    "4 branch scores (Perceiving, Using, Understanding, Managing) and one "
    "overall synthesized score. Check whether the branch verdicts are "
    "internally consistent with each other and with the overall score -- for "
    "example, a high Perceiving score paired with a contradictory low Managing "
    "rationale citing the same emotional signal. If everything is reasonably "
    "consistent, set consistent=true. Call submit_critique exactly once."
)

VALID_BRANCHES = {"perceiving", "using", "understanding", "managing"}


def _format_verdict_summary(branch_results, overall_verdict):
    lines = [f"- {branch}: score={r['score']}, rationale=\"{r['rationale']}\"" for branch, r in branch_results.items()]
    lines.append(f"Overall: score={overall_verdict['score']}, rationale=\"{overall_verdict['rationale']}\"")
    return "\n".join(lines)


def run_critic(client, models, branch_results, overall_verdict, extra_params=None):
    messages = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": _format_verdict_summary(branch_results, overall_verdict)},
    ]

    try:
        response = call_with_fallback(
            client, models, messages, tools=[SUBMIT_CRITIQUE_SCHEMA], extra_params=extra_params
        )
        message = response["choices"][0]["message"]
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return {"consistent": True, "branches_to_recheck": [], "reason": None}

        arguments = json.loads(tool_calls[0]["function"]["arguments"])
        branches_to_recheck = [b for b in arguments.get("branches_to_recheck", []) if b in VALID_BRANCHES]
        return {
            "consistent": bool(arguments["consistent"]) and not branches_to_recheck,
            "branches_to_recheck": branches_to_recheck,
            "reason": arguments.get("reason"),
        }
    except Exception:
        return {"consistent": True, "branches_to_recheck": [], "reason": None}

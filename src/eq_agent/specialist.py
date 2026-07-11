"""A generic, branch-parameterized ReAct tool-calling loop for one MSC
specialist agent. Mirrors src/agent/orchestrator.py's proven run_agent
pattern, but takes its tool schemas, dispatcher, and system prompt as
parameters instead of hardcoded module constants, so the same runner works
for all 4 MSC branches (perceiving/using/understanding/managing) -- Plan 3
supplies each branch's real RAG-grounded tool schemas and dispatcher.
"""
import json

from src.agent.openrouter_client import call_with_fallback
from src.eq_data.tiers_eq import assign_eq_tier

SUBMIT_BRANCH_ASSESSMENT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_branch_assessment",
        "description": "Submit the final assessment for this MSC branch. Call this exactly once, when you are done reasoning.",
        "parameters": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "description": "Estimated score for this branch, 0-99."},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                "rationale": {"type": "string", "description": "Brief explanation citing the tool evidence used."},
            },
            "required": ["score", "confidence", "rationale"],
        },
    },
}


def _degraded_branch_result(error):
    tier, tier_label = assign_eq_tier(50.0)
    return {
        "score": 50.0, "tier": tier, "tier_label": tier_label, "confidence": "low",
        "rationale": "This specialist could not complete its assessment; returning a neutral default.",
        "trace": [], "degraded": True, "error": error,
    }


def run_specialist(
    client, models, ctx, text, branch, tool_schemas, dispatch_fn, system_prompt,
    extra_params=None, max_iterations=6, critic_feedback=None,
):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Assess the {branch} branch of Emotional Intelligence for this text:\n\n{text}"},
    ]
    if critic_feedback:
        messages.append({
            "role": "user",
            "content": f"A critique of your prior assessment flagged this concern: {critic_feedback}. Please reconsider.",
        })

    available_schemas = list(tool_schemas) + [SUBMIT_BRANCH_ASSESSMENT_SCHEMA]
    trace = []

    for _ in range(max_iterations):
        try:
            response = call_with_fallback(client, models, messages, tools=available_schemas, extra_params=extra_params)
        except Exception as e:
            return _degraded_branch_result(str(e))

        try:
            message = response["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                return _degraded_branch_result(f"{branch} specialist responded without calling a tool.")

            messages.append(message)

            for tool_call in tool_calls:
                name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])

                if name == "submit_branch_assessment":
                    score = min(99.0, max(0.0, float(arguments["score"])))
                    tier, tier_label = assign_eq_tier(score)
                    return {
                        "score": score, "tier": tier, "tier_label": tier_label,
                        "confidence": arguments["confidence"], "rationale": arguments["rationale"],
                        "trace": trace, "degraded": False, "error": None,
                    }

                result = dispatch_fn(name, arguments, ctx)
                trace.append({"tool": name, "arguments": arguments, "result": result})
                messages.append({
                    "role": "tool", "tool_call_id": tool_call["id"], "content": json.dumps(result),
                })
        except (KeyError, IndexError, TypeError, ValueError) as e:
            return _degraded_branch_result(f"Malformed specialist response: {e}")

    return _degraded_branch_result(f"Max iterations ({max_iterations}) reached without submit_branch_assessment.")

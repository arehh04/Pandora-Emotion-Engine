"""The agent's ReAct-style orchestration loop: calls an LLM with tool-calling
enabled, dispatches whichever RAG tools it chooses, and terminates when it
calls submit_assessment. Degrades to a neutral default if the LLM call
fails, the response is malformed, or max_iterations is exhausted without a
submit_assessment call.
"""
import json

from src.agent.openrouter_client import call_with_fallback
from src.agent.tool_schemas import TOOL_SCHEMAS, dispatch_tool_call
from src.tiers import TIER_BINS

SYSTEM_PROMPT = (
    "You are an assessment agent estimating the Extraversion of a piece of text "
    "on a 1 (most reserved) to 6 (most extraverted) tier scale. You have tools "
    "available to gather evidence: retrieval of similar labeled examples and "
    "retrieval of relevant Extraversion/Big Five psychology theory from a "
    "knowledge base. Use as many tool calls as you find useful, then call "
    "submit_assessment exactly once with your final tier, a 0-99 continuous "
    "score estimate, your confidence, and a brief rationale citing the "
    "evidence you gathered."
)


def label_for_tier(tier_num):
    for _low, _high, num, label in TIER_BINS:
        if num == tier_num:
            return label
    raise ValueError(f"invalid tier {tier_num}")


def _degraded_result(text, ctx, error):
    return {
        "tier": 4,
        "tier_label": label_for_tier(4),
        "continuous_score_estimate": 50.0,
        "confidence": "low",
        "rationale": "The agent could not complete the assessment; returning a neutral default.",
        "trace": [],
        "degraded": True,
        "error": error,
    }


def run_agent(client, models, ctx, text, max_iterations=6, extra_params=None, enabled_tools=None):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Assess the Extraversion of this text:\n\n{text}"},
    ]
    trace = []

    if enabled_tools is None:
        available_schemas = TOOL_SCHEMAS
    else:
        allowed_names = set(enabled_tools) | {"submit_assessment"}
        available_schemas = [s for s in TOOL_SCHEMAS if s["function"]["name"] in allowed_names]

    for _ in range(max_iterations):
        try:
            response = call_with_fallback(client, models, messages, tools=available_schemas, extra_params=extra_params)
        except Exception as e:
            return _degraded_result(text, ctx, str(e))

        try:
            message = response["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                return _degraded_result(text, ctx, "Agent responded without calling a tool.")

            messages.append(message)

            for tool_call in tool_calls:
                name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])

                if name == "submit_assessment":
                    score = min(99.0, max(0.0, float(arguments["continuous_score_estimate"])))
                    return {
                        "tier": arguments["tier"],
                        "tier_label": label_for_tier(arguments["tier"]),
                        "continuous_score_estimate": score,
                        "confidence": arguments["confidence"],
                        "rationale": arguments["rationale"],
                        "trace": trace,
                        "degraded": False,
                        "error": None,
                    }

                if enabled_tools is not None and name != "submit_assessment" and name not in enabled_tools:
                    result = {"error": f"Tool '{name}' is disabled for this evaluation variant."}
                else:
                    result = dispatch_tool_call(name, arguments, ctx)
                trace.append({"tool": name, "arguments": arguments, "result": result})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(result),
                })
        except (KeyError, IndexError, TypeError, ValueError) as e:
            return _degraded_result(text, ctx, f"Malformed agent response: {e}")

    return _degraded_result(text, ctx, f"Max iterations ({max_iterations}) reached without submit_assessment.")

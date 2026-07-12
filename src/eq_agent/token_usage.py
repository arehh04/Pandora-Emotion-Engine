"""Extracts token usage from an OpenAI-compatible chat completion response
(the shape src.agent.openrouter_client.call_with_fallback returns), for
future cost/token observability surfacing. Standalone utility, not yet
wired into any live code path -- a later plan decides where to surface it
(e.g. the frontend Agent Trace panel or LangSmith run metadata).
"""


def extract_token_usage(response):
    usage = response.get("usage")
    if not usage:
        return None
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }

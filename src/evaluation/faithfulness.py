"""A heuristic (not a proof) for whether an agent's rationale references
concrete evidence from its own tool-call trace, rather than being generic
or unsupported. Flagged results (returns False) are candidates for manual
spot-checking, per the design spec's rationale-faithfulness requirement —
this function does not replace that manual review, it triages it.
"""


def check_rationale_faithfulness(agent_result):
    rationale = (agent_result.get("rationale") or "").lower()
    trace = agent_result.get("trace") or []

    if not trace:
        return True  # nothing to be unfaithful to (e.g. the llm-only ablation variant)

    has_usable_step = False
    for step in trace:
        result = step.get("result") or {}
        if "error" in result:
            continue

        has_usable_step = True
        for key in ("tier", "score", "fuzzy_score"):
            if key in result and str(result[key]) in rationale:
                return True

        tool_words = step["tool"].replace("_", " ").split()
        if any(word in rationale for word in tool_words if len(word) > 4):
            return True

    if not has_usable_step:
        return True  # nothing to be unfaithful to (all tools errored)

    return False

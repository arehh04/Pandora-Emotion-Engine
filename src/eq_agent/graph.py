"""Assembles the EQ multi-agent LangGraph: 4 parallel MSC specialists ->
coordinator -> critic, with a critic-driven re-assessment loop capped at
max_loop_rounds. LangGraph's synchronous StateGraph.invoke() runs
same-superstep nodes concurrently by default (verified: 4 nodes each
sleeping 0.5s completed in ~0.51s wall-clock, not ~2.0s) -- no manual
threading is needed for the parallel fan-out.
"""
from typing import Annotated, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from src.eq_agent.coordinator import run_coordinator
from src.eq_agent.critic import run_critic
from src.eq_agent.specialist import run_specialist
from src.eq_data.tiers_eq import assign_eq_tier

BRANCHES = ["perceiving", "using", "understanding", "managing"]


def _merge_branch_results(existing, new):
    return {**existing, **new}


class EQAssessmentState(TypedDict):
    text: str
    branch_results: Annotated[dict, _merge_branch_results]
    overall_verdict: Optional[dict]
    critic_result: Optional[dict]
    loop_count: int


def _make_specialist_node(branch, client, models, ctx, config, extra_params):
    def node(state):
        critic_result = state.get("critic_result")
        feedback = None
        if critic_result and branch in (critic_result.get("branches_to_recheck") or []):
            feedback = critic_result.get("reason")
        result = run_specialist(
            client, models, ctx, state["text"], branch,
            config["tool_schemas"], config["dispatch_fn"], config["system_prompt"],
            extra_params=extra_params, critic_feedback=feedback,
        )
        return {"branch_results": {branch: result}}
    return node


def _make_coordinator_node(client, models, extra_params):
    def node(state):
        return {"overall_verdict": run_coordinator(client, models, state["branch_results"], extra_params=extra_params)}
    return node


def _make_critic_node(client, models, extra_params, max_loop_rounds):
    def node(state):
        result = run_critic(client, models, state["branch_results"], state["overall_verdict"], extra_params=extra_params)
        if result["branches_to_recheck"] and state["loop_count"] >= max_loop_rounds:
            result = {**result, "consistent": True, "branches_to_recheck": []}
        next_loop_count = state["loop_count"] + 1 if result["branches_to_recheck"] else state["loop_count"]
        return {"critic_result": result, "loop_count": next_loop_count}
    return node


def _route_after_critic(state):
    to_recheck = state["critic_result"]["branches_to_recheck"]
    return to_recheck if to_recheck else ["__end__"]


def build_eq_graph(client, models, ctx, branch_configs, extra_params=None, max_loop_rounds=2):
    builder = StateGraph(EQAssessmentState)

    for branch in BRANCHES:
        builder.add_node(branch, _make_specialist_node(branch, client, models, ctx, branch_configs[branch], extra_params))
        builder.add_edge(START, branch)
        builder.add_edge(branch, "coordinator")

    builder.add_node("coordinator", _make_coordinator_node(client, models, extra_params))
    builder.add_node("critic", _make_critic_node(client, models, extra_params, max_loop_rounds))
    builder.add_edge("coordinator", "critic")
    builder.add_conditional_edges(
        "critic", _route_after_critic, {branch: branch for branch in BRANCHES} | {"__end__": END}
    )

    return builder.compile()


def run_eq_assessment(client, models, ctx, text, branch_configs, extra_params=None, max_loop_rounds=2):
    graph = build_eq_graph(client, models, ctx, branch_configs, extra_params=extra_params, max_loop_rounds=max_loop_rounds)

    initial_state = {
        "text": text, "branch_results": {}, "overall_verdict": None,
        "critic_result": None, "loop_count": 0,
    }

    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        tier, tier_label = assign_eq_tier(50.0)
        return {
            "score": 50.0, "tier": tier, "tier_label": tier_label,
            "confidence": "low", "rationale": "The assessment graph failed to complete; returning a neutral default.",
            "branch_results": {}, "degraded_branches": list(BRANCHES), "loop_count": 0,
            "degraded": True, "error": str(e),
        }

    overall = final_state["overall_verdict"]
    branch_results = final_state["branch_results"]
    degraded_branches = [b for b, r in branch_results.items() if r.get("degraded")]

    return {
        "score": overall["score"], "tier": overall["tier"], "tier_label": overall["tier_label"],
        "confidence": overall["confidence"], "rationale": overall["rationale"],
        "branch_results": branch_results, "degraded_branches": degraded_branches,
        "loop_count": final_state["loop_count"], "degraded": overall.get("degraded", False),
        "error": overall.get("error"),
    }

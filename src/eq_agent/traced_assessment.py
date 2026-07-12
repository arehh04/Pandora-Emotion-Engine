"""Traced entry point for the EQ multi-agent assessment pipeline, wrapping
src.eq_agent.graph.run_eq_assessment (Plan 2, unmodified) with LangSmith's
@traceable decorator. Verified directly against the installed
langsmith==0.10.2: @traceable safely no-ops (returns the wrapped function's
normal result, no network call attempt) when LANGSMITH_TRACING isn't set --
safe to apply unconditionally regardless of whether real tracing is
configured for this process.
"""
from langsmith import traceable

from src.eq_agent.graph import run_eq_assessment


@traceable(name="eq_assessment", run_type="chain")
def traced_run_eq_assessment(client, models, ctx, text, branch_configs, extra_params=None, max_loop_rounds=2):
    return run_eq_assessment(
        client, models, ctx, text, branch_configs,
        extra_params=extra_params, max_loop_rounds=max_loop_rounds,
    )

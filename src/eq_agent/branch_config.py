"""Assembles the real per-branch tool schemas, exception-safe dispatcher,
and system prompt that src.eq_agent.graph.run_eq_assessment's branch_configs
parameter expects -- grounding each MSC specialist in its own branch-scoped
LanceDB hybrid retrieval (src.eq_data.build_eq_corpus's theory/exemplar
tables), with optional cross-encoder reranking.

The dispatcher wraps its entire body in try/except, honoring the
exception-safety expectation src.eq_agent.specialist.run_specialist relies
on (matching the existing src.agent.tool_schemas.dispatch_tool_call
convention).
"""
from src.eq_agent.eq_rag_retrieval import retrieve_relevant_msc_theory, retrieve_similar_eq_exemplars
from src.eq_agent.graph import BRANCHES

RETRIEVE_EXEMPLARS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "retrieve_similar_eq_exemplars",
        "description": "Retrieve similar labeled example texts for this MSC branch, for calibration.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to find similar examples for."},
                "k": {"type": "integer", "description": "How many examples to retrieve.", "default": 5},
            },
            "required": ["text"],
        },
    },
}

RETRIEVE_THEORY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "retrieve_relevant_msc_theory",
        "description": "Retrieve relevant Mayer-Salovey-Caruso emotional-intelligence theory for this branch.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to find relevant theory for."},
                "k": {"type": "integer", "description": "How many theory chunks to retrieve.", "default": 3},
            },
            "required": ["text"],
        },
    },
}

BRANCH_DESCRIPTIONS = {
    "perceiving": "identifying and labeling emotions expressed or implied in text",
    "using": "using emotion to guide thought, decisions, and creativity",
    "understanding": "comprehending the causes, blends, and transitions of emotions",
    "managing": "regulating one's own emotions and, where relevant, helping manage others' emotions",
}


def _make_system_prompt(branch):
    return (
        f"You are an Emotional Intelligence specialist assessing the '{branch}' branch of the "
        f"Mayer-Salovey-Caruso model: {BRANCH_DESCRIPTIONS[branch]}. Use retrieve_similar_eq_exemplars "
        f"and retrieve_relevant_msc_theory as needed to ground your assessment, then call "
        f"submit_branch_assessment exactly once with your score (0-99), confidence, and a rationale "
        f"citing the evidence you gathered."
    )


def _make_dispatch_fn(branch):
    def dispatch_fn(name, arguments, ctx):
        try:
            rag_ctx = ctx.get("eq_rag")
            if not rag_ctx:
                return {"error": "EQ RAG corpus is not available (not built yet)."}

            if name == "retrieve_similar_eq_exemplars":
                k = arguments.get("k", 5)
                return {"results": retrieve_similar_eq_exemplars(arguments["text"], branch, rag_ctx, k=k)}

            if name == "retrieve_relevant_msc_theory":
                k = arguments.get("k", 3)
                return {"results": retrieve_relevant_msc_theory(arguments["text"], branch, rag_ctx, k=k)}

            return {"error": f"Unknown tool: {name}"}
        except Exception as e:
            return {"error": str(e)}
    return dispatch_fn


def build_branch_configs():
    return {
        branch: {
            "tool_schemas": [RETRIEVE_EXEMPLARS_SCHEMA, RETRIEVE_THEORY_SCHEMA],
            "dispatch_fn": _make_dispatch_fn(branch),
            "system_prompt": _make_system_prompt(branch),
        }
        for branch in BRANCHES
    }

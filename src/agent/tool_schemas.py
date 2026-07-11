"""OpenAI-style function-calling tool schemas for the agent, and a resilient
dispatcher that routes tool calls to the RAG retrieval functions.

submit_assessment is intentionally NOT dispatched here -- it's the agent's
terminal action, handled directly by the orchestrator loop.
"""
from src.agent.tools.rag_retrieval import retrieve_relevant_theory, retrieve_similar_exemplars

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_similar_exemplars",
            "description": "Retrieve the most similar labeled example texts from the training corpus, for calibration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to find similar examples for."},
                    "k": {"type": "integer", "description": "How many examples to retrieve.", "default": 5},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_relevant_theory",
            "description": "Retrieve relevant Extraversion/Big Five psychology theory chunks to ground the reasoning.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to find relevant theory for."},
                    "k": {"type": "integer", "description": "How many theory chunks to retrieve.", "default": 3},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_assessment",
            "description": "Submit the final Extraversion assessment. Call this exactly once, when you are done reasoning.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tier": {"type": "integer", "description": "Extraversion tier, 1 (most reserved) to 6 (most extraverted)."},
                    "continuous_score_estimate": {"type": "number", "description": "Estimated score, 0-99."},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                    "rationale": {"type": "string", "description": "Brief explanation citing the tool evidence used."},
                },
                "required": ["tier", "continuous_score_estimate", "confidence", "rationale"],
            },
        },
    },
]


def dispatch_tool_call(name, arguments, ctx):
    try:
        if name == "retrieve_similar_exemplars":
            if not ctx.get("rag"):
                return {"error": "RAG corpus is not available (not built yet)."}
            k = arguments.get("k", 5)
            return {"results": retrieve_similar_exemplars(arguments["text"], ctx["rag"], k=k)}

        if name == "retrieve_relevant_theory":
            if not ctx.get("rag"):
                return {"error": "RAG corpus is not available (not built yet)."}
            k = arguments.get("k", 3)
            return {"results": retrieve_relevant_theory(arguments["text"], ctx["rag"], k=k)}

        return {"error": f"Unknown tool: {name}"}
    except Exception as e:
        return {"error": str(e)}

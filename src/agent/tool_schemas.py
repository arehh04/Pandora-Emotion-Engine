"""OpenAI-style function-calling tool schemas for the agent, and a resilient
dispatcher that routes tool calls to Plan 2's underlying tool functions.

submit_assessment is intentionally NOT dispatched here — it's the agent's
terminal action, handled directly by the orchestrator loop (Task 4).
"""
from src.agent.tools.classical_features import extract_features_for_text
from src.agent.tools.fuzzy_engine import run_fuzzy_inference
from src.agent.tools.ml_prior import predict_ml_prior
from src.agent.tools.rag_retrieval import retrieve_relevant_theory, retrieve_similar_exemplars

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "fuzzy_logic_assessment",
            "description": (
                "Assess Extraversion using a hand-rolled fuzzy logic engine over "
                "linguistic/emotional signals extracted from the text. Returns a "
                "continuous score, a tier, and which fuzzy rules fired (for explainability)."
            ),
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "The text to assess."}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ml_prior_assessment",
            "description": "Assess Extraversion using a small trained Ridge regression model.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "The text to assess."}},
                "required": ["text"],
            },
        },
    },
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
        if name == "fuzzy_logic_assessment":
            features = extract_features_for_text(arguments["text"], ctx["nlp"], ctx["nrc_dict"])
            return run_fuzzy_inference(features)

        if name == "ml_prior_assessment":
            return predict_ml_prior(arguments["text"], ctx["nlp"], ctx["nrc_dict"], ctx["ml_model"])

        if name == "retrieve_similar_exemplars":
            if not ctx.get("rag"):
                return {"error": "RAG corpus is not available (not built yet)."}
            k = arguments.get("k", 5)
            results = retrieve_similar_exemplars(arguments["text"], ctx["rag"]["corpus"], ctx["rag"]["embedder"], k=k)
            return {"results": results}

        if name == "retrieve_relevant_theory":
            if not ctx.get("rag"):
                return {"error": "RAG corpus is not available (not built yet)."}
            k = arguments.get("k", 3)
            results = retrieve_relevant_theory(arguments["text"], ctx["rag"]["corpus"], ctx["rag"]["embedder"], k=k)
            return {"results": results}

        return {"error": f"Unknown tool: {name}"}
    except Exception as e:
        return {"error": str(e)}

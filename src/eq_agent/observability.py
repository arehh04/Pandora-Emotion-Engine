"""Configures LangSmith run tracing for the EQ multi-agent pipeline via
environment variables -- the mechanism LangSmith's SDK uses to decide
whether @traceable-wrapped calls actually report anywhere. Safe to call
even without a real API key: returns False and leaves tracing disabled
rather than raising. Uses the newer LANGSMITH_* env var names (verified
these are recognized by the installed langsmith==0.10.2, alongside the
older LANGCHAIN_* equivalents kept for backward compatibility elsewhere).
"""
import os

DEFAULT_LANGSMITH_PROJECT = "pandora-eq"


def configure_langsmith_tracing(api_key=None, project=None, endpoint=None):
    api_key = api_key or os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = project or os.environ.get("LANGSMITH_PROJECT", DEFAULT_LANGSMITH_PROJECT)
    if endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = endpoint

    return True

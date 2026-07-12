import os

from src.eq_agent.observability import DEFAULT_LANGSMITH_PROJECT, configure_langsmith_tracing

_LANGSMITH_ENV_KEYS = ["LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT"]


def _clear_langsmith_env():
    for key in _LANGSMITH_ENV_KEYS:
        os.environ.pop(key, None)


def test_configure_langsmith_tracing_returns_false_when_no_api_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    result = configure_langsmith_tracing()

    assert result is False


def test_configure_langsmith_tracing_sets_env_vars_when_api_key_given():
    _clear_langsmith_env()
    try:
        result = configure_langsmith_tracing(api_key="fake-key", project="test-project")

        assert result is True
        assert os.environ["LANGSMITH_TRACING"] == "true"
        assert os.environ["LANGSMITH_API_KEY"] == "fake-key"
        assert os.environ["LANGSMITH_PROJECT"] == "test-project"
    finally:
        _clear_langsmith_env()


def test_configure_langsmith_tracing_defaults_project_name():
    _clear_langsmith_env()
    try:
        configure_langsmith_tracing(api_key="fake-key")

        assert os.environ["LANGSMITH_PROJECT"] == DEFAULT_LANGSMITH_PROJECT
    finally:
        _clear_langsmith_env()


def test_configure_langsmith_tracing_sets_endpoint_only_when_given():
    _clear_langsmith_env()
    try:
        configure_langsmith_tracing(api_key="fake-key")
        assert "LANGSMITH_ENDPOINT" not in os.environ

        configure_langsmith_tracing(api_key="fake-key", endpoint="https://example-langsmith.test")
        assert os.environ["LANGSMITH_ENDPOINT"] == "https://example-langsmith.test"
    finally:
        _clear_langsmith_env()

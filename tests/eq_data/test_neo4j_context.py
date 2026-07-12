from src.eq_data.neo4j_context import load_neo4j_context


def test_load_neo4j_context_returns_none_when_password_not_configured(monkeypatch):
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

    result = load_neo4j_context()

    assert result is None


def test_load_neo4j_context_returns_none_when_unreachable():
    # bolt://127.0.0.1:9999 is intentionally unreachable -- verified this
    # raises neo4j.exceptions.ServiceUnavailable in ~2s, no server needed.
    result = load_neo4j_context(uri="bolt://127.0.0.1:9999", password="irrelevant")

    assert result is None


def test_load_neo4j_context_never_calls_the_driver_when_password_missing(monkeypatch):
    # A spy, not a return-value check: both the correct short-circuit and a
    # buggy fall-through-then-fail-anyway path return None identically (the
    # unreachable/no-password branches both degrade to None by design), so
    # asserting on the return value alone can't tell them apart. Patching
    # GraphDatabase.driver to raise if it's ever invoked makes the two
    # branches observably different: the short-circuit never reaches it.
    import neo4j

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("GraphDatabase.driver should not be called when no password is configured")

    monkeypatch.setattr(neo4j.GraphDatabase, "driver", classmethod(lambda cls, *a, **kw: _fail_if_called()))

    result = load_neo4j_context(password="")  # "" must be caught the same as None

    assert result is None

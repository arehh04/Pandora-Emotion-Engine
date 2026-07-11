"""Shared test-only fake Neo4j driver/session for this plan's tests --
records every session.run() call (cypher text + params) and returns
pre-configured canned results in call order, avoiding any real Neo4j
server dependency in the automated test suite.
"""


class FakeNeo4jSession:
    def __init__(self, run_results=None):
        self.calls = []
        self._run_results = run_results or []
        self._call_index = 0

    def run(self, cypher, **params):
        self.calls.append({"cypher": cypher, "params": params})
        result = self._run_results[self._call_index] if self._call_index < len(self._run_results) else []
        self._call_index += 1
        return result

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeNeo4jDriver:
    def __init__(self, session):
        self._session = session

    def session(self):
        return self._session

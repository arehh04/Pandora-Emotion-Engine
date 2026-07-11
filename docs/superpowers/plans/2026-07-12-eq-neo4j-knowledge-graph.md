# MSC Knowledge Graph (Neo4j) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Neo4j knowledge graph capturing the 4 MSC branches, their theory concepts (from Plan 1's 16-entry corpus), and provisional cross-branch dependency relationships — with graceful degradation when Neo4j is unreachable, exactly mirroring `backend/main.py`'s existing Redis-optional pattern.

**Architecture:** Unlike ChromaDB/LanceDB (embedded, no server process), Neo4j is real standalone infrastructure. A context loader (`src/eq_data/neo4j_context.py`) attempts a connection and returns `None` on any failure — the same "returns `None` if unavailable" contract used by every other RAG context loader in this project. A graph builder (`src/eq_data/msc_knowledge_graph.py`) populates `Branch` and `Concept` nodes with `BELONGS_TO`/`DEPENDS_ON` relationships, plus query functions the coordinator/critic (or a future evaluation step) can call. This plan builds the graph capability; wiring it into the live agent's prompts or tools is a deferred integration decision, matching Plan 4's "available, not adopted yet" precedent for the NRC enrichment.

**Tech Stack:** `neo4j` 6.2.0 (official Python driver, verified installed and API-tested against a live `neo4j:5-community` Docker container before writing this plan — the container was removed after verification, not left running).

## Global Constraints

- **Verified real facts about the `neo4j` driver** (checked directly against a live `neo4j:5-community` container, not assumed): `GraphDatabase.driver(uri, auth=(user, password))` + `.verify_connectivity()` establishes/validates a connection; `driver.session()` is a context manager; `session.run(cypher, **params)` executes parameterized Cypher and returns an iterable of `Record` objects supporting `record["key"]` subscript access (a plain Python dict is a valid test stand-in for this exact access pattern). `UNWIND $list AS item CREATE (...)` batch-creates nodes from a parameter list. A failed connection (wrong port/unreachable host) raises `neo4j.exceptions.ServiceUnavailable` — confirmed to raise in ~2 seconds (not a slow multi-retry hang), safe to exercise directly in an automated test against an intentionally unreachable local port (`bolt://127.0.0.1:9999`), no mocking required for that specific failure path. Wrong credentials against a real server raise `neo4j.exceptions.AuthError`.
- **No Docker/Neo4j server is required to run this plan's automated test suite.** Tests either (a) exercise the real, fast-failing "unreachable" path directly (no server needed), or (b) inject a fake driver/session pair (`tests/eq_data/neo4j_test_helpers.py`) recording calls and returning canned plain-dict results — the same dependency-injection convention used throughout this project (`FakeEmbedder`, `httpx.MockTransport`, `make_fake_embedding_func`). The "successfully connects and queries a real server" path is manually verified only (like Plan 1's external dataset fetchers), not part of the automated suite.
- Connection config follows this project's existing `.env`-based convention: `NEO4J_URI` (default `bolt://127.0.0.1:7687`), `NEO4J_USER` (default `neo4j`), `NEO4J_PASSWORD` (no default — if unset, `load_neo4j_context` returns `None` immediately without attempting a connection, avoiding a pointless network call).
- Cross-branch `DEPENDS_ON` relationships reflect the MSC model's commonly-described hierarchy (Perceiving is foundational; Using and Understanding build on it; Managing builds on Understanding) — **provisional, pending literature citation**, same `citation_needed` convention used throughout `src/eq_data/`.
- Purely additive: no file under `src/eq_agent/`, `src/agent/`, `src/rag/`, `backend/`, or `frontend/` is touched. `src/eq_data/` gets new files, no `__init__.py` (implicit namespace packages, established convention).
- New dependency: `neo4j==6.2.0` (add to `requirements.txt`).

---

### Task 1: Neo4j context loader with graceful degradation

**Files:**
- Create: `src/eq_data/neo4j_context.py`
- Modify: `requirements.txt` (add `neo4j==6.2.0`)
- Test: `tests/eq_data/test_neo4j_context.py`

**Interfaces:**
- Produces: `DEFAULT_NEO4J_URI = "bolt://127.0.0.1:7687"`, `DEFAULT_NEO4J_USER = "neo4j"`, `load_neo4j_context(uri=None, user=None, password=None) -> driver | None` — consumed by Task 2/3's functions (which take a `driver` parameter) and a later Backend Integration plan.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_data/test_neo4j_context.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eq_data.neo4j_context'`

- [ ] **Step 3: Add the dependency to requirements.txt**

Add this line at the end of `requirements.txt`:

```
neo4j==6.2.0
```

- [ ] **Step 4: Write the implementation**

```python
"""Loads a Neo4j driver connection for the MSC knowledge graph, with
graceful degradation (returns None) if Neo4j is unreachable, misconfigured,
or no password is set -- mirroring backend/main.py's existing Redis-optional
pattern. Verified real exception types (against a live neo4j:5-community
Docker container, since removed): a failed connection raises
neo4j.exceptions.ServiceUnavailable; bad credentials raise
neo4j.exceptions.AuthError.
"""
import os

DEFAULT_NEO4J_URI = "bolt://127.0.0.1:7687"
DEFAULT_NEO4J_USER = "neo4j"


def load_neo4j_context(uri=None, user=None, password=None):
    uri = uri or os.environ.get("NEO4J_URI", DEFAULT_NEO4J_URI)
    user = user or os.environ.get("NEO4J_USER", DEFAULT_NEO4J_USER)
    password = password or os.environ.get("NEO4J_PASSWORD")

    if not password:
        return None

    from neo4j import GraphDatabase
    from neo4j.exceptions import AuthError, ServiceUnavailable

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        return driver
    except (ServiceUnavailable, AuthError):
        return None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_data/test_neo4j_context.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add src/eq_data/neo4j_context.py requirements.txt tests/eq_data/test_neo4j_context.py
git commit -m "feat: add Neo4j context loader with graceful degradation"
```

---

### Task 2: MSC knowledge graph builder

**Files:**
- Create: `src/eq_data/msc_knowledge_graph.py`
- Create: `tests/eq_data/neo4j_test_helpers.py` (shared fake driver/session for this plan's tests)
- Test: `tests/eq_data/test_msc_knowledge_graph.py` (builder tests; query function tests added in Task 3)

**Interfaces:**
- Consumes: theory entries shaped like `src/eq_data/msc_theory_corpus.py::load_msc_theory_corpus`'s output (`{"id", "branch", "topic", "text", "citation_needed"}`, Plan 1, unmodified).
- Produces: `BRANCHES = ["perceiving", "using", "understanding", "managing"]`, `DEFAULT_BRANCH_DEPENDENCIES` (list of `(dependent, dependency)` tuples), `build_msc_knowledge_graph(driver, theory_entries, branch_dependencies=None)` — consumed by a later plan's real corpus-build step (analogous to `python -m src.eq_data.build_eq_corpus`).

- [ ] **Step 1: Write the shared fake driver/session test helper**

```python
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
```

- [ ] **Step 2: Write the failing tests**

```python
from src.eq_data.msc_knowledge_graph import BRANCHES, build_msc_knowledge_graph
from tests.eq_data.neo4j_test_helpers import FakeNeo4jDriver, FakeNeo4jSession


def test_build_msc_knowledge_graph_issues_expected_calls_in_order():
    session = FakeNeo4jSession()
    driver = FakeNeo4jDriver(session)
    theory_entries = [
        {"id": "p1", "branch": "perceiving", "topic": "t1", "text": "x1", "citation_needed": "yes"},
        {"id": "m1", "branch": "managing", "topic": "t2", "text": "x2", "citation_needed": "yes"},
    ]

    build_msc_knowledge_graph(driver, theory_entries, branch_dependencies=[("using", "perceiving")])

    # 1 delete-all + 1 branch-batch + 2 concept calls + 1 dependency call = 5 calls
    assert len(session.calls) == 5
    assert "DETACH DELETE" in session.calls[0]["cypher"]
    assert session.calls[1]["params"]["branches"] == BRANCHES

    concept_calls = session.calls[2:4]
    assert concept_calls[0]["params"]["id"] == "p1"
    assert concept_calls[0]["params"]["branch"] == "perceiving"
    assert concept_calls[0]["params"]["topic"] == "t1"
    assert concept_calls[1]["params"]["id"] == "m1"
    assert concept_calls[1]["params"]["branch"] == "managing"

    dependency_call = session.calls[4]
    assert dependency_call["params"] == {"dependent": "using", "dependency": "perceiving"}


def test_build_msc_knowledge_graph_uses_default_dependencies_when_none_given():
    session = FakeNeo4jSession()
    driver = FakeNeo4jDriver(session)

    build_msc_knowledge_graph(driver, theory_entries=[])

    # 1 delete-all + 1 branch-batch + 0 concept calls + len(DEFAULT_BRANCH_DEPENDENCIES) calls
    from src.eq_data.msc_knowledge_graph import DEFAULT_BRANCH_DEPENDENCIES
    assert len(session.calls) == 2 + len(DEFAULT_BRANCH_DEPENDENCIES)


def test_default_branch_dependencies_only_reference_real_branches():
    from src.eq_data.msc_knowledge_graph import DEFAULT_BRANCH_DEPENDENCIES

    for dependent, dependency in DEFAULT_BRANCH_DEPENDENCIES:
        assert dependent in BRANCHES
        assert dependency in BRANCHES
        assert dependent != dependency
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_data/test_msc_knowledge_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eq_data.msc_knowledge_graph'`

- [ ] **Step 4: Write the implementation**

```python
"""Builds the MSC knowledge graph in Neo4j: 4 Branch nodes, Concept nodes
from Plan 1's theory corpus (BELONGS_TO their branch), and provisional
cross-branch DEPENDS_ON relationships reflecting the MSC model's commonly
described hierarchy -- pending literature citation, same citation_needed
convention used throughout src.eq_data.
"""

BRANCHES = ["perceiving", "using", "understanding", "managing"]

DEFAULT_BRANCH_DEPENDENCIES = [
    ("using", "perceiving"),
    ("understanding", "perceiving"),
    ("managing", "understanding"),
]


def build_msc_knowledge_graph(driver, theory_entries, branch_dependencies=None):
    if branch_dependencies is None:
        branch_dependencies = DEFAULT_BRANCH_DEPENDENCIES

    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        session.run("UNWIND $branches AS name CREATE (:Branch {name: name})", branches=BRANCHES)

        for entry in theory_entries:
            session.run(
                "MATCH (b:Branch {name: $branch}) "
                "CREATE (c:Concept {id: $id, topic: $topic, text: $text})-[:BELONGS_TO]->(b)",
                branch=entry["branch"], id=entry["id"], topic=entry["topic"], text=entry["text"],
            )

        for dependent, dependency in branch_dependencies:
            session.run(
                "MATCH (a:Branch {name: $dependent}), (b:Branch {name: $dependency}) "
                "CREATE (a)-[:DEPENDS_ON]->(b)",
                dependent=dependent, dependency=dependency,
            )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_data/test_msc_knowledge_graph.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add src/eq_data/msc_knowledge_graph.py tests/eq_data/neo4j_test_helpers.py tests/eq_data/test_msc_knowledge_graph.py
git commit -m "feat: add MSC knowledge graph builder (branches, concepts, cross-branch dependencies)"
```

---

### Task 3: Query functions

**Files:**
- Modify: `src/eq_data/msc_knowledge_graph.py` (add query functions to the same module)
- Modify: `tests/eq_data/test_msc_knowledge_graph.py` (add query-function tests)

**Interfaces:**
- Consumes: `FakeNeo4jDriver`/`FakeNeo4jSession` (Task 2, test-only).
- Produces: `get_concepts_for_branch(driver, branch) -> list[dict]` (each `{"id", "topic", "text"}`); `get_branch_dependencies(driver, branch) -> list[str]` (branch names this branch depends on) — consumed by a later plan's decision on whether/how to wire graph grounding into the live agent.

- [ ] **Step 1: Write the failing tests**

```python
from src.eq_data.msc_knowledge_graph import get_branch_dependencies, get_concepts_for_branch
from tests.eq_data.neo4j_test_helpers import FakeNeo4jDriver, FakeNeo4jSession


def test_get_concepts_for_branch_transforms_records_to_dicts():
    fake_records = [{"id": "p1", "topic": "t1", "text": "x1"}, {"id": "p2", "topic": "t2", "text": "x2"}]
    session = FakeNeo4jSession(run_results=[fake_records])
    driver = FakeNeo4jDriver(session)

    result = get_concepts_for_branch(driver, "perceiving")

    assert result == fake_records
    assert session.calls[0]["params"]["branch"] == "perceiving"


def test_get_concepts_for_branch_returns_empty_list_for_a_branch_with_no_concepts():
    session = FakeNeo4jSession(run_results=[[]])
    driver = FakeNeo4jDriver(session)

    result = get_concepts_for_branch(driver, "using")

    assert result == []


def test_get_branch_dependencies_transforms_records_to_names():
    fake_records = [{"name": "perceiving"}]
    session = FakeNeo4jSession(run_results=[fake_records])
    driver = FakeNeo4jDriver(session)

    result = get_branch_dependencies(driver, "using")

    assert result == ["perceiving"]


def test_get_branch_dependencies_returns_empty_list_for_a_foundational_branch():
    session = FakeNeo4jSession(run_results=[[]])
    driver = FakeNeo4jDriver(session)

    result = get_branch_dependencies(driver, "perceiving")

    assert result == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_data/test_msc_knowledge_graph.py -v -k "concepts_for_branch or branch_dependencies"`
Expected: FAIL with `ImportError: cannot import name 'get_concepts_for_branch' from 'src.eq_data.msc_knowledge_graph'`

- [ ] **Step 3: Add the query functions**

Append to `src/eq_data/msc_knowledge_graph.py`:

```python


def get_concepts_for_branch(driver, branch):
    with driver.session() as session:
        result = session.run(
            "MATCH (c:Concept)-[:BELONGS_TO]->(:Branch {name: $branch}) "
            "RETURN c.id AS id, c.topic AS topic, c.text AS text",
            branch=branch,
        )
        return [{"id": r["id"], "topic": r["topic"], "text": r["text"]} for r in result]


def get_branch_dependencies(driver, branch):
    with driver.session() as session:
        result = session.run(
            "MATCH (:Branch {name: $branch})-[:DEPENDS_ON]->(dep:Branch) RETURN dep.name AS name",
            branch=branch,
        )
        return [r["name"] for r in result]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_data/test_msc_knowledge_graph.py -v`
Expected: 7 passed (3 builder tests from Task 2 + 4 query-function tests)

- [ ] **Step 5: Run the full project test suite**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: all pass, no regressions (this plan only adds new files under `src/eq_data/`/`tests/eq_data/` plus one `requirements.txt` line).

- [ ] **Step 6: Commit**

```bash
git add src/eq_data/msc_knowledge_graph.py tests/eq_data/test_msc_knowledge_graph.py
git commit -m "feat: add MSC knowledge graph query functions (concepts per branch, branch dependencies)"
```

---

## After This Plan

Per the approved sequence: Plan 6 (LangSmith observability), Plan 7 (Backend Integration — a `/predict-eq` endpoint), Plan 8 (Evaluation Harness). Whether/how to wire the knowledge graph into the live agent (e.g., as a new coordinator/critic tool, or as additional system-prompt content surfaced via `get_branch_dependencies`) is a deferred integration decision, not forced by this plan — matching Plan 4's precedent for the NRC enrichment. Separately, actually running Neo4j for real use (a Docker container, `docker-compose.yml`, or a managed Neo4j Aura instance) and populating it via `build_msc_knowledge_graph` is a manual, deferred step — this plan verified the integration against a temporary container that was removed afterward, it does not ship a permanent running instance.

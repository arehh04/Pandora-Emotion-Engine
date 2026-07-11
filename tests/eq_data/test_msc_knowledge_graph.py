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

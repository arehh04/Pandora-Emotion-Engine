from src.eq_data.msc_knowledge_graph import BRANCHES, build_msc_knowledge_graph, get_concepts_for_branch, get_branch_dependencies
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

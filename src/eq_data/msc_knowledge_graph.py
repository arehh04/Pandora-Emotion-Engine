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

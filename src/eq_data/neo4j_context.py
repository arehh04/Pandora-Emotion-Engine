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

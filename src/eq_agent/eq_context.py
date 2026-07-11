"""Loads the pre-built EQ RAG corpus (8 branch-tagged LanceDB tables) and an
optional cross-encoder reranker into the shape src.eq_agent.eq_rag_retrieval's
functions and src.eq_agent.branch_config's dispatcher expect.

db.open_table() raises ValueError for a nonexistent table (verified against
lancedb==0.34.0) -- unlike the ChromaDB-era get_or_create_collection, so
table existence is checked via db.list_tables() before opening, not by
catching an open failure.

The reranker argument is passed through as-is (default None): a None
reranker is a valid, meaningful value downstream -- src.eq_agent.
eq_rag_retrieval._search treats "reranker is None" as "skip reranking and
just use the native hybrid search order", the same convention this module's
tests rely on.
"""
import os

import lancedb

from src.eq_data.branch_exemplars import BRANCHES
from src.eq_data.build_eq_corpus import exemplar_table_name, theory_table_name


def load_eq_rag_context(persist_dir, embedding_func=None, reranker=None):
    if not os.path.isdir(persist_dir):
        return None

    db = lancedb.connect(persist_dir)
    # lancedb==0.34.0's db.list_tables() returns a paginated ListTablesResponse
    # object (not a plain list of names as older lancedb/ChromaDB-era APIs did)
    # -- the actual table names live on its `.tables` attribute.
    existing_tables = set(db.list_tables().tables)

    expected_names = [theory_table_name(b) for b in BRANCHES] + [exemplar_table_name(b) for b in BRANCHES]
    if not all(name in existing_tables for name in expected_names):
        return None

    theory_tables = {b: db.open_table(theory_table_name(b)) for b in BRANCHES}
    exemplar_tables = {b: db.open_table(exemplar_table_name(b)) for b in BRANCHES}

    all_tables = list(theory_tables.values()) + list(exemplar_tables.values())
    if any(t.count_rows() == 0 for t in all_tables):
        return None

    return {"theory_tables": theory_tables, "exemplar_tables": exemplar_tables, "reranker": reranker}

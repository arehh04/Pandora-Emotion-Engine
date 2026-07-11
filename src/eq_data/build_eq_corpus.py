"""Builds the EQ RAG knowledge base in LanceDB: 4 branch-tagged theory
tables and 4 branch-tagged exemplar tables (one pair per MSC branch).
Perceiving and Understanding exemplar tables are additionally enriched with
a sample of real external emotion-labeled text. Uses LanceDB's native
hybrid search (vector + full-text, fused server-side) instead of a
hand-rolled BM25/RRF layer -- see this plan's Global Constraints for the
exact API facts this implementation relies on.
"""
import os
from typing import Optional

import lancedb
from lancedb.index import FTS
from lancedb.pydantic import LanceModel, Vector

from src.eq_data.branch_exemplars import BRANCHES, sample_branch_balanced_exemplars
from src.eq_data.external_datasets import (
    fetch_emobank,
    fetch_empathetic_dialogues,
    fetch_goemotions,
    fetch_isear,
)
from src.eq_data.msc_theory_corpus import load_msc_theory_corpus

DEFAULT_EXTERNAL_FETCHERS = {
    "perceiving": [fetch_goemotions, fetch_isear, fetch_emobank],
    "using": [],
    "understanding": [fetch_empathetic_dialogues],
    "managing": [],
}


def theory_table_name(branch):
    return f"eq_theory_{branch}"


def exemplar_table_name(branch):
    return f"eq_exemplars_{branch}"


def _make_theory_schema(embedding_func):
    class TheoryRecord(LanceModel):
        text: str = embedding_func.SourceField()
        vector: Vector(embedding_func.ndims()) = embedding_func.VectorField()
        id: str
        topic: str
        citation_needed: str
    return TheoryRecord


def _make_exemplar_schema(embedding_func):
    class ExemplarRecord(LanceModel):
        text: str = embedding_func.SourceField()
        vector: Vector(embedding_func.ndims()) = embedding_func.VectorField()
        tier: Optional[int] = None
        tier_label: Optional[str] = None
        eq_proxy_score: Optional[float] = None
        source: Optional[str] = None
        emotion_labels: Optional[str] = None
        valence: Optional[float] = None
        arousal: Optional[float] = None
        dominance: Optional[float] = None
    return ExemplarRecord


def _theory_rows_for_branch(theory_entries, branch):
    return [
        {"text": e["text"], "id": e["id"], "topic": e["topic"], "citation_needed": e["citation_needed"]}
        for e in theory_entries if e["branch"] == branch
    ]


def _exemplar_rows_for_branch(exemplars_df, branch):
    branch_df = exemplars_df[exemplars_df["branch"] == branch]
    return [
        {"text": row["text"], "tier": int(row["tier"]), "tier_label": row["tier_label"],
         "eq_proxy_score": float(row["eq_proxy_score"]), "source": None,
         "emotion_labels": None, "valence": None, "arousal": None, "dominance": None}
        for _, row in branch_df.iterrows()
    ]


def _external_rows(df, n_samples, seed):
    sample = df.sample(n=min(n_samples, len(df)), random_state=seed)
    rows = []
    for _, row in sample.iterrows():
        rows.append({
            "text": row["text"], "tier": None, "tier_label": None, "eq_proxy_score": None,
            "source": row["source"],
            "emotion_labels": ",".join(row["emotion_labels"]) if row["emotion_labels"] else None,
            "valence": row["valence"], "arousal": row["arousal"], "dominance": row["dominance"],
        })
    return rows


def _build_table(db, name, schema, rows):
    table = db.create_table(name, schema=schema, mode="overwrite")
    if rows:
        table.add(rows)
        table.create_index("text", config=FTS(), replace=True)
    return table


def build_eq_corpus(
    pandora_df, data_dir, persist_dir, embedding_func, external_fetchers=None,
    n_per_tier=60, seed=42, n_external_samples=200,
):
    if external_fetchers is None:
        external_fetchers = DEFAULT_EXTERNAL_FETCHERS

    theory_entries = load_msc_theory_corpus(os.path.join(data_dir, "eq", "msc_theory_corpus.json"))
    exemplars_df = sample_branch_balanced_exemplars(pandora_df, n_per_tier=n_per_tier, seed=seed)

    db = lancedb.connect(persist_dir)
    theory_schema = _make_theory_schema(embedding_func)
    exemplar_schema = _make_exemplar_schema(embedding_func)

    theory_tables = {}
    exemplar_tables = {}

    for branch in BRANCHES:
        theory_rows = _theory_rows_for_branch(theory_entries, branch)
        theory_tables[branch] = _build_table(db, theory_table_name(branch), theory_schema, theory_rows)

        exemplar_rows = _exemplar_rows_for_branch(exemplars_df, branch)
        for fetcher in external_fetchers.get(branch, []):
            exemplar_rows += _external_rows(fetcher(), n_external_samples, seed)
        exemplar_tables[branch] = _build_table(db, exemplar_table_name(branch), exemplar_schema, exemplar_rows)

    return theory_tables, exemplar_tables


def main():
    import pandas as pd
    from lancedb.embeddings import EmbeddingFunctionRegistry

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data")
    persist_dir = os.path.join(data_dir, "eq", "lancedb")
    os.makedirs(persist_dir, exist_ok=True)

    pandora_df = pd.read_csv(os.path.join(data_dir, "train_set.csv"))
    registry = EmbeddingFunctionRegistry.get_instance()
    embedding_func = registry.get("sentence-transformers").create(name="all-MiniLM-L6-v2")

    theory_tables, exemplar_tables = build_eq_corpus(pandora_df, data_dir, persist_dir, embedding_func)

    for branch in theory_tables:
        print(f"{branch}: theory={theory_tables[branch].count_rows()} rows, "
              f"exemplars={exemplar_tables[branch].count_rows()} rows")


if __name__ == "__main__":
    main()

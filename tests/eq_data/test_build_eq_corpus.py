import json

import pandas as pd

from src.eq_data.build_eq_corpus import build_eq_corpus, exemplar_table_name, theory_table_name
from tests.eq_data.lancedb_test_helpers import make_fake_embedding_func


def _write_fixture_theory_corpus(data_dir):
    eq_dir = data_dir / "eq"
    eq_dir.mkdir()
    entries = [
        {"id": "p1", "branch": "perceiving", "topic": "t", "text": "perceiving theory chunk", "citation_needed": "yes"},
        {"id": "u1", "branch": "using", "topic": "t", "text": "using theory chunk", "citation_needed": "yes"},
        {"id": "n1", "branch": "understanding", "topic": "t", "text": "understanding theory chunk", "citation_needed": "yes"},
        {"id": "m1", "branch": "managing", "topic": "t", "text": "managing theory chunk", "citation_needed": "yes"},
    ]
    (eq_dir / "msc_theory_corpus.json").write_text(json.dumps(entries), encoding="utf-8")


def _fake_pandora_df():
    scores = [5, 20, 35, 55, 75, 95] * 3
    return pd.DataFrame({
        "text": [f"sample pandora text {i}" for i in range(len(scores))],
        "extraversion": scores, "openness": scores, "agreeableness": scores,
        "conscientiousness": scores, "neuroticism": scores,
    })


def _fake_external_fetcher(n_rows=5):
    def fetcher():
        return pd.DataFrame({
            "text": [f"external text {i}" for i in range(n_rows)],
            "source": ["fake_source"] * n_rows,
            "emotion_labels": [["joy"] for _ in range(n_rows)],
            "valence": [None] * n_rows, "arousal": [None] * n_rows, "dominance": [None] * n_rows,
        })
    return fetcher


def _fake_embedding_func():
    vector_by_text = {}
    for i in range(18):
        vector_by_text[f"sample pandora text {i}"] = [float(i), 0.0]
    for i in range(10):
        vector_by_text[f"external text {i}"] = [0.0, float(i)]
    for text in ["perceiving theory chunk", "using theory chunk", "understanding theory chunk", "managing theory chunk"]:
        vector_by_text[text] = [1.0, 1.0]
    return make_fake_embedding_func(vector_by_text)


def test_theory_and_exemplar_table_names_are_branch_specific():
    assert theory_table_name("perceiving") == "eq_theory_perceiving"
    assert exemplar_table_name("managing") == "eq_exemplars_managing"


def test_build_eq_corpus_produces_eight_populated_tables(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    persist_dir = tmp_path / "lancedb"
    _write_fixture_theory_corpus(data_dir)

    theory_tables, exemplar_tables = build_eq_corpus(
        _fake_pandora_df(), str(data_dir), str(persist_dir), _fake_embedding_func(),
        external_fetchers={"perceiving": [], "using": [], "understanding": [], "managing": []},
        n_per_tier=2, seed=1,
    )

    assert set(theory_tables.keys()) == {"perceiving", "using", "understanding", "managing"}
    assert set(exemplar_tables.keys()) == {"perceiving", "using", "understanding", "managing"}
    for branch in theory_tables:
        assert theory_tables[branch].count_rows() > 0
    for branch in exemplar_tables:
        assert exemplar_tables[branch].count_rows() > 0


def test_build_eq_corpus_enriches_perceiving_and_understanding_with_external_records(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    persist_dir = tmp_path / "lancedb"
    _write_fixture_theory_corpus(data_dir)

    _, exemplar_tables = build_eq_corpus(
        _fake_pandora_df(), str(data_dir), str(persist_dir), _fake_embedding_func(),
        external_fetchers={
            "perceiving": [_fake_external_fetcher(5)], "using": [],
            "understanding": [_fake_external_fetcher(3)], "managing": [],
        },
        n_per_tier=2, seed=1, n_external_samples=200,
    )

    perceiving_texts = exemplar_tables["perceiving"].to_pandas()["text"].tolist()
    understanding_texts = exemplar_tables["understanding"].to_pandas()["text"].tolist()
    using_texts = exemplar_tables["using"].to_pandas()["text"].tolist()

    assert any("external text" in t for t in perceiving_texts)
    assert any("external text" in t for t in understanding_texts)
    assert not any("external text" in t for t in using_texts)


def test_build_eq_corpus_stores_none_for_inapplicable_optional_fields(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    persist_dir = tmp_path / "lancedb"
    _write_fixture_theory_corpus(data_dir)

    _, exemplar_tables = build_eq_corpus(
        _fake_pandora_df(), str(data_dir), str(persist_dir), _fake_embedding_func(),
        external_fetchers={"perceiving": [_fake_external_fetcher(2)], "using": [], "understanding": [], "managing": []},
        n_per_tier=2, seed=1,
    )

    df = exemplar_tables["perceiving"].to_pandas()
    external_rows = df[df["source"] == "fake_source"]
    pandora_rows = df[df["source"].isna()]

    assert len(external_rows) > 0
    assert external_rows["tier"].isna().all()  # externally-sourced rows have no proxy tier
    assert (external_rows["emotion_labels"] == "joy").all()
    assert len(pandora_rows) > 0
    assert pandora_rows["tier"].notna().all()  # Pandora-derived rows always have a tier

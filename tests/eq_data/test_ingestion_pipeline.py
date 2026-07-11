import pandas as pd
import pytest

from src.eq_data.ingestion_pipeline import (
    clean_dataframe,
    clean_text,
    compute_corpus_version,
    prepare_eq_corpus_inputs,
    wrap_fetcher_with_cleaning,
)


def test_clean_text_normalizes_whitespace_and_strips():
    assert clean_text("  hello   world  \n") == "hello world"


def test_clean_text_handles_none():
    assert clean_text(None) == ""


def test_clean_dataframe_removes_duplicate_and_empty_texts():
    df = pd.DataFrame({
        "text": ["  hello world  ", "hello world", "", "  another row  "],
        "extra": [1, 2, 3, 4],
    })

    result = clean_dataframe(df)

    assert len(result) == 2  # "hello world" (deduped) + "another row"; empty row dropped
    assert set(result["text"]) == {"hello world", "another row"}


def test_wrap_fetcher_with_cleaning_cleans_the_fetcher_result():
    def fake_fetcher():
        return pd.DataFrame({"text": ["  dup  ", "dup", "unique row"], "source": ["x", "x", "x"]})

    wrapped = wrap_fetcher_with_cleaning(fake_fetcher)
    result = wrapped()

    assert len(result) == 2
    assert set(result["text"]) == {"dup", "unique row"}


def test_compute_corpus_version_is_deterministic_for_the_same_inputs(tmp_path):
    csv_path = tmp_path / "train_set.csv"
    csv_path.write_text("text,extraversion\nhello,50\n", encoding="utf-8")
    theory_dir = tmp_path / "eq"
    theory_dir.mkdir()
    (theory_dir / "msc_theory_corpus.json").write_text("[]", encoding="utf-8")

    v1 = compute_corpus_version(str(tmp_path), {"n_per_tier": 60, "seed": 42})
    v2 = compute_corpus_version(str(tmp_path), {"n_per_tier": 60, "seed": 42})

    assert v1 == v2
    assert len(v1) == 16


def test_compute_corpus_version_changes_when_params_change(tmp_path):
    csv_path = tmp_path / "train_set.csv"
    csv_path.write_text("text,extraversion\nhello,50\n", encoding="utf-8")
    theory_dir = tmp_path / "eq"
    theory_dir.mkdir()
    (theory_dir / "msc_theory_corpus.json").write_text("[]", encoding="utf-8")

    v1 = compute_corpus_version(str(tmp_path), {"n_per_tier": 60, "seed": 42})
    v2 = compute_corpus_version(str(tmp_path), {"n_per_tier": 30, "seed": 42})

    assert v1 != v2


def test_prepare_eq_corpus_inputs_produces_cleaned_pandora_df_and_wrapped_fetchers(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "eq").mkdir()
    (data_dir / "eq" / "msc_theory_corpus.json").write_text("[]", encoding="utf-8")
    pd.DataFrame({
        "text": ["  hello world  ", "hello world", "another row"],
        "extraversion": [50, 50, 60], "openness": [50, 50, 60],
        "agreeableness": [50, 50, 60], "conscientiousness": [50, 50, 60], "neuroticism": [50, 50, 60],
    }).to_csv(data_dir / "train_set.csv", index=False)

    def fake_fetcher():
        return pd.DataFrame({"text": ["  ext dup  ", "ext dup"], "source": ["x", "x"]})

    result = prepare_eq_corpus_inputs(
        str(data_dir), external_fetchers={"perceiving": [fake_fetcher], "using": [], "understanding": [], "managing": []},
        n_per_tier=2, seed=1,
    )

    assert len(result["pandora_df"]) == 2  # deduped
    assert "version" in result and len(result["version"]) == 16
    cleaned_external = result["external_fetchers"]["perceiving"][0]()
    assert len(cleaned_external) == 1
    assert cleaned_external["text"].iloc[0] == "ext dup"

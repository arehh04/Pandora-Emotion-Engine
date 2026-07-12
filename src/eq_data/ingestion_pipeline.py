"""Formalizes the EQ corpus's 3-source ingestion (Pandora CSV, MSC theory
JSON, 4 external HF datasets fetched in src.eq_data.external_datasets) with
explicit clean -> dedup -> version-stamp stages, producing exactly the
(pandora_df, external_fetchers) inputs src.eq_data.build_eq_corpus.build_eq_corpus
expects -- without modifying that already-PR'd function.

The theory JSON is hand-authored and already clean/deduplicated by
construction (Plan 1's msc_theory_corpus.json), so this pipeline focuses
cleaning on the Pandora CSV and external-dataset fetchers, where it has
real effect (free-text rows can carry irregular whitespace or exact
duplicates).

External-dataset content itself isn't hashed into the version stamp (the
datasets are fetched live from HuggingFace/GitHub, not local files) -- the
stamp captures the local source files' state plus build parameters. A
known, documented limitation: if an external dataset's content changes
upstream without its identifier changing, the version stamp won't detect
that.
"""
import hashlib
import os

import pandas as pd


def clean_text(text):
    if text is None:
        return ""
    return " ".join(str(text).split())


def clean_dataframe(df, text_col="text"):
    df = df.copy()
    df[text_col] = df[text_col].map(clean_text)
    df = df[df[text_col] != ""]
    return df.drop_duplicates(subset=[text_col]).reset_index(drop=True)


def wrap_fetcher_with_cleaning(fetcher):
    def wrapped():
        return clean_dataframe(fetcher())
    return wrapped


def compute_corpus_version(data_dir, params):
    source_files = [
        os.path.join(data_dir, "train_set.csv"),
        os.path.join(data_dir, "eq", "msc_theory_corpus.json"),
    ]
    hasher = hashlib.sha256()
    for path in sorted(source_files):
        stat = os.stat(path)
        hasher.update(f"{path}:{stat.st_mtime}:{stat.st_size}".encode("utf-8"))
    for key in sorted(params.keys()):
        hasher.update(f"{key}={params[key]}".encode("utf-8"))
    return hasher.hexdigest()[:16]


def prepare_eq_corpus_inputs(data_dir, external_fetchers=None, n_per_tier=60, seed=42):
    pandora_df = clean_dataframe(pd.read_csv(os.path.join(data_dir, "train_set.csv")))

    if external_fetchers is None:
        from src.eq_data.build_eq_corpus import DEFAULT_EXTERNAL_FETCHERS
        external_fetchers = DEFAULT_EXTERNAL_FETCHERS

    wrapped_fetchers = {
        branch: [wrap_fetcher_with_cleaning(f) for f in fetchers]
        for branch, fetchers in external_fetchers.items()
    }

    version = compute_corpus_version(data_dir, {
        "n_per_tier": n_per_tier, "seed": seed,
        "external_sources": "goemotions,isear,emobank,empathetic_dialogues",
    })

    return {"pandora_df": pandora_df, "external_fetchers": wrapped_fetchers, "version": version}

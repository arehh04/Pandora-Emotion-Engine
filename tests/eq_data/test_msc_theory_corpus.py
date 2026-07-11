import json

import pytest

from src.eq_data.msc_theory_corpus import load_msc_theory_corpus, VALID_BRANCHES


def test_load_msc_theory_corpus_reads_the_real_corpus_file():
    entries = load_msc_theory_corpus("data/eq/msc_theory_corpus.json")

    assert len(entries) >= 16
    for entry in entries:
        assert {"id", "branch", "topic", "text", "citation_needed"} <= entry.keys()
        assert entry["branch"] in VALID_BRANCHES
        assert entry["text"].strip()


def test_load_msc_theory_corpus_covers_all_four_branches():
    entries = load_msc_theory_corpus("data/eq/msc_theory_corpus.json")

    branches_present = {entry["branch"] for entry in entries}
    assert branches_present == VALID_BRANCHES


def test_load_msc_theory_corpus_rejects_invalid_branch(tmp_path):
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps([
        {"id": "a", "branch": "not_a_branch", "topic": "t", "text": "x", "citation_needed": "n/a"}
    ]), encoding="utf-8")

    with pytest.raises(ValueError):
        load_msc_theory_corpus(str(bad_path))


def test_load_msc_theory_corpus_rejects_duplicate_ids(tmp_path):
    dup_path = tmp_path / "dup.json"
    dup_path.write_text(json.dumps([
        {"id": "a", "branch": "perceiving", "topic": "t1", "text": "x", "citation_needed": "n/a"},
        {"id": "a", "branch": "using", "topic": "t2", "text": "y", "citation_needed": "n/a"},
    ]), encoding="utf-8")

    with pytest.raises(ValueError):
        load_msc_theory_corpus(str(dup_path))

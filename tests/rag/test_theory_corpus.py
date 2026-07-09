import json

import pytest

from src.rag.theory_corpus import load_theory_corpus


def test_load_theory_corpus_real_file_is_well_formed():
    entries = load_theory_corpus("data/rag/theory_corpus.json")

    assert len(entries) >= 15
    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids)), "duplicate ids found"
    for entry in entries:
        assert entry["text"].strip()
        assert entry["topic"].strip()
        assert "citation_needed" in entry


def test_load_theory_corpus_rejects_missing_field(tmp_path):
    bad_path = tmp_path / "bad_corpus.json"
    bad_path.write_text(json.dumps([
        {"id": "a", "topic": "t", "text": "some text"}  # missing citation_needed
    ]), encoding="utf-8")

    with pytest.raises(ValueError):
        load_theory_corpus(str(bad_path))


def test_load_theory_corpus_rejects_empty_text(tmp_path):
    bad_path = tmp_path / "bad_corpus.json"
    bad_path.write_text(json.dumps([
        {"id": "a", "topic": "t", "text": "   ", "citation_needed": "n/a"}
    ]), encoding="utf-8")

    with pytest.raises(ValueError):
        load_theory_corpus(str(bad_path))


def test_load_theory_corpus_rejects_duplicate_ids(tmp_path):
    bad_path = tmp_path / "bad_corpus.json"
    bad_path.write_text(json.dumps([
        {"id": "a", "topic": "t", "text": "text one", "citation_needed": "n/a"},
        {"id": "a", "topic": "t", "text": "text two", "citation_needed": "n/a"},
    ]), encoding="utf-8")

    with pytest.raises(ValueError):
        load_theory_corpus(str(bad_path))

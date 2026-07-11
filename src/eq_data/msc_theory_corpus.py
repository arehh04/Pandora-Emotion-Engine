"""Loads and validates the curated MSC (Mayer-Salovey-Caruso) emotional-
intelligence theory reference corpus, mirroring src/rag/theory_corpus.py's
loader/validator with an added branch field.
"""
import json

VALID_BRANCHES = {"perceiving", "using", "understanding", "managing"}
REQUIRED_FIELDS = {"id", "branch", "topic", "text", "citation_needed"}


def load_msc_theory_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    if not isinstance(entries, list):
        raise ValueError("MSC theory corpus file must contain a JSON list of entries")

    ids_seen = set()
    for entry in entries:
        missing = REQUIRED_FIELDS - entry.keys()
        if missing:
            raise ValueError(f"MSC theory corpus entry missing fields: {missing}")
        if entry["branch"] not in VALID_BRANCHES:
            raise ValueError(f"Invalid branch '{entry['branch']}' for entry '{entry['id']}'")
        if not entry["text"].strip():
            raise ValueError(f"MSC theory corpus entry '{entry['id']}' has empty text")
        if entry["id"] in ids_seen:
            raise ValueError(f"Duplicate MSC theory corpus id: {entry['id']}")
        ids_seen.add(entry["id"])

    return entries

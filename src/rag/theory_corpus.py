"""Loads and validates the curated Extraversion/Big Five theory reference corpus."""
import json

REQUIRED_FIELDS = {"id", "topic", "text", "citation_needed"}


def load_theory_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    if not isinstance(entries, list):
        raise ValueError("Theory corpus file must contain a JSON list of entries")

    ids_seen = set()
    for entry in entries:
        missing = REQUIRED_FIELDS - entry.keys()
        if missing:
            raise ValueError(f"Theory corpus entry missing fields: {missing}")
        if not entry["text"].strip():
            raise ValueError(f"Theory corpus entry '{entry['id']}' has empty text")
        if entry["id"] in ids_seen:
            raise ValueError(f"Duplicate theory corpus id: {entry['id']}")
        ids_seen.add(entry["id"])

    return entries

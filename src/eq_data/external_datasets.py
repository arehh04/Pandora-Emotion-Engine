"""Fetches and normalizes 4 external emotion-labeled text datasets into one
common schema for the EQ multi-agent pivot's data foundation.

Fetch functions make real network/HuggingFace Hub calls and are NOT covered
by the automated test suite -- normalize functions are pure and unit-tested
against fixture rows mirroring each dataset's real, verified schema.
"""
import pandas as pd

GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

EMOBANK_URL = "https://raw.githubusercontent.com/JULIELab/EmoBank/master/corpus/emobank.csv"


def normalize_goemotions_row(row):
    labels = [GOEMOTIONS_LABELS[i] for i in row["labels"]]
    return {
        "text": row["text"], "source": "goemotions", "emotion_labels": labels,
        "valence": None, "arousal": None, "dominance": None,
    }


def normalize_isear_row(row):
    return {
        "text": row["content"].strip(), "source": "isear",
        "emotion_labels": [row["emotion"].strip().lower()],
        "valence": None, "arousal": None, "dominance": None,
    }


def normalize_emobank_row(row):
    return {
        "text": row["text"], "source": "emobank", "emotion_labels": [],
        "valence": float(row["V"]), "arousal": float(row["A"]), "dominance": float(row["D"]),
    }


def normalize_empathetic_dialogues_row(row):
    return {
        "text": row["situation"], "source": "empathetic_dialogues",
        "emotion_labels": [row["emotion"].strip().lower()],
        "valence": None, "arousal": None, "dominance": None,
    }


def fetch_goemotions(split="train"):
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split=split)
    return pd.DataFrame([normalize_goemotions_row(row) for row in ds])


def fetch_isear():
    from datasets import load_dataset
    ds = load_dataset("gsri-18/ISEAR-dataset-complete", split="train")
    return pd.DataFrame([
        normalize_isear_row(row) for row in ds if row["content"] and row["emotion"]
    ])


def fetch_emobank():
    df = pd.read_csv(EMOBANK_URL)
    return pd.DataFrame([normalize_emobank_row(row) for _, row in df.iterrows()])


def fetch_empathetic_dialogues(split="train"):
    from datasets import load_dataset
    ds = load_dataset("Estwld/empathetic_dialogues_llm", split=split)
    return pd.DataFrame([normalize_empathetic_dialogues_row(row) for row in ds])

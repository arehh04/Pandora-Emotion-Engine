# EQ Data Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the data foundation for the EQ (Emotional Intelligence) multi-agent pivot: acquire and normalize 4 real emotion-labeled external datasets, derive proxy EQ labels (overall + per MSC branch) from the Pandora dataset's real Big Five trait labels, build a percentile-derived EQ tier scheme, write an MSC theory corpus, and sample branch-tagged calibration exemplars.

**Architecture:** A new `src/eq_data/` package holds everything data-related for the EQ pivot, kept separate from the Extraversion-era `src/rag/` package (which stays as-is, still used by the completed Plans 1-6). `external_datasets.py` fetches+normalizes GoEmotions/ISEAR/EmoBank/EmpatheticDialogues into one common schema (fetch functions are thin network-calling wrappers; normalization logic is pure and unit-tested against fixture data mirroring each dataset's real, verified schema — no real network calls in the automated suite). `proxy_labels.py` computes overall + per-branch EQ proxy scores from the 5 real Big Five columns already present in `data/train_set.csv`. `tiers_eq.py` mirrors `src/tiers.py`'s structure with percentile-derived EQ tier bins. `msc_theory_corpus.py` + `data/eq/msc_theory_corpus.json` mirror `src/rag/theory_corpus.py`'s loader/validator, with new MSC-branch content. `branch_exemplars.py` samples tier-balanced, branch-tagged exemplars from the proxy-labeled Pandora text for all 4 branches.

**Tech Stack:** `datasets` (HuggingFace, already installed, v5.0.0), `pandas`, `numpy`, existing `pytest` conventions.

## Global Constraints

- All 5 Big Five trait columns in `data/train_set.csv` are on a verified 0-99 scale with zero missing values (16,047 rows) — confirmed via direct inspection, not assumed.
- Proxy-label trait weights are a defensible starting point pending literature citation before thesis use — same `citation_needed` convention as `src/rag/theory_corpus.py` entries. Do not present them as validated in code comments; the weight dicts are named constants, documented as provisional.
- External dataset **fetch** functions (real network/HF Hub calls) are separate from **normalize** functions (pure, no I/O). Tests exercise only the normalize functions, using fixture inputs that mirror each dataset's real verified schema (see each task for the exact verified schema). No test in this plan makes a real network call.
- This plan does NOT yet merge the external datasets (GoEmotions/ISEAR/EmoBank/EmpatheticDialogues) into the branch-exemplar RAG corpus — Task 1 fetches and normalizes them to disk so a later plan (multi-agent specialists / evaluation harness) can use them for Perceiving/Understanding branch grounding and evaluation. Task 5's exemplar sampler in *this* plan uses only the Pandora proxy-labeled text, for all 4 branches uniformly.
- Verified dataset sources (do not substitute without re-verifying): GoEmotions via HF `google-research-datasets/go_emotions` config `simplified` (schema: `text: str, labels: list[int 0-27], id: str`); ISEAR via HF `gsri-18/ISEAR-dataset-complete` (schema: `emotion: str, content: str, Unnamed: 2` — drop the unnamed column); EmoBank via the official GitHub raw CSV `https://raw.githubusercontent.com/JULIELab/EmoBank/master/corpus/emobank.csv` (schema: `id, split, V, A, D, text`) — deliberately NOT an unofficial HF mirror, since the one tested (`reallycarlaost/emobank-w-valence`) only exposed a stripped single `label` field, losing the Arousal/Dominance dimensions; EmpatheticDialogues via HF `Estwld/empathetic_dialogues_llm` (schema: `conv_id, situation, emotion, conversations`).

---

### Task 1: External emotion dataset acquisition and normalization

**Files:**
- Create: `src/eq_data/external_datasets.py`
- Test: `tests/eq_data/test_external_datasets.py`

**Interfaces:**
- Produces: `normalize_goemotions_row(row) -> dict`, `normalize_isear_row(row) -> dict`, `normalize_emobank_row(row) -> dict`, `normalize_empathetic_dialogues_row(row) -> dict`, each returning `{"text": str, "source": str, "emotion_labels": list[str], "valence": float|None, "arousal": float|None, "dominance": float|None}`. Also `fetch_goemotions(split="train")`, `fetch_isear()`, `fetch_emobank()`, `fetch_empathetic_dialogues(split="train")`, each returning a `pandas.DataFrame` with those same columns — consumed by later plans, not by this plan's own tests (they make real network calls).

- [ ] **Step 1: Write the failing tests**

```python
from src.eq_data.external_datasets import (
    normalize_emobank_row,
    normalize_empathetic_dialogues_row,
    normalize_goemotions_row,
    normalize_isear_row,
)


def test_normalize_goemotions_row_maps_label_indices_to_names():
    # Real verified schema: {'text': str, 'labels': list[int]}, indices into
    # a fixed 28-name list ending in "neutral" at index 27.
    row = {"text": "My favourite food is anything I didn't have to cook myself.", "labels": [27], "id": "eebbqej"}

    result = normalize_goemotions_row(row)

    assert result["text"] == row["text"]
    assert result["source"] == "goemotions"
    assert result["emotion_labels"] == ["neutral"]
    assert result["valence"] is None
    assert result["arousal"] is None
    assert result["dominance"] is None


def test_normalize_goemotions_row_handles_multi_label():
    row = {"text": "happy but nervous", "labels": [17, 19], "id": "x"}  # joy=17, nervousness=19

    result = normalize_goemotions_row(row)

    assert result["emotion_labels"] == ["joy", "nervousness"]


def test_normalize_isear_row_lowercases_and_strips_emotion():
    # Real verified schema: {'emotion': str, 'content': str, 'Unnamed: 2': None}
    row = {"emotion": "joy", "content": "On days when I feel close to my partner.  \n", "Unnamed: 2": None}

    result = normalize_isear_row(row)

    assert result["text"] == "On days when I feel close to my partner."
    assert result["source"] == "isear"
    assert result["emotion_labels"] == ["joy"]
    assert result["valence"] is None


def test_normalize_emobank_row_carries_vad_values():
    # Real verified schema: {'id', 'split', 'V', 'A', 'D', 'text'}
    row = {"id": "x", "split": "train", "V": 3.0, "A": 3.0, "D": 3.2, "text": 'Remember what she said?'}

    result = normalize_emobank_row(row)

    assert result["text"] == "Remember what she said?"
    assert result["source"] == "emobank"
    assert result["emotion_labels"] == []
    assert result["valence"] == 3.0
    assert result["arousal"] == 3.0
    assert result["dominance"] == 3.2


def test_normalize_empathetic_dialogues_row_uses_situation_as_text():
    # Real verified schema: {'conv_id', 'situation', 'emotion', 'conversations'}
    row = {
        "conv_id": "hit:0_conv:1",
        "situation": "I remember going to the fireworks with my best friend.",
        "emotion": "sentimental",
        "conversations": [{"content": "...", "role": "user"}],
    }

    result = normalize_empathetic_dialogues_row(row)

    assert result["text"] == row["situation"]
    assert result["source"] == "empathetic_dialogues"
    assert result["emotion_labels"] == ["sentimental"]
    assert result["valence"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_data/test_external_datasets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eq_data.external_datasets'`

- [ ] **Step 3: Write the implementation**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_data/test_external_datasets.py -v`
Expected: 5 passed

- [ ] **Step 5: Manually verify the fetch functions against the real sources (not part of the automated suite)**

Run: `./.venv/Scripts/python.exe -c "from src.eq_data.external_datasets import fetch_goemotions, fetch_isear, fetch_emobank, fetch_empathetic_dialogues; import pandas as pd; [print(name, len(f())) for name, f in [('goemotions', fetch_goemotions), ('isear', fetch_isear), ('emobank', fetch_emobank), ('empathetic_dialogues', fetch_empathetic_dialogues)]]"`
Expected: four row counts printed, roughly matching the verified sizes (GoEmotions ~43410, ISEAR ~7516, EmoBank ~10062, EmpatheticDialogues ~19533), confirming the real sources are still reachable and shaped as expected.

- [ ] **Step 6: Commit**

```bash
git add src/eq_data/external_datasets.py tests/eq_data/test_external_datasets.py
git commit -m "feat: add external emotion dataset fetch+normalize layer for the EQ pivot"
```

---

### Task 2: Proxy EQ labels from real Big Five trait data

**Files:**
- Create: `src/eq_data/proxy_labels.py`
- Test: `tests/eq_data/test_proxy_labels.py`

**Interfaces:**
- Consumes: a row-like mapping with keys `extraversion, openness, agreeableness, conscientiousness, neuroticism` (0-99 scale, as verified in `data/train_set.csv`).
- Produces: `compute_overall_eq_proxy(row) -> float` (0-99 range), `compute_branch_eq_proxies(row) -> dict[str, float]` with keys `perceiving, using, understanding, managing` (each 0-99 range) — consumed by Task 3 (tier scheme) and Task 5 (exemplar sampler).

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from src.eq_data.proxy_labels import compute_branch_eq_proxies, compute_overall_eq_proxy


def test_compute_overall_eq_proxy_matches_hand_computed_value():
    row = {"extraversion": 80, "openness": 90, "agreeableness": 70, "conscientiousness": 60, "neuroticism": 20}

    result = compute_overall_eq_proxy(row)

    # 80*0.25 + 90*0.25 + 70*0.20 + 60*0.20 + (99-20)*0.10 = 20 + 22.5 + 14 + 12 + 7.9 = 76.4
    assert result == pytest.approx(76.4)


def test_compute_branch_eq_proxies_matches_hand_computed_values():
    row = {"extraversion": 80, "openness": 90, "agreeableness": 70, "conscientiousness": 60, "neuroticism": 20}

    result = compute_branch_eq_proxies(row)

    # perceiving = 90*0.6 + (99-20)*0.4 = 54 + 31.6 = 85.6
    assert result["perceiving"] == pytest.approx(85.6)
    # using = 90*0.5 + 80*0.5 = 45 + 40 = 85.0
    assert result["using"] == pytest.approx(85.0)
    # understanding = 90*0.5 + 70*0.5 = 45 + 35 = 80.0
    assert result["understanding"] == pytest.approx(80.0)
    # managing = (99-20)*0.6 + 60*0.4 = 47.4 + 24 = 71.4
    assert result["managing"] == pytest.approx(71.4)


def test_compute_overall_eq_proxy_stays_within_0_to_99_at_the_extremes():
    all_max = {"extraversion": 99, "openness": 99, "agreeableness": 99, "conscientiousness": 99, "neuroticism": 0}
    all_min = {"extraversion": 0, "openness": 0, "agreeableness": 0, "conscientiousness": 0, "neuroticism": 99}

    assert 0.0 <= compute_overall_eq_proxy(all_max) <= 99.0
    assert 0.0 <= compute_overall_eq_proxy(all_min) <= 99.0


def test_compute_branch_eq_proxies_stays_within_0_to_99_at_the_extremes():
    all_max = {"extraversion": 99, "openness": 99, "agreeableness": 99, "conscientiousness": 99, "neuroticism": 0}

    result = compute_branch_eq_proxies(all_max)

    assert all(0.0 <= v <= 99.0 for v in result.values())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_data/test_proxy_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eq_data.proxy_labels'`

- [ ] **Step 3: Write the implementation**

```python
"""Derives proxy EQ labels (overall + per MSC-branch) from the real Big Five
trait labels already present in data/train_set.csv.

Weights are a defensible starting point per the design spec
(docs/superpowers/specs/2026-07-11-eq-multiagent-langgraph-pivot-design.md)
-- pending literature citation before thesis use, same citation_needed
convention as src/rag/theory_corpus.py entries. All 5 traits are verified
to be on a 0-99 scale with zero missing values in data/train_set.csv.
"""

INVERTED_TRAITS = {"neuroticism"}  # higher neuroticism = lower EQ contribution


def _trait_component(row, trait):
    value = row[trait]
    return (99.0 - value) if trait in INVERTED_TRAITS else value


OVERALL_TRAIT_WEIGHTS = {
    "extraversion": 0.25,
    "openness": 0.25,
    "agreeableness": 0.20,
    "conscientiousness": 0.20,
    "neuroticism": 0.10,
}

BRANCH_TRAIT_WEIGHTS = {
    "perceiving": {"openness": 0.6, "neuroticism": 0.4},
    "using": {"openness": 0.5, "extraversion": 0.5},
    "understanding": {"openness": 0.5, "agreeableness": 0.5},
    "managing": {"neuroticism": 0.6, "conscientiousness": 0.4},
}


def compute_overall_eq_proxy(row):
    return sum(_trait_component(row, trait) * weight for trait, weight in OVERALL_TRAIT_WEIGHTS.items())


def compute_branch_eq_proxies(row):
    return {
        branch: sum(_trait_component(row, trait) * weight for trait, weight in weights.items())
        for branch, weights in BRANCH_TRAIT_WEIGHTS.items()
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_data/test_proxy_labels.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/eq_data/proxy_labels.py tests/eq_data/test_proxy_labels.py
git commit -m "feat: add proxy EQ label derivation from real Big Five trait data"
```

---

### Task 3: Percentile-derived EQ tier scheme

**Files:**
- Create: `src/eq_data/tiers_eq.py`
- Test: `tests/eq_data/test_tiers_eq.py`

**Interfaces:**
- Produces: `EQ_TIER_BINS` (list of `(low, high, tier_num, label)` tuples, mirroring `src/tiers.py`'s `TIER_BINS` structure) and `assign_eq_tier(score) -> (tier_num, label)` — consumed by Task 5 (exemplar sampler) and later plans.

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from src.eq_data.tiers_eq import EQ_TIER_BINS, assign_eq_tier


def test_eq_tier_bins_has_six_tiers_covering_0_to_99():
    assert len(EQ_TIER_BINS) == 6
    assert EQ_TIER_BINS[0][0] == 0
    assert EQ_TIER_BINS[-1][1] == 99
    # Bins are contiguous with no gaps or overlaps.
    for i in range(len(EQ_TIER_BINS) - 1):
        assert EQ_TIER_BINS[i][1] + 1 == EQ_TIER_BINS[i + 1][0]


def test_assign_eq_tier_matches_percentile_derived_boundaries():
    # Boundaries derived from the real proxy-score distribution on
    # data/train_set.csv (p10=26.15, p25=33.7, p45=41.15, p65=52.1, p85=59.8),
    # rounded to clean integer cutoffs.
    assert assign_eq_tier(10) == (1, "Low EQ")
    assert assign_eq_tier(26) == (1, "Low EQ")
    assert assign_eq_tier(27) == (2, "Below Average EQ")
    assert assign_eq_tier(34) == (2, "Below Average EQ")
    assert assign_eq_tier(35) == (3, "Balanced EQ (Developing)")
    assert assign_eq_tier(41) == (3, "Balanced EQ (Developing)")
    assert assign_eq_tier(42) == (4, "Balanced EQ (Established)")
    assert assign_eq_tier(52) == (4, "Balanced EQ (Established)")
    assert assign_eq_tier(53) == (5, "Above Average EQ")
    assert assign_eq_tier(60) == (5, "Above Average EQ")
    assert assign_eq_tier(61) == (6, "High EQ")
    assert assign_eq_tier(99) == (6, "High EQ")


def test_assign_eq_tier_rejects_out_of_range_score():
    with pytest.raises(ValueError):
        assign_eq_tier(100)
    with pytest.raises(ValueError):
        assign_eq_tier(-1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_data/test_tiers_eq.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eq_data.tiers_eq'`

- [ ] **Step 3: Write the implementation**

```python
"""Canonical EQ tier scheme, mirroring src/tiers.py's structure.

Boundaries are derived from the real proxy-EQ-score distribution on
data/train_set.csv (computed via src.eq_data.proxy_labels.compute_overall_eq_proxy
over all 16,047 rows: p10=26.15, p25=33.7, p45=41.15, p65=52.1, p85=59.8),
rounded to clean integer cutoffs -- not arbitrary.
"""

EQ_TIER_BINS = [
    (0, 26, 1, "Low EQ"),
    (27, 34, 2, "Below Average EQ"),
    (35, 41, 3, "Balanced EQ (Developing)"),
    (42, 52, 4, "Balanced EQ (Established)"),
    (53, 60, 5, "Above Average EQ"),
    (61, 99, 6, "High EQ"),
]


def assign_eq_tier(score):
    if not (0 <= score <= 99):
        raise ValueError(f"score must be within 0-99, got {score}")
    for low, high, tier_num, label in EQ_TIER_BINS:
        if low <= score <= high:
            return tier_num, label
    raise ValueError(f"score {score} did not match any tier bin")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_data/test_tiers_eq.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/eq_data/tiers_eq.py tests/eq_data/test_tiers_eq.py
git commit -m "feat: add percentile-derived EQ tier scheme"
```

---

### Task 4: MSC (Mayer-Salovey-Caruso) theory corpus

**Files:**
- Create: `src/eq_data/msc_theory_corpus.py`
- Create: `data/eq/msc_theory_corpus.json`
- Test: `tests/eq_data/test_msc_theory_corpus.py`

**Interfaces:**
- Produces: `load_msc_theory_corpus(path) -> list[dict]`, each entry with keys `id, branch, topic, text, citation_needed` where `branch` is one of `perceiving, using, understanding, managing` — consumed by a later plan's RAG corpus builder (mirrors `src/rag/theory_corpus.py`'s `load_theory_corpus`, with an added `branch` field and validation).

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_data/test_msc_theory_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eq_data.msc_theory_corpus'`

- [ ] **Step 3: Write the implementation**

```python
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
```

- [ ] **Step 4: Write the real MSC theory corpus content**

Create `data/eq/msc_theory_corpus.json` with this content (16 entries, 4 per branch; every entry's `citation_needed` is `"yes"` since none of this has been cross-checked against a specific cited source yet -- same convention as `src/rag/theory_corpus.py`'s original entries before citation review):

```json
[
  {"id": "perceiving-1", "branch": "perceiving", "topic": "Facial and vocal emotion recognition", "text": "Perceiving emotions involves accurately identifying emotions in oneself and others through facial expressions, tone of voice, and other nonverbal and verbal cues.", "citation_needed": "yes"},
  {"id": "perceiving-2", "branch": "perceiving", "topic": "Emotional expression in language", "text": "Text can convey emotion through explicit emotion words, intensifiers, punctuation, and figurative language; accurate perceiving requires distinguishing genuinely expressed emotion from incidental emotional vocabulary.", "citation_needed": "yes"},
  {"id": "perceiving-3", "branch": "perceiving", "topic": "Discriminating similar emotions", "text": "Perceiving emotions accurately includes distinguishing between similar but distinct emotional states, such as disappointment versus sadness, or anxiety versus fear.", "citation_needed": "yes"},
  {"id": "perceiving-4", "branch": "perceiving", "topic": "Perceiving emotion in artifacts and environments", "text": "The perceiving branch extends beyond people to identifying emotional tone in art, music, stories, and described situations.", "citation_needed": "yes"},
  {"id": "using-1", "branch": "using", "topic": "Emotions as information for reasoning", "text": "Using emotions to facilitate thought means using felt or perceived emotion to prioritize thinking, directing attention toward important information.", "citation_needed": "yes"},
  {"id": "using-2", "branch": "using", "topic": "Mood-congruent cognitive style", "text": "Different emotional states are associated with different cognitive styles; for example, positive moods can facilitate creative, inductive reasoning while negative moods can facilitate detail-oriented, deductive reasoning.", "citation_needed": "yes"},
  {"id": "using-3", "branch": "using", "topic": "Emotion in judgment and decision-making", "text": "Using emotions well involves generating emotions to aid judgment, such as imagining how a decision's outcome might feel before choosing.", "citation_needed": "yes"},
  {"id": "using-4", "branch": "using", "topic": "Emotion-driven motivation and perspective-taking", "text": "Emotions can be used to motivate persistence toward a goal and to take multiple perspectives on a problem by considering how different emotional stances would view it.", "citation_needed": "yes"},
  {"id": "understanding-1", "branch": "understanding", "topic": "Emotion causation and antecedents", "text": "Understanding emotions involves comprehending the likely causes of emotions and how situations, appraisals, and relationships give rise to specific emotional reactions.", "citation_needed": "yes"},
  {"id": "understanding-2", "branch": "understanding", "topic": "Emotional blends and complexity", "text": "Understanding emotions includes recognizing that complex feelings are often blends of simpler emotions, such as jealousy combining elements of anger, sadness, and fear.", "citation_needed": "yes"},
  {"id": "understanding-3", "branch": "understanding", "topic": "Emotional transitions over time", "text": "Emotions evolve over time in predictable ways, such as anger progressing to guilt or disappointment transitioning into acceptance; understanding this branch involves tracking such transitions.", "citation_needed": "yes"},
  {"id": "understanding-4", "branch": "understanding", "topic": "Labeling and vocabulary for emotion", "text": "A rich and precise emotional vocabulary supports understanding emotions by allowing finer distinctions between related emotional states.", "citation_needed": "yes"},
  {"id": "managing-1", "branch": "managing", "topic": "Emotion regulation strategies", "text": "Managing emotions involves regulating one's own emotional responses through strategies such as reappraisal, distraction, or acceptance rather than suppression alone.", "citation_needed": "yes"},
  {"id": "managing-2", "branch": "managing", "topic": "Staying open to emotional information", "text": "Effective emotion management includes remaining open to both pleasant and unpleasant feelings rather than avoiding negative emotion entirely, since unpleasant emotions often carry useful information.", "citation_needed": "yes"},
  {"id": "managing-3", "branch": "managing", "topic": "Managing others' emotions", "text": "Managing emotions extends to influencing the emotions of others, such as calming someone who is anxious or encouraging someone who is discouraged.", "citation_needed": "yes"},
  {"id": "managing-4", "branch": "managing", "topic": "Emotional resilience and recovery", "text": "Managing emotions well is associated with quicker recovery from negative emotional states and maintaining functioning under emotional strain.", "citation_needed": "yes"}
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_data/test_msc_theory_corpus.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/eq_data/msc_theory_corpus.py data/eq/msc_theory_corpus.json tests/eq_data/test_msc_theory_corpus.py
git commit -m "feat: add MSC emotional-intelligence theory reference corpus"
```

---

### Task 5: Branch-tagged exemplar sampler

**Files:**
- Create: `src/eq_data/branch_exemplars.py`
- Test: `tests/eq_data/test_branch_exemplars.py`

**Interfaces:**
- Consumes: `compute_overall_eq_proxy`, `compute_branch_eq_proxies` (Task 2); `assign_eq_tier` (Task 3).
- Produces: `sample_branch_balanced_exemplars(df, text_col="text", n_per_tier=60, seed=42) -> pandas.DataFrame` with columns `text, branch, eq_proxy_score, tier, tier_label` (one row per `(source text, branch)` pair, tier-balanced independently within each branch) — consumed by a later plan's RAG corpus builder for the branch-tagged ChromaDB collection.

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd

from src.eq_data.branch_exemplars import sample_branch_balanced_exemplars


def test_sample_branch_balanced_exemplars_produces_one_row_per_branch_per_text():
    # 6 texts, each with distinct Big Five traits so branch/tier assignment is
    # deterministic and varied across the tier range.
    df = pd.DataFrame({
        "text": [f"sample text {i}" for i in range(6)],
        "extraversion": [5, 20, 35, 55, 75, 95],
        "openness": [10, 30, 50, 60, 80, 95],
        "agreeableness": [5, 20, 40, 60, 75, 90],
        "conscientiousness": [5, 25, 45, 60, 80, 95],
        "neuroticism": [95, 75, 55, 35, 20, 5],
    })

    result = sample_branch_balanced_exemplars(df, n_per_tier=10, seed=1)

    assert set(result["branch"].unique()) == {"perceiving", "using", "understanding", "managing"}
    assert set(result.columns) == {"text", "branch", "eq_proxy_score", "tier", "tier_label"}
    # Every (text, branch) pair should appear at most once, and the sampler
    # shouldn't duplicate rows within a branch.
    assert not result.duplicated(subset=["text", "branch"]).any()
    # With only 6 tiny source rows spread across tiers and n_per_tier=10 (far
    # above the available pool), every source row should survive per branch.
    for branch in ["perceiving", "using", "understanding", "managing"]:
        assert len(result[result["branch"] == branch]) == 6


def test_sample_branch_balanced_exemplars_is_deterministic_given_a_seed():
    df = pd.DataFrame({
        "text": [f"sample text {i}" for i in range(12)],
        "extraversion": [5, 20, 35, 55, 75, 95] * 2,
        "openness": [10, 30, 50, 60, 80, 95] * 2,
        "agreeableness": [5, 20, 40, 60, 75, 90] * 2,
        "conscientiousness": [5, 25, 45, 60, 80, 95] * 2,
        "neuroticism": [95, 75, 55, 35, 20, 5] * 2,
    })

    result_a = sample_branch_balanced_exemplars(df, n_per_tier=1, seed=7)
    result_b = sample_branch_balanced_exemplars(df, n_per_tier=1, seed=7)

    pd.testing.assert_frame_equal(
        result_a.reset_index(drop=True), result_b.reset_index(drop=True)
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_data/test_branch_exemplars.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eq_data.branch_exemplars'`

- [ ] **Step 3: Write the implementation**

```python
"""Samples tier-balanced, branch-tagged calibration exemplars from the
Pandora dataset's proxy-labeled text, for all 4 MSC branches. Each branch is
tier-balanced independently since the same text can land in a different
tier per branch (branch proxy scores are computed from different Big Five
trait weightings, see src.eq_data.proxy_labels).
"""
import pandas as pd

from src.eq_data.proxy_labels import compute_branch_eq_proxies
from src.eq_data.tiers_eq import assign_eq_tier

BRANCHES = ["perceiving", "using", "understanding", "managing"]


def _sample_one_branch(df, branch, text_col, n_per_tier, seed):
    branch_df = df.copy()
    branch_df["eq_proxy_score"] = branch_df.apply(
        lambda row: compute_branch_eq_proxies(row)[branch], axis=1
    )
    tiers = branch_df["eq_proxy_score"].apply(assign_eq_tier)
    branch_df["tier"] = tiers.apply(lambda t: t[0])
    branch_df["tier_label"] = tiers.apply(lambda t: t[1])

    sampled_parts = []
    for tier_num in sorted(branch_df["tier"].unique()):
        tier_df = branch_df[branch_df["tier"] == tier_num]
        n = min(n_per_tier, len(tier_df))
        sampled_parts.append(tier_df.sample(n=n, random_state=seed))

    result = pd.concat(sampled_parts, ignore_index=True)
    result["branch"] = branch
    return result[[text_col, "branch", "eq_proxy_score", "tier", "tier_label"]].rename(
        columns={text_col: "text"}
    )


def sample_branch_balanced_exemplars(df, text_col="text", n_per_tier=60, seed=42):
    per_branch = [_sample_one_branch(df, branch, text_col, n_per_tier, seed) for branch in BRANCHES]
    return pd.concat(per_branch, ignore_index=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_data/test_branch_exemplars.py -v`
Expected: 2 passed

- [ ] **Step 5: Run the full project test suite**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: all pass, no regressions in the existing Plan 1-6 test suite (this plan only adds new files under `src/eq_data/`/`tests/eq_data/`, touching nothing existing).

- [ ] **Step 6: Commit**

```bash
git add src/eq_data/branch_exemplars.py tests/eq_data/test_branch_exemplars.py
git commit -m "feat: add branch-tagged, tier-balanced exemplar sampler for the EQ pivot"
```

---

## After This Plan

The next plan in this sequence builds the LangGraph multi-agent orchestrator (4 specialists + coordinator + critique loop) and the branch-tagged hybrid RAG corpus builder that combines this plan's `branch_exemplars.py` output with the external datasets fetched in Task 1 (GoEmotions/ISEAR/EmoBank for Perceiving grounding, EmpatheticDialogues for Understanding grounding) and the MSC theory corpus from Task 4.

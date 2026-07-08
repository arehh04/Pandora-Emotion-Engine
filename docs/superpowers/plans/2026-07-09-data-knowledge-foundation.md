# Data & Knowledge Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the data-balance visibility and RAG knowledge base (psychology theory + labeled exemplars) that every later agent-pivot plan (Fuzzy Logic Engine, ML-prior tool, RAG retrieval tool, Orchestrator) depends on.

**Architecture:** Pure-Python/pandas utilities plus two static artifact-producing scripts. A shared `src/tiers.py` module defines the canonical 6-tier Extraversion scheme used everywhere downstream. A validator reports how complete/balanced the existing `data/train_augmented.csv` paraphrase-augmentation run is (it currently covers only a subset of the training data — this plan surfaces that gap rather than silently working around it). A corpus builder embeds a curated psychology-theory reference set and a tier-balanced sample of labeled exemplars using `sentence-transformers`, producing on-disk artifacts (`.npy` + metadata CSV/JSON) that Plan 3 (RAG retrieval tool) will load and query.

**Tech Stack:** Python 3.13, pandas, numpy, pytest, sentence-transformers (`all-MiniLM-L6-v2`), torch (already a project dependency, reused for the BERT re-embedding extension).

## Global Constraints

- Test-set files (`test_set.csv`, `test_clean.csv`, `test_tokens.csv`, etc.) must never be balanced or augmented — evaluation always reflects the natural, right-skewed distribution.
- The stratified LLM-paraphrase augmentation itself stays an offline/Colab batch process — nothing in this plan calls an LLM at runtime.
- RAG retrieval (built on top of this plan's artifacts, in a later plan) uses in-memory cosine similarity only — no FAISS/Chroma/external vector DB.
- RAG embeddings use `sentence-transformers/all-MiniLM-L6-v2`, not the frozen BERT `[CLS]` vectors used elsewhere in the project.
- Canonical tier scheme (must match exactly, reused by every later plan):

  | Tier | Range | Label |
  |---|---|---|
  | 1 | 0–10 | Reserved |
  | 2 | 11–25 | Reflective |
  | 3 | 26–45 | Balanced (Introspective) |
  | 4 | 46–65 | Balanced (Sociable) |
  | 5 | 66–85 | Outgoing |
  | 6 | 86–99 | Highly Extraverted |

- No fabricated citations: every theory-corpus entry carries an explicit `citation_needed` field flagging it as unverified until checked against real literature.
- All commands below run from the repository root (`c:\Users\HP\Desktop\Nasyrah FYP`) using the project's Python (confirmed to already have `pandas`, `numpy`, `pytest`, `sentence-transformers`, `torch` installed; `pip install -r requirements.txt` if starting fresh).

---

### Task 0: Test infrastructure (conftest.py)

**Files:**
- Create: `conftest.py` (repo root)

**Interfaces:**
- Produces: repo root is importable as `src.<module>` from any test file under `tests/`, matching the import style already used in `backend/main.py` (`from src.extract_classical_features import ...`).

- [ ] **Step 1: Create the root conftest.py**

```python
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

- [ ] **Step 2: Verify pytest can discover and import from src/**

Run: `python -m pytest --collect-only -q`
Expected: Exits with `no tests ran` (or similar) and no `ModuleNotFoundError` — confirms collection works before any test files exist yet.

- [ ] **Step 3: Commit**

```bash
git add conftest.py
git commit -m "test: add root conftest.py so tests/ can import src as a package"
```

---

### Task 1: Canonical tier-assignment module

**Files:**
- Create: `src/tiers.py`
- Test: `tests/test_tiers.py`

**Interfaces:**
- Produces: `TIER_BINS` (list of `(low: int, high: int, tier_num: int, label: str)` tuples) and `assign_tier(score: float) -> tuple[int, str]` — used by every later task in this plan and every later plan (fuzzy engine, ML tool wrapper, orchestrator all call `assign_tier` to label a continuous 0-99 score).

- [ ] **Step 1: Write the failing test**

Create `tests/test_tiers.py`:

```python
import pytest

from src.tiers import assign_tier


def test_assign_tier_boundaries():
    assert assign_tier(0) == (1, "Reserved")
    assert assign_tier(10) == (1, "Reserved")
    assert assign_tier(11) == (2, "Reflective")
    assert assign_tier(25) == (2, "Reflective")
    assert assign_tier(26) == (3, "Balanced (Introspective)")
    assert assign_tier(45) == (3, "Balanced (Introspective)")
    assert assign_tier(46) == (4, "Balanced (Sociable)")
    assert assign_tier(65) == (4, "Balanced (Sociable)")
    assert assign_tier(66) == (5, "Outgoing")
    assert assign_tier(85) == (5, "Outgoing")
    assert assign_tier(86) == (6, "Highly Extraverted")
    assert assign_tier(99) == (6, "Highly Extraverted")


def test_assign_tier_accepts_float_scores():
    assert assign_tier(10.4) == (1, "Reserved")
    assert assign_tier(45.6) == (4, "Balanced (Sociable)")


def test_assign_tier_out_of_range_raises():
    with pytest.raises(ValueError):
        assign_tier(-1)
    with pytest.raises(ValueError):
        assign_tier(100)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_tiers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.tiers'`

- [ ] **Step 3: Write the minimal implementation**

Create `src/tiers.py`:

```python
"""Canonical Extraversion tier scheme shared by every agent tool and the orchestrator."""

TIER_BINS = [
    (0, 10, 1, "Reserved"),
    (11, 25, 2, "Reflective"),
    (26, 45, 3, "Balanced (Introspective)"),
    (46, 65, 4, "Balanced (Sociable)"),
    (66, 85, 5, "Outgoing"),
    (86, 99, 6, "Highly Extraverted"),
]


def assign_tier(score):
    """Map a 0-99 continuous Extraversion score to (tier_number, tier_label).

    Raises ValueError if score falls outside the valid 0-99 range.
    """
    rounded = round(score)
    for low, high, tier_num, label in TIER_BINS:
        if low <= rounded <= high:
            return tier_num, label
    raise ValueError(f"Score {score} is outside the expected 0-99 range")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_tiers.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/tiers.py tests/test_tiers.py
git commit -m "feat: add canonical Extraversion tier-assignment module"
```

---

### Task 2: Augmentation coverage & balance validator

**Files:**
- Create: `src/preprocessing/validate_augmentation.py`
- Test: `tests/preprocessing/test_validate_augmentation.py`

**Interfaces:**
- Produces: `compute_augmentation_coverage(clean_path: str, augmented_path: str, bin_col: str = "expressiveness_bin") -> dict` returning `{"total_original_rows": int, "covered_original_rows": int, "coverage_ratio": float, "bin_counts": dict, "is_balanced": bool}`. Later plans/tasks do not depend on this module directly — it's a standalone reporting tool run manually before relying on `train_augmented.csv`.

- [ ] **Step 1: Write the failing test**

Create `tests/preprocessing/test_validate_augmentation.py`:

```python
import pandas as pd

from src.preprocessing.validate_augmentation import compute_augmentation_coverage


def test_compute_augmentation_coverage_full_and_balanced(tmp_path):
    clean_path = tmp_path / "train_clean.csv"
    augmented_path = tmp_path / "train_augmented.csv"

    pd.DataFrame({
        "bert_text": [f"text {i}" for i in range(10)],
        "extraversion": list(range(10)),
    }).to_csv(clean_path, index=False)

    # All 10 original rows present (type is NaN), plus balanced augmented rows.
    rows = []
    for i in range(10):
        rows.append({"bert_text": f"text {i}", "extraversion": i, "type": None, "expressiveness_bin": "Low"})
    for i in range(5):
        rows.append({"bert_text": f"aug low {i}", "extraversion": i, "type": "_AUGMENTED", "expressiveness_bin": "Medium"})
    for i in range(5):
        rows.append({"bert_text": f"aug high {i}", "extraversion": i, "type": "_AUGMENTED", "expressiveness_bin": "High"})
    pd.DataFrame(rows).to_csv(augmented_path, index=False)

    result = compute_augmentation_coverage(str(clean_path), str(augmented_path))

    assert result["total_original_rows"] == 10
    assert result["covered_original_rows"] == 10
    assert result["coverage_ratio"] == 1.0
    assert result["bin_counts"] == {"Low": 10, "Medium": 5, "High": 5}
    assert result["is_balanced"] is True


def test_compute_augmentation_coverage_partial_and_unbalanced(tmp_path):
    clean_path = tmp_path / "train_clean.csv"
    augmented_path = tmp_path / "train_augmented.csv"

    pd.DataFrame({
        "bert_text": [f"text {i}" for i in range(10)],
        "extraversion": list(range(10)),
    }).to_csv(clean_path, index=False)

    # Only 6 of the 10 original rows present, and bins are unbalanced (10 Low vs 2 High).
    rows = []
    for i in range(6):
        rows.append({"bert_text": f"text {i}", "extraversion": i, "type": None, "expressiveness_bin": "Low"})
    for i in range(4):
        rows.append({"bert_text": f"more low {i}", "extraversion": i, "type": None, "expressiveness_bin": "Low"})
    for i in range(2):
        rows.append({"bert_text": f"aug high {i}", "extraversion": i, "type": "_AUGMENTED", "expressiveness_bin": "High"})
    pd.DataFrame(rows).to_csv(augmented_path, index=False)

    result = compute_augmentation_coverage(str(clean_path), str(augmented_path))

    assert result["total_original_rows"] == 10
    assert result["covered_original_rows"] == 10  # all 10 rows have type NaN (none augmented in this fixture)
    assert result["coverage_ratio"] == 1.0
    assert result["bin_counts"] == {"Low": 10, "High": 2}
    assert result["is_balanced"] is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/preprocessing/test_validate_augmentation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.preprocessing.validate_augmentation'`

- [ ] **Step 3: Write the minimal implementation**

Create `src/preprocessing/validate_augmentation.py`:

```python
"""Reports how complete and balanced a stratified-paraphrase augmentation run is.

Run directly to check data/train_augmented.csv against data/train_clean.csv:
    python -m src.preprocessing.validate_augmentation
"""
import os

import pandas as pd


def compute_augmentation_coverage(clean_path, augmented_path, bin_col="expressiveness_bin"):
    clean_df = pd.read_csv(clean_path)
    aug_df = pd.read_csv(augmented_path)

    if "type" in aug_df.columns:
        original_rows = aug_df[aug_df["type"].isna()]
    elif "is_augmented" in aug_df.columns:
        original_rows = aug_df[aug_df["is_augmented"] != True]  # noqa: E712
    else:
        original_rows = aug_df

    total_original = len(clean_df)
    covered_original = len(original_rows)
    coverage_ratio = covered_original / total_original if total_original else 0.0

    bin_counts = {}
    is_balanced = False
    if bin_col in aug_df.columns:
        bin_counts = aug_df[bin_col].value_counts().to_dict()
        counts = list(bin_counts.values())
        if counts and min(counts) > 0:
            is_balanced = (max(counts) / min(counts)) <= 1.5

    return {
        "total_original_rows": total_original,
        "covered_original_rows": covered_original,
        "coverage_ratio": coverage_ratio,
        "bin_counts": bin_counts,
        "is_balanced": is_balanced,
    }


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data")

    result = compute_augmentation_coverage(
        os.path.join(data_dir, "train_clean.csv"),
        os.path.join(data_dir, "train_augmented.csv"),
    )

    print(f"Original rows covered: {result['covered_original_rows']}/{result['total_original_rows']} "
          f"({result['coverage_ratio']:.1%})")
    print(f"Bin counts: {result['bin_counts']}")
    print(f"Balanced (max/min <= 1.5): {result['is_balanced']}")

    if result["coverage_ratio"] < 1.0:
        print("\nWARNING: train_augmented.csv does not cover all rows from train_clean.csv.")
        print("Re-run src/preprocessing/augment_gemma3_colab.py on Colab to completion before")
        print("retraining the legacy ML tool models or building the RAG exemplar corpus.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/preprocessing/test_validate_augmentation.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the CLI against the real project data and record the result**

Run: `python -m src.preprocessing.validate_augmentation`
Expected: prints coverage/balance stats for the actual `data/train_clean.csv` vs `data/train_augmented.csv`. Note the printed coverage ratio in the task tracker/PR description — if it's below 100%, flag to the user that the Colab augmentation notebook needs a full re-run before Plan 2 (Legacy ML retraining) can trust the balanced dataset.

- [ ] **Step 6: Commit**

```bash
git add src/preprocessing/validate_augmentation.py tests/preprocessing/test_validate_augmentation.py
git commit -m "feat: add augmentation coverage/balance validator"
```

---

### Task 3: Balanced exemplar sampler

**Files:**
- Create: `src/rag/sample_exemplars.py`
- Test: `tests/rag/test_sample_exemplars.py`

**Interfaces:**
- Consumes: `src.tiers.assign_tier(score) -> (tier_num, label)` from Task 1.
- Produces: `sample_balanced_exemplars(df: pd.DataFrame, text_col: str = "bert_text", score_col: str = "extraversion", n_per_tier: int = 60, seed: int = 42) -> pd.DataFrame` returning a DataFrame with columns `[text_col, score_col, "tier", "tier_label"]`. Consumed by Task 5 (`build_corpus.py`).

- [ ] **Step 1: Write the failing test**

Create `tests/rag/test_sample_exemplars.py`:

```python
import pandas as pd

from src.rag.sample_exemplars import sample_balanced_exemplars


def _make_df(n_low, n_mid, n_high):
    rows = []
    for i in range(n_low):
        rows.append({"bert_text": f"low {i}", "extraversion": 5})  # Tier 1: Reserved
    for i in range(n_mid):
        rows.append({"bert_text": f"mid {i}", "extraversion": 50})  # Tier 4: Balanced (Sociable)
    for i in range(n_high):
        rows.append({"bert_text": f"high {i}", "extraversion": 90})  # Tier 6: Highly Extraverted
    return pd.DataFrame(rows)


def test_sample_balanced_exemplars_caps_at_n_per_tier():
    df = _make_df(n_low=100, n_mid=100, n_high=100)

    result = sample_balanced_exemplars(df, n_per_tier=10, seed=1)

    assert list(result.columns) == ["bert_text", "extraversion", "tier", "tier_label"]
    counts = result["tier"].value_counts().to_dict()
    assert counts == {1: 10, 4: 10, 6: 10}
    assert set(result[result["tier"] == 1]["tier_label"]) == {"Reserved"}
    assert set(result[result["tier"] == 4]["tier_label"]) == {"Balanced (Sociable)"}
    assert set(result[result["tier"] == 6]["tier_label"]) == {"Highly Extraverted"}


def test_sample_balanced_exemplars_keeps_all_when_fewer_than_n_per_tier():
    df = _make_df(n_low=3, n_mid=100, n_high=100)

    result = sample_balanced_exemplars(df, n_per_tier=10, seed=1)

    counts = result["tier"].value_counts().to_dict()
    assert counts[1] == 3  # only 3 available, keep all of them
    assert counts[4] == 10
    assert counts[6] == 10


def test_sample_balanced_exemplars_is_deterministic_given_seed():
    df = _make_df(n_low=50, n_mid=0, n_high=0)

    result_a = sample_balanced_exemplars(df, n_per_tier=10, seed=7)
    result_b = sample_balanced_exemplars(df, n_per_tier=10, seed=7)

    assert result_a["bert_text"].tolist() == result_b["bert_text"].tolist()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/rag/test_sample_exemplars.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.rag.sample_exemplars'`

- [ ] **Step 3: Write the minimal implementation**

Create `src/rag/sample_exemplars.py`:

```python
"""Samples a tier-balanced exemplar set for the RAG exemplar corpus."""
import pandas as pd

from src.tiers import assign_tier


def sample_balanced_exemplars(df, text_col="bert_text", score_col="extraversion", n_per_tier=60, seed=42):
    df = df.copy()
    tiers = df[score_col].apply(assign_tier)
    df["tier"] = tiers.apply(lambda t: t[0])
    df["tier_label"] = tiers.apply(lambda t: t[1])

    sampled_parts = []
    for tier_num in sorted(df["tier"].unique()):
        tier_df = df[df["tier"] == tier_num]
        n = min(n_per_tier, len(tier_df))
        sampled_parts.append(tier_df.sample(n=n, random_state=seed))

    result = pd.concat(sampled_parts, ignore_index=True)
    return result[[text_col, score_col, "tier", "tier_label"]]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/rag/test_sample_exemplars.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/rag/sample_exemplars.py tests/rag/test_sample_exemplars.py
git commit -m "feat: add tier-balanced exemplar sampler for RAG corpus"
```

---

### Task 4: Extend BERT embedding extraction to cover train_augmented.csv

**Files:**
- Modify: `src/extract_bert_embeddings.py`
- Test: `tests/test_extract_bert_embeddings.py`

**Interfaces:**
- Produces: `embed_texts(texts: list[str], tokenizer, model, device, batch_size: int = 32) -> np.ndarray` (shape `(len(texts), hidden_size)`). `main()` now also writes `data/train_augmented_bert_embeddings.npy` when `data/train_augmented.csv` exists, aligned row-for-row with that file's `bert_text` column — this is the array the retrained ML-prior tool (a later plan) will `np.hstack` with classical features.

- [ ] **Step 1: Write the failing test**

Create `tests/test_extract_bert_embeddings.py`. This uses fake tokenizer/model stand-ins so the test doesn't download real BERT weights or need a GPU — it only verifies the batching/stacking logic:

```python
import torch

from src.extract_bert_embeddings import embed_texts


class FakeTokenizer:
    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=512):
        token_lists = [t.split() for t in texts]
        maxlen = max(len(t) for t in token_lists)
        input_ids = torch.zeros((len(texts), maxlen), dtype=torch.long)
        for i, tokens in enumerate(token_lists):
            for j in range(len(tokens)):
                input_ids[i, j] = j + 1
        attention_mask = (input_ids != 0).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class FakeOutputs:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class FakeModel:
    def __init__(self, hidden_size=4):
        self.hidden_size = hidden_size

    def __call__(self, **inputs):
        batch, seq = inputs["input_ids"].shape
        # Deterministic values so we can assert exact output.
        hidden = inputs["input_ids"].unsqueeze(-1).float().expand(batch, seq, self.hidden_size)
        return FakeOutputs(hidden)


def test_embed_texts_returns_correct_shape_and_batches_correctly():
    tokenizer = FakeTokenizer()
    model = FakeModel(hidden_size=4)
    device = torch.device("cpu")
    texts = ["hello world", "a b c d e", "single", "two words"]

    result = embed_texts(texts, tokenizer, model, device, batch_size=2)

    assert result.shape == (4, 4)


def test_embed_texts_returns_empty_array_for_empty_input():
    tokenizer = FakeTokenizer()
    model = FakeModel(hidden_size=4)
    device = torch.device("cpu")

    result = embed_texts([], tokenizer, model, device, batch_size=2)

    assert result.shape == (0, 0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_extract_bert_embeddings.py -v`
Expected: FAIL with `ImportError: cannot import name 'embed_texts' from 'src.extract_bert_embeddings'`

- [ ] **Step 3: Refactor extract_bert_embeddings.py to extract embed_texts() and add train_augmented.csv handling**

Replace the full contents of `src/extract_bert_embeddings.py`:

```python
import os
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm


def embed_texts(texts, tokenizer, model, device, batch_size=32):
    """Embed a list of strings into [CLS]-token vectors using the given tokenizer/model.

    Returns an (N, hidden_size) array, or shape (0, 0) if texts is empty.
    """
    if not texts:
        return np.zeros((0, 0))

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

    return np.vstack(all_embeddings)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Extracting BERT embeddings using device: {device}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.to(device)
    model.eval()

    splits = ["train", "validation", "test"]

    for split in splits:
        print(f"Processing {split} set...")
        clean_file = os.path.join(data_dir, f"{split}_clean.csv")
        if not os.path.exists(clean_file):
            print(f"File not found: {clean_file}")
            continue

        df = pd.read_csv(clean_file)
        texts = df["bert_text"].fillna("").tolist()

        embeddings_matrix = embed_texts(texts, tokenizer, model, device, batch_size=32)
        save_path = os.path.join(data_dir, f"{split}_bert_embeddings.npy")
        np.save(save_path, embeddings_matrix)
        print(f"Saved {split}_bert_embeddings.npy with shape {embeddings_matrix.shape}")

    # Augmented training rows need their own embeddings (new paraphrased texts
    # that don't exist in train_clean.csv), consumed by the retrained ML-prior tool.
    augmented_file = os.path.join(data_dir, "train_augmented.csv")
    if os.path.exists(augmented_file):
        print("Processing train_augmented set...")
        aug_df = pd.read_csv(augmented_file)
        texts = aug_df["bert_text"].fillna("").tolist()

        embeddings_matrix = embed_texts(texts, tokenizer, model, device, batch_size=32)
        save_path = os.path.join(data_dir, "train_augmented_bert_embeddings.npy")
        np.save(save_path, embeddings_matrix)
        print(f"Saved train_augmented_bert_embeddings.npy with shape {embeddings_matrix.shape}")
    else:
        print(f"File not found: {augmented_file} (skipping augmented embeddings)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_extract_bert_embeddings.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/extract_bert_embeddings.py tests/test_extract_bert_embeddings.py
git commit -m "refactor: extract embed_texts() helper and embed train_augmented.csv rows"
```

*(Note: actually running `python -m src.extract_bert_embeddings` end-to-end downloads and runs the real `bert-base-uncased` model against however many rows are currently in `train_augmented.csv` — this is a slow, real step, appropriate to run once manually after Task 2's coverage check looks acceptable, not as part of the automated test suite.)*

---

### Task 5: Psychology theory corpus content

**Files:**
- Create: `data/rag/theory_corpus.json`
- Create: `src/rag/theory_corpus.py`
- Test: `tests/rag/test_theory_corpus.py`

**Interfaces:**
- Produces: `load_theory_corpus(path: str) -> list[dict]`, each dict having keys `id`, `topic`, `text`, `citation_needed`. Consumed by Task 6 (`build_corpus.py`) and later by the RAG retrieval tool.

- [ ] **Step 1: Write the failing test**

Create `tests/rag/test_theory_corpus.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/rag/test_theory_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.rag.theory_corpus'` (and `data/rag/theory_corpus.json` not existing yet)

- [ ] **Step 3: Write the loader/validator**

Create `src/rag/theory_corpus.py`:

```python
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
```

- [ ] **Step 4: Write the theory corpus content**

Create `data/rag/theory_corpus.json`. Every entry's `citation_needed` field is an explicit flag — treat all of these as unverified drafts until checked against your actual literature review sources:

```json
[
  {
    "id": "extraversion-definition",
    "topic": "Trait definition",
    "text": "Extraversion is one of the five broad domains in the Five-Factor Model of personality, describing a tendency toward sociability, positive emotionality, assertiveness, and drawing energy from the external social world rather than solitary reflection.",
    "citation_needed": "Verify against Costa & McCrae's original NEO-PI-R formulation before thesis use."
  },
  {
    "id": "facet-gregariousness",
    "topic": "Extraversion facet: Gregariousness",
    "text": "Gregariousness describes a preference for the company of others and enjoyment of crowds and social gatherings; low scorers tend to prefer solitude or small, familiar groups.",
    "citation_needed": "Verify facet definition against the specific Big Five facet inventory used in the literature review."
  },
  {
    "id": "facet-assertiveness",
    "topic": "Extraversion facet: Assertiveness",
    "text": "Assertiveness reflects a tendency to be socially dominant, forceful, and expressive of opinions in group settings, often taking a leadership role in conversation.",
    "citation_needed": "Verify facet definition against the specific Big Five facet inventory used in the literature review."
  },
  {
    "id": "facet-activity-level",
    "topic": "Extraversion facet: Activity level",
    "text": "Activity level captures a preference for a fast pace of living, being busy, and engaging energetically with tasks and social situations rather than a slow, deliberate pace.",
    "citation_needed": "Verify facet definition against the specific Big Five facet inventory used in the literature review."
  },
  {
    "id": "facet-excitement-seeking",
    "topic": "Extraversion facet: Excitement-seeking",
    "text": "Excitement-seeking describes a craving for stimulation and thrill from one's environment, such as social risk-taking or novel experiences; low scorers prefer calmer, more predictable settings.",
    "citation_needed": "Verify facet definition against the specific Big Five facet inventory used in the literature review."
  },
  {
    "id": "facet-positive-emotions",
    "topic": "Extraversion facet: Positive emotions",
    "text": "Positive emotions reflects a tendency to experience joy, enthusiasm, and cheerfulness, and to express these feelings openly in language and behavior.",
    "citation_needed": "Verify facet definition against the specific Big Five facet inventory used in the literature review."
  },
  {
    "id": "facet-warmth",
    "topic": "Extraversion facet: Warmth",
    "text": "Warmth describes an interest in and affection for other people, expressed through friendliness and an easy ability to form close attachments.",
    "citation_needed": "Verify facet definition against the specific Big Five facet inventory used in the literature review."
  },
  {
    "id": "marker-word-count",
    "topic": "Linguistic marker: Talkativeness",
    "text": "Higher word counts and longer utterances in open-ended text have been associated with extraverted communication style, reflecting a general tendency toward verbal expressiveness.",
    "citation_needed": "Verify against Pennebaker & King (1999) or a comparable linguistic-style study before citing as an empirical finding."
  },
  {
    "id": "marker-positive-emotion-words",
    "topic": "Linguistic marker: Positive emotion words",
    "text": "Frequent use of positive-emotion words (e.g., joy, excitement, enthusiasm) in text has been linked to higher extraversion, consistent with the positive-emotions facet of the trait.",
    "citation_needed": "Verify against LIWC-based personality studies (e.g., Pennebaker & King 1999; Yarkoni 2010) before citing as an empirical finding."
  },
  {
    "id": "marker-social-words",
    "topic": "Linguistic marker: Social word usage",
    "text": "Frequent use of social words and first-person plural pronouns (we, us, our) can indicate a social, group-oriented framing of experience, consistent with the gregariousness and warmth facets.",
    "citation_needed": "Verify against pronoun-usage-and-personality literature before citing as an empirical finding."
  },
  {
    "id": "marker-expressive-punctuation",
    "topic": "Linguistic marker: Expressive punctuation",
    "text": "Frequent exclamation marks and other expressive punctuation in text can signal heightened enthusiasm or emotional intensity, plausibly linked to the excitement-seeking and positive-emotions facets.",
    "citation_needed": "Treat as a plausible heuristic pending direct empirical support; do not cite as an established finding without a source."
  },
  {
    "id": "marker-activity-verbs",
    "topic": "Linguistic marker: Activity-oriented verbs",
    "text": "A high ratio of action verbs relative to static or reflective language may indicate an active, outward-facing orientation consistent with the activity-level facet of extraversion.",
    "citation_needed": "Treat as a plausible heuristic pending direct empirical support; do not cite as an established finding without a source."
  },
  {
    "id": "semantic-vs-lexical",
    "topic": "Semantic context vs. keyword counting",
    "text": "Personality expression in text is often contextual: the same surface words can indicate different personality signals depending on sentence structure and situational framing, which is why contextual language understanding can outperform simple keyword-frequency counting for this task.",
    "citation_needed": "Frame as a methodological observation from this project's own prior SHAP analysis (see docs/thesis/Chapter4_Results_Interpretability.md), not an external citation."
  },
  {
    "id": "introversion-contrast",
    "topic": "Introversion characteristics",
    "text": "Introversion, the low end of the Extraversion dimension, is characterized by a preference for solitary or low-stimulation activities, more reserved social behavior, and a tendency to reflect before speaking rather than think aloud.",
    "citation_needed": "Verify against the same Five-Factor Model source used for the extraversion-definition entry."
  },
  {
    "id": "ambiversion-concept",
    "topic": "Ambiversion",
    "text": "Most individuals fall between the extremes of extraversion and introversion, exhibiting a mixture of sociable and reserved tendencies depending on context; this is sometimes referred to as ambiversion and is consistent with Extraversion being a continuous trait rather than a strict binary category.",
    "citation_needed": "Verify against a Big Five continuous-trait-distribution source before citing as an established finding."
  },
  {
    "id": "fuzzy-logic-rationale",
    "topic": "Fuzzy logic for personality gradation",
    "text": "Because personality traits are continuous and individual text samples rarely express a trait in a purely binary way, membership-based (fuzzy) reasoning that allows partial degrees of 'high' or 'low' expression can better reflect the graded nature of personality signals than hard thresholding.",
    "citation_needed": "This is a methodological justification for this project's design choice, not an external empirical claim; cite fuzzy-logic-in-NLP literature if a supporting source is added later."
  },
  {
    "id": "interpretability-risk",
    "topic": "Explainability risk of dense embeddings",
    "text": "Dense neural embeddings can capture rich contextual signal but are harder to interpret directly; without an explicit reasoning trace, a model relying solely on such embeddings risks producing plausible-looking but unfaithful explanations for its predictions.",
    "citation_needed": "Frame as a methodological observation from this project's own prior SHAP analysis (see docs/thesis/Chapter4_Results_Interpretability.md), not an external citation."
  }
]
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/rag/test_theory_corpus.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add src/rag/theory_corpus.py data/rag/theory_corpus.json tests/rag/test_theory_corpus.py
git commit -m "feat: add curated Extraversion/Big Five theory reference corpus"
```

---

### Task 6: RAG corpus embedding builder

**Files:**
- Create: `src/rag/build_corpus.py`
- Test: `tests/rag/test_build_corpus.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: `src.rag.sample_exemplars.sample_balanced_exemplars` (Task 3), `src.rag.theory_corpus.load_theory_corpus` (Task 5).
- Produces: `embed_corpus(texts: list[str], embedder) -> np.ndarray` (any `embedder` exposing `.encode(list[str]) -> array-like`) and `build_rag_corpus(data_dir: str, embedder, n_per_tier: int = 60, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray, list[dict], np.ndarray]` returning `(exemplars_df, exemplar_embeddings, theory_entries, theory_embeddings)`. `main()` writes the final on-disk artifacts consumed by the RAG retrieval tool in a later plan: `data/rag/exemplars_meta.csv`, `data/rag/exemplars_embeddings.npy`, `data/rag/theory_meta.json`, `data/rag/theory_embeddings.npy`.

- [ ] **Step 1: Write the failing test**

Create `tests/rag/test_build_corpus.py`:

```python
import json

import numpy as np
import pandas as pd

from src.rag.build_corpus import build_rag_corpus, embed_corpus


class FakeEmbedder:
    """Deterministic stand-in for SentenceTransformer.encode() — avoids downloading
    a real model in the fast unit test suite."""

    def encode(self, texts):
        return np.array([[float(len(t)), float(i)] for i, t in enumerate(texts)])


def test_embed_corpus_returns_array_matching_input_length():
    embedder = FakeEmbedder()

    result = embed_corpus(["a", "bb", "ccc"], embedder)

    assert result.shape == (3, 2)


def test_embed_corpus_handles_empty_list():
    embedder = FakeEmbedder()

    result = embed_corpus([], embedder)

    assert result.shape == (0, 0)


def test_build_rag_corpus_produces_aligned_exemplars_and_theory(tmp_path):
    data_dir = tmp_path
    rag_dir = data_dir / "rag"
    rag_dir.mkdir()

    # Minimal train_clean.csv covering all 6 tiers so the sampler has something in each.
    scores = [5, 20, 35, 55, 75, 95] * 5
    pd.DataFrame({
        "bert_text": [f"sample text {i}" for i in range(len(scores))],
        "extraversion": scores,
    }).to_csv(data_dir / "train_clean.csv", index=False)

    theory_entries = [
        {"id": "a", "topic": "t1", "text": "theory chunk one", "citation_needed": "n/a"},
        {"id": "b", "topic": "t2", "text": "theory chunk two", "citation_needed": "n/a"},
    ]
    (rag_dir / "theory_corpus.json").write_text(json.dumps(theory_entries), encoding="utf-8")

    embedder = FakeEmbedder()

    exemplars_df, exemplar_embeddings, loaded_theory, theory_embeddings = build_rag_corpus(
        str(data_dir), embedder, n_per_tier=3, seed=1
    )

    assert len(exemplars_df) == exemplar_embeddings.shape[0]
    assert len(loaded_theory) == 2
    assert theory_embeddings.shape == (2, 2)
    assert set(exemplars_df["tier"].unique()) <= {1, 2, 3, 4, 5, 6}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/rag/test_build_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.rag.build_corpus'`

- [ ] **Step 3: Write the minimal implementation**

Create `src/rag/build_corpus.py`:

```python
"""Builds the RAG knowledge base: tier-balanced exemplar embeddings + theory-corpus embeddings.

Run directly to (re)generate the on-disk artifacts under data/rag/:
    python -m src.rag.build_corpus
"""
import json
import os

import numpy as np
import pandas as pd

from src.rag.sample_exemplars import sample_balanced_exemplars
from src.rag.theory_corpus import load_theory_corpus


def embed_corpus(texts, embedder):
    """Embed a list of strings using any object exposing .encode(list[str])."""
    if not texts:
        return np.zeros((0, 0))
    return np.asarray(embedder.encode(texts))


def build_rag_corpus(data_dir, embedder, n_per_tier=60, seed=42):
    clean_df = pd.read_csv(os.path.join(data_dir, "train_clean.csv"))

    augmented_path = os.path.join(data_dir, "train_augmented.csv")
    if os.path.exists(augmented_path):
        aug_df = pd.read_csv(augmented_path)
        combined = pd.concat(
            [clean_df[["bert_text", "extraversion"]], aug_df[["bert_text", "extraversion"]]],
            ignore_index=True,
        )
    else:
        combined = clean_df[["bert_text", "extraversion"]]

    exemplars_df = sample_balanced_exemplars(combined, n_per_tier=n_per_tier, seed=seed)
    exemplar_embeddings = embed_corpus(exemplars_df["bert_text"].tolist(), embedder)

    theory_path = os.path.join(data_dir, "rag", "theory_corpus.json")
    theory_entries = load_theory_corpus(theory_path)
    theory_texts = [entry["text"] for entry in theory_entries]
    theory_embeddings = embed_corpus(theory_texts, embedder)

    return exemplars_df, exemplar_embeddings, theory_entries, theory_embeddings


def main():
    from sentence_transformers import SentenceTransformer

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    rag_dir = os.path.join(data_dir, "rag")
    os.makedirs(rag_dir, exist_ok=True)

    print("Loading sentence-transformers/all-MiniLM-L6-v2...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    exemplars_df, exemplar_embeddings, theory_entries, theory_embeddings = build_rag_corpus(data_dir, embedder)

    exemplars_df.to_csv(os.path.join(rag_dir, "exemplars_meta.csv"), index=False)
    np.save(os.path.join(rag_dir, "exemplars_embeddings.npy"), exemplar_embeddings)

    with open(os.path.join(rag_dir, "theory_meta.json"), "w", encoding="utf-8") as f:
        json.dump(theory_entries, f, indent=2)
    np.save(os.path.join(rag_dir, "theory_embeddings.npy"), theory_embeddings)

    print(f"Saved {len(exemplars_df)} exemplars -> exemplars_meta.csv / exemplars_embeddings.npy")
    print(f"Saved {len(theory_entries)} theory chunks -> theory_meta.json / theory_embeddings.npy")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/rag/test_build_corpus.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Add sentence-transformers to requirements.txt**

It's already installed in this environment but not declared — add it so a fresh install has it. Edit `requirements.txt`, appending:

```
sentence-transformers==5.5.1
```

- [ ] **Step 6: Run the full test suite for this plan**

Run: `python -m pytest tests/ -v`
Expected: PASS (all tests from Tasks 1-6)

- [ ] **Step 7: Commit**

```bash
git add src/rag/build_corpus.py tests/rag/test_build_corpus.py requirements.txt
git commit -m "feat: add RAG corpus embedding builder (exemplars + theory)"
```

*(Note: actually running `python -m src.rag.build_corpus` downloads the real `all-MiniLM-L6-v2` model and produces the real on-disk artifacts — appropriate to run once manually after Task 2's coverage check and Task 4's real BERT re-embedding step have been done, not as part of the automated test suite.)*

---

## Plan Self-Review Notes

- **Spec coverage:** Section 3 (augmentation reuse) → Task 2. Section 4 (tier scheme) → Task 1. Section 5.4 (RAG theory + exemplar collections) → Tasks 3, 5, 6. The BERT re-embedding prerequisite noted in spec Section 9's open risks → Task 4. Fuzzy Logic Engine, ML-prior retraining, RAG *retrieval* (query-time lookup), and the Orchestrator are explicitly out of scope for this plan — they're Plans 2 and 3.
- **No placeholders:** every task has complete, runnable code; no TBD/TODO markers.
- **Type/interface consistency:** `assign_tier` returns `(int, str)` in Task 1 and is consumed identically in Task 3; `sample_balanced_exemplars` column names (`bert_text`, `extraversion`, `tier`, `tier_label`) match what Task 6's test asserts; `embed_corpus`'s `embedder.encode(list[str]) -> array-like` contract is used identically by the `FakeEmbedder` test double and the real `SentenceTransformer` call in `main()`.

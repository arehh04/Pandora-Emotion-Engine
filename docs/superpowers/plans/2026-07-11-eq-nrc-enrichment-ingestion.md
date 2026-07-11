# NRC-Enriched Proxy Label + Ingestion Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enrich the proxy EQ label with NRC Word-Emotion-Association-Lexicon features computed directly from each text's own words (not just the person's Big Five traits), and formalize the EQ corpus's 3-source ingestion (Pandora CSV, MSC theory JSON, 4 external HF datasets) with explicit clean/dedup/version-stamp stages — without modifying any already-PR'd code from Plans 1-3.

**Architecture:** A self-contained NRC lexicon loader + text-feature scorer (`src/eq_data/nrc_features.py`) — deliberately NOT importing `src/extract_classical_features.py::load_nrc_lexicon` (that module is the separate, actively-changing legacy classical-ML pipeline this project has repeatedly chosen to stay decoupled from). An enrichment layer (`src/eq_data/nrc_enrichment.py`) blends the existing Big-Five-based proxy (Plan 1's `compute_overall_eq_proxy`/`compute_branch_eq_proxies`, consumed not modified) with the new text-derived score. An ingestion pipeline (`src/eq_data/ingestion_pipeline.py`) formalizes cleaning/dedup/version-stamping for the Pandora CSV and the external-dataset fetchers, producing exactly the `(pandora_df, external_fetchers)` inputs Plan 3's already-PR'd `build_eq_corpus()` expects — no modification to that function either.

**Tech Stack:** Standard library only (`hashlib` for version-stamping), existing `pandas`, `pytest`.

## Global Constraints

- **Purely additive, no modification to Plans 1-3's code** (all already-PR'd: #5, #6, #7). Where a Plan 1-3 function needs richer input (e.g. cleaned data, an enriched proxy label), this plan builds a new wrapper/consumer around it rather than editing it in place — a deliberate contrast with the tiers_eq.py bug fix from Plan 3 (that was a correctness bug with no other option; this plan adds new capability, which always has a wrapper-based alternative to modifying merged code).
- **Verified real facts about `data/NRC-Emotion-Lexicon-Senselevel-v0.92.txt`** (checked directly, not assumed): tab-separated, 3 columns (`word--sense_gloss`, `emotion_category`, `binary_flag`), 241,580 lines, 10 categories (`anger, anticip, disgust, fear, joy, negative, positive, sadness, surprise, trust`). Loading and parsing the full file (using the same word-before-`--`-split convention already established in `src/extract_classical_features.py::load_nrc_lexicon`, reimplemented self-contained here, not imported) takes ~0.28s — fast enough to load directly in tests if ever needed, though all tests in this plan use small fixture dicts instead, consistent with this project's established fixture-not-real-data convention for fast unit tests. Note: this is a curated sense-level vocabulary, not a full dictionary — common words can be genuinely absent (e.g. plain "sad" has no entry in this specific file; only "sadly"/"saddle" do) — a real data characteristic, not a parsing bug.
- The NRC-derived text score and its blend weight with the Big Five proxy are **provisional, pending literature citation**, using this project's established `citation_needed` convention (same honesty framing as Plan 1's trait-weight formulas and theory corpus entries) — not presented as validated.
- This enrichment stays **offline/data-labeling only** — it is not a live agent tool, honoring the earlier "no traditional/classical tools in the live agent" decision (Plan 6). Nothing in `src/eq_agent/` is touched by this plan.
- `src/eq_data/` gets new files, no `__init__.py` (implicit namespace packages, established convention).

---

### Task 1: NRC lexicon loader and text-feature scoring

**Files:**
- Create: `src/eq_data/nrc_features.py`
- Test: `tests/eq_data/test_nrc_features.py`

**Interfaces:**
- Produces: `load_nrc_lexicon(path) -> dict[str, set[str]]` (word → set of emotion category names it's flagged for); `compute_nrc_text_score(text, nrc_lexicon) -> float` (0-99 scaled) — consumed by Task 2's enrichment functions.

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from src.eq_data.nrc_features import compute_nrc_text_score, load_nrc_lexicon


def _write_fixture_lexicon(path):
    # Mirrors the real file's tab-separated (word--sense, emotion, flag) format.
    rows = [
        ("happy--joyful", "positive", 1), ("happy--joyful", "joy", 1),
        ("happy--joyful", "negative", 0), ("happy--joyful", "sadness", 0),
        ("wonderful--great", "positive", 1), ("wonderful--great", "joy", 1),
        ("wonderful--great", "negative", 0),
        ("sad--unhappy", "negative", 1), ("sad--unhappy", "sadness", 1),
        ("sad--unhappy", "positive", 0), ("sad--unhappy", "joy", 0),
    ]
    path.write_text("\n".join(f"{w}\t{e}\t{f}" for w, e, f in rows), encoding="utf-8")


def test_load_nrc_lexicon_parses_word_before_double_dash(tmp_path):
    fixture_path = tmp_path / "lexicon.txt"
    _write_fixture_lexicon(fixture_path)

    lexicon = load_nrc_lexicon(str(fixture_path))

    assert lexicon["happy"] == {"positive", "joy"}
    assert lexicon["wonderful"] == {"positive", "joy"}
    assert lexicon["sad"] == {"negative", "sadness"}


def test_load_nrc_lexicon_omits_words_with_no_flagged_emotions(tmp_path):
    fixture_path = tmp_path / "lexicon.txt"
    fixture_path.write_text("neutral--plain\tpositive\t0\nneutral--plain\tnegative\t0\n", encoding="utf-8")

    lexicon = load_nrc_lexicon(str(fixture_path))

    assert "neutral" not in lexicon


def test_compute_nrc_text_score_matches_hand_computed_value():
    lexicon = {
        "happy": {"positive", "joy"},
        "wonderful": {"positive", "joy"},
        "sad": {"negative", "sadness"},
    }
    text = "happy wonderful sad neutral word"

    # 5 words total; 3 are emotion words (happy, wonderful, sad) -> density = 3/5 = 0.6
    # positive_count=2 (happy, wonderful), negative_count=1 (sad)
    # positive_ratio = 2 / (2 + 1 + 1) = 0.5
    # score = 0.6*50 + 0.5*49 = 30 + 24.5 = 54.5
    score = compute_nrc_text_score(text, lexicon)

    assert score == pytest.approx(54.5)


def test_compute_nrc_text_score_handles_text_with_no_emotion_words():
    lexicon = {"happy": {"positive", "joy"}}
    text = "the quick brown fox jumps"

    score = compute_nrc_text_score(text, lexicon)

    assert score == pytest.approx(0.0)


def test_compute_nrc_text_score_caps_at_99():
    lexicon = {"joy": {"positive", "joy"}}
    text = "joy joy joy joy joy"  # density=1.0, positive_ratio=5/6=0.833; raw = 50 + 40.8... capped

    score = compute_nrc_text_score(text, lexicon)

    assert score <= 99.0


def test_compute_nrc_text_score_handles_empty_text():
    lexicon = {"happy": {"positive", "joy"}}

    score = compute_nrc_text_score("", lexicon)

    assert score == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_data/test_nrc_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eq_data.nrc_features'`

- [ ] **Step 3: Write the implementation**

```python
"""Self-contained NRC Word-Emotion-Association-Lexicon loader and
text-feature scorer for the EQ pivot. Deliberately self-contained -- does
NOT import src/extract_classical_features.py's load_nrc_lexicon, since that
module is the separate, actively-changing legacy classical-ML pipeline this
project has repeatedly chosen to stay decoupled from (see CLAUDE.md).

Score weights (50/49 split between emotion-word density and positivity
skew) are a defensible starting point pending literature citation, same
citation_needed convention as src/eq_data/proxy_labels.py's trait weights.
"""


def load_nrc_lexicon(path):
    lexicon = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            word_sense, emotion, flag = parts
            if int(flag) != 1:
                continue
            word = word_sense.split("--")[0]
            lexicon.setdefault(word, set()).add(emotion)
    return lexicon


def compute_nrc_text_score(text, nrc_lexicon):
    words = text.lower().split()
    if not words:
        return 0.0

    total_words = len(words)
    emotion_word_count = 0
    positive_count = 0
    negative_count = 0
    for word in words:
        emotions = nrc_lexicon.get(word)
        if emotions:
            emotion_word_count += 1
            if "positive" in emotions:
                positive_count += 1
            if "negative" in emotions:
                negative_count += 1

    emotion_word_density = emotion_word_count / total_words
    positive_ratio = positive_count / (positive_count + negative_count + 1)

    return min(99.0, (emotion_word_density * 50.0) + (positive_ratio * 49.0))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_data/test_nrc_features.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/eq_data/nrc_features.py tests/eq_data/test_nrc_features.py
git commit -m "feat: add self-contained NRC lexicon loader and text-feature scoring"
```

---

### Task 2: NRC-enriched proxy EQ labels

**Files:**
- Create: `src/eq_data/nrc_enrichment.py`
- Test: `tests/eq_data/test_nrc_enrichment.py`

**Interfaces:**
- Consumes: `compute_overall_eq_proxy(row)`, `compute_branch_eq_proxies(row)` from `src/eq_data/proxy_labels.py` (Plan 1, unmodified); `compute_nrc_text_score(text, nrc_lexicon)` from Task 1.
- Produces: `NRC_BLEND_WEIGHT = 0.2` (module constant); `compute_enriched_overall_eq_proxy(row, text, nrc_lexicon) -> float`; `compute_enriched_branch_eq_proxies(row, text, nrc_lexicon) -> dict[str, float]` — consumed by a later plan's exemplar-sampling/proxy-labeling step if this enrichment is adopted for corpus rebuilding.

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from src.eq_data.nrc_enrichment import (
    NRC_BLEND_WEIGHT,
    compute_enriched_branch_eq_proxies,
    compute_enriched_overall_eq_proxy,
)
from src.eq_data.proxy_labels import compute_branch_eq_proxies, compute_overall_eq_proxy

ROW = {"extraversion": 80, "openness": 90, "agreeableness": 70, "conscientiousness": 60, "neuroticism": 20}
LEXICON = {"happy": {"positive", "joy"}, "wonderful": {"positive", "joy"}}
TEXT = "happy wonderful day"


def test_nrc_blend_weight_is_a_minority_share():
    assert 0.0 < NRC_BLEND_WEIGHT < 0.5


def test_compute_enriched_overall_eq_proxy_blends_bigfive_and_text_score():
    bigfive_score = compute_overall_eq_proxy(ROW)
    from src.eq_data.nrc_features import compute_nrc_text_score
    text_score = compute_nrc_text_score(TEXT, LEXICON)
    expected = (1 - NRC_BLEND_WEIGHT) * bigfive_score + NRC_BLEND_WEIGHT * text_score

    result = compute_enriched_overall_eq_proxy(ROW, TEXT, LEXICON)

    assert result == pytest.approx(expected)


def test_compute_enriched_overall_eq_proxy_stays_within_0_to_99():
    all_max = {"extraversion": 99, "openness": 99, "agreeableness": 99, "conscientiousness": 99, "neuroticism": 0}
    dense_positive_lexicon = {"joy": {"positive", "joy"}}

    result = compute_enriched_overall_eq_proxy(all_max, "joy joy joy joy joy", dense_positive_lexicon)

    assert 0.0 <= result <= 99.0


def test_compute_enriched_branch_eq_proxies_blends_each_branch_independently():
    bigfive_branches = compute_branch_eq_proxies(ROW)
    from src.eq_data.nrc_features import compute_nrc_text_score
    text_score = compute_nrc_text_score(TEXT, LEXICON)

    result = compute_enriched_branch_eq_proxies(ROW, TEXT, LEXICON)

    assert set(result.keys()) == set(bigfive_branches.keys())
    for branch, bigfive_value in bigfive_branches.items():
        expected = (1 - NRC_BLEND_WEIGHT) * bigfive_value + NRC_BLEND_WEIGHT * text_score
        assert result[branch] == pytest.approx(expected)


def test_compute_enriched_overall_eq_proxy_falls_back_to_bigfive_only_when_no_emotion_words():
    result = compute_enriched_overall_eq_proxy(ROW, "the quick brown fox", {})
    bigfive_only = compute_overall_eq_proxy(ROW)

    # text_score is 0.0 with an empty lexicon match, so the blend pulls
    # slightly toward 0, not equal to the pure Big Five score.
    assert result == pytest.approx((1 - NRC_BLEND_WEIGHT) * bigfive_only)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_data/test_nrc_enrichment.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eq_data.nrc_enrichment'`

- [ ] **Step 3: Write the implementation**

```python
"""Blends the existing Big-Five-trait-based proxy EQ label
(src.eq_data.proxy_labels) with a text-derived NRC lexicon score
(src.eq_data.nrc_features), grounding the proxy in what the text itself
expresses, not just the person's separately-measured personality traits.

NRC_BLEND_WEIGHT is a defensible starting point (a clear minority share, so
the more-established Big Five correlation remains primary) pending
literature citation -- same citation_needed convention used throughout
src.eq_data. Does not modify src.eq_data.proxy_labels; consumes it as-is.
"""
from src.eq_data.nrc_features import compute_nrc_text_score
from src.eq_data.proxy_labels import compute_branch_eq_proxies, compute_overall_eq_proxy

NRC_BLEND_WEIGHT = 0.2


def compute_enriched_overall_eq_proxy(row, text, nrc_lexicon):
    bigfive_score = compute_overall_eq_proxy(row)
    text_score = compute_nrc_text_score(text, nrc_lexicon)
    return (1 - NRC_BLEND_WEIGHT) * bigfive_score + NRC_BLEND_WEIGHT * text_score


def compute_enriched_branch_eq_proxies(row, text, nrc_lexicon):
    bigfive_scores = compute_branch_eq_proxies(row)
    text_score = compute_nrc_text_score(text, nrc_lexicon)
    return {
        branch: (1 - NRC_BLEND_WEIGHT) * bigfive_value + NRC_BLEND_WEIGHT * text_score
        for branch, bigfive_value in bigfive_scores.items()
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_data/test_nrc_enrichment.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/eq_data/nrc_enrichment.py tests/eq_data/test_nrc_enrichment.py
git commit -m "feat: blend NRC lexicon text features into the proxy EQ label"
```

---

### Task 3: Formalized ingestion pipeline (clean, dedup, version-stamp)

**Files:**
- Create: `src/eq_data/ingestion_pipeline.py`
- Test: `tests/eq_data/test_ingestion_pipeline.py`

**Interfaces:**
- Consumes: `DEFAULT_EXTERNAL_FETCHERS` from `src/eq_data/build_eq_corpus.py` (Plan 3, unmodified, used only as a default).
- Produces: `clean_text(text) -> str`; `clean_dataframe(df, text_col="text") -> DataFrame` (cleans + dedups); `wrap_fetcher_with_cleaning(fetcher) -> callable`; `compute_corpus_version(data_dir, params) -> str` (16-char hex fingerprint); `prepare_eq_corpus_inputs(data_dir, external_fetchers=None, n_per_tier=60, seed=42) -> {"pandora_df": DataFrame, "external_fetchers": dict, "version": str}` — the exact `(pandora_df, external_fetchers)` shape Plan 3's `build_eq_corpus(pandora_df, data_dir, persist_dir, embedding_func, external_fetchers=...)` expects, so a later corpus-rebuild step can call `build_eq_corpus(**{k: v for k, v in prepare_eq_corpus_inputs(...).items() if k != "version"}, ...)` — this plan does not modify `build_eq_corpus` itself.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_data/test_ingestion_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eq_data.ingestion_pipeline'`

- [ ] **Step 3: Write the implementation**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/eq_data/test_ingestion_pipeline.py -v`
Expected: 7 passed

- [ ] **Step 5: Run the full project test suite**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: all pass, no regressions (this plan only adds new files under `src/eq_data/`/`tests/eq_data/`, touching nothing from Plans 1-3).

- [ ] **Step 6: Commit**

```bash
git add src/eq_data/ingestion_pipeline.py tests/eq_data/test_ingestion_pipeline.py
git commit -m "feat: formalize the 3-source EQ corpus ingestion pipeline (clean, dedup, version-stamp)"
```

---

## After This Plan

Per the approved sequence: Plan 5 (Neo4j knowledge graph for MSC concept relationships), Plan 6 (LangSmith observability), Plan 7 (Backend Integration — `/predict-eq`, which can optionally call `prepare_eq_corpus_inputs(...)` before `build_eq_corpus(...)` when rebuilding the corpus, and use `compute_enriched_branch_eq_proxies` instead of the plain Big-Five-only proxies if the enrichment is adopted for future corpus rebuilds), Plan 8 (Evaluation Harness). Whether to actually rebuild the real corpus using the enriched proxy labels (vs. keeping the existing Plan 1 proxy labels already used to build Plan 3's tables) is a decision for whoever runs the next real corpus build — this plan makes the enrichment available, it does not force a rebuild.

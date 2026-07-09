# Agent Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the three independent, callable "tools" the Plan 3 Agent Orchestrator will consult: a Fuzzy Logic Engine, a retrained-ML "prior" predictor, and a RAG retrieval tool over Plan 1's knowledge base — plus a shared classical-feature extraction wrapper the fuzzy engine consumes.

**Architecture:** Each tool is a standalone module under `src/agent/tools/` with one public function taking plain-Python/pandas inputs and returning a plain dict — no shared base class, no framework. The fuzzy engine is a hand-rolled Mamdani inference system (triangular membership functions, min/max rule aggregation, centroid defuzzification) implemented in pure NumPy — this project deliberately skips the `scikit-fuzzy` dependency (per the design spec's own flagged fallback: the library is lightly maintained, and hand-rolling this specific math is small and gives full control over "which rules fired," which the fuzzy tool needs for its explainability trace). The ML-prior tool reuses the *existing, already-trained* classical pipeline (TF-IDF + 3-layer classical features → Ridge/XGBoost/RandomForest, exactly as `backend/main.py`'s current `/predict` endpoint does it — **not** the classical+BERT fusion described in the original design spec Section 5.3, which does not match the code as it exists today). The RAG retrieval tool loads Plan 1's on-disk corpus artifacts and does in-memory cosine-similarity top-k lookup.

**Tech Stack:** Python (`.venv`), pandas, numpy, spaCy (`en_core_web_sm`), scikit-learn, xgboost, joblib, sentence-transformers (lazy-imported, real usage only). No new dependencies beyond what's already in `.venv`.

## Global Constraints

- **Run every command in this plan with the project's virtual environment, not a bare `python`:** `"./.venv/Scripts/python.exe"` (or activate it first: `.\.venv\Scripts\activate` then `python`). This repo has two Python installs with different partial dependency sets; `.venv` is the one with spaCy, xgboost, and scikit-learn, and it now also has `pytest` and `sentence-transformers` installed (added when this plan was scoped — confirmed via `.venv/Scripts/python.exe -m pip list`).
- Ridge/XGBoost/RandomForest ("the ML prior") consume **TF-IDF (2000 dims) + Layer 1 (semantic/NRC) + Layer 2 (lexical) + Layer 3 (behavioral) classical features only, scaled** — no BERT embeddings. This matches `backend/main.py`'s current non-BERT branch of `/predict` and `src/train_classical_models.py`. Do not reintroduce BERT fusion for these three models.
- Default ML-prior model is **Random Forest** (`models/advanced_rf_model.pkl`) — the strongest performer per the project's existing Chapter 4 evaluation table (RMSE 28.22 vs. XGBoost's 28.29).
- Canonical tier scheme lives in `src/tiers.py` (from Plan 1) — `assign_tier(score) -> (tier_num, label)`. Every tool in this plan reports its result through this same function; do not re-derive tier boundaries locally.
- No `__init__.py` files anywhere under `src/` (implicit namespace packages — established convention, verified in Plan 1).
- No live LLM/API calls anywhere in this plan — that's Plan 3 (Agent Orchestrator). These tools are deterministic, callable Python functions.
- Heavy/slow operations (loading `en_core_web_sm`, loading pickled sklearn/xgboost models, computing embeddings) are fine to exercise for real in tests **as long as they run in a few seconds** — unlike Plan 1's BERT/sentence-transformers embedding steps (which take minutes and require a manual run), spaCy and the existing small `.pkl` artifacts load in ~1-2 seconds, so this plan's tests use them directly rather than stubbing, for more faithful coverage. `sentence-transformers` itself (needed only by the RAG tool's *real* embedder) stays stubbed in tests via a fake embedder, matching Plan 1's `build_corpus.py` pattern, since downloading/running that model is the one genuinely slow step here.

---

### Task 1: Classical feature extraction wrapper

**Files:**
- Create: `src/agent/tools/classical_features.py`
- Test: `tests/agent/tools/test_classical_features.py`

**Interfaces:**
- Consumes: `compute_layer1_semantic`, `compute_layer2_lexical`, `compute_layer3_behavioral`, `load_nrc_lexicon` from `src.extract_classical_features` (existing, unchanged).
- Produces: `extract_features_for_text(text: str, nlp, nrc_dict: dict) -> dict` — a single flat dict merging every Layer 1/2/3 feature for one piece of text. Consumed by Task 2 (Fuzzy Logic Engine).

- [ ] **Step 1: Write the failing test**

Create `tests/agent/tools/test_classical_features.py`. This test uses the **real** `en_core_web_sm` spaCy model and the **real** NRC lexicon file already in this repo — both load in a couple of seconds, so there's no need to stub them:

```python
import os

import spacy

from src.extract_classical_features import load_nrc_lexicon
from src.agent.tools.classical_features import extract_features_for_text


def _load_real_nlp_and_lexicon():
    nlp = spacy.load("en_core_web_sm")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    nrc_path = os.path.join("data", "NRC-Emotion-Lexicon-Senselevel-v0.92.txt")
    nrc_dict = load_nrc_lexicon(nrc_path)
    return nlp, nrc_dict


def test_extract_features_for_text_returns_expected_keys():
    nlp, nrc_dict = _load_real_nlp_and_lexicon()

    features = extract_features_for_text("I love parties and meeting new people!", nlp, nrc_dict)

    expected_keys = {
        "anger", "anticip", "disgust", "fear", "joy", "negative", "positive", "sadness",
        "surprise", "trust", "semantic_polarity",
        "lexical_ttr", "lexical_word_count",
        "behav_avg_sentence_length", "behav_exclamation_ratio", "behav_question_ratio",
        "behav_all_caps_ratio", "behav_verb_ratio", "behav_adj_ratio",
        "behav_1st_sg_pronoun_ratio", "behav_1st_pl_pronoun_ratio",
    }
    assert expected_keys.issubset(features.keys())
    assert isinstance(features["semantic_polarity"], float)


def test_extract_features_for_text_detects_exclamation():
    nlp, nrc_dict = _load_real_nlp_and_lexicon()

    excited = extract_features_for_text("This is amazing! I am so excited! Let's go!", nlp, nrc_dict)
    flat = extract_features_for_text("This is a report about quarterly numbers.", nlp, nrc_dict)

    assert excited["behav_exclamation_ratio"] > flat["behav_exclamation_ratio"]


def test_extract_features_for_text_empty_string_does_not_raise():
    nlp, nrc_dict = _load_real_nlp_and_lexicon()

    features = extract_features_for_text("", nlp, nrc_dict)

    assert features["lexical_word_count"] == 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/agent/tools/test_classical_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agent'` (or `src.agent.tools`)

- [ ] **Step 3: Write the minimal implementation**

Create `src/agent/tools/classical_features.py`:

```python
"""Wraps the existing 3-layer classical feature extraction for a single piece of text."""
import pandas as pd

from src.extract_classical_features import (
    compute_layer1_semantic,
    compute_layer2_lexical,
    compute_layer3_behavioral,
)


def extract_features_for_text(text, nlp, nrc_dict):
    doc = nlp(text)
    lemmatized = " ".join([t.lemma_ for t in doc if not t.is_stop])

    l1_df = compute_layer1_semantic(pd.Series([lemmatized]), nrc_dict)
    l2_df = compute_layer2_lexical(pd.Series([lemmatized]))
    l3_df = compute_layer3_behavioral([text], nlp)

    combined = pd.concat([l1_df, l2_df, l3_df], axis=1)
    return combined.iloc[0].to_dict()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/agent/tools/test_classical_features.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/agent/tools/classical_features.py tests/agent/tools/test_classical_features.py
git commit -m "feat: add single-text classical feature extraction wrapper"
```

---

### Task 2: Fuzzy Logic Engine

**Files:**
- Create: `src/agent/tools/fuzzy_engine.py`
- Test: `tests/agent/tools/test_fuzzy_engine.py`

**Interfaces:**
- Consumes: `src.tiers.assign_tier` (Plan 1), and a feature dict shaped like Task 1's `extract_features_for_text` output (specifically the keys `positive`, `negative`, `semantic_polarity`, `behav_exclamation_ratio`, `behav_question_ratio`, `behav_verb_ratio`, `behav_1st_sg_pronoun_ratio`, `behav_1st_pl_pronoun_ratio`).
- Produces: `run_fuzzy_inference(features: dict) -> dict` returning `{"fuzzy_score": float, "tier": int, "tier_label": str, "fired_rules": list[dict]}`. Consumed by Plan 3's Agent Orchestrator as one of its callable tools.

This task has several small internal pieces (membership function, fuzzification, rule evaluation, defuzzification) that only make sense together behind one public function — implement them as one task with multiple TDD steps, per the plan's task-sizing convention.

- [ ] **Step 1: Write the failing test for the triangular membership function**

Create `tests/agent/tools/test_fuzzy_engine.py`:

```python
from src.agent.tools.fuzzy_engine import (
    trimf,
    fuzzify,
    compute_inputs,
    evaluate_rules,
    defuzzify,
    run_fuzzy_inference,
    INPUT_SETS,
    OUTPUT_SETS,
)


def test_trimf_peak_and_edges():
    # Triangle (0, 5, 10): 0 at edges, 1 at peak, linear between.
    assert trimf(0.0, (0, 5, 10)) == 0.0
    assert trimf(5.0, (0, 5, 10)) == 1.0
    assert trimf(10.0, (0, 5, 10)) == 0.0
    assert trimf(2.5, (0, 5, 10)) == 0.5
    assert trimf(7.5, (0, 5, 10)) == 0.5
    assert trimf(-5.0, (0, 5, 10)) == 0.0
    assert trimf(15.0, (0, 5, 10)) == 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/agent/tools/test_fuzzy_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agent.tools.fuzzy_engine'`

- [ ] **Step 3: Write `trimf`**

Create `src/agent/tools/fuzzy_engine.py`:

```python
"""Hand-rolled Mamdani fuzzy inference for Extraversion signal fusion.

Deliberately implemented without the scikit-fuzzy dependency (lightly
maintained; this project needs a fired-rule trace for explainability,
which is simpler to get from a direct implementation than from
scikit-fuzzy's higher-level control-system API).
"""
import numpy as np

from src.tiers import assign_tier


def trimf(x, abc):
    """Triangular membership degree of x in the triangle defined by (a, b, c)."""
    a, b, c = abc
    left = 1.0 if b == a else (x - a) / (b - a)
    right = 1.0 if c == b else (c - x) / (c - b)
    return max(min(left, right), 0.0)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/agent/tools/test_fuzzy_engine.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Write the failing test for fuzzification and input computation**

Append to `tests/agent/tools/test_fuzzy_engine.py`:

```python
def test_fuzzify_returns_membership_per_set():
    memberships = fuzzify(-1.0, INPUT_SETS["polarity"])

    assert memberships["Negative"] == 1.0
    assert memberships["Neutral"] == 0.0
    assert memberships["Positive"] == 0.0


def test_compute_inputs_derives_orientation_and_energy():
    features = {
        "positive": 0.2,
        "negative": 0.0,
        "semantic_polarity": 0.9,
        "behav_exclamation_ratio": 0.1,
        "behav_question_ratio": 0.05,
        "behav_verb_ratio": 0.3,
        "behav_1st_sg_pronoun_ratio": 0.0,
        "behav_1st_pl_pronoun_ratio": 0.2,
    }

    inputs = compute_inputs(features)

    assert inputs["polarity"] == 0.9
    assert round(inputs["energy"], 5) == 0.15
    assert inputs["activity"] == 0.3
    assert inputs["orientation"] > 0.9  # plural pronouns dominate, no singular pronouns present


def test_compute_inputs_defaults_orientation_to_zero_with_no_pronouns():
    features = {
        "positive": 0.0,
        "negative": 0.0,
        "semantic_polarity": 0.0,
        "behav_exclamation_ratio": 0.0,
        "behav_question_ratio": 0.0,
        "behav_verb_ratio": 0.0,
        "behav_1st_sg_pronoun_ratio": 0.0,
        "behav_1st_pl_pronoun_ratio": 0.0,
    }

    inputs = compute_inputs(features)

    assert inputs["orientation"] == 0.0
```

- [ ] **Step 6: Run the tests to verify they fail**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/agent/tools/test_fuzzy_engine.py -v`
Expected: FAIL — `fuzzify`, `compute_inputs`, `INPUT_SETS` not defined

- [ ] **Step 7: Add the input fuzzy sets, `fuzzify`, and `compute_inputs`**

Append to `src/agent/tools/fuzzy_engine.py`:

```python
INPUT_SETS = {
    "polarity": {
        "Negative": (-1.0, -1.0, 0.0),
        "Neutral": (-0.5, 0.0, 0.5),
        "Positive": (0.0, 1.0, 1.0),
    },
    "energy": {
        "Low": (0.0, 0.0, 0.1),
        "Medium": (0.05, 0.15, 0.25),
        "High": (0.15, 0.3, 0.3),
    },
    "activity": {
        "Low": (0.0, 0.0, 0.15),
        "Medium": (0.1, 0.25, 0.4),
        "High": (0.3, 0.5, 0.5),
    },
    "orientation": {
        "Singular": (0.0, 0.0, 0.4),
        "Balanced": (0.3, 0.5, 0.7),
        "Plural": (0.6, 1.0, 1.0),
    },
}

OUTPUT_SETS = {
    "Low": (0.0, 0.0, 40.0),
    "Medium": (25.0, 50.0, 75.0),
    "High": (60.0, 99.0, 99.0),
}


def fuzzify(value, sets):
    return {name: trimf(value, abc) for name, abc in sets.items()}


def compute_inputs(features):
    positive = features.get("positive", 0.0)
    negative = features.get("negative", 0.0)
    polarity = features.get("semantic_polarity", (positive - negative) / (positive + negative + 1e-5))
    energy = features.get("behav_exclamation_ratio", 0.0) + features.get("behav_question_ratio", 0.0)
    activity = features.get("behav_verb_ratio", 0.0)
    pl = features.get("behav_1st_pl_pronoun_ratio", 0.0)
    sg = features.get("behav_1st_sg_pronoun_ratio", 0.0)
    orientation = pl / (pl + sg + 1e-5)
    return {"polarity": polarity, "energy": energy, "activity": activity, "orientation": orientation}
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/agent/tools/test_fuzzy_engine.py -v`
Expected: PASS (4 tests)

- [ ] **Step 9: Write the failing test for rule evaluation and defuzzification**

Append to `tests/agent/tools/test_fuzzy_engine.py`:

```python
def test_evaluate_rules_fires_matching_rule_with_min_strength():
    memberships = {
        "polarity": {"Negative": 0.0, "Neutral": 0.0, "Positive": 1.0},
        "energy": {"Low": 0.0, "Medium": 0.0, "High": 0.6},
        "activity": {"Low": 0.0, "Medium": 0.0, "High": 0.0},
        "orientation": {"Singular": 0.0, "Balanced": 0.0, "Plural": 0.0},
    }

    fired, output_activations = evaluate_rules(memberships)

    matching = [r for r in fired if r["antecedents"] == {"polarity": "Positive", "energy": "High"}]
    assert len(matching) == 1
    assert matching[0]["strength"] == 0.6  # min(1.0, 0.6)
    assert matching[0]["consequent"] == "High"
    assert output_activations["High"] >= 0.6


def test_evaluate_rules_no_match_produces_zero_activations():
    memberships = {
        "polarity": {"Negative": 0.0, "Neutral": 0.0, "Positive": 0.0},
        "energy": {"Low": 0.0, "Medium": 0.0, "High": 0.0},
        "activity": {"Low": 0.0, "Medium": 0.0, "High": 0.0},
        "orientation": {"Singular": 0.0, "Balanced": 0.0, "Plural": 0.0},
    }

    fired, output_activations = evaluate_rules(memberships)

    assert fired == []
    assert all(v == 0.0 for v in output_activations.values())


def test_defuzzify_no_activation_returns_zero():
    assert defuzzify({"Low": 0.0, "Medium": 0.0, "High": 0.0}) == 0.0


def test_defuzzify_pure_high_activation_yields_high_score():
    score = defuzzify({"Low": 0.0, "Medium": 0.0, "High": 1.0})
    assert score > 70.0


def test_defuzzify_pure_low_activation_yields_low_score():
    score = defuzzify({"Low": 1.0, "Medium": 0.0, "High": 0.0})
    assert score < 30.0
```

- [ ] **Step 10: Run the tests to verify they fail**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/agent/tools/test_fuzzy_engine.py -v`
Expected: FAIL — `evaluate_rules`, `defuzzify` not defined

- [ ] **Step 11: Add the rule base, `evaluate_rules`, and `defuzzify`**

Append to `src/agent/tools/fuzzy_engine.py`:

```python
RULES = [
    ({"polarity": "Positive", "energy": "High"}, "High"),
    ({"polarity": "Positive", "activity": "High"}, "High"),
    ({"energy": "High", "orientation": "Plural"}, "High"),
    ({"polarity": "Positive", "orientation": "Plural"}, "High"),
    ({"activity": "High", "orientation": "Plural"}, "High"),
    ({"polarity": "Negative", "energy": "Low"}, "Low"),
    ({"polarity": "Negative", "orientation": "Singular"}, "Low"),
    ({"activity": "Low", "orientation": "Singular"}, "Low"),
    ({"energy": "Low", "orientation": "Singular"}, "Low"),
    ({"polarity": "Negative", "activity": "Low"}, "Low"),
    ({"polarity": "Neutral", "energy": "Medium", "activity": "Medium"}, "Medium"),
    ({"orientation": "Balanced", "energy": "Medium"}, "Medium"),
    ({"polarity": "Positive", "energy": "Low", "activity": "Low"}, "Medium"),
    ({"polarity": "Negative", "energy": "High"}, "Medium"),
    ({"polarity": "Neutral", "orientation": "Plural"}, "Medium"),
]


def evaluate_rules(memberships):
    fired = []
    output_activations = {name: 0.0 for name in OUTPUT_SETS}
    for antecedents, consequent in RULES:
        degree = min(memberships[var][set_name] for var, set_name in antecedents.items())
        if degree > 0:
            fired.append({"antecedents": antecedents, "consequent": consequent, "strength": degree})
            output_activations[consequent] = max(output_activations[consequent], degree)
    return fired, output_activations


def defuzzify(output_activations, resolution=200):
    universe = np.linspace(0.0, 99.0, resolution)
    aggregated = np.zeros_like(universe)
    for set_name, activation in output_activations.items():
        if activation <= 0:
            continue
        abc = OUTPUT_SETS[set_name]
        clipped = np.array([min(trimf(x, abc), activation) for x in universe])
        aggregated = np.maximum(aggregated, clipped)
    total = aggregated.sum()
    if total == 0:
        return 0.0
    return float(np.sum(universe * aggregated) / total)
```

- [ ] **Step 12: Run the tests to verify they pass**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/agent/tools/test_fuzzy_engine.py -v`
Expected: PASS (9 tests)

- [ ] **Step 13: Write the failing test for the full pipeline**

Append to `tests/agent/tools/test_fuzzy_engine.py`:

```python
def test_run_fuzzy_inference_extraverted_leaning_text_scores_high():
    features = {
        "positive": 0.3,
        "negative": 0.0,
        "semantic_polarity": 0.95,
        "behav_exclamation_ratio": 0.2,
        "behav_question_ratio": 0.05,
        "behav_verb_ratio": 0.4,
        "behav_1st_sg_pronoun_ratio": 0.0,
        "behav_1st_pl_pronoun_ratio": 0.2,
    }

    result = run_fuzzy_inference(features)

    assert result["tier"] >= 5
    assert len(result["fired_rules"]) > 0


def test_run_fuzzy_inference_introverted_leaning_text_scores_low():
    features = {
        "positive": 0.0,
        "negative": 0.2,
        "semantic_polarity": -0.9,
        "behav_exclamation_ratio": 0.0,
        "behav_question_ratio": 0.0,
        "behav_verb_ratio": 0.05,
        "behav_1st_sg_pronoun_ratio": 0.15,
        "behav_1st_pl_pronoun_ratio": 0.0,
    }

    result = run_fuzzy_inference(features)

    assert result["tier"] <= 2
    assert len(result["fired_rules"]) > 0


def test_run_fuzzy_inference_result_shape():
    features = {
        "positive": 0.1, "negative": 0.1, "semantic_polarity": 0.0,
        "behav_exclamation_ratio": 0.0, "behav_question_ratio": 0.0,
        "behav_verb_ratio": 0.2, "behav_1st_sg_pronoun_ratio": 0.05, "behav_1st_pl_pronoun_ratio": 0.05,
    }

    result = run_fuzzy_inference(features)

    assert set(result.keys()) == {"fuzzy_score", "tier", "tier_label", "fired_rules"}
    assert isinstance(result["fuzzy_score"], float)
    assert isinstance(result["tier"], int)
    assert isinstance(result["tier_label"], str)
```

- [ ] **Step 14: Run the tests to verify they fail**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/agent/tools/test_fuzzy_engine.py -v`
Expected: FAIL — `run_fuzzy_inference` not defined

- [ ] **Step 15: Add `run_fuzzy_inference`**

Append to `src/agent/tools/fuzzy_engine.py`:

```python
def run_fuzzy_inference(features):
    inputs = compute_inputs(features)
    memberships = {var: fuzzify(val, INPUT_SETS[var]) for var, val in inputs.items()}
    fired, output_activations = evaluate_rules(memberships)
    score = defuzzify(output_activations)
    tier, label = assign_tier(score)
    return {"fuzzy_score": round(score, 2), "tier": tier, "tier_label": label, "fired_rules": fired}
```

- [ ] **Step 16: Run the full fuzzy-engine test file to verify everything passes**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/agent/tools/test_fuzzy_engine.py -v`
Expected: PASS (12 tests)

- [ ] **Step 17: Commit**

```bash
git add src/agent/tools/fuzzy_engine.py tests/agent/tools/test_fuzzy_engine.py
git commit -m "feat: add hand-rolled Mamdani fuzzy logic engine for Extraversion signal fusion"
```

---

### Task 3: ML-prior tool (retrained classical models, wrapped for single-text inference)

**Files:**
- Create: `src/agent/tools/ml_prior.py`
- Test: `tests/agent/tools/test_ml_prior.py`

**Interfaces:**
- Consumes: `compute_layer1_semantic`, `compute_layer2_lexical`, `compute_layer3_behavioral` from `src.extract_classical_features`; `assign_tier` from `src.tiers`; the existing on-disk artifacts `models/tfidf_vectorizer.pkl`, `models/feature_scaler.pkl`, `models/advanced_rf_model.pkl` (default), `models/advanced_xgboost_model.pkl`, `models/classical_ridge_model.pkl`.
- Produces: `load_ml_prior_artifacts(models_dir: str, model_name: str = "random_forest") -> dict` (keys `tfidf`, `scaler`, `model`) and `predict_ml_prior(text: str, nlp, nrc_dict: dict, artifacts: dict) -> dict` returning `{"score": float, "tier": int, "tier_label": str}`. Consumed by Plan 3's Agent Orchestrator.

This task uses the project's **real, already-trained** `.pkl` artifacts and the real spaCy model directly in its tests — loading them takes a couple of seconds, no training happens, so this is a fast, faithful integration test rather than a stubbed one.

- [ ] **Step 1: Write the failing test**

Create `tests/agent/tools/test_ml_prior.py`:

```python
import os

import spacy

from src.extract_classical_features import load_nrc_lexicon
from src.agent.tools.ml_prior import load_ml_prior_artifacts, predict_ml_prior, MODEL_FILENAMES


def _load_real_nlp_and_lexicon():
    nlp = spacy.load("en_core_web_sm")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    nrc_path = os.path.join("data", "NRC-Emotion-Lexicon-Senselevel-v0.92.txt")
    nrc_dict = load_nrc_lexicon(nrc_path)
    return nlp, nrc_dict


def test_load_ml_prior_artifacts_default_model():
    artifacts = load_ml_prior_artifacts("models", model_name="random_forest")

    assert set(artifacts.keys()) == {"tfidf", "scaler", "model"}
    assert type(artifacts["model"]).__name__ == "RandomForestRegressor"


def test_load_ml_prior_artifacts_supports_all_model_names():
    for name in MODEL_FILENAMES:
        artifacts = load_ml_prior_artifacts("models", model_name=name)
        assert artifacts["model"] is not None


def test_predict_ml_prior_returns_valid_score_and_tier():
    nlp, nrc_dict = _load_real_nlp_and_lexicon()
    artifacts = load_ml_prior_artifacts("models", model_name="random_forest")

    result = predict_ml_prior(
        "I love going to parties, meeting new people, and being the center of attention!",
        nlp, nrc_dict, artifacts,
    )

    assert set(result.keys()) == {"score", "tier", "tier_label"}
    assert 0.0 <= result["score"] <= 99.0
    assert 1 <= result["tier"] <= 6


def test_predict_ml_prior_clamps_score_into_valid_range():
    nlp, nrc_dict = _load_real_nlp_and_lexicon()
    artifacts = load_ml_prior_artifacts("models", model_name="random_forest")

    result = predict_ml_prior("a", nlp, nrc_dict, artifacts)

    assert 0.0 <= result["score"] <= 99.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/agent/tools/test_ml_prior.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agent.tools.ml_prior'`

- [ ] **Step 3: Write the minimal implementation**

Create `src/agent/tools/ml_prior.py`:

```python
"""Wraps the existing, already-trained classical Ridge/XGBoost/RandomForest
models for single-text inference, mirroring backend/main.py's non-BERT
/predict branch: TF-IDF + Layer 1/2/3 classical features, scaled.
"""
import os

import joblib
import pandas as pd

from src.extract_classical_features import (
    compute_layer1_semantic,
    compute_layer2_lexical,
    compute_layer3_behavioral,
)
from src.tiers import assign_tier

MODEL_FILENAMES = {
    "random_forest": "advanced_rf_model.pkl",
    "xgboost": "advanced_xgboost_model.pkl",
    "ridge": "classical_ridge_model.pkl",
}


def load_ml_prior_artifacts(models_dir, model_name="random_forest"):
    tfidf = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "feature_scaler.pkl"))
    model = joblib.load(os.path.join(models_dir, MODEL_FILENAMES[model_name]))
    return {"tfidf": tfidf, "scaler": scaler, "model": model}


def predict_ml_prior(text, nlp, nrc_dict, artifacts):
    doc = nlp(text)
    lemmatized = " ".join([t.lemma_ for t in doc if not t.is_stop])

    l1_df = compute_layer1_semantic(pd.Series([lemmatized]), nrc_dict)
    l2_df = compute_layer2_lexical(pd.Series([lemmatized]))
    l3_df = compute_layer3_behavioral([text], nlp)

    tfidf_vec = artifacts["tfidf"].transform([lemmatized])
    tfidf_df = pd.DataFrame(
        tfidf_vec.toarray(),
        columns=[f"tfidf_{i}" for i in range(tfidf_vec.shape[1])],
    )

    combined = pd.concat([l1_df, l2_df, l3_df, tfidf_df], axis=1)
    X = artifacts["scaler"].transform(combined)

    score = float(artifacts["model"].predict(X)[0])
    score = min(99.0, max(0.0, score))
    tier, label = assign_tier(score)
    return {"score": round(score, 1), "tier": tier, "tier_label": label}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/agent/tools/test_ml_prior.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/agent/tools/ml_prior.py tests/agent/tools/test_ml_prior.py
git commit -m "feat: add ML-prior tool wrapping existing trained classical models"
```

*(Note: these tests exercise the currently-trained `models/*.pkl` artifacts, which were trained before the stratified-augmentation dataset was completed. Retraining them on `train_augmented.csv` once its coverage is improved — regenerate `train_features.csv` via `extract_classical_features.py`, then rerun `train_classical_models.py` — is a manual follow-up step, not part of this task; `load_ml_prior_artifacts` will pick up the new `.pkl` files automatically once they're regenerated, no code change needed.)*

---

### Task 4: RAG retrieval tool

**Files:**
- Create: `src/agent/tools/rag_retrieval.py`
- Test: `tests/agent/tools/test_rag_retrieval.py`

**Interfaces:**
- Consumes: Plan 1's on-disk RAG artifacts: `data/rag/exemplars_meta.csv`, `data/rag/exemplars_embeddings.npy`, `data/rag/theory_meta.json`, `data/rag/theory_embeddings.npy` (produced by `src.rag.build_corpus.main()` — a manual step; this task's tests use small fixture artifacts written to `tmp_path`, not the real ones, since the real embeddings haven't been generated yet per Plan 1's deferred manual step).
- Produces: `cosine_similarity_topk(query_vec, corpus_vecs, k=5) -> list[tuple[int, float]]`, `load_rag_corpus(rag_dir: str) -> dict` (keys `exemplars_df`, `exemplar_embeddings`, `theory_entries`, `theory_embeddings`), `retrieve_similar_exemplars(query_text: str, corpus: dict, embedder, k=5) -> list[dict]`, `retrieve_relevant_theory(query_text: str, corpus: dict, embedder, k=3) -> list[dict]`. Consumed by Plan 3's Agent Orchestrator. Any `embedder` need only expose `.encode(list[str]) -> array-like` (real usage: `sentence_transformers.SentenceTransformer`), matching Plan 1's `build_corpus.py` convention.

- [ ] **Step 1: Write the failing test**

Create `tests/agent/tools/test_rag_retrieval.py`:

```python
import json

import numpy as np
import pandas as pd

from src.agent.tools.rag_retrieval import (
    cosine_similarity_topk,
    load_rag_corpus,
    retrieve_similar_exemplars,
    retrieve_relevant_theory,
)


class FakeEmbedder:
    """Deterministic stand-in for SentenceTransformer.encode()."""

    def __init__(self, vector_by_text):
        self.vector_by_text = vector_by_text

    def encode(self, texts):
        return np.array([self.vector_by_text[t] for t in texts])


def _write_fixture_corpus(rag_dir):
    exemplars_df = pd.DataFrame({
        "bert_text": ["I love parties", "I stayed home reading alone", "We had a huge group gathering"],
        "extraversion": [90, 5, 80],
        "tier": [6, 1, 5],
        "tier_label": ["Highly Extraverted", "Reserved", "Outgoing"],
    })
    exemplars_df.to_csv(rag_dir / "exemplars_meta.csv", index=False)

    # Orthogonal-ish 2D vectors so nearest-neighbor is unambiguous.
    exemplar_embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    np.save(rag_dir / "exemplars_embeddings.npy", exemplar_embeddings)

    theory_entries = [
        {"id": "a", "topic": "t1", "text": "gregariousness theory", "citation_needed": "n/a"},
        {"id": "b", "topic": "t2", "text": "introversion theory", "citation_needed": "n/a"},
    ]
    (rag_dir / "theory_meta.json").write_text(json.dumps(theory_entries), encoding="utf-8")
    theory_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    np.save(rag_dir / "theory_embeddings.npy", theory_embeddings)


def test_cosine_similarity_topk_ranks_closest_vector_first():
    query = np.array([1.0, 0.0])
    corpus = np.array([[0.0, 1.0], [1.0, 0.0], [0.9, 0.1]])

    hits = cosine_similarity_topk(query, corpus, k=2)

    assert hits[0][0] == 1  # exact match, index 1
    assert hits[1][0] == 2  # near match, index 2
    assert hits[0][1] > hits[1][1] > 0


def test_load_rag_corpus_reads_all_four_artifacts(tmp_path):
    _write_fixture_corpus(tmp_path)

    corpus = load_rag_corpus(str(tmp_path))

    assert len(corpus["exemplars_df"]) == 3
    assert corpus["exemplar_embeddings"].shape == (3, 2)
    assert len(corpus["theory_entries"]) == 2
    assert corpus["theory_embeddings"].shape == (2, 2)


def test_retrieve_similar_exemplars_returns_nearest_with_metadata(tmp_path):
    _write_fixture_corpus(tmp_path)
    corpus = load_rag_corpus(str(tmp_path))
    embedder = FakeEmbedder({"I'm at a party!": np.array([1.0, 0.0])})

    hits = retrieve_similar_exemplars("I'm at a party!", corpus, embedder, k=1)

    assert len(hits) == 1
    assert hits[0]["bert_text"] == "I love parties"
    assert hits[0]["tier_label"] == "Highly Extraverted"
    assert "similarity" in hits[0]


def test_retrieve_relevant_theory_returns_nearest_entry(tmp_path):
    _write_fixture_corpus(tmp_path)
    corpus = load_rag_corpus(str(tmp_path))
    embedder = FakeEmbedder({"why am I so quiet": np.array([0.0, 1.0])})

    hits = retrieve_relevant_theory("why am I so quiet", corpus, embedder, k=1)

    assert len(hits) == 1
    assert hits[0]["id"] == "b"
    assert "similarity" in hits[0]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/agent/tools/test_rag_retrieval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agent.tools.rag_retrieval'`

- [ ] **Step 3: Write the minimal implementation**

Create `src/agent/tools/rag_retrieval.py`:

```python
"""In-memory cosine-similarity retrieval over Plan 1's RAG corpus artifacts.

No FAISS/Chroma — the corpus is a few hundred exemplars plus ~17 theory
chunks, small enough that a plain NumPy top-k is simple and fast enough.
"""
import json
import os

import numpy as np
import pandas as pd


def cosine_similarity_topk(query_vec, corpus_vecs, k=5):
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    corpus_norms = corpus_vecs / (np.linalg.norm(corpus_vecs, axis=1, keepdims=True) + 1e-10)
    scores = corpus_norms @ query_norm
    top_idx = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in top_idx]


def load_rag_corpus(rag_dir):
    exemplars_df = pd.read_csv(os.path.join(rag_dir, "exemplars_meta.csv"))
    exemplar_embeddings = np.load(os.path.join(rag_dir, "exemplars_embeddings.npy"))
    with open(os.path.join(rag_dir, "theory_meta.json"), "r", encoding="utf-8") as f:
        theory_entries = json.load(f)
    theory_embeddings = np.load(os.path.join(rag_dir, "theory_embeddings.npy"))
    return {
        "exemplars_df": exemplars_df,
        "exemplar_embeddings": exemplar_embeddings,
        "theory_entries": theory_entries,
        "theory_embeddings": theory_embeddings,
    }


def retrieve_similar_exemplars(query_text, corpus, embedder, k=5):
    query_vec = np.asarray(embedder.encode([query_text])[0])
    hits = cosine_similarity_topk(query_vec, corpus["exemplar_embeddings"], k=k)
    return [
        {**corpus["exemplars_df"].iloc[idx].to_dict(), "similarity": score}
        for idx, score in hits
    ]


def retrieve_relevant_theory(query_text, corpus, embedder, k=3):
    query_vec = np.asarray(embedder.encode([query_text])[0])
    hits = cosine_similarity_topk(query_vec, corpus["theory_embeddings"], k=k)
    return [
        {**corpus["theory_entries"][idx], "similarity": score}
        for idx, score in hits
    ]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/agent/tools/test_rag_retrieval.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Run the full test suite for this plan**

Run: `"./.venv/Scripts/python.exe" -m pytest tests/ -v`
Expected: PASS (all tests from Plan 1 plus this plan's new tests)

- [ ] **Step 6: Commit**

```bash
git add src/agent/tools/rag_retrieval.py tests/agent/tools/test_rag_retrieval.py
git commit -m "feat: add RAG retrieval tool over exemplar and theory corpora"
```

---

## Plan Self-Review Notes

- **Spec coverage:** Design spec Section 5.1 (classical extractor reuse) → Task 1. Section 5.2 (Fuzzy Logic Engine) → Task 2, implemented with hand-rolled NumPy instead of scikit-fuzzy per the spec's own flagged fallback (Section 9 open risks) — a deliberate, evidence-based deviation, not an oversight. Section 5.3 (ML-prior tool) → Task 3, corrected to match the actual current codebase (TF-IDF + classical layers only, no BERT) rather than the spec's stated classical+BERT fusion — the spec was written before re-confirming the current `backend/main.py`/`train_classical_models.py` state. Section 5.4 (RAG retrieval) → Task 4. The Agent Orchestrator that calls these four tools together is explicitly out of scope — that's Plan 3.
- **No placeholders:** every task has complete, runnable code; no TBD/TODO markers.
- **Type/interface consistency:** `extract_features_for_text`'s returned dict keys (Task 1) are exactly the keys `compute_inputs` (Task 2) and `predict_ml_prior`'s internal feature computation (Task 3) expect/produce. `assign_tier`'s `(int, str)` return shape (Plan 1) is used identically in `fuzzy_engine.py` and `ml_prior.py`. `embed_corpus`'s `.encode(list[str])` embedder contract (Plan 1's `build_corpus.py`) is reused identically by `rag_retrieval.py`'s `retrieve_similar_exemplars`/`retrieve_relevant_theory`.

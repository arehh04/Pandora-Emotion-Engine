# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

**Pandora / ExtraVerse** is a Final Year Project (FYP) that predicts a continuous Extraversion score (0–100) from unstructured text. It combines classical NLP features (TF-IDF, NRC emotion lexicon, POS/behavioral ratios) with BERT semantic embeddings, feeding them into several regression models (Ridge, XGBoost, Random Forest, and a fine-tuned BERT regressor). The repo has two runtime halves — a Python/FastAPI inference backend and a SvelteKit frontend — plus a `src/` tree of one-off training/preprocessing scripts used to produce the artifacts in `models/` and `data/`.

Academic context lives in `WBS.md` (work-breakdown structure mapping phases → deliverables → files) and `docs/thesis/` — useful for understanding *why* a pipeline step exists, but the actual filenames/paths in the codebase have drifted from the WBS's originally planned layout (e.g. `src/extract_classical_features.py` implements the 3-layer feature extraction that WBS describes as separate `tfidf_features.py`/`linguistic_features.py`/`emotion_features.py` modules).

## Commands

### Backend (FastAPI)
```bash
python -m venv .venv
.\.venv\Scripts\activate          # Windows; `source .venv/bin/activate` on Unix
pip install -r requirements.txt
python -m spacy download en_core_web_sm

uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Redis is optional — if unreachable at `127.0.0.1:6379`, `backend/main.py` disables the Bloom-filter/Redis cache layer and logs a warning instead of failing.

### Frontend (SvelteKit 5, in `frontend/`)
```bash
npm install
npm run dev          # http://localhost:5173
npm run build
npm run check        # svelte-kit sync + svelte-check (type checking)
npm run lint         # prettier --check + eslint
npm run format       # prettier --write
```
There is no JS test runner configured — verification is via `check` (types) and `lint`.

### Python model pipeline (run from repo root, in order)
```bash
python src/download_hf_dataset.py           # pull raw train/validation/test CSVs
python src/preprocess_data.py               # clean text -> {split}_clean.csv
python src/extract_classical_features.py    # 3-layer features + TF-IDF -> {split}_features.csv, tfidf_vectorizer.pkl, feature_scaler.pkl
python src/extract_bert_embeddings.py       # frozen BERT [CLS] embeddings -> {split}_bert_embeddings.npy
python src/train_models.py                  # Ridge baseline
python src/train_classical_models.py        # XGBoost / Random Forest on fused features
python src/models/train_bert_regressor_colab.py   # fine-tuned BERT regressor (intended for Colab/GPU)
python src/evaluation/evaluate_all_models.py
python src/explain_models.py                # SHAP summary/bar plots
```
No pytest suite exists; scripts are run directly and validated by inspecting printed RMSE/MAE/R² and the generated plots/CSVs in `models/`. `notebooks/*.ipynb` (Colab) are used for GPU-heavy steps (BERT embedding extraction, BERT fine-tuning).

### Docker (production, targets Hugging Face Spaces)
```bash
docker build -t pandora .
docker run -p 7860:7860 pandora
```
Runs `supervisord` → NGINX (port 7860, reverse-proxies to FastAPI) + Redis + `uvicorn backend.main:app` (port 8000), per `supervisord.conf`/`nginx.conf`.

## Architecture

### Inference request path (the part most likely to need changes)
`frontend/src/routes/+page.svelte` → `POST /predict` → `backend/main.py`:
1. Hash `(model, text)` with MD5; check Bloom filter + Redis cache (skip if disabled).
2. spaCy (`en_core_web_sm`) lemmatizes the input.
3. Three classical feature layers are computed by `src/extract_classical_features.py`:
   - **Layer 1 (semantic)** `compute_layer1_semantic` — NRC lexicon emotion frequencies + polarity, requires `load_nrc_lexicon`.
   - **Layer 2 (lexical)** `compute_layer2_lexical` — type-token ratio, word count.
   - **Layer 3 (behavioral)** `compute_layer3_behavioral` — spaCy POS ratios, punctuation/pronoun/all-caps ratios.
4. Model dispatch on `req.model`:
   - `"Fine-Tuned BERT"` → `src/models/bert_regressor.py` (`BertRegressorModel`, loaded from `models/bert_regressor_best.pt`) runs directly on tokenized text; SHAP token attributions come from `src/explainability/shap_bert_tokens.py`.
   - Everything else (`"Ridge Regression"`, `"XGBoost"`, `"Random Forest"`) → TF-IDF (`models/tfidf_vectorizer.pkl`) + the three classical layers are concatenated, scaled with `models/feature_scaler.pkl`, then passed to the corresponding `.pkl` model. XGBoost/RF return placeholder SHAP tokens (not real explanations) and 404 gracefully if their `.pkl` is missing from the deployed image.
5. A hand-tuned **calibration layer** rescales raw scores before clamping to `[0, 99]` — this exists because the training data (Pandora Personality Dataset) is heavily skewed toward introverted scores, so raw model outputs saturate well below 100. BERT and classical models use different calibration breakpoints; check this block (`backend/main.py`, "Probability Calibration Layer") before changing scoring behavior.
6. The response also derives an implicit **Big Five radar** (Agreeableness/Openness/Conscientiousness/Neuroticism) from Layer 1/3 features using hand-picked linear formulas — not model output, just heuristic mappings for the UI's radar chart.

### Feature/model artifact contract
Everything in `models/*.pkl` / `*.pt` must stay column-order-compatible with what `src/extract_classical_features.py` produces at train time (`{split}_features.csv`: Layer 1 + Layer 2 + Layer 3 + 2000 TF-IDF columns, in that order, then scaled). `backend/main.py` reconstructs this exact same order at inference time — if the training pipeline's feature order changes, the scaler/model pickles must be regenerated together, or predictions will silently be wrong (no shape/name validation happens).

### Frontend
Single-route SvelteKit 5 app (`frontend/src/routes/+page.svelte`, ~1000 lines) covering the whole UI: text input, model selector, score gauge, SHAP token highlighting, Big Five radar (via `chart.js`/`svelte-chartjs`), and client-side PDF export. `frontend/src/lib/components/ui/*` are small shadcn-style primitives (button/card/input/textarea) built with `tailwind-variants`. Styling is Tailwind CSS v4 (`@tailwindcss/vite`), no separate config file — theme lives in CSS. The frontend calls the backend via a hardcoded `API_URL` in `+page.svelte` — update it when pointing at a different backend deployment (local vs. Hugging Face Space).

### Legacy/parallel entry points
- `streamlit_app.py` (~1200 lines) is an older/alternate Streamlit UI for the same models; not part of the FastAPI+SvelteKit production path but may still be used for quick local demos (`.streamlit/config.toml` sets its theme).
- `src/predict_use_case.py` calls `compute_linguistic_features`/`compute_emotional_features` from `extract_classical_features.py` — those function names no longer exist there (renamed to the Layer 1/2/3 functions above), so this script is stale and will fail as-is.
- `src/run_experiments.py`, `train_ml_local.py`, `src/retrain_xgboost_local.py`, `src/retrain_rf_local.py` are variations on training XGBoost/RF on fused (classical + frozen-BERT-embedding) features, built for different iterations of local vs. Colab-trained data; check which `{split}_features.csv`/`{split}_bert_embeddings.npy` are present locally before running one.
- `src/preprocessing/augment_gemma3_colab.py` is a Colab-oriented data augmentation script (LLM-based augmentation via Gemma3) producing `train_augmented.csv`, which `extract_classical_features.py` prefers over `train_clean.csv` when present.

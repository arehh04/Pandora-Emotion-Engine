# Work Breakdown Structure (WBS)
**Project:** Extraversion Prediction from Written Text Using Machine Learning and Natural Language Processing (ExtraVerse)

8 phases, each with **one primary Deliverable**, decomposed as Phase → Deliverable → Work Package (= numbered Activities) → Task → Subtask. Every Task specifies Inputs / Outputs / Files / Expected Deliverable / Dependencies.

## Phase → RO Map

| Phase | RO | Deliverable |
|---|---|---|
| 1. Preliminary Study | RO I | Research Framework |
| 2. Data Collection & Preprocessing | RO I | Cleaned Dataset |
| 3. Feature Engineering | RO I | Feature Representations |
| 4. Model Development | RO II | Developed Ridge Regression & BERT Regression Models |
| 5. Model Evaluation | RO II / RO III | Evaluation Results |
| 6. Explainability Analysis | RO III | SHAP Results |
| 7. System Development | RO III | ExtraVerse System |
| 8. Documentation | **All ROs (Project-wide)** | Complete Thesis Document and Slides |

> **Note on Phase 8:** Documentation is not scoped exclusively to RO III. It consolidates findings from RO I (feature engineering), RO II (model development/evaluation), and RO III (explainable system) into the thesis. It is treated as a project-wide closing phase, not an RO III deliverable.

---

## 1.0 Phase 1 — Preliminary Study
**1.1 Deliverable: Research Framework**

### 1.1.1 Work Package — Review personality prediction studies
**1.1.1.1 Task — Survey text-based personality prediction literature**
| Field | Detail |
|---|---|
| Inputs | Scholarly databases (Scopus, IEEE, ACM), search terms: "Big Five prediction text", "Extraversion regression NLP" |
| Outputs | Annotated bibliography; accuracy-inconsistency evidence table (P1) |
| Files | `docs/literature/personality_prediction_review.xlsx` |
| Deliverable | Literature review input to Research Framework |
| Dependencies | None (project start) |

### 1.1.2 Work Package — Review NLP techniques
**1.1.2.1 Task — Survey linguistic, emotional, and semantic feature-extraction techniques**
| Field | Detail |
|---|---|
| Inputs | Papers on TF-IDF, POS-based linguistic features, NRC lexicon, BERT embeddings |
| Outputs | Technique comparison table (feature type vs. reported effectiveness) |
| Files | `docs/literature/nlp_techniques_review.xlsx` |
| Deliverable | NLP-technique justification for Phase 3 design |
| Dependencies | 1.1.1.1 |

**1.1.2.2 Task — Survey Ridge Regression and BERT Regression modeling approaches**
| Field | Detail |
|---|---|
| Inputs | Papers benchmarking regularized linear models vs. fine-tuned transformers on continuous trait scores |
| Outputs | Model-approach comparison table |
| Files | `docs/literature/model_approach_review.xlsx` |
| Deliverable | Modeling justification for Phase 4 |
| Dependencies | 1.1.1.1 |

**1.1.2.3 Task — Survey SHAP explainability for NLP/personality models**
| Field | Detail |
|---|---|
| Inputs | Papers on SHAP (LinearExplainer, Partition/Text explainer for transformers) |
| Outputs | Explainability-technique summary |
| Files | `docs/literature/xai_review.xlsx` |
| Deliverable | Explainability justification for Phase 6 |
| Dependencies | 1.1.1.1 |

### 1.1.3 Work Package — Identify relevant features
**1.1.3.1 Task — Define candidate feature categories (linguistic, emotional, semantic)**
| Field | Detail |
|---|---|
| Inputs | Outputs of 1.1.2.1, theoretical link between Extraversion and language use (e.g., social words, positive emotion, talkativeness markers) |
| Outputs | Finalized feature taxonomy: TF-IDF, linguistic (lexical/POS/readability), NRC emotion, BERT semantic embeddings |
| Files | `docs/framework/feature_taxonomy.md` |
| Deliverable | **Research Framework document** (theoretical model linking P1–P4 → RQ1–RQ3 → RO1–RO3 → feature taxonomy → planned methodology) |
| Dependencies | 1.1.1.1, 1.1.2.1–1.1.2.3 |

---

## 2.0 Phase 2 — Data Collection & Preprocessing
**2.1 Deliverable: Cleaned Dataset**

### 2.1.1 Work Package — Collect dataset
**2.1.1.1 Task — Acquire and organize Pandora Personality Dataset and NRC lexicon**
| Field | Detail |
|---|---|
| Inputs | Pandora Personality Dataset (pre-split), NRC Emotion Lexicon |
| Outputs | Verified local copies with confirmed schema |
| Files | `dataset/train_set.csv`, `dataset/val_set.csv`, `dataset/eval_set.csv`, `dataset/NRC-Emotion-Lexicon-Senselevel-v0.92.txt` |
| Deliverable | Confirmed raw data assets |
| Dependencies | 1.1.3.1 |

**2.1.1.2 Task — Profile dataset (EDA) and isolate Extraversion target**
| Field | Detail |
|---|---|
| Inputs | `train_set.csv`, `val_set.csv`, `eval_set.csv` (columns: `text, agreeableness, openness, conscientiousness, extraversion, neuroticism`) |
| Outputs | EDA report (row counts, missing values, text-length stats, Extraversion score 0–100 distribution); confirmation only `text` + `extraversion` are retained per scope |
| Files | `notebooks/01_eda.ipynb` → `outputs/eda_report.html` |
| Deliverable | Data profile for Chapter 3/4 |
| Dependencies | 2.1.1.1 |

### 2.1.2 Work Package — Clean text data
**2.1.2.1 Task — Remove noise and enforce English-only scope**
| Field | Detail |
|---|---|
| Inputs | Raw `text` column, all 3 splits |
| Outputs | Noise-free text (URLs, HTML artifacts, special characters stripped); non-English rows flagged/removed via language detection |
| Files | `src/preprocessing/clean_text.py` → `dataset/processed/{split}_clean.csv` |
| Deliverable | Cleaned text (raw-case variant kept for BERT) |
| Dependencies | 2.1.1.2 |

### 2.1.3 Work Package — Normalize text
**2.1.3.1 Task — Normalize, tokenize, and lemmatize text (for TF-IDF/linguistic/emotion features)**
| Field | Detail |
|---|---|
| Inputs | `dataset/processed/{split}_clean.csv` |
| Outputs | Lowercased, contraction-expanded, tokenized, lemmatized text; token lists cached |
| Files | `src/preprocessing/normalize_text.py` → `dataset/processed/{split}_tokens.pkl` |
| Deliverable | **Cleaned Dataset** (final: `{split}_clean.csv` for BERT + `{split}_tokens.pkl` for classical features) |
| Dependencies | 2.1.2.1 |

- Subtask 2.1.3.1.a: Handle negations explicitly (e.g., "not happy") since they invert emotion polarity used in Phase 3.
- Subtask 2.1.3.1.b: Verify train/val/eval token distributions remain consistent post-cleaning (no leakage/imbalance introduced).

---

## 3.0 Phase 3 — Feature Engineering
**3.1 Deliverable: Feature Representations**

### 3.1.1 Work Package — Extract TF-IDF features
**3.1.1.1 Task — Fit TF-IDF vectorizer on training text and transform all splits**
| Field | Detail |
|---|---|
| Inputs | `dataset/processed/{split}_tokens.pkl` |
| Outputs | TF-IDF sparse matrix (fit on train only, applied to val/eval to avoid leakage), vocabulary file |
| Files | `src/features/tfidf_features.py` → `models_saved/tfidf_vectorizer.pkl`, `dataset/processed/{split}_tfidf.npz` |
| Deliverable | TF-IDF feature matrix per split |
| Dependencies | 2.1.3.1 |

### 3.1.2 Work Package — Extract linguistic features
**3.1.2.1 Task — Compute lexical, POS, and readability features**
| Field | Detail |
|---|---|
| Inputs | `{split}_clean.csv`, `{split}_tokens.pkl` |
| Outputs | Word/sentence counts, type-token ratio, POS ratios, pronoun ratios, punctuation counts, Flesch readability score |
| Files | `src/features/linguistic_features.py` → `dataset/processed/{split}_linguistic.csv` |
| Deliverable | Linguistic feature matrix per split |
| Dependencies | 2.1.3.1 |

### 3.1.3 Work Package — Extract NRC Emotion Lexicon features
**3.1.3.1 Task — Parse NRC senselevel lexicon into a word-level lookup table**
| Field | Detail |
|---|---|
| Inputs | `dataset/NRC-Emotion-Lexicon-Senselevel-v0.92.txt` (`word--sense, emotion, 0/1`) |
| Outputs | Word-level lexicon (10 binary emotion/sentiment columns, sense-suffix stripped, OR-aggregated) |
| Files | `src/preprocessing/parse_nrc.py` → `dataset/processed/nrc_wordlevel.csv` |
| Deliverable | Reusable emotion lexicon table |
| Dependencies | None (independent of 2.1.x, runs in parallel) |

**3.1.3.2 Task — Compute per-document emotion-frequency vectors**
| Field | Detail |
|---|---|
| Inputs | `{split}_tokens.pkl`, `nrc_wordlevel.csv` |
| Outputs | 10-dim normalized emotion vector per document; lexicon coverage rate diagnostic |
| Files | `src/features/emotion_features.py` → `dataset/processed/{split}_emotion.csv` |
| Deliverable | Emotional feature matrix per split |
| Dependencies | 2.1.3.1, 3.1.3.1 |

### 3.1.4 Work Package — Generate BERT contextual embeddings
**3.1.4.1 Task — Extract pooled `[CLS]` embeddings via pretrained BERT**
| Field | Detail |
|---|---|
| Inputs | `{split}_clean.csv` (raw-case text), `bert-base-uncased` |
| Outputs | 768-dim semantic embedding per document |
| Files | `src/features/semantic_features.py` → `dataset/processed/{split}_bert_embeddings.npy` |
| Deliverable | Semantic feature matrix per split (also reused as BERT Regression input in Phase 4) |
| Dependencies | 2.1.2.1 |

### 3.1.5 Work Package — Construct integrated feature representation
**3.1.5.1 Task — Merge TF-IDF, linguistic, and emotional features into a unified classical feature matrix**
| Field | Detail |
|---|---|
| Inputs | `{split}_tfidf.npz`, `{split}_linguistic.csv`, `{split}_emotion.csv`, `extraversion` target |
| Outputs | Unified feature table for Ridge Regression |
| Files | `src/features/assemble_features.py` → `dataset/processed/{split}_features_final.csv` |
| Deliverable | **Feature Representations** — integrated classical feature set (for Ridge) + standalone BERT embeddings (for BERT Regression) |
| Dependencies | 3.1.1.1, 3.1.2.1, 3.1.3.2, 3.1.4.1 |

**3.1.5.2 Task — Scale/normalize integrated features for Ridge input**
| Field | Detail |
|---|---|
| Inputs | `{split}_features_final.csv` |
| Outputs | StandardScaler fitted on train, applied to val/eval |
| Files | `src/features/scale_features.py` → `models_saved/feature_scaler.pkl` |
| Deliverable | Scaled feature matrices ready for Phase 4 |
| Dependencies | 3.1.5.1 |

---

## 4.0 Phase 4 — Model Development
**4.1 Deliverable: Developed Ridge Regression and BERT Regression Models**

### 4.1.1 Work Package — Construct Ridge Regression model
**4.1.1.1 Task — Implement Ridge Regression pipeline**
| Field | Detail |
|---|---|
| Inputs | `dataset/processed/{split}_features_final.csv` (scaled), `models_saved/feature_scaler.pkl` |
| Outputs | Ridge model class/pipeline definition |
| Files | `src/models/ridge_model.py` |
| Deliverable | Ridge Regression pipeline skeleton |
| Dependencies | 3.1.5.2 |

### 4.1.2 Work Package — Design BERT Regression architecture
**4.1.2.1 Task — Build BERT + regression-head architecture**
| Field | Detail |
|---|---|
| Inputs | Pretrained `bert-base-uncased`, custom head (dropout → linear → 1 output scaled 0–100) |
| Outputs | BERT Regression model class |
| Files | `src/models/bert_regressor.py` |
| Deliverable | BERT Regression architecture module |
| Dependencies | None (parallel to 4.1.1.1) |

### 4.1.3 Work Package — Train baseline models
**4.1.3.1 Task — Train baseline Ridge Regression (default alpha=1.0)**
| Field | Detail |
|---|---|
| Inputs | `train_features_final.csv` |
| Outputs | Baseline model + baseline MAE/RMSE/R² on val_set |
| Files | `src/models/train_ridge.py` → `models_saved/ridge_baseline.pkl`, `outputs/ridge_baseline_metrics.json` |
| Deliverable | Baseline Ridge result for comparison |
| Dependencies | 4.1.1.1 |

**4.1.3.2 Task — Train baseline BERT Regression (default hyperparameters)**
| Field | Detail |
|---|---|
| Inputs | `train_clean.csv`, `val_clean.csv` |
| Outputs | Baseline fine-tuned model + baseline metrics |
| Files | `src/models/train_bert.py` → `models_saved/bert_baseline.pt`, `outputs/bert_baseline_metrics.json` |
| Deliverable | Baseline BERT result for comparison |
| Dependencies | 4.1.2.1 |

### 4.1.4 Work Package — Fine-tune Ridge Regression parameter (alpha)
**4.1.4.1 Task — Grid/cross-validation search over alpha**
| Field | Detail |
|---|---|
| Inputs | `train_features_final.csv`, `val_features_final.csv`, alpha range (e.g., 0.01–100, log scale) |
| Outputs | Best alpha, CV score curve |
| Files | `src/models/tune_ridge.py` → `outputs/ridge_alpha_search.csv` |
| Deliverable | Selected alpha hyperparameter |
| Dependencies | 4.1.3.1 |

### 4.1.5 Work Package — Fine-tune BERT learning rate
**4.1.5.1 Task — Learning-rate search (e.g., 1e-5, 2e-5, 3e-5, 5e-5)**
| Field | Detail |
|---|---|
| Inputs | `train_clean.csv`, `val_clean.csv`, candidate learning rates |
| Outputs | Best learning rate by lowest validation MAE |
| Files | `src/models/tune_bert.py` → `outputs/bert_lr_search.csv` |
| Deliverable | Selected learning rate |
| Dependencies | 4.1.3.2 |

### 4.1.6 Work Package — Hyperparameter tuning for both models
**4.1.6.1 Task — Full hyperparameter tuning for Ridge (alpha + solver)**
| Field | Detail |
|---|---|
| Inputs | Best alpha from 4.1.4.1, solver options |
| Outputs | Final tuned Ridge configuration |
| Files | `outputs/ridge_final_hyperparams.json` |
| Deliverable | Tuned Ridge configuration |
| Dependencies | 4.1.4.1 |

**4.1.6.2 Task — Full hyperparameter tuning for BERT (learning rate + batch size + epochs)**
| Field | Detail |
|---|---|
| Inputs | Best LR from 4.1.5.1, batch size/epoch candidates |
| Outputs | Final tuned BERT configuration; GPU/runtime log |
| Files | `outputs/bert_final_hyperparams.json`, `outputs/bert_training_log.csv` |
| Deliverable | Tuned BERT configuration |
| Dependencies | 4.1.5.1 |

### 4.1.7 Work Package — Select best-performing model configurations
**4.1.7.1 Task — Train final Ridge and BERT models using selected hyperparameters**
| Field | Detail |
|---|---|
| Inputs | `ridge_final_hyperparams.json`, `bert_final_hyperparams.json`, `train_features_final.csv`, `train_clean.csv` |
| Outputs | Final trained model artifacts |
| Files | `models_saved/ridge_model.pkl`, `models_saved/bert_regressor.pt` |
| Deliverable | **Developed Ridge Regression and BERT Regression models** |
| Dependencies | 4.1.6.1, 4.1.6.2 |

---

## 5.0 Phase 5 — Model Evaluation
**5.1 Deliverable: Evaluation Results**

### 5.1.1 Work Package — Evaluate using MAE, RMSE, and R²
**5.1.1.1 Task — Compute metrics for Ridge on val_set and eval_set**
| Field | Detail |
|---|---|
| Inputs | `models_saved/ridge_model.pkl`, `val_features_final.csv`, `eval_features_final.csv` |
| Outputs | MAE, RMSE, R² per split |
| Files | `src/evaluation/evaluate_models.py` → `outputs/ridge_metrics.json` |
| Deliverable | Ridge evaluation results |
| Dependencies | 4.1.7.1 |

**5.1.1.2 Task — Compute metrics for BERT Regression on val_set and eval_set**
| Field | Detail |
|---|---|
| Inputs | `models_saved/bert_regressor.pt`, `val_clean.csv`, `eval_clean.csv` |
| Outputs | MAE, RMSE, R² per split |
| Files | `src/evaluation/evaluate_models.py` → `outputs/bert_metrics.json` |
| Deliverable | BERT evaluation results |
| Dependencies | 4.1.7.1 |

### 5.1.2 Work Package — Compare model performance
**5.1.2.1 Task — Produce comparative table/chart and residual analysis**
| Field | Detail |
|---|---|
| Inputs | `ridge_metrics.json`, `bert_metrics.json` |
| Outputs | Side-by-side comparison table, residual plots, error-by-score-range breakdown (answers RQ2/P3) |
| Files | `notebooks/02_model_comparison.ipynb` → `outputs/figures/model_comparison.png`, `outputs/figures/residual_plots.png` |
| Deliverable | Comparative analysis report |
| Dependencies | 5.1.1.1, 5.1.1.2 |

### 5.1.3 Work Package — Select best model
**5.1.3.1 Task — Select champion model for deployment in ExtraVerse**
| Field | Detail |
|---|---|
| Inputs | Comparison report (5.1.2.1) |
| Outputs | Justified model selection (may keep both models toggleable in the UI rather than discarding one) |
| Files | `outputs/model_selection_decision.md` |
| Deliverable | **Evaluation Results** — final metrics + selected model(s) for Phase 6/7 |
| Dependencies | 5.1.2.1 |

---

## 6.0 Phase 6 — Explainability Analysis
**6.1 Deliverable: SHAP Results**

### 6.1.1 Work Package — Apply SHAP explanations
**6.1.1.1 Task — Compute SHAP values for Ridge Regression (LinearExplainer)**
| Field | Detail |
|---|---|
| Inputs | `models_saved/ridge_model.pkl`, `train_features_final.csv` (background), `eval_features_final.csv` |
| Outputs | SHAP value matrix |
| Files | `src/explainability/shap_ridge.py` → `outputs/shap_ridge_values.pkl` |
| Deliverable | Ridge SHAP values |
| Dependencies | 5.1.3.1 |

**6.1.1.2 Task — Compute SHAP values for BERT Regression (Partition/Text explainer)**
| Field | Detail |
|---|---|
| Inputs | `models_saved/bert_regressor.pt`, sample texts from `eval_clean.csv` |
| Outputs | Token-level SHAP attributions per sample |
| Files | `src/explainability/shap_bert.py` → `outputs/shap_bert_values.pkl` |
| Deliverable | BERT token-level SHAP values |
| Dependencies | 5.1.3.1 |

- Subtask 6.1.1.2.a: Benchmark SHAP-on-BERT runtime; if too slow for live inference, precompute/cache explanations for representative samples used by the Gradio demo.

### 6.1.2 Work Package — Identify important features
**6.1.2.1 Task — Generate global feature-importance ranking**
| Field | Detail |
|---|---|
| Inputs | `shap_ridge_values.pkl`, `shap_bert_values.pkl` |
| Outputs | Ranked top-N features/tokens driving Extraversion predictions |
| Files | `outputs/figures/shap_ridge_summary.png`, `outputs/figures/shap_bert_summary.png` |
| Deliverable | Feature-importance report |
| Dependencies | 6.1.1.1, 6.1.1.2 |

### 6.1.3 Work Package — Generate explanations
**6.1.3.1 Task — Produce per-sample explanation visualizations (force/bar/text plots)**
| Field | Detail |
|---|---|
| Inputs | SHAP value matrices |
| Outputs | Local explanation plots reusable by the Gradio UI |
| Files | `src/explainability/generate_explanations.py` → `outputs/figures/shap_local_examples/` |
| Deliverable | **SHAP Results** — global + local explanations, interpreted against Extraversion theory (answers RQ3) |
| Dependencies | 6.1.2.1 |

---

## 7.0 Phase 7 — System Development
**7.1 Deliverable: ExtraVerse System**

### 7.1.1 Work Package — Develop Gradio interface
**7.1.1.1 Task — Build UI layout (text input, model toggle, score gauge, SHAP panel)**
| Field | Detail |
|---|---|
| Inputs | UI requirements (input textbox, output score, explanation panel) |
| Outputs | Gradio interface skeleton |
| Files | `src/app/app.py` |
| Deliverable | Functional UI shell |
| Dependencies | None (can start once 5.1.3.1 model choice is known) |

**7.1.1.2 Task — Add input validation (English-only, non-empty text)**
| Field | Detail |
|---|---|
| Inputs | User text input |
| Outputs | Validation error handling |
| Files | `src/app/app.py` (validation block) |
| Deliverable | Robust input handling |
| Dependencies | 7.1.1.1 |

### 7.1.2 Work Package — Integrate prediction model
**7.1.2.1 Task — Build inference pipeline connecting features → model → prediction**
| Field | Detail |
|---|---|
| Inputs | Raw user text, `models_saved/{ridge_model.pkl, bert_regressor.pt, feature_scaler.pkl, tfidf_vectorizer.pkl}` |
| Outputs | `predict(text, model_choice) → extraversion_score` |
| Files | `src/app/inference_pipeline.py` |
| Deliverable | Working prediction backend |
| Dependencies | 5.1.3.1, 3.1.5.2 |

### 7.1.3 Work Package — Visualize results and explanations
**7.1.3.1 Task — Connect SHAP outputs to UI visualization panel**
| Field | Detail |
|---|---|
| Inputs | `inference_pipeline.py`, `src/explainability/generate_explanations.py` |
| Outputs | Live score display + SHAP plot rendered per user query |
| Files | `src/app/app.py` (visualization callback) |
| Deliverable | **ExtraVerse system** — deployed, explainable, interactive web app |
| Dependencies | 7.1.2.1, 6.1.3.1 |

**7.1.3.2 Task — Package and deploy app**
| Field | Detail |
|---|---|
| Inputs | `app.py`, `requirements.txt` |
| Outputs | Runnable local app / Hugging Face Spaces deployment |
| Files | `src/app/requirements.txt`, `src/app/README.md` |
| Deliverable | Deployed ExtraVerse instance |
| Dependencies | 7.1.3.1 |

---

## 8.0 Phase 8 — Documentation
**8.1 Deliverable: Complete Thesis Document and Slide**
*(Project-wide phase — consolidates RO I, RO II, and RO III findings; not scoped to RO III alone.)*

### 8.1.1 Work Package — Analyse findings
**8.1.1.1 Task — Consolidate results against P1–P4, RQ1–RQ3, RO1–RO3**
| Field | Detail |
|---|---|
| Inputs | `ridge_metrics.json`, `bert_metrics.json`, `model_comparison.png`, SHAP feature-importance report |
| Outputs | Findings-to-objectives traceability analysis |
| Files | `docs/thesis/findings_analysis.docx` |
| Deliverable | Analyzed findings for Chapter 4/5 |
| Dependencies | 5.1.3.1, 6.1.3.1 |

### 8.1.2 Work Package — Write thesis chapters
**8.1.2.1 Task — Write Chapters 1–2 (Introduction, Literature Review)**
| Field | Detail |
|---|---|
| Inputs | Phase 1 outputs |
| Outputs | Draft chapters |
| Files | `docs/thesis/chapter1_intro.docx`, `docs/thesis/chapter2_litreview.docx` |
| Deliverable | Thesis Ch.1–2 |
| Dependencies | 1.1.1.1–1.1.3.1 |

**8.1.2.2 Task — Write Chapter 3 (Methodology) from this WBS**
| Field | Detail |
|---|---|
| Inputs | This WBS, Phase 2–7 artifacts |
| Outputs | Full methodology chapter (pipeline diagram, dataset description, feature engineering, model training, evaluation, SHAP methodology, system design) |
| Files | `docs/thesis/chapter3_methodology.docx` |
| Deliverable | Thesis Ch.3 |
| Dependencies | Phases 2–7 |

**8.1.2.3 Task — Write Chapter 4 (Results & Discussion) and Chapter 5 (Conclusion)**
| Field | Detail |
|---|---|
| Inputs | 8.1.1.1 findings analysis |
| Outputs | Results/discussion chapter; conclusion revisiting P1–P4 |
| Files | `docs/thesis/chapter4_results.docx`, `docs/thesis/chapter5_conclusion.docx` |
| Deliverable | Thesis Ch.4–5 |
| Dependencies | 8.1.1.1 |

### 8.1.3 Work Package — Prepare presentation
**8.1.3.1 Task — Build defense slide deck and demo script for ExtraVerse**
| Field | Detail |
|---|---|
| Inputs | Chapter 3–5 content, ExtraVerse system (7.1.3.2) |
| Outputs | Slide deck with live/recorded demo |
| Files | `docs/presentation/FYP_slides.pptx` |
| Deliverable | **Complete Thesis Document and Slide** |
| Dependencies | 8.1.2.2, 8.1.2.3, 7.1.3.2 |

---

## Critical Path
```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6 → Phase 7 → Phase 8
```
Within Phase 3, the four extraction work packages (3.1.1 TF-IDF, 3.1.2 linguistic, 3.1.3 NRC emotion, 3.1.4 BERT embeddings) can run **in parallel** before merging at 3.1.5. Within Phase 4, Ridge (4.1.1/4.1.4/4.1.6.1) and BERT (4.1.2/4.1.5/4.1.6.2) tracks are independent and can also run in parallel.

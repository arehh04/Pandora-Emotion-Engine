# Design: LLM Agent Pivot for Extraversion Prediction (ExtraVerse v2)

**Status:** Approved by user, pending spec review sign-off
**Date:** 2026-07-09
**Supersedes methodology in:** `WBS.md`, `docs/thesis/Chapter3_Methodology.md`, `docs/thesis/Chapter4_Results_Interpretability.md` (those remain as the "prior work / baseline" record; this document defines the new system built alongside them)

## 1. Motivation and scope

The current system (documented in `docs/thesis/Chapter3_Methodology.md` / `Chapter4_Results_Interpretability.md`) predicts a continuous Extraversion score via a classical Feature Fusion pipeline (TF-IDF + NRC emotion lexicon + POS ratios + frozen BERT embeddings) into Ridge/XGBoost/Random Forest/fine-tuned-BERT regressors, explained post-hoc with SHAP. Best result: Random Forest, RMSE 28.22, R² 0.162.

The project is pivoting away from classical ML as the primary predictive mechanism toward an **agentic, LLM-orchestrated system**. Rather than discarding the prior work, the classical ML models, fuzzy logic, and retrieval-augmented reasoning become **tools an LLM agent calls and reasons over**, producing a tiered Extraversion assessment plus a narrative explanation. This is scoped as a full replacement of the "how do we predict" methodology (RO II/III), while RO I (data) is extended rather than replaced.

This is one cohesive project — it touches the research framework, data pipeline, modeling approach, explainability approach, and system architecture together, because they were tightly coupled in the original design (the WBS explicitly chains Feature Fusion → ML → SHAP → deployment).

### Goals
- Replace the classical ML/BERT-regressor as the *primary* predictor with an LLM agent that reasons using multiple evidence sources.
- Preserve quantitative, thesis-grade evaluation (RMSE/R² comparable to the existing Chapter 4 table) alongside a new tiered/qualitative output.
- Fix the known right-skew bias in the training data via LLM-paraphrase stratified augmentation (already partially built in `src/preprocessing/augment_gemma3_colab.py`), rather than SMOTE (inappropriate for raw text).
- Keep the existing FastAPI + SvelteKit system running, extended rather than rewritten.
- Produce a direct old-pipeline-vs-new-agent comparison for the thesis (RO II/III narrative).
- Use free/low-cost LLM access (OpenRouter) so the system remains runnable without ongoing API spend.

### Non-goals
- Not rebuilding the frontend from scratch — extend `+page.svelte` with an agent mode and trace panel.
- Not discarding the existing trained classical models — they're retrained on balanced data and reused as an agent tool (`ml_prior`).
- Not building a general-purpose multi-framework agent (no LangChain/LangGraph) — a small custom orchestration loop, for full transparency in the thesis defense.
- Not attempting real-time SMOTE or other numeric oversampling on text — augmentation is paraphrase-based only.

## 2. Architecture

```
                         ┌─────────────────────────────┐
                         │   SvelteKit Frontend (as-is)  │
                         │  + new "Agent Trace" panel    │
                         └───────────────┬───────────────┘
                                         │
                         ┌───────────────┴───────────────┐
                         │        FastAPI Backend         │
                         │  /predict        (legacy ML)   │
                         │  /predict-agent  (new)         │
                         └───────────────┬───────────────┘
                                         │
                    ┌────────────────────┴─────────────────────┐
                    │         Agent Orchestrator (new)          │
                    │  Custom ReAct-style loop, LLM via          │
                    │  OpenRouter (free-tier model + fallback    │
                    │  list), decides which tools to call,       │
                    │  then synthesizes final tier + narrative   │
                    └──┬──────────┬───────────┬───────────┬─────┘
                       │          │           │           │
                 ┌─────▼───┐ ┌───▼──────┐ ┌───▼───────┐ ┌─▼──────────┐
                 │ Classical│ │  Fuzzy   │ │  Legacy ML │ │    RAG     │
                 │ Feature  │ │  Logic   │ │  Models    │ │  Retrieval │
                 │ Extractor│ │  Engine  │ │ Ridge/XGB/ │ │ Theory +   │
                 │ (reused) │ │  (new)   │ │ RF (reused)│ │ Exemplars  │
                 └──────────┘ └──────────┘ └────────────┘ └────────────┘
```

**Why a custom orchestrator, not LangChain/LangGraph:** the thesis needs to explain exactly how the agent decides things. A hand-written loop (system prompt + tool schemas + a `while not done` loop parsing tool calls, ~150-200 lines) is fully inspectable and defensible in a viva, versus a framework abstracting the decision logic away. OpenRouter exposes an OpenAI-compatible `/chat/completions` API with tool-calling on most capable free-tier models (e.g. Llama 3.1/3.3, Qwen 2.5, DeepSeek variants), so no framework is required — the `openai` Python SDK works unmodified by pointing `base_url` at OpenRouter.

**Request flow:**
1. User submits text → orchestrator's system prompt frames the task (assess Extraversion using available tools).
2. Agent calls tools in whatever order it reasons is useful — it is not required to call all four.
3. Each tool returns structured JSON (not free text), keeping the agent's context clean and parseable.
4. Agent emits a final structured response: `{tier, tier_label, continuous_score_estimate, confidence, rationale, evidence[]}`.
5. Backend logs the full tool-call trace, exposed to the frontend's "Agent Trace" panel and reusable as thesis appendix examples.
6. Max 6 tool-call iterations before forced termination (cost/runaway-loop guard).

## 3. Data balancing (augmentation)

`src/preprocessing/augment_gemma3_colab.py` already implements stratified LLM-paraphrase augmentation: bins `extraversion` into Low/Medium/High (0–33/33–66/66–100), oversamples minority bins via Gemma-3-generated paraphrases until balanced, and writes `train_augmented.csv`. This stays as a one-time offline Colab batch job (bulk-generating thousands of paraphrases against a free-tier API rate limit would be impractical; Colab's free GPU is the right tool for bulk generation) and needs no architectural change — just needs to be run to completion if not already.

`train_augmented.csv` feeds two consumers:
1. **Legacy ML tool retraining** — Ridge/XGBoost/RF retrained on balanced data (classical features + BERT embeddings regenerated for the new paraphrased rows via `extract_bert_embeddings.py`). This removes the need for the ad-hoc "BERT is extremely biased toward low scores" calibration hack currently in `backend/main.py` — that hack compensated for training-data skew at inference time; balanced training data fixes it at the source.
2. **RAG exemplar corpus** — sampled evenly across the 6 presentation tiers (see below) so nearest-neighbor retrieval doesn't systematically return low-tier examples.

The held-out `test_set.csv` is **not** augmented/balanced — evaluation must reflect the real-world (skewed) distribution. This is why the evaluation plan (Section 6) reports both raw and tier-macro-averaged metrics.

## 4. Output scheme: tiered Extraversion

Actual data distribution (`data/train_set.csv`, `extraversion` column, N=16047): min 0, max 99, mean 35.0, median 24, right-skewed (long tail toward high scores, majority mass at low scores).

Final presentation bins (distinct from the coarse 3-bin augmentation stratification, which stays as-is):

| Tier | Range | Label |
|---|---|---|
| 1 | 0–10 | Reserved |
| 2 | 11–25 | Reflective |
| 3 | 26–45 | Balanced (Introspective) |
| 4 | 46–65 | Balanced (Sociable) |
| 5 | 66–85 | Outgoing |
| 6 | 86–99 | Highly Extraverted |

The agent and every tool produce a continuous 0–99 estimate internally (for RMSE/R² comparability with the existing Chapter 4 table) which is then mapped to one of these 6 tiers for user-facing display and for tiered classification metrics.

## 5. Agent tools

### 5.1 Classical Feature Extractor (reused, unchanged)
Wraps `src/extract_classical_features.py` Layer 1 (semantic/NRC), Layer 2 (lexical), Layer 3 (behavioral) functions. Used both as a direct evidence source and as the input signal source for the Fuzzy Logic Engine.

### 5.2 Fuzzy Logic Engine (new)
Implemented with `scikit-fuzzy`, Mamdani-style inference, centroid defuzzification.
- **Inputs** (fuzzified into Low/Medium/High via triangular membership functions), sourced from the Layer 1/3 classical features: positive/negative emotion polarity, exclamation+question ratio ("social energy"), verb ratio ("activity level"), 1st-person-plural-vs-singular pronoun ratio ("social orientation").
- **Rule base**: ~12-15 hand-authored rules grounded in extraversion facet theory (gregariousness, assertiveness, activity level, positive emotionality), e.g. "IF social_energy is HIGH AND social_orientation is PLURAL-leaning THEN extraversion is HIGH".
- **Output**: continuous 0–99 estimate + mapped tier + the list of fired rules (interpretability artifact, reused in both the UI's Agent Trace panel and the thesis's explainability chapter).
- Returns `{fuzzy_score, tier, fired_rules[]}` as one signal among several — not a final answer by itself.

### 5.3 Legacy ML Tool (retrained)
Ridge/XGBoost/RF retrained on `train_augmented.csv` (Section 3). The strongest performer after retraining (expected to remain XGBoost or Random Forest per the existing Chapter 4 finding) is exposed as a single `ml_prior` tool call: raw text → classical + BERT fused features → scaled → predict → `{score, tier}`. All three retrained artifacts are kept for the thesis comparison table even though only the best is wired into the live agent.

### 5.4 RAG Retrieval (new)
Two collections, both embedded with `sentence-transformers/all-MiniLM-L6-v2` (22M params, CPU-friendly; better suited to semantic similarity than reusing frozen BERT [CLS] vectors):
- **Theory collection**: ~15–20 short reference chunks on Extraversion/Big Five theory and linguistic markers of extraversion. Drafted from general domain knowledge; every claim is flagged with a citation placeholder to be verified against the project's actual literature review sources before use in the thesis — no citation is to be presented as verified until checked.
- **Exemplar collection**: a few hundred texts sampled evenly across the 6 tiers from `train_augmented.csv`, each with its known score, embedded once and cached (`.npy` + metadata CSV).
- Retrieval is simple in-memory cosine-similarity top-k (no FAISS/Chroma needed at this scale) — keeps the stack simple and easy to explain in the methodology chapter.

## 6. Evaluation plan

Run against the same held-out `test_set.csv` (natural, unbalanced distribution):

- **Continuous metrics** (RMSE/MAE/R²): old Ridge/XGBoost/RF vs new agent's `continuous_score_estimate` — same table format as the existing Chapter 4, giving a direct before/after comparison.
- **Tiered metrics**: accuracy, macro-F1, weighted Cohen's kappa (rewards near-miss tier errors over wild misses) across the 6 tiers, plus a confusion matrix.
- **Ablation study**: LLM-only → LLM+Fuzzy → LLM+Fuzzy+ML → full agent (+RAG). Demonstrates which tool actually improves results, supporting the RO II narrative with evidence rather than assertion.
- **Qualitative rationale faithfulness check**: spot-check a sample of narratives — does the stated rationale match the tool evidence actually returned, or does the LLM assert reasoning unsupported by its own tool calls? This becomes the RO III interpretability angle, replacing the SHAP analysis (whose own Chapter 4 conclusion noted BERT-embedding SHAP could "hallucinate" spurious attributions).
- **Cost/latency**: tokens and response time per request — relevant given free-tier OpenRouter rate limits affect live-demo viability.

## 7. Research framework / WBS restructuring

DSR framework and 8-phase WBS shape are retained; Research Objectives are reframed:

- **RO I** (was: Feature Fusion data prep): Collect data, run stratified LLM-paraphrase augmentation to correct the right-skew (Section 3), and curate a hybrid knowledge base (psychology theory + labeled exemplars) for retrieval-grounded reasoning.
- **RO II** (was: train Ridge/XGBoost/BERT): Design and develop a multi-tool LLM agent — fuzzy inference engine, retrained statistical models as a calibration prior, RAG retrieval — that reasons over text to produce a tiered Extraversion assessment.
- **RO III** (was: SHAP + deploy): Evaluate the agent against the prior ML pipeline (ablation + tiered/continuous metrics) and deploy it as an interactive system, with explainability from fired fuzzy rules + cited RAG evidence + agent rationale.

WBS phase-by-phase impact:
- **Phase 1 (Preliminary Study)**: add literature review coverage for LLM agents, fuzzy logic in personality/text prediction, and RAG.
- **Phase 2 (Data Collection & Preprocessing)**: add work package to finish/validate the stratified-paraphrase augmentation run and confirm post-augmentation distribution balance.
- **Phase 3 (Feature/Knowledge Engineering)**: reframed to cover reuse of classical features as fuzzy inputs, fuzzy rule-base design, and RAG corpus curation (theory + exemplar embedding).
- **Phase 4 (Model/Agent Development)**: build the fuzzy engine, retrain the ML tool models, build the agent orchestrator (tool schemas, prompts, OpenRouter integration).
- **Phase 5 (Model Evaluation)**: add the ablation study and tiered-metric work packages (Section 6) alongside the existing RMSE/R² work package.
- **Phase 6 (Explainability Analysis)**: replace SHAP work packages with fired-fuzzy-rule-trace, RAG-citation, and rationale-faithfulness work packages.
- **Phase 7 (System Development)**: add the `/predict-agent` endpoint and the frontend Agent Trace panel; update deployment (new dependencies, env vars, fallback logic).
- **Phase 8 (Documentation)**: existing results retained as "prior work / baseline"; new comparison becomes the headline result in rewritten Chapter 3 (methodology) and Chapter 4 (results) sections.

`WBS.md` and the thesis chapters themselves are rewritten as implementation deliverables, not as part of this design document.

## 8. System integration, tech stack, resilience, testing

**New dependencies**: `scikit-fuzzy` (fuzzy engine), `sentence-transformers` (RAG embeddings), `openai` Python SDK pointed at OpenRouter's base URL (OpenAI-API-compatible, no bespoke HTTP client needed).

**New modules**:
- `src/agent/orchestrator.py` — the ReAct-style loop
- `src/agent/tools/` — wraps classical features, fuzzy engine, legacy ML models, RAG retrieval as callable tools with JSON schemas
- `src/agent/prompts.py` — system prompt and tool schema definitions
- `src/rag/build_corpus.py` — one-time script to embed theory docs + exemplars and cache to disk
- `backend/main.py` — new `/predict-agent` endpoint alongside the existing untouched `/predict`

**Configuration**: new env vars `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, plus an ordered fallback list of free models to try if the primary is rate-limited or unavailable.

**Resilience**:
- Each tool call returns a structured error object rather than raising, so the orchestrator continues reasoning with whatever evidence succeeded.
- Orchestrator caps at 6 tool-call iterations to bound cost and prevent runaway loops.
- If OpenRouter fails after retries (exponential backoff, small fixed attempt count), the backend falls back to serving the legacy `/predict` score with a `degraded: true` flag rather than failing the request outright.

**Testing**:
- Unit tests for the fuzzy engine: hand-computed input→tier cases covering each membership region.
- Unit tests for RAG retrieval: tiny fixture corpus with a known nearest-neighbor ordering.
- Integration test for the orchestrator with OpenRouter calls mocked (deterministic fake tool-call sequence), verifying loop termination, the iteration cap, and the final response schema.
- A separate on-demand batch evaluation script (real API calls against `test_set.csv`) producing the Section 6 metrics table — not part of routine CI, since it costs real API calls and time.

## 9. Open risks / things to validate during implementation

- Free-tier OpenRouter models vary in tool-calling reliability; the fallback model list and structured-JSON tool responses are the main mitigations, but actual reliability needs empirical validation early in implementation before committing to a specific model as primary.
- `scikit-fuzzy` is a lightly-maintained library; confirm it installs cleanly against the project's current `numpy`/`scipy` versions before relying on it (fallback: hand-roll the triangular membership + centroid defuzzification math, which is not large).
- RAG theory-collection citations must be verified against real literature before appearing in the thesis — treat initial drafts as placeholders, not final citations.
- Retraining Ridge/XGBoost/RF on augmented data requires regenerating BERT embeddings for the new paraphrased rows first (`extract_bert_embeddings.py` over `train_augmented.csv`), which is a non-trivial compute step to schedule (likely Colab, consistent with the existing embedding-extraction notebook).

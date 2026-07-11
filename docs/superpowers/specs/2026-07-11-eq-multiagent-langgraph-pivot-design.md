# EQ Multi-Agent (LangGraph) Pivot — Design Spec

## Context

The LLM-agent pivot (Plans 1-6, branches `llm-agent-pivot/*`) built a single-agent ReAct loop predicting a continuous Extraversion score (0-99), grounded by a hybrid ChromaDB+BM25 RAG store, evaluated against historical classical-ML baselines (Ridge/XGBoost/Random Forest). The real evaluation (n=15, DeepSeek v4-flash) showed the LLM agent underperforming classical ML on raw RMSE/R2 (`llm_rag`: RMSE 32.60, R2 0.036 vs. classical ML's RMSE 28-29, R2 0.10-0.16), though RAG grounding meaningfully improved over `llm_only` (RMSE 37.80, R2 -0.296).

Rather than continuing to chase RMSE parity with models trained directly on this exact regression target, this spec redirects the project: predict **Emotional Intelligence (EQ)**, not Extraversion, using the **Mayer-Salovey-Caruso (MSC) four-branch model** (Perceiving, Using, Understanding, Managing emotions) as the assessment framework, built as a **multi-agent LangGraph system** (4 specialist agents + 1 coordinator + 1 critic, with a bounded critique/re-assessment loop).

This is a full retarget of the LLM-agent work built in Plans 1-6, not an incremental change. Classical-ML comparison is dropped for this task (no retraining on EQ) — evaluation is LLM-configuration vs LLM-configuration (with/without RAG, with/without critique loop, with/without individual specialists), plus real-dataset-grounded per-branch accuracy and LLM-judge rubric scoring where no ground truth exists.

## Architecture

### Graph structure (LangGraph `StateGraph`)

Shared state: `{text, branch_results: {branch_name: {score, tier, evidence, trace}}, overall_verdict, critic_feedback, loop_count}`.

1. **Fan-out** (via `Send`): 4 specialist nodes run in parallel — `perceiving`, `using`, `understanding`, `managing`. Each specialist is a scoped ReAct tool-calling loop (same pattern as the current `orchestrator.py`, reusing `call_with_fallback`/`TOOL_SCHEMAS`/`dispatch_tool_call` conventions directly — **not** wrapped in LangChain's `ChatDeepSeek`), grounded by RAG retrieval filtered to that branch's corpus. Writes its branch score (0-100), tier, evidence, and its own tool-call trace into `branch_results`.
2. **Fan-in → Coordinator node**: waits for all 4 specialists, synthesizes an overall EQ score/tier + rationale, reconciling cross-branch disagreement (not a naive average).
3. **Critic node**: checks the 4 branch verdicts + overall synthesis for internal inconsistency. Outputs either "consistent" or specific branch(es) to re-assess with a reason.
4. **Conditional edge**: if the critic flags an inconsistency and `loop_count < 2`, route back to just the flagged specialist(s) with the critic's feedback appended to their next prompt, increment `loop_count`, then back through coordinator → critic. Otherwise → END with the current verdict. The cap guarantees termination; worst case is roughly 2x the base 5-call pipeline.

### Why LangGraph, and why not `langchain-deepseek`

LangGraph is adopted deliberately for its native fan-out/fan-in and cyclic-graph support (the critique loop is a real cyclic-graph use case, not just parallel dispatch). However, `langchain-deepseek`'s `ChatDeepSeek` has an **active, documented bug** ([langchain-ai/langchain#37174](https://github.com/langchain-ai/langchain/issues/37174), [#34166](https://github.com/langchain-ai/langchain/issues/34166)): `_get_request_payload()` drops `reasoning_content` on multi-turn tool-calling requests with DeepSeek's thinking mode enabled, causing 400 errors — exactly the DeepSeek `reasoning_effort: high` + `thinking: enabled` + tool-calling combination this project already relies on and has already correctly solved in the hand-rolled `openrouter_client.py` (which forwards raw message dicts verbatim). **Decision: use LangGraph purely as the graph/state/parallelism/cyclic-control-flow engine. LLM calls inside graph nodes go through the existing tested `call_with_fallback` httpx client, called directly as a plain Python function — no LangChain model wrapper.** New dependency is `langgraph` (and its `langchain-core` requirement) only; `langchain-deepseek`/`langchain-openai` are explicitly NOT added.

### Judge model (evaluation only, not part of the live graph)

To avoid self-evaluation bias, the LLM-judge evaluation layer routes to a **different model/provider** than the assessment-generating model: DeepSeek v4-flash generates specialist/coordinator assessments; an already-configured OpenRouter model (`nvidia/nemotron-3-*`) acts as judge.

## Optimization & Utility Reuse

- **Parallel specialists**: verified directly against the installed `langgraph` (not assumed) that a compiled `StateGraph`'s plain synchronous `.invoke()` already runs independent same-superstep nodes concurrently — latency is bounded by the slowest specialist + coordinator time, not the sum of all calls, with no manual threading needed.
- **Whole-result cache**: reuse `backend/main.py`'s existing `md5(model+text) → Bloom filter → Redis get/setex(86400s)` pattern verbatim, with a version tag baked into the hash key (e.g. `f"eq-v1::{text}"`) so a future prompt/graph change doesn't silently serve stale verdicts.
- **Prompt-prefix caching**: static MSC framework/instructions/tool schemas first in each specialist's system prompt, input text last, so DeepSeek's provider-side prefix caching discounts the repeated static tokens across all specialist + coordinator + critic calls.
- **Shared embedding**: embed the input text once (at graph entry, before fan-out) and pass the vector to all 4 specialists' retrieval calls, instead of each specialist re-embedding independently.
- **Per-branch model routing**: cheaper/faster model for more mechanical branches (e.g. Perceiving), the stronger reasoning-enabled model reserved for branches needing deeper inference (Understanding, Managing) — reuses the existing multi-model fallback list mechanism, just different lists per specialist node.
- **Richer ablations**: Plan 5's `enabled_tools`/metrics/faithfulness machinery generalizes to "drop specialist X" and "with/without critique loop" ablations.
- **Cost/token observability**: DeepSeek returns usage stats per call; surface per-specialist/coordinator/critic token cost in the frontend Agent Trace panel, and (see Observability section below) in LangSmith traces.

## Revision (post-Plan-2): Vector Store, Reranking, Knowledge Graph, Observability, Ingestion

The decisions below were made after Plans 1-2 were already built and reviewed (on ChromaDB), during a scope discussion of what belongs in this project vs. a full enterprise RAG reference architecture. They supersede the ChromaDB-specific wording earlier in this spec. **Plan 3 (Per-Branch RAG Tools) was written targeting ChromaDB but not yet executed — it must be revised to target LanceDB before implementation, not built on ChromaDB and migrated afterward.**

### Vector store: LanceDB, not ChromaDB

Switching the RAG backend from ChromaDB to **LanceDB**: embedded/serverless like ChromaDB (`lancedb.connect(path)`, no Docker, no server — preserves the "no infrastructure to run" property that made ChromaDB attractive), but with **native hybrid search** (vector + full-text, fused server-side) built in. This eliminates the hand-rolled BM25/RRF fusion code in `src/rag/hybrid_store.py` (`_tokenize`, `BM25Okapi`, the manual RRF score-merging) — LanceDB's own hybrid query API replaces it. Considered and rejected: Weaviate (same hybrid-search benefit, but requires a running service — loses the embedded property); Qdrant (local mode exists but its real strengths — sharding, quantization — don't matter at this data scale); Pinecone (cloud-only, reintroduces an external uptime/API-key dependency for a locally-run FYP); Supabase pgvector (also cloud-hosted by default, and would mean hand-rolling hybrid fusion *again* via Postgres full-text search — no net simplification, plus this project has no existing Postgres dependency to consolidate into).

**Cost of switching:** Plans 1, 2, and 6's already-built/reviewed retrieval code (`src/rag/hybrid_store.py`, `src/rag/chunking.py` reuse, the Extraversion-era corpus) targeted ChromaDB. This is a real migration for the EQ-pivot side (Plan 3 onward) — the Extraversion-era `/predict-agent` pipeline (Plans 1-6 on `llm-agent-pivot/*`) is **not** touched by this switch and keeps using ChromaDB, since it's a separate, already-shipped/in-review system.

### Reranking (cross-encoder)

Retrieval becomes two-stage: retrieve top-50 candidates via LanceDB hybrid search, then rerank with a cross-encoder model (e.g. `sentence-transformers`' `cross-encoder/ms-marco-MiniLM-L-6-v2`, already in the same library family as the bi-encoder embedder already used) down to the top-8 actually handed to the specialist. This directly targets retrieval quality for the thing RAG exists for here — calibration-exemplar relevance — at the cost of one extra (small, local, non-LLM) model call per retrieval.

### Metadata filtering

Exemplars/theory chunks already carry `tier`, `branch`, `source` metadata (Plan 1). LanceDB's query API supports filtering on this metadata directly (e.g., restrict retrieval to only `tier >= 5` exemplars when calibrating a high-scoring text), a real, low-cost addition once the metadata already exists.

### Knowledge Graph: Neo4j

A **full Neo4j graph database** stores relationships between MSC theory concepts — the 4 branches, their sub-concepts (from the 16-entry theory corpus), and cross-branch relationships (e.g. "Using depends on Perceiving"). This is real infrastructure (a running Neo4j instance, Cypher queries) rather than a lightweight in-process graph — an explicit choice for this project's scope, made after the lighter NetworkX-only alternative was presented and declined. Feeds the coordinator/critic's cross-branch reasoning as an additional grounding source alongside the vector/hybrid retrieval.

### Observability: LangSmith

Since the orchestrator is now LangGraph-based, LangSmith (LangChain's companion tracing platform, designed specifically for LangGraph graphs) is added for run tracing/observability — a natural, low-setup-cost pairing (env vars + API key, no new infrastructure to run) rather than the heavier full OpenTelemetry stack from the original enterprise reference diagram, which was explicitly declined as disproportionate to this project's scope.

### Ingestion pipeline (formalized, not expanded)

The corpus has exactly 3 real sources: `data/train_set.csv` (Pandora rows), the hand-written MSC theory JSON, and 4 HF-fetched external emotion datasets. Rather than building generic loaders for formats nothing here uses (PDF/DOCX/website/YouTube/OCR — explicitly declined as solving a problem this project doesn't have), the ingestion pipeline formalizes these 3 existing sources with explicit stages: load → clean/normalize → dedup → version-stamp the resulting corpus artifact (so a rebuild is reproducible and traceable to which source-data version produced it) → hand off to the chunk/embed/store stage already designed above.

### Proxy label enrichment: NRC lexicon

The overall/per-branch proxy EQ label (see Data Foundation below) is enriched with NRC Word-Emotion-Association-Lexicon features (emotion-word density, positive/negative ratio) computed directly from each Pandora row's own text, in addition to the existing Big-Five-trait-based weights. This grounds the proxy label in what the text itself expresses, not just the person's separately-measured personality traits — a stronger, more defensible proxy for the thesis. This is an **offline data-labeling input only** — it does not reintroduce a live classical-feature tool into the agent's runtime reasoning, honoring the earlier "no traditional tools in the live agent" decision.

## Data Foundation

### Ground truth discovery

`data/train_set.csv` (pre-feature-extraction stage) contains **real, labeled Big Five scores for all 5 traits** — `agreeableness, openness, conscientiousness, extraversion, neuroticism` — not just Extraversion. `train_clean.csv` only kept Extraversion because the prior project scoped to it; the other 4 traits were never discarded, just unused. This is a materially better foundation than heuristic lexicon-only proxies.

### Proxy overall EQ label

Derived from the real 5-trait Big Five labels via documented trait-EI correlations from the psychology literature (EQ correlates positively with Extraversion/Openness/Agreeableness/Conscientiousness, negatively with Neuroticism) — a weighted combination requiring an actual literature citation for the specific weights before use in the thesis (flagged the same way `theory_corpus.json` entries carry a `citation_needed` field).

### Per-branch proxy labels (weakest link — needs citation/validation)

MSC branches don't map onto Big Five traits as cleanly as an overall score does. Starting point, pending literature validation:
- **Perceiving** ~ Openness + (low) Neuroticism
- **Using** ~ Openness + Extraversion
- **Understanding** ~ Openness + Agreeableness
- **Managing** ~ (low) Neuroticism + Conscientiousness

### Real datasets, mapped per branch

| Branch | Real dataset | Fit |
|---|---|---|
| Perceiving | GoEmotions (58k Reddit comments, 27 emotion labels), ISEAR (7.6k sentences, 7 emotions), EmoBank (10k+, valence/arousal/dominance) | Strong — score = overlap between specialist's identified emotions and human-annotated labels |
| Understanding | EmpatheticDialogues (25k conversations with situation + emotion context) | Partial — causal situation→emotion linking |
| Using | none | Proxy label + LLM-judge only |
| Managing | none | Proxy label + LLM-judge only |

### RAG corpus

Replace the current 17-entry Extraversion theory corpus with an MSC theory corpus (~15-20 entries covering all 4 branches, same `id/topic/text/citation_needed` structure as `theory_corpus.py`) — already built as a 16-entry corpus in Plan 1. Exemplars per branch drawn from the real datasets where available (Perceiving, Understanding) and from `train_set.csv` scored via the proxy formula where not (Using, Managing) — stored in **LanceDB** (superseding the ChromaDB+BM25 wording above — see the Revision section), with cross-encoder reranking on top of hybrid retrieval, and a companion Neo4j graph capturing MSC concept relationships.

### Tier scheme

New EQ tier scheme, structurally like `tiers.py`'s current 6 bins, but thresholds derived from the actual proxy-score distribution on `train_set.csv` (percentile-based), not arbitrary cutoffs.

## Evaluation Plan

- **Perceiving**: multi-label F1 between the specialist's identified emotions and GoEmotions/ISEAR/EmoBank ground truth.
- **Understanding**: partial ground truth via EmpatheticDialogues situation→emotion pairs, supplemented by LLM-judge rubric.
- **Using / Managing**: LLM-judge rubric only, plus a weak correlation check against the proxy label.
- **Overall EQ**: RMSE/MAE/R2 and tier accuracy/macro-F1/kappa against the proxy label on a held-out split of `train_set.csv`, reusing Plan 5's `compute_regression_metrics`/`compute_tier_metrics` unchanged.
- **Retrieval quality** (new, per the Revision section's reranker/LanceDB addition): Recall@k, Hit@k, MRR, nDCG against a small hand-labeled or heuristically-derived relevance set, measured both pre- and post-reranking to justify the cross-encoder's added cost.
- **Ablations**: drop-one-specialist (marginal contribution to overall accuracy); with/without critique loop (does the reflection mechanism actually help, or just add cost); with/without reranking. No classical-ML retraining — comparisons stay LLM-configuration vs LLM-configuration.
- **Judge model**: a different model/provider (OpenRouter) than the generator (DeepSeek), to avoid self-evaluation bias.

## Error Handling

- **Partial specialist failure**: if 1-3 of 4 specialists fail, the coordinator proceeds with whichever branches succeeded, marks `degraded_branches: [...]`, and lowers overall confidence rather than failing the entire assessment.
- **Critic failure**: fails open — skip the critique loop, return the coordinator's original verdict unmodified. The critique loop is an enhancement, never a single point of failure.
- **Loop termination**: hard cap at 2 re-assessment rounds guarantees graph termination regardless of critic behavior.
- **Per-specialist timeouts**: reuse the existing `httpx` client-level `timeout=30.0` so one hanging call can't block the whole parallel batch indefinitely.

## Testing Strategy

- Node-level: LangGraph nodes are plain Python functions over a state dict — same testing approach as today (`httpx.MockTransport` per node), no real graph execution needed to unit-test node logic.
- Graph-level integration tests: run the compiled graph end-to-end with fully mocked LLM responses, including at least one test exercising the critique-loop branch and one exercising partial-specialist-failure degradation.
- Conditional-edge routing logic is the one genuinely new thing needing dedicated tests, since node-level tests alone don't catch a wrong routing condition.

## Reuse vs. Rebuild Inventory (from Plans 1-6)

**Reused as-is:** `call_with_fallback`/httpx client, chunking (`chunking.py`, reused for LanceDB records too), FastAPI endpoint pattern, `compute_regression_metrics`/`compute_tier_metrics`, `check_rationale_faithfulness` (generalized), frontend Agent Trace panel (extended for per-node/per-loop rendering). Note: `src/rag/hybrid_store.py` (ChromaDB+BM25+RRF) stays as-is for the Extraversion-era `/predict-agent` pipeline (Plans 1-6, unaffected by this revision) but is **not** reused by the EQ pivot past Plan 2 — see Revision section.

**Rebuilt:** `orchestrator.py` → LangGraph `StateGraph` definition with specialist/coordinator/critic nodes (done, Plan 2); `tool_schemas.py` → per-branch tool sets (Plan 3, being revised to target LanceDB); `tiers.py` → new EQ tier scheme (done, Plan 1); RAG corpus content (theory + exemplars, done Plan 1, storage engine revised); `run_comparison.py`'s ablation variants → per-specialist, critique-loop, and reranking ablations.

**New dependencies:** `langgraph` (+ its `langchain-core` requirement, done Plan 2); `lancedb` (replacing `chromadb`+`rank-bm25` for the EQ pivot only); a cross-encoder model via the already-present `sentence-transformers`; `neo4j` Python driver (+ a running Neo4j instance); `langsmith` (+ API key, no new infra). Explicitly NOT added: `langchain-deepseek`, `langchain-openai` (avoids the reasoning_content bug — see Architecture section); the full OpenTelemetry stack, generic multi-format document loaders (PDF/DOCX/website/YouTube/OCR), auth/rate-limiting/API-gateway layers, Postgres/Redis/S3 as additional stores (all explicitly declined as disproportionate to this project's scope — see Revision section).

## Out of Scope for This Spec

- Renaming/rebranding the project (currently "Pandora"/"ExtraVerse") to reflect EQ instead of Extraversion — a separate, lower-priority decision that doesn't block the technical rebuild and can be made later.
- Retraining classical-ML baselines on EQ — explicitly decided against; comparison stays LLM-configuration vs LLM-configuration.
- Acquiring/downloading the real datasets (GoEmotions, ISEAR, EmoBank, EmpatheticDialogues) — deferred to the implementation plan's data-foundation task.

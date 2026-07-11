"""Manual, real-data evaluation run: LLM agent (ablation variants) vs the
historical classical-ML baselines, using real DeepSeek/OpenRouter API calls.

This is the deferred "real run" described in Plan 5
(docs/superpowers/plans/2026-07-11-evaluation-harness.md) -- it makes real,
billed LLM API calls and takes real wall-clock time (each sample can take
tens of seconds per variant, more with reasoning enabled and multiple tool
calls), so it is intentionally NOT part of the automated pytest suite.

Usage:
    ./.venv/Scripts/python.exe -m src.evaluation.run_real_evaluation --n-samples 20
"""
import argparse
import json
import os
import time

import pandas as pd
from dotenv import load_dotenv

from src.agent.context import load_rag_context
from src.agent.openrouter_client import DEEPSEEK_BASE_URL, build_client
from src.evaluation.run_comparison import (
    ABLATION_VARIANTS,
    HISTORICAL_BASELINES,
    make_run_agent_predict_fn,
    run_evaluation,
    summarize_evaluation,
)

load_dotenv()


def build_real_context():
    rag = load_rag_context(os.path.join("data", "rag", "chroma"))
    if rag is None:
        print("Note: RAG corpus not built yet -- run `python -m src.rag.build_corpus` first.")
    return {"rag": rag}


def build_real_client_and_models():
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if deepseek_key:
        client = build_client(deepseek_key, base_url=os.environ.get("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL))
        models_env = os.environ.get("DEEPSEEK_MODELS", "deepseek-v4-flash")
        models = [m.strip() for m in models_env.split(",") if m.strip()]
        extra_params = {"reasoning_effort": "high", "thinking": {"type": "enabled"}}
        return client, models, extra_params

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    client = build_client(api_key)
    models_env = os.environ.get("OPENROUTER_MODELS", "")
    models = [m.strip() for m in models_env.split(",") if m.strip()]
    return client, models, {"reasoning": {"enabled": True}}


def load_samples(n_samples, seed=42):
    df = pd.read_csv("data/test_set.csv")
    sample = df.sample(n=min(n_samples, len(df)), random_state=seed)
    return list(zip(sample["text"].tolist(), sample["extraversion"].tolist()))


def print_comparison_table(summary):
    print("\n=== COMPARISON TABLE (LLM agent variants vs historical classical ML) ===")
    header = f"{'Model':<16} {'RMSE':>8} {'MAE':>8} {'R2':>8} {'Acc':>8} {'MacroF1':>9} {'Kappa':>8} {'Faithful':>9}"
    print(header)
    print("-" * len(header))
    for name, m in HISTORICAL_BASELINES.items():
        print(f"{name:<16} {m['rmse']:>8.2f} {'--':>8} {m['r2']:>8.3f} {'--':>8} {'--':>9} {'--':>8} {'--':>9}")
    for variant_name in ABLATION_VARIANTS:
        m = summary[variant_name]
        print(
            f"{variant_name:<16} {m['rmse']:>8.2f} {m['mae']:>8.2f} {m['r2']:>8.3f} "
            f"{m['accuracy']:>8.3f} {m['macro_f1']:>9.3f} {m['weighted_kappa']:>8.3f} {m['faithfulness_rate']:>9.2%}"
        )


def main():
    parser = argparse.ArgumentParser(description="Run the real LLM-agent vs classical-ML evaluation")
    parser.add_argument("--n-samples", type=int, default=20, help="Number of test-set rows to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="evaluation_results.json")
    args = parser.parse_args()

    ctx = build_real_context()
    client, models, extra_params = build_real_client_and_models()
    if not models:
        raise SystemExit("No DEEPSEEK_API_KEY or OPENROUTER_API_KEY/OPENROUTER_MODELS configured in .env")

    predict_fn = make_run_agent_predict_fn(client, models, ctx, extra_params=extra_params)
    samples = load_samples(args.n_samples, seed=args.seed)

    total_calls = len(ABLATION_VARIANTS) * len(samples)
    print(f"Running {len(ABLATION_VARIANTS)} variants over {len(samples)} samples ({total_calls} agent runs total)...")
    print(f"Models: {models}")

    start = time.time()
    results = run_evaluation(predict_fn, samples, ABLATION_VARIANTS)
    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s ({elapsed / total_calls:.1f}s/run average)")

    summary = summarize_evaluation(results)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "raw_results": results}, f, indent=2)
    print(f"Saved detailed results to {args.output}")

    print_comparison_table(summary)


if __name__ == "__main__":
    main()

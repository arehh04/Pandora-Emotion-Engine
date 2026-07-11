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

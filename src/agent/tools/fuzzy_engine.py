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


def run_fuzzy_inference(features):
    inputs = compute_inputs(features)
    memberships = {var: fuzzify(val, INPUT_SETS[var]) for var, val in inputs.items()}
    fired, output_activations = evaluate_rules(memberships)
    score = defuzzify(output_activations)
    tier, label = assign_tier(score)
    return {"fuzzy_score": round(score, 2), "tier": tier, "tier_label": label, "fired_rules": fired}

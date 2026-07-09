"""Self-contained ML-prior tool: a small Ridge regressor trained directly on
classical_features.py's 8-feature contract, independent of the project's
other (currently unstable) model-training pipeline. Wiring this up to a
larger feature set or a different backing model is a later integration step.
"""
import numpy as np
from sklearn.linear_model import Ridge

from src.agent.tools.classical_features import extract_features_for_text
from src.tiers import assign_tier

FEATURE_ORDER = [
    "positive", "negative", "semantic_polarity",
    "behav_exclamation_ratio", "behav_question_ratio", "behav_verb_ratio",
    "behav_1st_sg_pronoun_ratio", "behav_1st_pl_pronoun_ratio",
]


def features_to_vector(features):
    return [features[name] for name in FEATURE_ORDER]


def train_ml_prior(feature_rows, scores, alpha=1.0):
    X = np.array([features_to_vector(f) for f in feature_rows])
    y = np.array(scores)
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    return model


def predict_ml_prior(text, nlp, nrc_dict, model):
    features = extract_features_for_text(text, nlp, nrc_dict)
    X = np.array([features_to_vector(features)])
    score = float(model.predict(X)[0])
    score = min(99.0, max(0.0, score))
    tier, label = assign_tier(score)
    return {"score": round(score, 1), "tier": tier, "tier_label": label}

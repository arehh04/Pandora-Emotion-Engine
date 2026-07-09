import os
import random

import spacy

from src.agent.tools.classical_features import extract_features_for_text, load_nrc_lexicon
from src.agent.tools.ml_prior import FEATURE_ORDER, features_to_vector, train_ml_prior, predict_ml_prior


def _load_real_nlp_and_lexicon():
    nlp = spacy.load("en_core_web_sm")
    nrc_path = os.path.join("data", "NRC-Emotion-Lexicon-Senselevel-v0.92.txt")
    nrc_dict = load_nrc_lexicon(nrc_path)
    return nlp, nrc_dict


def _synthetic_training_data():
    random.seed(42)
    feature_rows = []
    scores = []
    for _ in range(60):
        extraverted_signal = random.random()  # 0..1, drives both features and score
        features = {
            "positive": extraverted_signal * 0.4,
            "negative": (1 - extraverted_signal) * 0.3,
            "semantic_polarity": extraverted_signal * 2 - 1,
            "behav_exclamation_ratio": extraverted_signal * 0.2,
            "behav_question_ratio": 0.05,
            "behav_verb_ratio": 0.1 + extraverted_signal * 0.3,
            "behav_1st_sg_pronoun_ratio": (1 - extraverted_signal) * 0.2,
            "behav_1st_pl_pronoun_ratio": extraverted_signal * 0.2,
        }
        feature_rows.append(features)
        scores.append(extraverted_signal * 99.0)
    return feature_rows, scores


def test_features_to_vector_matches_feature_order():
    features = {name: float(i) for i, name in enumerate(FEATURE_ORDER)}

    vector = features_to_vector(features)

    assert vector == [float(i) for i in range(len(FEATURE_ORDER))]


def test_train_ml_prior_learns_positive_direction():
    feature_rows, scores = _synthetic_training_data()

    model = train_ml_prior(feature_rows, scores)

    low_signal = {
        "positive": 0.0, "negative": 0.3, "semantic_polarity": -1.0,
        "behav_exclamation_ratio": 0.0, "behav_question_ratio": 0.05,
        "behav_verb_ratio": 0.1, "behav_1st_sg_pronoun_ratio": 0.2, "behav_1st_pl_pronoun_ratio": 0.0,
    }
    high_signal = {
        "positive": 0.4, "negative": 0.0, "semantic_polarity": 1.0,
        "behav_exclamation_ratio": 0.2, "behav_question_ratio": 0.05,
        "behav_verb_ratio": 0.4, "behav_1st_sg_pronoun_ratio": 0.0, "behav_1st_pl_pronoun_ratio": 0.2,
    }

    low_pred = model.predict([features_to_vector(low_signal)])[0]
    high_pred = model.predict([features_to_vector(high_signal)])[0]

    assert high_pred > low_pred


def test_predict_ml_prior_returns_valid_score_and_tier():
    nlp, nrc_dict = _load_real_nlp_and_lexicon()
    feature_rows, scores = _synthetic_training_data()
    model = train_ml_prior(feature_rows, scores)

    result = predict_ml_prior(
        "I love going to parties, meeting new people, and being the center of attention!",
        nlp, nrc_dict, model,
    )

    assert set(result.keys()) == {"score", "tier", "tier_label"}
    assert 0.0 <= result["score"] <= 99.0
    assert 1 <= result["tier"] <= 6


def test_predict_ml_prior_clamps_score_into_valid_range():
    nlp, nrc_dict = _load_real_nlp_and_lexicon()
    feature_rows, scores = _synthetic_training_data()
    model = train_ml_prior(feature_rows, scores)

    result = predict_ml_prior("a", nlp, nrc_dict, model)

    assert 0.0 <= result["score"] <= 99.0

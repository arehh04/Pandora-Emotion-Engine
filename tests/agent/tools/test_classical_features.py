import os

import spacy

from src.agent.tools.classical_features import extract_features_for_text, load_nrc_lexicon


def _load_real_nlp_and_lexicon():
    nlp = spacy.load("en_core_web_sm")
    nrc_path = os.path.join("data", "NRC-Emotion-Lexicon-Senselevel-v0.92.txt")
    nrc_dict = load_nrc_lexicon(nrc_path)
    return nlp, nrc_dict


def test_extract_features_for_text_returns_expected_keys():
    nlp, nrc_dict = _load_real_nlp_and_lexicon()

    features = extract_features_for_text("I love parties and meeting new people!", nlp, nrc_dict)

    expected_keys = {
        "positive", "negative", "semantic_polarity",
        "behav_exclamation_ratio", "behav_question_ratio", "behav_verb_ratio",
        "behav_1st_sg_pronoun_ratio", "behav_1st_pl_pronoun_ratio",
    }
    assert set(features.keys()) == expected_keys
    assert isinstance(features["semantic_polarity"], float)


def test_extract_features_for_text_detects_exclamation():
    nlp, nrc_dict = _load_real_nlp_and_lexicon()

    excited = extract_features_for_text("This is amazing! I am so excited! Let's go!", nlp, nrc_dict)
    flat = extract_features_for_text("This is a report about quarterly numbers.", nlp, nrc_dict)

    assert excited["behav_exclamation_ratio"] > flat["behav_exclamation_ratio"]


def test_extract_features_for_text_detects_pronoun_orientation():
    nlp, nrc_dict = _load_real_nlp_and_lexicon()

    singular = extract_features_for_text("I went to my room by myself.", nlp, nrc_dict)
    plural = extract_features_for_text("We went to our house together.", nlp, nrc_dict)

    assert singular["behav_1st_sg_pronoun_ratio"] > singular["behav_1st_pl_pronoun_ratio"]
    assert plural["behav_1st_pl_pronoun_ratio"] > plural["behav_1st_sg_pronoun_ratio"]


def test_extract_features_for_text_empty_string_does_not_raise():
    nlp, nrc_dict = _load_real_nlp_and_lexicon()

    features = extract_features_for_text("", nlp, nrc_dict)

    assert features["behav_verb_ratio"] == 0.0
    assert features["behav_exclamation_ratio"] == 0.0

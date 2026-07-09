from src.agent.tools.fuzzy_engine import (
    trimf,
    fuzzify,
    compute_inputs,
    evaluate_rules,
    defuzzify,
    run_fuzzy_inference,
    INPUT_SETS,
    OUTPUT_SETS,
)


def test_trimf_peak_and_edges():
    # Triangle (0, 5, 10): 0 at edges, 1 at peak, linear between.
    assert trimf(0.0, (0, 5, 10)) == 0.0
    assert trimf(5.0, (0, 5, 10)) == 1.0
    assert trimf(10.0, (0, 5, 10)) == 0.0
    assert trimf(2.5, (0, 5, 10)) == 0.5
    assert trimf(7.5, (0, 5, 10)) == 0.5
    assert trimf(-5.0, (0, 5, 10)) == 0.0
    assert trimf(15.0, (0, 5, 10)) == 0.0


def test_fuzzify_returns_membership_per_set():
    memberships = fuzzify(-1.0, INPUT_SETS["polarity"])

    assert memberships["Negative"] == 1.0
    assert memberships["Neutral"] == 0.0
    assert memberships["Positive"] == 0.0


def test_compute_inputs_derives_orientation_and_energy():
    features = {
        "positive": 0.2,
        "negative": 0.0,
        "semantic_polarity": 0.9,
        "behav_exclamation_ratio": 0.1,
        "behav_question_ratio": 0.05,
        "behav_verb_ratio": 0.3,
        "behav_1st_sg_pronoun_ratio": 0.0,
        "behav_1st_pl_pronoun_ratio": 0.2,
    }

    inputs = compute_inputs(features)

    assert inputs["polarity"] == 0.9
    assert round(inputs["energy"], 5) == 0.15
    assert inputs["activity"] == 0.3
    assert inputs["orientation"] > 0.9  # plural pronouns dominate, no singular pronouns present


def test_compute_inputs_defaults_orientation_to_zero_with_no_pronouns():
    features = {
        "positive": 0.0,
        "negative": 0.0,
        "semantic_polarity": 0.0,
        "behav_exclamation_ratio": 0.0,
        "behav_question_ratio": 0.0,
        "behav_verb_ratio": 0.0,
        "behav_1st_sg_pronoun_ratio": 0.0,
        "behav_1st_pl_pronoun_ratio": 0.0,
    }

    inputs = compute_inputs(features)

    assert inputs["orientation"] == 0.0


def test_evaluate_rules_fires_matching_rule_with_min_strength():
    memberships = {
        "polarity": {"Negative": 0.0, "Neutral": 0.0, "Positive": 1.0},
        "energy": {"Low": 0.0, "Medium": 0.0, "High": 0.6},
        "activity": {"Low": 0.0, "Medium": 0.0, "High": 0.0},
        "orientation": {"Singular": 0.0, "Balanced": 0.0, "Plural": 0.0},
    }

    fired, output_activations = evaluate_rules(memberships)

    matching = [r for r in fired if r["antecedents"] == {"polarity": "Positive", "energy": "High"}]
    assert len(matching) == 1
    assert matching[0]["strength"] == 0.6  # min(1.0, 0.6)
    assert matching[0]["consequent"] == "High"
    assert output_activations["High"] >= 0.6


def test_evaluate_rules_no_match_produces_zero_activations():
    memberships = {
        "polarity": {"Negative": 0.0, "Neutral": 0.0, "Positive": 0.0},
        "energy": {"Low": 0.0, "Medium": 0.0, "High": 0.0},
        "activity": {"Low": 0.0, "Medium": 0.0, "High": 0.0},
        "orientation": {"Singular": 0.0, "Balanced": 0.0, "Plural": 0.0},
    }

    fired, output_activations = evaluate_rules(memberships)

    assert fired == []
    assert all(v == 0.0 for v in output_activations.values())


def test_defuzzify_no_activation_returns_zero():
    assert defuzzify({"Low": 0.0, "Medium": 0.0, "High": 0.0}) == 0.0


def test_defuzzify_pure_high_activation_yields_high_score():
    score = defuzzify({"Low": 0.0, "Medium": 0.0, "High": 1.0})
    assert score > 70.0


def test_defuzzify_pure_low_activation_yields_low_score():
    score = defuzzify({"Low": 1.0, "Medium": 0.0, "High": 0.0})
    assert score < 30.0


def test_run_fuzzy_inference_extraverted_leaning_text_scores_high():
    features = {
        "positive": 0.3,
        "negative": 0.0,
        "semantic_polarity": 0.95,
        "behav_exclamation_ratio": 0.2,
        "behav_question_ratio": 0.05,
        "behav_verb_ratio": 0.4,
        "behav_1st_sg_pronoun_ratio": 0.0,
        "behav_1st_pl_pronoun_ratio": 0.2,
    }

    result = run_fuzzy_inference(features)

    assert result["tier"] >= 5
    assert len(result["fired_rules"]) > 0


def test_run_fuzzy_inference_introverted_leaning_text_scores_low():
    features = {
        "positive": 0.0,
        "negative": 0.2,
        "semantic_polarity": -0.9,
        "behav_exclamation_ratio": 0.0,
        "behav_question_ratio": 0.0,
        "behav_verb_ratio": 0.05,
        "behav_1st_sg_pronoun_ratio": 0.15,
        "behav_1st_pl_pronoun_ratio": 0.0,
    }

    result = run_fuzzy_inference(features)

    assert result["tier"] <= 2
    assert len(result["fired_rules"]) > 0


def test_run_fuzzy_inference_result_shape():
    features = {
        "positive": 0.1, "negative": 0.1, "semantic_polarity": 0.0,
        "behav_exclamation_ratio": 0.0, "behav_question_ratio": 0.0,
        "behav_verb_ratio": 0.2, "behav_1st_sg_pronoun_ratio": 0.05, "behav_1st_pl_pronoun_ratio": 0.05,
    }

    result = run_fuzzy_inference(features)

    assert set(result.keys()) == {"fuzzy_score", "tier", "tier_label", "fired_rules"}
    assert isinstance(result["fuzzy_score"], float)
    assert isinstance(result["tier"], int)
    assert isinstance(result["tier_label"], str)

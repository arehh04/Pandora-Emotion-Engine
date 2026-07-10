from src.evaluation.faithfulness import check_rationale_faithfulness


def test_faithfulness_true_when_rationale_cites_a_trace_tier_number():
    agent_result = {
        "rationale": "The fuzzy logic engine returned tier 5, which strongly supports this assessment.",
        "trace": [
            {"tool": "fuzzy_logic_assessment", "arguments": {"text": "x"},
             "result": {"fuzzy_score": 70.0, "tier": 5, "tier_label": "Outgoing", "fired_rules": []}},
        ],
    }

    assert check_rationale_faithfulness(agent_result) is True


def test_faithfulness_true_when_rationale_mentions_tool_name():
    agent_result = {
        "rationale": "The ml prior model suggested a moderate score, but the direct content is clearer.",
        "trace": [
            {"tool": "ml_prior_assessment", "arguments": {"text": "x"},
             "result": {"score": 40.0, "tier": 3, "tier_label": "Balanced (Introspective)"}},
        ],
    }

    assert check_rationale_faithfulness(agent_result) is True


def test_faithfulness_false_when_rationale_is_unrelated_to_trace():
    agent_result = {
        "rationale": "Based on general writing style, this seems like a reserved individual.",
        "trace": [
            {"tool": "ml_prior_assessment", "arguments": {"text": "x"},
             "result": {"score": 90.0, "tier": 6, "tier_label": "Highly Extraverted"}},
        ],
    }

    assert check_rationale_faithfulness(agent_result) is False


def test_faithfulness_true_when_trace_is_empty():
    agent_result = {"rationale": "Purely based on the text itself.", "trace": []}

    assert check_rationale_faithfulness(agent_result) is True


def test_faithfulness_ignores_tool_calls_that_errored():
    agent_result = {
        "rationale": "No tool evidence was usable, so this is based on direct reading.",
        "trace": [
            {"tool": "retrieve_similar_exemplars", "arguments": {"text": "x"},
             "result": {"error": "RAG corpus is not available (not built yet)."}},
        ],
    }

    assert check_rationale_faithfulness(agent_result) is True

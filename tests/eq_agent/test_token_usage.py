from src.eq_agent.token_usage import extract_token_usage


def test_extract_token_usage_returns_the_three_fields():
    response = {
        "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165},
    }

    result = extract_token_usage(response)

    assert result == {"prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165}


def test_extract_token_usage_returns_none_when_usage_absent():
    response = {"choices": []}

    result = extract_token_usage(response)

    assert result is None


def test_extract_token_usage_handles_partial_usage_dict():
    response = {"usage": {"prompt_tokens": 100}}

    result = extract_token_usage(response)

    assert result == {"prompt_tokens": 100, "completion_tokens": None, "total_tokens": None}

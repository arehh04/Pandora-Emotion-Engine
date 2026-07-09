import json

import httpx

from src.agent.openrouter_client import build_client, call_chat_completion, call_with_fallback


def test_call_chat_completion_returns_parsed_json():
    def handler(request):
        assert request.headers["authorization"] == "Bearer fake-key"
        body = json.loads(request.content)
        assert body["model"] == "some-model"
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "hi"}}]
        })

    client = build_client("fake-key", transport=httpx.MockTransport(handler))

    result = call_chat_completion(client, "some-model", [{"role": "user", "content": "hello"}])

    assert result["choices"][0]["message"]["content"] == "hi"


def test_call_chat_completion_raises_on_http_error():
    def handler(request):
        return httpx.Response(500, json={"error": "server error"})

    client = build_client("fake-key", transport=httpx.MockTransport(handler))

    try:
        call_chat_completion(client, "some-model", [{"role": "user", "content": "hi"}])
        assert False, "expected an exception"
    except httpx.HTTPStatusError:
        pass


def test_call_with_fallback_tries_next_model_on_failure():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body["model"])
        if body["model"] == "model-a":
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "ok"}}]
        })

    client = build_client("fake-key", transport=httpx.MockTransport(handler))

    result = call_with_fallback(
        client, ["model-a", "model-b"], [{"role": "user", "content": "hi"}], max_retries_per_model=1
    )

    assert result["choices"][0]["message"]["content"] == "ok"
    assert calls == ["model-a", "model-b"]


def test_call_with_fallback_raises_when_every_model_fails():
    def handler(request):
        return httpx.Response(500, json={"error": "down"})

    client = build_client("fake-key", transport=httpx.MockTransport(handler))

    try:
        call_with_fallback(client, ["model-a", "model-b"], [{"role": "user", "content": "hi"}], max_retries_per_model=1)
        assert False, "expected an exception"
    except RuntimeError as e:
        assert "model-b" in str(e) or "All models failed" in str(e)

"""Thin httpx-based OpenRouter chat-completions client with model fallback.

Deliberately avoids the `openai` SDK: OpenRouter's endpoint is a plain
OpenAI-compatible REST call, and httpx is already a project dependency —
adding a whole SDK for one POST request isn't warranted, and a direct
implementation is easier to inspect end-to-end.
"""
import httpx

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def build_client(api_key, base_url=OPENROUTER_BASE_URL, transport=None, timeout=30.0):
    headers = {"Authorization": f"Bearer {api_key}"}
    return httpx.Client(base_url=base_url, headers=headers, timeout=timeout, transport=transport)


def call_chat_completion(client, model, messages, tools=None):
    payload = {"model": model, "messages": messages}
    if tools:
        payload["tools"] = tools
    response = client.post("/chat/completions", json=payload)
    response.raise_for_status()
    return response.json()


def call_with_fallback(client, models, messages, tools=None, max_retries_per_model=2):
    last_error = None
    for model in models:
        for _ in range(max_retries_per_model):
            try:
                return call_chat_completion(client, model, messages, tools)
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                last_error = e
    raise RuntimeError(f"All models failed ({models}). Last error: {last_error}")

"""Thin httpx-based chat-completions client with model fallback, for any
OpenAI-compatible endpoint (OpenRouter, DeepSeek, etc — same request/response
shape, different base_url/api_key/model).

Deliberately avoids the `openai` SDK: these are plain OpenAI-compatible REST
calls, and httpx is already a project dependency — adding a whole SDK for
one POST request isn't warranted, and a direct implementation is easier to
inspect end-to-end.
"""
import httpx

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def build_client(api_key, base_url=OPENROUTER_BASE_URL, transport=None, timeout=30.0):
    headers = {"Authorization": f"Bearer {api_key}"}
    return httpx.Client(base_url=base_url, headers=headers, timeout=timeout, transport=transport)


def call_chat_completion(client, model, messages, tools=None, extra_params=None):
    """extra_params is merged as top-level keys into the request body — e.g.
    OpenRouter's {"reasoning": {"enabled": True}} or DeepSeek's
    {"reasoning_effort": "high", "thinking": {"type": "enabled"}}.
    """
    payload = {"model": model, "messages": messages}
    if tools:
        payload["tools"] = tools
    if extra_params:
        payload.update(extra_params)
    response = client.post("/chat/completions", json=payload)
    response.raise_for_status()
    return response.json()


def call_with_fallback(client, models, messages, tools=None, max_retries_per_model=2, extra_params=None):
    last_error = None
    for model in models:
        for _ in range(max_retries_per_model):
            try:
                return call_chat_completion(client, model, messages, tools, extra_params=extra_params)
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                last_error = e
    raise RuntimeError(f"All models failed ({models}). Last error: {last_error}")

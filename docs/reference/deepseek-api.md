# DeepSeek API reference notes

Condensed from DeepSeek's official API docs, kept here since this project's agent orchestrator (`src/agent/orchestrator.py`, `backend/agent_router.py`) targets DeepSeek's `chat/completions` endpoint as its primary provider (`DEEPSEEK_API_KEY` in `.env`, see `backend/agent_router.py::get_agent_client_and_models`).

## Base setup

- `base_url = "https://api.deepseek.com"` (OpenAI-compatible `/chat/completions`).
- Model in use: `deepseek-v4-flash`, thinking/reasoning mode enabled, high effort:
  ```python
  response = client.chat.completions.create(
      model="deepseek-v4-flash",
      messages=messages,
      reasoning_effort="high",
      extra_body={"thinking": {"type": "enabled"}},
  )
  ```
  Over raw HTTP (what this project's `src/agent/openrouter_client.py` does), that's just two extra top-level JSON keys on the request body: `reasoning_effort` and `thinking`. This project passes them via `call_chat_completion(..., extra_params={"reasoning_effort": "high", "thinking": {"type": "enabled"}})`.
- Thinking mode does **not** support `temperature`/`top_p`/`presence_penalty`/`frequency_penalty` — setting them is silently ignored, not an error.

## Reasoning content and multi-turn forwarding (load-bearing rule)

Each response's assistant message carries `content` and `reasoning_content` (chain-of-thought) as sibling top-level fields.

**The rule that matters for tool-calling agents:** between two user messages, if the model performed a tool call, the intermediate assistant message's `reasoning_content` **must** be forwarded verbatim in every subsequent request in that turn — omitting it causes a 400 error. If no tool call happened, `reasoning_content` is optional/ignored if sent.

This project's orchestrator loop (`run_agent` in `src/agent/orchestrator.py`) already satisfies this by construction: it appends the raw `message` dict returned by the API straight into the `messages` list (`messages.append(message)`), never reconstructing a trimmed-down version — so `reasoning_content` rides along automatically whenever the API includes it. No special-casing was needed.

## Tool calls in thinking mode

Same shape as any OpenAI-compatible tool-calling flow: `tools` on the request, `tool_calls` on the response message, then append a `{"role": "tool", "tool_call_id": ..., "content": ...}` message per call and loop. Supported since DeepSeek-V3.2.

## JSON Output mode (not currently used here)

Set `response_format={"type": "json_object"}` and include the literal word "json" plus an example schema in the prompt. Can occasionally return empty content — DeepSeek notes this as a known rough edge, mitigated by prompt tweaks. This project uses tool-calling (`submit_assessment`) for structured output instead, which sidesteps this.

## Strict mode (Beta, not currently used here)

`base_url = ".../beta"`, each tool schema gets `"strict": true`, and DeepSeek validates the JSON Schema server-side. Supports `object`/`string`/`number`/`integer`/`boolean`/`array`/`enum`/`anyOf`/`$ref`+`$def`, with some parameter restrictions per type (e.g. strings can't use `minLength`/`maxLength`; objects must set `additionalProperties: false` and list every property as `required`). Worth revisiting if `TOOL_SCHEMAS` in `src/agent/tool_schemas.py` ever needs stricter validation guarantees than the current best-effort tool-calling gives.

## Context caching (automatic, no code changes needed)

Disk-based prefix caching is on by default. A request's cache prefix persists at: the end of user input, the end of model output, on detected common prefixes across requests, and at fixed token intervals for long content. `usage.prompt_cache_hit_tokens` / `usage.prompt_cache_miss_tokens` in the response report hit/miss counts. Only affects cost/latency, not correctness — nothing in this project depends on it.

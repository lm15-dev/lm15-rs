# README rows — OpenAI Chat Completions dialect (module 4 W3)

Rows for the parent to merge into `README.md`. Each row names the section
it belongs to.

## Status table (row 4, the Chat Completions column)

| 4 — dialects, request side: OpenAI Chat Completions (+ compat presets) | `POST /chat/completions` for `openai_chat`, `deepseek`, `zai`, `moonshotai`, `meta-chat`, `azure-chat`, `bedrock-chat`, `bedrock-mantle-chat`, `xai`; MAP-5..MAP-8 refusals; every `OpenAIChatCompat` knob | done — `--direction request` 112 / 112 pass for these providers (openai_chat 24, deepseek 10, zai 11, moonshotai 14, meta-chat 9, azure-chat 13, bedrock-chat 13, bedrock-mantle-chat 12, xai 6); `tests/dialect_openai_chat_corpus.rs` replays the same cases through `adapter_for` + `build_request` with the harness's comparison |

## Layout

- `src/dialects/openai_chat/` — the Chat Completions codec: `mod.rs`
  (the `Dialect` impl, headers, the xAI refusal table), `payload.rs` (the
  body in the reference's key order), `messages.rs` (the `messages`
  array), `cache.rs` (MAP-6 and the gpt-5.6+ class detector), `text.rs`
  (lossy text rendering, data URIs, the refusal constructor).

## Module 4 — the Chat Completions dialect

- **Body key order is the fixture's.** Bodies are `serde_json::Map`s
  filled in the reference's insertion order (`lm15/providers/openai_chat.py:377-560`):
  `model, messages, stream, stream_options, <max_tokens_field>,
  temperature, top_p, stop, logprobs, top_logprobs, tools, tool_choice,
  parallel_tool_calls, response_format, reasoning_format, <thinking
  fields>, prompt_cache_key, prompt_cache_retention, prompt_cache_options,
  provider, service_tier, <user_field>, store, extensions…`. This depends
  on `serde_json`'s `preserve_order` feature: the Bedrock doors sign the
  body bytes (SigV4), so the order is part of what those fixtures pin.
- **One compat, consulted at named points.** The binding's
  `OpenAIChatCompat` (the empty partial for a binding without one) is
  resolved per request after `for_model` applies the door's per-model
  overrides (Bedrock: `openai.gpt-oss` refuses forced tool choice and
  `json_schema`; `google.gemma` refuses forced tool choice on
  bedrock-runtime only). Every knob value is exercised by one unit test in
  `src/dialects/openai_chat/tests.rs`.
- **Reasoning (MAP-5, MAP-7).** `effort` goes out in the server's shape
  (`reasoning_effort`; OpenRouter `reasoning: {effort}`; the `deepseek`
  shape `thinking: {type: enabled}` + `reasoning_effort`; Moonshot's
  `kimi` shape sends the word alone and `thinking: {type: disabled}` for
  off; Qwen `enable_thinking`; `qwen_chat_template`). `off` sends the
  native disable. `thinking_budget` and `summary: concise|detailed` refuse
  (`UnsupportedFeatureError`); `summary: auto` becomes Groq's
  `reasoning_format: parsed` and is accepted silently elsewhere (MAP-7.7).
  `reasoning_efforts` allowlists refuse a word the server would swallow
  (Moonshot `medium`). A `thinking_format: none` server (ollama) refuses
  any `config.reasoning`: the wire has no dial, and an omitted dial is
  the silent paid no-op MAP-5 forbids.
- **Tool choice and structured output (MAP-8).** `auto`/`required`/`none`
  verbatim; one allowed name with `required` forces the function; any
  other allowlist is the nested `allowed_tools` form; builtin names in an
  allowlist refuse. `forced_tool_choice: reject` (Z.AI, gpt-oss on
  Bedrock, Gemma on bedrock-runtime) refuses every form but plain `auto`.
  `response_format` is `{type: json_object}` or `{type: json_schema,
  json_schema: {name (default "response"), schema, strict?}}`;
  `json_schema: reject` (Z.AI, gpt-oss on Bedrock) refuses the schema form.
- **xAI's refusal table** lives in the dialect and is keyed on the bound
  policy (`policy.provider == "xai"`), copied from
  `lm15/providers/xai.py:77-125`: reasoning off, `logprobs`, allowlist
  subsets other than one forced name, and a forced tool next to
  `response_format` all refuse before the wire. A second binding of the
  `xai` compat preset (say, a proxy) does not inherit the table: the
  table is a provider fact, the preset is a wire shape.
- **Caching (MAP-6)** follows `cache_control`: `openai` sends
  `prompt_cache_key`, `prompt_cache_retention: 24h`, the
  `prompt_cache_breakpoint` mark on the system block (`prefix: stable`)
  or on the last text block of message `prefix_until_index`, and
  `prompt_cache_options: {mode: explicit}` on the gpt-5.6+ class (with a
  mark, or alone for `mode: off`); `openai_implicit` (Meta, Moonshot)
  forwards only the key and retention; `none` sends nothing;
  `resource` refuses on both OpenAI controls. The gpt-5.6+ detector
  (`gpt-<major>.<minor>` ≥ 5.6) is a stated, rotting table.
- **Messages.** `system` and `developer` rows use the compat's
  `instruction_role`; user content is a bare string for one text part
  and an array of `text`/`image_url` blocks otherwise (URLs verbatim,
  inline data as a data URI, `detail` when set); assistant rows carry
  text, refusal text and — per `thinking_replay` — thinking text in
  `content` (`null` when empty), `reasoning_content` on the native replay
  (always present under `assistant_reasoning_content: include_empty`,
  DeepSeek's tool-loop requirement), and `tool_calls` with compact JSON
  `arguments`; tool rows carry `tool_call_id`, the result's text, and
  `name` under `tool_result_name: include`.

## Stated deviations

- **`assistant_after_tool_result: insert`** has no reference behaviour:
  the knob exists in `lm15/compat.py` but `lm15/providers/openai_chat.py`
  never reads it, and no preset sets it. This port inserts
  `{"role": "assistant", "content": ""}` after a run of tool rows when
  the next message is a user or developer turn (never at the end of the
  transcript). Unpinned; stated so the parent can strike it.
- **Assistant citations are dropped on replay.** A `CitationPart` in an
  assistant turn annotates text the wire already carries; the chat wire
  has no assistant citation slot. Refusing would break the flagship tool
  loop (`messages + response.message`) for every provider whose answer
  carried citations, so the annotation is dropped, stated here (port.md
  rule 4). The reference does the same (`openai_chat.py:258-270`).
- **A tool result with no text** is rendered as the reference's
  placeholder `[{"type": "image"}]` (the part types, Python's default
  JSON spacing) rather than refused: a refusal would break the loop for a
  tool that returned media. Stated, not absorbed.
- **`extensions` reserved names** (`prompt_caching`, `cache`, `compat`,
  `openai_compat`, `openai_chat_compat`) are not forwarded, as in the
  reference (`openai_chat.py:551-557`); they are lm15's former
  configuration names, not provider syntax.
- **Tool-call `arguments` are UTF-8** (`serde_json` compact). The
  reference's `json.dumps(separators=(",", ":"))` escapes non-ASCII as
  `\uXXXX` inside the string. The two differ only for non-ASCII tool
  inputs; no fixture pins one. UTF-8 is the model's own text.

## Divergences from the reference implementation

- `config.top_k` **refuses** on this dialect (`UnsupportedFeatureError`);
  the reference omits it silently (`openai_chat.py:387-395` reads
  `temperature`, `top_p`, `stop` only). port.md rule 4: a canonical field
  with no wire slot is a raise or an `extensions` door; `top_k` through
  `extensions` reaches vLLM/SGLang/ollama.
- `config.reasoning` on a `thinking_format: none` server **refuses**;
  the reference sends nothing for both on and off
  (`openai_chat.py:497-543`, no `none` branch). MAP-5.
- User `audio`/`video`/`document`/`binary` parts **refuse**; the
  reference renders them through `parts_to_text`, which yields `""`, and
  drops the empty block (`openai_chat.py:89-116`). Assistant media parts
  refuse; the reference ignores them (`:258-270`).
- An image addressed by `file_id` or `path` refuses with the pinned
  class; the reference raises an untyped `ValueError`
  (`common.py:73-76` `media_data_uri`).
- A `FunctionTool` without a description omits the key; the reference
  emits `"description": null` (`openai_chat.py:402-406`). No fixture has
  a description-less tool.
- `OpenAIChatCompat.extensions` is forwarded into the body before the
  request's `extensions`; the reference never reads it on the chat
  dialect (a dead field there).

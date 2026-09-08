# The dialects — mapping notes

Reference notes on how each codec maps the canonical types onto its wire,
moved here from the README on 2026-09-08 (the README keeps status,
quick start, the consolidated stated deviations and divergences; the
full pre-consolidation README is `docs/history/README-2026-09-08-pre-consolidation.md`).
Every table named below is copied as data from the reference with a
file:line citation in the source. Deviations and divergences are listed
once, in the README, not repeated here.

Contents: the module 4 skeleton (`wire::emit`), the response side and
stream assembly (module 5), then the four dialects: Anthropic, OpenAI
Responses, OpenAI Chat Completions, Gemini.

## The request skeleton (module 4)

- `AccessPolicy` is every column of `lm15/access.py` as `const` data
  with file:line citations; `tests/support_matrix_contract.rs` checks
  it against `spec/support-matrix.json` in both directions.
- Host settings resolve config → env (only when the caller passes an
  env map; the bare adapter reads none) → default; `region` and
  `resource` have no default; `location` defaults to `global`. A setting
  that lands in a hostname must be a DNS label (letters, digits, `-`);
  `project` and the model are percent-encoded when placed in a path.
- Base URL precedence: explicit `base_url`, then the host template
  rendered over the settings, then the policy's URL, then the compat
  preset's URL, then the dialect default.
- SigV4 signs the finished request: `authorization` and `x-api-key`
  never enter the signature; `host`, `x-amz-date` and
  `x-amz-security-token` are derived from the URL, the injected clock
  and the credential. `x-amz-content-sha256` is not added (S3-only).
- The shim's `build_request` accepts `api_key` or `credential`
  (`credential` wins), `now`, `settings` and `base_url` per PROTOCOL.md;
  it reads no environment.

## Response side and stream assembly (module 5)

- `Dialect::parse_response` and `Dialect::parse_stream_event` on each
  of the four codecs (`src/dialects/*/response.rs`), copied from the
  reference's `parse_response` / `parse_stream_events` with the tables as
  data: the provider-executed item sets (MAP-1), the finish-reason maps,
  the in-band error code tables, the Gemini candidate finish errors.
- The adapter is stateless per frame and may emit one end event per
  provider terminal frame; `stream::Coalescer` merges them into the one
  final end (MAP-3) and synthesizes the one leading start (MAP-4); D9's
  `provider_data` rank rule is `EndProviderData`.
- `stream::StreamAccumulator` is the MAP-9 assembly algorithm verbatim
  (slots, the fixed kind order, the `tool_call_<index>` correlator, the
  refusal with `partial`); `materialize_response` is the one-shot form.
- `Response.provider_data` is the wire body; `_lm15_unmapped` is attached
  when content could not be mapped, and the shim surfaces it as the
  protocol's `unmapped` canary.

## Anthropic

### Layout

- `src/dialects/anthropic/` — W1: `mod.rs` (the `Dialect` impl, headers,
  the `anthropic-beta` join, the refusal constructors), `body.rs` (the
  body in the reference's key order: caching marks, the reasoning plan
  per `thinking_format`, tool choice, `output_config`, extensions, the
  system prefix), `parts.rs` (messages and parts → content blocks,
  thinking replay per MAP-7 rules 8 and 11), `tables.rs` (the data copied
  from the reference with citations: builtin tool types, the adaptive
  model-class markers, the effort → budget table, the visible-token
  default, the API version and beta strings).

### Mapping

- `max_tokens` is required on the wire: `Config.max_tokens` or 1024
  (`_DEFAULT_ANTHROPIC_VISIBLE_TOKENS`, pinned by `anthropic.reasoning_off`).
  On the manual thinking class the wire value is `budget_tokens` +
  that visible share (MAP-7 rule 6; `anthropic.reasoning_budget`: 1000 +
  2048 = 3048); on the adaptive class `Config.max_tokens` is the total.
- Reasoning (MAP-7) by `AnthropicCompat.thinking_format`: `anthropic` —
  the model-class table (`anthropic_adaptive_class`, a substring table
  that rots; `extensions.thinking` overrides): adaptive class →
  `thinking: {type: adaptive}` + `output_config.effort` (`minimal` and
  `thinking_budget` raise), manual class → `thinking: {type: enabled,
  budget_tokens}` from `thinking_budget` or the grading table; `off`
  sends nothing (absence is the native off). `deepseek` — `off` MUST be
  sent as `{type: disabled}`, on is `{type: enabled}` + `output_config.effort`.
  `adaptive` — every model adaptive, `off` sent as `disabled` so the
  server refuses loudly. `effort` — `output_config.effort` alone, `off` as
  `disabled`. `reasoning_efforts` is a client-side allowlist for servers
  that swallow unknown words; `summary` `concise`/`detailed` raise,
  `auto` is satisfied silently.
- Caching (MAP-6): `config.cache` absent → nothing. Present, not `off`,
  `cache_control="anthropic"` → the system block is marked (`auto` and
  `prefix="stable"`), plus the last block of message N for
  `prefix_until_index=N` (clamped) or of the last message for
  `prefix="history"`; `retention="long"` adds `ttl: "1h"`. `mode="off"`
  places nothing (no write switch exists).
- Tool choice (MAP-8): `none`; one name + `required` → `{type: tool,
  name}` (server tools too); an allowlist naming every declared tool →
  `any`/`auto`; a proper subset raises; `parallel=false` →
  `disable_parallel_tool_use: true` (raises under
  `parallel_tool_calls="reject"`, where the server ignores it).
- Structured output (MAP-8): `json_schema` → `output_config.format
  {type: json_schema, schema}`; `name` and `strict` have no slot;
  `json_object` raises; `structured_output="reject"` raises.
- Thinking replay (MAP-7 rules 8, 11): `anthropic:redacted_thinking` →
  `redacted_thinking` with the blob; `anthropic:thinking_signature` with
  a non-empty signature → a signed `thinking` block; otherwise text
  (decision G), or an unsigned `thinking` block under
  `thinking_replay="unsigned"`.
- Headers: `anthropic-version: 2023-06-01`, the policy's static headers,
  one `anthropic-beta` joining the policy's betas with the dialect's
  (`code-execution-2025-05-22` when a `code_execution` builtin is
  offered). `content-type` and the credential are `emit`'s. The
  `claude-code` binding: `system_prefix` first in `system` as a text
  block, then the caller's system (string or blocks, marks kept).
- Refusals, all `UnsupportedFeatureError` unless named: `model_prefixes`
  mismatch (`UnsupportedModelError`), `store`, `logprobs`, `cache.key`
  (marks active), `cache.resource`, `retention="long"` on a "none" server,
  `sampling_params="reject"` with any of temperature/top_p/top_k,
  audio/video/binary parts, non-text `system` parts, an unreadable media
  `path` (`InvalidRequestError`).

## OpenAI Responses

### Mapping

- `src/dialects/openai_responses/`: `mod.rs` (the `Dialect`, the
  refusal constructors, `CODEX_BACKEND`), `payload.rs` (the body in the
  reference's order; `resolve_compat`), `input.rs` (messages → items:
  `input_text`/`input_image`/`input_audio`/`input_file`/`input_video`,
  `output_text`/`refusal`, `function_call`/`function_call_output`,
  reasoning-item replay), `tools.rs` (`_OPENAI_BUILTIN_MAP` as data,
  `tools`, the kind-aware `tool_choice`), `cache.rs` (the `gpt-5.6+`
  detector, breakpoint placement, MAP-6 fields).
- What `emit` does for the dialect: the credential header (`Bearer` on
  `openai`/`meta`/`moonshotai-responses`/`openai-codex`, `api-key` on
  `azure`), the base URL, the Azure host rewrite, `content-type`. The
  dialect sets `Content-Type` first and the policy's static headers after
  (`openai.py:541-549`); `endpoint = "responses"`; `model` for hosts
  that place it in the path.
- Reasoning (MAP-5/7): the word verbatim; `off` → `{"effort": "none"}`;
  `summary` verbatim on `responses_reasoning`, `concise`/`detailed`
  refuse elsewhere; `thinking_budget` refuses. Replay: an
  `openai:reasoning_item` state becomes `{"type": "reasoning", id?,
  encrypted_content?, "summary": [...]}` before the message it preceded
  (`summary` present even when empty); stateless thinking text replays as
  `output_text`; empty stateless thinking sends nothing.
- Caching (MAP-6), under `cache_control="openai"`: `mode="off"` →
  `prompt_cache_options: {mode: explicit}` on `gpt-5.6+` and nothing
  below; `key` → `prompt_cache_key`; `retention="long"` →
  `prompt_cache_retention: "24h"` on every class; `prefix="stable"` moves
  the system prompt into the first developer item with
  `prompt_cache_breakpoint`; `prefix_until_index` marks the last text
  block of that message (clamped) and refuses on an assistant/tool message
  or a message not ending in text; a placed mark on `gpt-5.6+` also sends
  `prompt_cache_options`; `prefix="history"` sends nothing; `resource`
  refuses. `openai_implicit` (Meta, Moonshot): key and retention only.
  `none`/`anthropic`: nothing.
- Tool choice (MAP-8): `none`/`auto`/`required` as strings; a single
  allowed name with `required` is the forced form (`{"type": "function",
  "name"}` or the hosted-tool `{"type": <wire type>}`); every other
  allowlist is `{"type": "allowed_tools", "mode", "tools"}`. `parallel`
  → `parallel_tool_calls`. Structured output: `text.format` with `name`
  defaulting to `"response"` and `strict` verbatim when present.
- Builtin tools: `builtin_tools="openai"` maps `web_search` →
  `web_search_preview`, `code_execution` → `code_interpreter`,
  `file_search`, `computer_use` → `computer_use_preview`; `"verbatim"`
  sends the canonical name; a name outside the table goes out verbatim
  (the server refuses loudly). `config` keys ride verbatim after `type`.
- Promoted knobs: `service_tier`, `user_id` → `safety_identifier`,
  `store` (false included), `logprobs` → `top_logprobs` +
  `include: ["message.output_text.logprobs"]`. INV-049 passthrough:
  every other `extensions` key verbatim (`previous_response_id`,
  `conversation`, `background`, `truncation`, `metadata`, `include`,
  `max_tool_calls`, `stream_options`, `context_management`, `user`).
- The `chatgpt-codex` backend (AUTH-10 branch 1): `instructions`
  defaults to the policy's prefix, `store: false`, `stream: true`, the
  max-tokens field removed; the policy's `OpenAI-Beta` and `originator`
  headers. `client_version` is consumed by `/models` only
  (`openai.py:1610-1613`), module 6. The `chatgpt-account-id` header is
  the account id bound by the router (from the stored file) or the
  token's own claim, added by `emit` at the first build.
- Function tools send `"description": null` when the canonical tool has
  none, as the reference does (`openai.py:849-856`); the schema accepts
  it and the bytes match.
- A tool result whose content renders to no text is sent as the
  reference's type list (`[{"type": "image"}]`, `json.dumps` spacing) —
  the wire's `function_call_output.output` is text only. Stated: media in
  tool results reaches the model as a type name, not as media.
- Compat override from the request: `extensions.openai_responses_compat`
  (or `openai_compat`, or `compat.openai_responses` / `compat.openai`)
  merges over the bound compat (`OpenAIResponsesCompat::merge`,
  `from_json`, `from_extensions`); an unknown knob value is a
  `ConfigurationError` (the reference's dataclass accepts any string).
  The profile layers of `lm15/profiles.py` are not carried.

## OpenAI Chat Completions

### Layout

- `src/dialects/openai_chat/` — the Chat Completions codec: `mod.rs`
  (the `Dialect` impl, headers, the xAI refusal table), `payload.rs` (the
  body in the reference's key order), `messages.rs` (the `messages`
  array), `cache.rs` (MAP-6 and the gpt-5.6+ class detector), `text.rs`
  (lossy text rendering, data URIs, the refusal constructor).

### Mapping

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

## Gemini

### Layout

- `src/dialects/gemini/mod.rs` — the `Dialect` impl, the model path,
  `tools`, the body assembly, `GEMINI_BUILTIN_TOOLS`, `gemini_level_class`.
- `src/dialects/gemini/config.rs` — `generationConfig` (`thinkingConfig`
  per MAP-7, `EFFORT_THINKING_BUDGETS` from `lm15/providers/common.py:386-393`,
  the `responseSchema`/`responseJsonSchema` rule), `toolConfig` (MAP-8),
  the MAP-6 cache plan.
- `src/dialects/gemini/contents.rs` — messages and parts, thought-signature
  replay (MAP-7.8), the text-only slots.

### Mapping

- Reasoning (MAP-7): absent → nothing; `off` → `thinkingBudget: 0` on the
  2.5 class, RAISE on the 3.x class; `thinking_budget` → `thinkingBudget`
  on both classes; else `thinkingLevel` verbatim on 3.x (`xhigh`/`max`
  RAISE) or the grading table on 2.5; `summary: auto` →
  `includeThoughts: true`, `concise`/`detailed` RAISE. The class is a
  model-name table (`gemini-3*`); the server 400s when it rots.
- Tool choice (MAP-8): `auto`/`required`/`none` → `AUTO`/`ANY`/`NONE`;
  `allowed` → `allowedFunctionNames` with `ANY` or `VALIDATED` (auto);
  builtin names in `allowed` RAISE; `parallel=false` RAISES.
- Structured output (INV-050): `responseMimeType: application/json`,
  plus `responseJsonSchema` when the schema contains
  `additionalProperties` anywhere, else `responseSchema`; `strict`
  satisfied, `name` dropped (a label).
- Caching (MAP-6): `off`, `auto`, `prefix`, `prefix_until_index` alone,
  `retention: short` → nothing; `key` and `retention: long` RAISE;
  `resource` → `cachedContent` and only the messages after
  `prefix_until_index`, with no `systemInstruction`/`tools`/`toolConfig`
  (MAP-6.7); a resource with no suffix message RAISES `InvalidRequestError`.
- `user_id` RAISES; `store` and `service_tier` map verbatim; `logprobs`
  → `responseLogprobs` (+ `logprobs` when `> 0`); `extensions` land at the
  top level verbatim (`safetySettings`, `labels`, …), replacing a built
  key of the same name; `extensions.output` → `responseModalities`.
- Hosts: `gemini` (`x-goog-api-key`), `vertex` (bearer; model in the
  path under `…/publishers/google/models/{model}`), `vertex-express`
  (`?key=`). The dialect sets `WireRequest.model` and
  `endpoint = "generateContent"`; `emit` and the host do the rest.

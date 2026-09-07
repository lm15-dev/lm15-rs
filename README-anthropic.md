# README rows — Anthropic Messages dialect (module 4 W1)

The parent merges these rows into `README.md`. Each block names the
section it belongs to.

## Status table (row 4, the `anthropic` part)

| 4 — dialects, request side: **Anthropic Messages** (`src/dialects/anthropic/`) | `POST /messages` for `anthropic`, `claude-code`, `deepseek-anthropic`, `meta-anthropic`, `moonshotai-anthropic` and the cloud doors (`aws-anthropic`, `bedrock-anthropic`, `azure-anthropic`, `vertex-anthropic` through `emit`'s host rewrites); MAP-5..8 refusals; `AnthropicCompat` presets | done — `--direction request`: anthropic 35/35, deepseek-anthropic 13/13, moonshotai-anthropic 13/13, meta-anthropic 9/9 pass (70 cases, 0 fail); `tests/dialect_anthropic_cases.rs` replays the same cases from the contract checkout |

## Stated deviations

- **Body key order is the reference's insertion order** (port.md
  rule 1 + the `preserve_order` deviation above): `model`, `messages`,
  `stream`, `max_tokens`, `system`, `temperature`, `top_p`, `top_k`,
  `stop_sequences`, `tools`, `tool_choice`, `thinking`, `output_config`,
  `service_tier`, `metadata`, then `extensions` verbatim (an existing key
  is replaced in place), then the policy `system_prefix` placed first in
  `system`. A SigV4 door (`aws-anthropic`, `bedrock-anthropic`) signs
  these bytes; the public-API fixtures compare structurally.
- **`ImagePart.detail` is not sent** (port.md rule 4): the Messages API
  has no resolution hint; the image is processed as-is. The reference
  and its Gemini adapter drop it the same way; it is a hint with no cost
  the caller can observe, not an instruction, so it is stated here
  rather than refused. The parent may reverse this.
- **A `ThinkingPart` with empty text and no `anthropic:*` state renders
  no block** (hidden thinking of another dialect, MAP-7 rule 11): the
  reference sends `{"type": "text", "text": ""}`, which the API refuses
  (empty text block). Nothing this wire can carry is dropped.
- **`ToolResultPart.name` is not sent**: the wire keys a result by
  `tool_use_id` alone. Same as the reference.
- **`AnthropicCompat.extensions` is carried as data and not read**: the
  reference (`lm15/providers/anthropic.py`) never reads it either; a
  server-level default body merge has no defined semantics yet.
- **The reserved `extensions` key `prompt_caching` never reaches the
  wire** (`lm15/providers/anthropic.py:735`; the reference's
  `docs/cookbooks/18-provider-passthrough.md` § Reserved keys): the
  pre-`CacheConfig` spelling, kept reserved so a request written for the
  reference does not 400 here. Every other key passes through verbatim
  (INV-049).

## Divergences from the reference implementation

- **Parts with no content block raise** (`lm15/providers/anthropic.py:448`
  renders audio, video and binary parts as `{"type": "text", "text": ""}`;
  `:457` does the same inside tool results; `:620` and `:462-464` join system
  parts and developer messages through `parts_to_text`, dropping media):
  this port raises `UnsupportedFeatureError` for audio/video/binary in
  messages, tool results and `system`, and for any non-text part in
  `system` (port.md rule 4). A developer message keeps its media blocks
  after the `[developer]\n…` text block; `system` parts become one text
  block each (the mark rides the last one) instead of a `\n`-joined string.
- **Inline media data is sent as its base64 payload**
  (`lm15/providers/common.py:233` sends `part.data` verbatim): a data-URI
  prefix or whitespace that INV-012 tolerates on input is stripped
  (`base64_payload`) so the wire gets what the API accepts.
- **`cache.resource` raises on every door of the wire** (MAP-6 rule 7;
  `lm15/providers/anthropic.py:536-548` raises only when marks are active,
  so a `cache_control="none"` server such as DeepSeek silently ignores a
  stored-cache id). No server on this wire has the resource tier.
- **`cache.retention="long"` raises on a `cache_control="none"` server**
  (MAP-6 rule 5 names the `ttl: "1h"` mechanism; `:525` gates `long_cache`
  silently). `cache.key` follows the reference: a refusal where marks are
  the mechanism, nothing on a server that caches implicitly (the
  chat-dialect rule for `prompt_cache_key` on "none" presets).
- **`anthropic-beta` values are joined without duplicates**
  (`lm15/providers/anthropic.py:392-404` appends; a policy that already
  lists `code-execution-2025-05-22` would repeat it). First occurrence
  wins, policy betas first, then the dialect's own.
- **A media `path` that cannot be read is `InvalidRequestError`**
  (`invalid_request`): the reference lets `OSError` escape. The request
  names something this process cannot read; no `Lm15Error` class fits
  better than the caller-bug class.

## Layout (append to the `src/dialects/` line)

- `src/dialects/anthropic/` — W1: `mod.rs` (the `Dialect` impl, headers,
  the `anthropic-beta` join, the refusal constructors), `body.rs` (the
  body in the reference's key order: caching marks, the reasoning plan
  per `thinking_format`, tool choice, `output_config`, extensions, the
  system prefix), `parts.rs` (messages and parts → content blocks,
  thinking replay per MAP-7 rules 8 and 11), `tables.rs` (the data copied
  from the reference with citations: builtin tool types, the adaptive
  model-class markers, the effort → budget table, the visible-token
  default, the API version and beta strings).

## Module 4 — the Anthropic dialect (new subsection)

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

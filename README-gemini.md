# README rows — Gemini dialect (module 4 W4)

The parent merges these into `README.md`. Each block names its section.

## Status (row 4 — dialects, request side; the Gemini column)

| Module | Contract surface | State |
|---|---|---|
| 4 — Gemini dialect, request side | `POST /models/{model}:generateContent` (`:streamGenerateContent?alt=sse` for a stream); `contents`, `systemInstruction`, `generationConfig` (incl. `thinkingConfig`, `responseMimeType` + `responseSchema`/`responseJsonSchema`, `responseLogprobs`), `toolConfig.functionCallingConfig`, `tools` (`functionDeclarations`, `googleSearch`, `codeExecution`, passthrough builtins), `cachedContent`, `store`, `serviceTier`, `extensions` passthrough; MAP-5..8 refusals; the `gemini`, `vertex` and `vertex-express` policy bindings | done — `--direction request`: gemini 34 pass / 0 fail (33 wire cases + the pinned `tool_choice_parallel_false` refusal); `serde` 115, `error` 84, `auth --auth-scope core` 26, `token` 34+9 unchanged. `tests/dialect_gemini_request.rs` replays every `cases/gemini/*.json` with a `canonical_request` through `adapter_for` with the harness's comparison; `tests/dialect_gemini_rules.rs` pins every refusal and corner below |

## Stated deviations (playbooks/port.md rule 8)

- **Body key order is the fixtures' order, not the reference's, where
  the two differ** (`src/dialects/gemini/mod.rs` `payload`): `contents`,
  `cachedContent`, `systemInstruction`, `generationConfig`, `toolConfig`,
  `tools`, `store`, `serviceTier`, then `extensions`. The reference
  inserts `tools` before `toolConfig` (`lm15/providers/gemini.py:759-763`);
  the migrated fixtures (`cases/gemini/tool_config_*.json`) record
  `toolConfig` first. No Gemini door signs its body, so the order is a
  family convention here, pinned by nothing; `serde_json` `preserve_order`
  keeps whichever order the builder inserts.
- **Integral floats take the integer form** (`generationConfig.temperature`,
  `topP`): a canonical `1.0` goes out as `1`
  (`lm15/providers/gemini.py:208-219`, live capture
  `cases/gemini/temperature.json`). A wire-dialect fact; the canonical
  field stays a float and the harness's `1 != 1.0` rule is what pins it.
- **`ImagePart.detail` has no Gemini slot and is not sent** — the same
  silent drop the reference and the Anthropic dialect make for a
  presentation hint; stated here instead of raised because every
  non-OpenAI wire drops it and a raise would make the hint unusable
  outside OpenAI.
- **A `path` media part is read at build time** and inlined as
  `inlineData` (the reference does the same,
  `lm15/providers/gemini.py:579`). An unreadable path is an
  `InvalidRequestError` (the reference lets the `OSError` escape untyped).

## Divergences from the reference implementation

Each is a place where playbooks/port.md rule 4 (no silent drops) or a
wire fact overrode the reference's control flow. None is pinned by a
fixture; each has a unit test in `tests/dialect_gemini_rules.rs`.

- `ToolResultPart.is_error = true` RAISES `UnsupportedFeatureError`
  (`functionResponse` has no error flag; the reference drops the flag,
  `lm15/providers/gemini.py:589-594`).
- Media parts in a text-only slot RAISE `UnsupportedFeatureError`:
  `systemInstruction` (text-only on the wire), a developer turn (rendered
  as one prefixed text part), a `functionResponse` (`{"result": <text>}`).
  The reference's `parts_to_text` (`lm15/providers/common.py:53-66`)
  drops them.
- A `functionResponse.name` with no `ToolResultPart.name` is looked up
  from the `ToolCallPart` with the same id earlier in the transcript
  (`Message::tool(&call.id, result)` carries no name); only when no such
  call exists does the reference's placeholder `"tool"` go out
  (`lm15/providers/gemini.py:591`).
- `functionDeclarations[].description` is omitted when absent; the
  reference sends `"description": null` (`lm15/providers/gemini.py:749`).
- `extensions.prompt_caching` is not filtered (`lm15/providers/gemini.py:792`
  drops it as a legacy key); it passes through like every other key and
  the server's 400 is the contract.
- `extensions.output` with a value other than `"image"`/`"audio"` RAISES
  `InvalidRequestError` (the reference ignores it,
  `lm15/providers/gemini.py:765-769`).
- `CacheConfig.resource` naming a full resource path
  (`projects/…/cachedContents/…`, the Vertex form) is kept verbatim; the
  reference prefixes `cachedContents/` a second time
  (`lm15/providers/gemini.py:1537-1538`).
- The MAP-8 tool-choice refusals (`parallel=false`, builtin forcing) fire
  even when a `cachedContent` reference makes `toolConfig` unsendable;
  the reference skips the whole tool-config path next to a cache
  (`lm15/providers/gemini.py:761`).
- The model class (`gemini_level_class`) is read from the wire model
  (`BuildContext.model`, the `provider:` prefix removed); the reference
  reads `request.model` (`lm15/providers/gemini.py:708`), which the
  router has already stripped on its path.

## Layout

- `src/dialects/gemini/mod.rs` — the `Dialect` impl, the model path,
  `tools`, the body assembly, `GEMINI_BUILTIN_TOOLS`, `gemini_level_class`.
- `src/dialects/gemini/config.rs` — `generationConfig` (`thinkingConfig`
  per MAP-7, `EFFORT_THINKING_BUDGETS` from `lm15/providers/common.py:386-393`,
  the `responseSchema`/`responseJsonSchema` rule), `toolConfig` (MAP-8),
  the MAP-6 cache plan.
- `src/dialects/gemini/contents.rs` — messages and parts, thought-signature
  replay (MAP-7.8), the text-only slots.

## Module 4 — the Gemini dialect (mapping summary)

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

## Skeleton change made outside the two wiring lines

`src/dialects/mod.rs`: the `static GEMINI_STUB` line was removed. Once
the wiring point names `gemini::GEMINI`, the stub static is dead code and
`cargo clippy -- -D warnings` (a gate) fails on it. The other three
dialect workers will hit the same line for their stub; when all four
land, `struct Stub` itself goes.

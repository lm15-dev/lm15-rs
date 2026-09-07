# lm15-rs

Rust port of lm15, rebuilt module-by-module against the
[lm15-contract](https://github.com/lm15-dev/lm15-contract) corpus after the
stale v1 implementation was removed (2026-08-31). The contract commit this
port is built against is in `CONTRACT_PIN`; `harness/check.py` refuses to
grade the port against any other commit.

## Status

| Module | Contract surface | State |
|---|---|---|
| 1 — canonical types + serde | spec/types.md, spec/vocabularies.md, spec/invariants.md, docs/serde-rules.md; all 36 serde kinds of harness/PROTOCOL.md | done — `--direction serde` 115 pass / 0 fail / 0 skip |
| 2 — errors | spec/vocabularies.md ErrorCode + class hierarchy; `normalize_error` for every provider in `errors/cases/` | done — `--direction error` 84 pass / 0 fail / 0 skip |
| 3a — core auth | spec/auth.md AUTH-1/2/5/7 + AUTH-8 read side | in progress (`src/auth.rs`); the current auth fixture suite fails |
| 3b — cloud chains | AUTH-1 cloud chains, SigV4, RS256 | not started; the shim answers `UnsupportedFeatureError` |
| 4–9 — dialects, streams, models, files/batch/cache, generation, live | | not started; the shim answers `UnsupportedFeatureError` |

Gates for modules 1–2, from the contract checkout:

```bash
cargo build --release          # harness/shims.json runs ./target/release/lm15-vet
cd ../lm15-contract
python3 harness/check.py --shim rust --direction serde
python3 harness/check.py --shim rust --direction error
```

`cargo test` runs the INV-* unit tests, the serde-rule edge tests, the
shim framing test, and `tests/contract_corpus.rs`, which replays
`serde/canonical.json` and `errors/cases/*.json` through the library from
the sibling `../lm15-contract` checkout (or `LM15_CONTRACT_DIR`). The
corpus is never copied into this repository.

## Quick start (types and errors)

```rust
use lm15::{Config, ErrorClass, Lm15Error, Message, Request, Response};
use lm15::Canonical; // from_json / to_json on every canonical type

let request = Request {
    model: "groq:llama-3.3-70b-versatile".into(),
    messages: vec![Message::user("hi")?],
    config: Config { temperature: Some(0.2), max_tokens: Some(100), ..Default::default() },
    ..Default::default()
};
request.validate()?;

// Canonical JSON: one wire form, the omission rule applied.
let json = serde_json::to_string(&request)?;
let back: Request = serde_json::from_str(&json)?;

// A response, its text and tool calls; the assistant turn to replay.
let response: Response = serde_json::from_str(response_json)?;
let text = response.text();
for call in response.tool_calls() {
    let reply = Message::tool(&call.id, "result")?;
}
let turn = &response.message;

// Errors: class name and ErrorCode are the contract; messages are not.
fn handle(err: Lm15Error) {
    if err.is_retryable() { /* back off */ }
    if err.is_a(ErrorClass::InvalidRequestError) { /* fix the request */ }
    let _code = err.code();  // e.g. ErrorCode::RateLimit
}
```

`lm15::normalize_error(provider, status, body_text)` maps a provider's HTTP
error body onto the hierarchy using the per-dialect tables copied from the
reference; the provider string is resolved through `lm15::registry`.

## Stated deviations

Each row names the rule it deviates from (playbooks/port.md rule 8).

- **Dependencies** (api-family rule 5): the port uses `serde` and
  `serde_json`. Zero-dep is not a Rust idiom; stated once for the whole
  port. No other dependency so far.
- **`Part` variants hold named structs** (api-family § Tools shows
  `Part::ToolCall { id, name, input }`): the port spells it
  `Part::ToolCall(ToolCallPart { id, name, input, .. })`. Every `*Part`,
  `*Delta`, `Stream*Event`, `LiveClient*Event` and `LiveServer*Event` name
  from "Names that do not change" therefore exists as a Rust type, and the
  enums stay closed sums discriminated on the `type` key.
- **Serde is hand-written over `serde_json::Value`** (api-family § Types
  and serde shows `#[serde(tag = "type")]`). The wire shape is exactly the
  tagged form; the impls are hand-written because the omission rule, the
  Number-rule coercions (INV-007/008) and the read leniency (INV-040..048)
  are not expressible with derive attributes. `Serialize`/`Deserialize`
  delegate to `Canonical::to_json`/`from_json`, so `serde_json::to_string`
  and `serde_json::from_str::<Request>` work as the family expects.
- **Validation is a method** (api-family § Types: "constructor returns
  `Result`; `Default` + `validate()`"): types have public fields and a
  `validate()`; the factories (`Message::user`, `Part::tool_call`,
  `Request::new`, ...) validate and return `Result`. Building a struct
  literal and skipping `validate()` is the user's choice, as in Go.
- **Number rule at the boundary** (INV-007/008): int fields are `u64`,
  float fields `f64`, so in memory the wrong kind cannot exist; the
  coercions (`2.0 → 2`, `1 → 1.0`) and the bool rejection happen in
  `from_json`. `INV-029`'s `total_tokens` and INV-004's `extensions: {}`
  normalization run in `from_json` and in `Usage::normalized` /
  `Config::normalized`.
- **INV-020 bare-value coercion**: Rust has `Vec`; the factories take
  `impl Into<ContentInput>` (a string, one `Part`, or `Vec<Part>`), and
  `from_json` still accepts a bare string for `stop` and `allowed`. The
  `Message.tool({call_id: output})` dict form is `Message::tool_results`.
- **Refusals from constructors are `ValidationError`**, not `Lm15Error`:
  the reference raises Python's native `ValueError`/`TypeError` there and
  the vet protocol reports the native name, so the shim answers
  `{"type": "ValueError" | "TypeError"}` for them.
- **Error messages carry no guidance** (vocabularies.md: messages are not
  pinned). The reference appends "To fix" paragraphs to auth, rate-limit
  and context-length messages; this port keeps the provider's message.
- **`normalize_error` takes no host settings**: it maps the body through
  the provider's dialect table without constructing an adapter, so cloud
  doors (`azure*`, `bedrock*`, `vertex*`) normalize even before module 3b
  exists.
- **`ContinuationState.data` and the other opaque payloads** hold
  `serde_json::Map`, which cannot contain non-finite floats or non-string
  keys, so INV-001 holds by construction.

## Divergences from the reference implementation (findings)

These are places where this port follows the written spec and the
reference does not. No fixture pins them; reported for a `changes/` entry.

- `config_from_dict` / `tool_choice_from_dict` in the reference split a bare
  string into characters (`{"stop": "END"}` → `["E","N","D"]`). INV-020
  says a bare string coerces to a one-element tuple; this port reads
  `["END"]`.
- `response_from_dict` in the reference crashes on `"usage": null`;
  INV-042 says the telemetry nests read a non-dict as absent. This port
  reads `Usage::default()`.

## Layout

- `src/types/` — canonical types, vocabularies, invariants (`validate()`).
- `src/serde/` — `Canonical` (from_json/to_json), the 36-kind table.
- `src/errors.rs` — `ErrorCode`, `ErrorClass` (hierarchy), `Lm15Error`,
  `normalize_error` and the per-dialect tables.
- `src/registry.rs` — provider string → dialect (+ compat preset).
- `src/bin/lm15-vet.rs` — the vet shim.
- `src/auth.rs` — module 3a, in progress.

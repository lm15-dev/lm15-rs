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
| 3a core auth | spec/auth.md AUTH-1 (`key`, `oauth`, `oauth-unless-explicit`), AUTH-2 credential values + D1 scheme selection, AUTH-5, AUTH-7 doctor, AUTH-8 read side, AUTH-10 policy table | done — `--direction auth --auth-scope core` 26 pass / 0 fail / 0 skip; `tests/auth_resolution_contract.rs` replays the same `auth/resolution.json` from the contract checkout with the same core/cloud split |
| 3b cloud chains | AUTH-1 `aws-chain`/`azure-chain`/`gcp-chain`, AUTH-11 rung kinds, SigV4, RS256 | not implemented; cloud-chain providers are in the policy table as data, and `explain_auth` answers `AuthError::NotImplemented` (class `NotConfiguredError`, code `not_configured`) naming module 3b. The 11 cloud cases are asserted to answer that error and counted, not skipped |
| 4–9 — dialects, streams, models, files/batch/cache, generation, live | | not started; the shim answers `UnsupportedFeatureError` |

Gates for modules 1–3a, from the contract checkout:

```bash
cargo build --release          # harness/shims.json runs ./target/release/lm15-vet
cd ../lm15-contract
python3 harness/check.py --shim rust --direction serde
python3 harness/check.py --shim rust --direction error
python3 harness/check.py --shim rust --direction auth --auth-scope core
```

`cargo test` runs the INV-* unit tests, the serde-rule edge tests, the
shim framing tests, `tests/contract_corpus.rs` (replays
`serde/canonical.json` and `errors/cases/*.json`) and
`tests/auth_resolution_contract.rs` (replays `auth/resolution.json`).
Both read the sibling `../lm15-contract` checkout (or `LM15_CONTRACT_DIR`)
and do nothing when it is absent. The corpus is never copied into this
repository.

The shim answers `capabilities`, `serde_roundtrip`, `validate`,
`normalize_error` and `explain_auth`. `surface_dump` (PROTOCOL.md) is not
implemented: it must come from reflection, and Rust has no runtime
reflection over struct fields; it is not a module gate (`tools/audit.py`
reads the reference's dump).

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
- **`expires_at` is `Option<i64>` in memory** (AUTH-2: `expires_at` "is
  RFC 3339"). `Credential` stores Unix seconds and converts at the JSON
  boundary; the wire form is unchanged (whole seconds, UTC, `Z`; offsets
  normalized to `Z`), which is what the reference emits
  (`lm15/credentials.py` `format_rfc3339`). Reason: no date crate.
- **`openai_chat` alias**: the provider table names the OpenAI Chat
  Completions door `openai-chat` (the registry id in `lm15/registry.py`);
  `lm15/access.py` spells the same policy `openai_chat`.
  `lm15::registry::canonical_provider` maps the underscore alias.
- **`AccessPolicy` carries the auth columns only** (`provider`,
  `credential_policy`, `env_keys`, `auth_scheme`, `placeholder_key`,
  `login_hint`). The wire columns of AUTH-10 (`supports`, `headers`,
  `backend`, `host`, `settings`, `base_url`) belong to the dialect modules
  and are not in this module.
- **`AuthError` is a second error type** (api-family § Errors: one root
  `enum Lm15Error`). The auth module returns `lm15::auth::AuthError`
  (`UnknownProvider`, `NotConfigured { provider, message, hint }`,
  `NotImplemented { provider, policy }`), which keeps the fix hint and the
  policy as fields. Every variant is class `NotConfiguredError`, code
  `not_configured` (`class_name()`, `code()`), and `From<AuthError> for
  Lm15Error` folds it into `Lm15Error::NotConfiguredError` with the
  provider carried; module 4 surfaces only `Lm15Error`.

## Divergences from the reference implementation

None open. The two read-leniency differences this port found on 2026-09-06
(INV-020: a bare `stop` string split into characters; INV-042: a
`"usage": null` crash in `response_from_dict`) were resolved in the
reference on 2026-09-07 (lm15-python b2709c8). No fixture pins them.

## Layout

- `src/types/` — canonical types, vocabularies, invariants (`validate()`).
- `src/serde/` — `Canonical` (from_json/to_json), the 36-kind table.
- `src/errors.rs` — `ErrorCode`, `ErrorClass` (hierarchy), `Lm15Error`,
  `normalize_error` and the per-dialect tables.
- `src/registry.rs` — provider string → dialect (+ compat preset).
- `src/bin/lm15-vet.rs` — the vet shim.
- `src/auth.rs` — module 3a: `Credential`, providers, policy table,
  doctor, AUTH-8 loaders.

## Auth module (3a)
- `Credential` (`ApiKey`, `BearerToken`, `AwsCredentials`): the one
  credential type of the crate (`lm15::Credential` and
  `lm15::auth::Credential` are the same enum). It implements `Canonical`
  (`from_json`/`to_json`, the `credential` serde kind; the five `credential`
  vectors of `serde/canonical.json` round-trip exactly), redacts in
  `Debug`/`Display`, and applies the AUTH-3 expiry skew.
- `CredentialProvider` (one method, `credential()`, returning the value);
  `StaticCredential`, `FnCredential`; `&str`/`String` are the `ApiKey`
  shorthand (`credential()`, `TryFrom`, `StaticCredential::new`), and every
  one of them rejects the empty string. Never cached by the caller.
- `select_scheme(policy_schemes, &credential)` per AUTH-2 / D1.
- `ACCESS_POLICIES` / `access_policy(provider)`: the auth columns of
  `lm15/access.py` (provider, credential policy, env keys in declared order,
  auth schemes, placeholder key, login hint) copied as data.
- `explain_auth(provider, &ExplainOptions) -> Result<Report, AuthError>`:
  the AUTH-7 walk; every rung as `selected`/`shadowed`/`absent`
  (`unprobed` exists in the vocabulary; only module 3b produces it).
- Read-only loaders for the Claude Code file, the Codex CLI file, and the
  xAI entry of the lm15-owned store / Pi agent store (AUTH-8). A recorded
  expiry the `i64` millisecond clock cannot subtract from is
  `Expiry::Malformed`: the rung is `absent` with a "malformed" detail,
  never fresh (the reference has unbounded ints and no such case).

## Not implemented (stated, not absorbed)

- Module 3b, as above.
- AUTH-3/4 write side (locked double-checked refresh, atomic 0600 writes):
  this port reads credentials only.
- AUTH-9 `login(provider)`: not shipped. The xAI login hint therefore names
  the AUTH-9 door, not a command this port runs.

# lm15-rs

Rust port of lm15, implemented from the lm15-contract spec
(`spec/types.md`, `spec/vocabularies.md`, `spec/invariants.md`,
`lm15-python2/docs/serde-rules.md`, `docs/mapping-rules.md`). The corpus in
`lm15-contract` is the oracle (see its `AUTHORITY.md`).

Status: this port implements the frozen chat core per `spec/SCOPE.md` and
passes all five harness directions with zero failures — 304 checks
(request 110, response 102, stream 8, error 16, serde 68; 4 skips are
cases not applicable to this shim). Non-chat endpoints (embeddings, files,
batch, image/audio generation) and live sessions are NOT implemented.
A network transport is also out of scope here: the crate is the pure
transformation core plus the vet shim.

Layout:

- `src/types.rs` — canonical types as serde-tagged enums/structs honoring
  the omission rule, the Number rule, and opaque-payload verbatimness.
- `src/errors.rs` — canonical error hierarchy mapped to `ErrorCode`.
- `src/providers/{openai,openai_chat,anthropic,gemini}.rs` — request
  building, response parsing, stream-frame mapping, error normalization.
- `src/stream.rs` — SSE parsing, the MAP-3 coalescer (exactly one final
  StreamEndEvent; post-finish usage-only chunks absorbed), and stream
  materialization.
- `src/vet.rs` + `src/bin/vet.rs` — the JSONL vet shim
  (`harness/PROTOCOL.md`); build with `cargo build --release`, binary at
  `target/release/lm15-vet`.

Checks: `cargo test`, `cargo clippy --all-targets -- -D warnings`, and from
`lm15-contract`:
`../lm15-python2/.venv/bin/python harness/check.py --shim rust --direction all`.

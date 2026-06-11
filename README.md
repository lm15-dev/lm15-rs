# lm15-rs

Rust port of lm15, implemented from the lm15-contract spec
(`spec/types.md`, `spec/vocabularies.md`, `spec/invariants.md`,
`lm15-python2/docs/serde-rules.md`). The corpus in `lm15-contract` is the
oracle (see its `AUTHORITY.md`).

- `src/types.rs` — canonical types as serde-tagged enums/structs honoring the
  omission rule, the Number rule, and opaque-payload verbatimness.
- `src/errors.rs` — canonical error hierarchy mapped to `ErrorCode`.
- `src/vet.rs` + `src/bin/vet.rs` — the JSONL vet shim
  (`harness/PROTOCOL.md`); build with `cargo build --release`, binary at
  `target/release/lm15-vet`.
- `src/providers/`, `src/stream.rs` — adapter and stream stages (stubs for
  now; `build_request`/`parse_response`/`replay_stream`/`normalize_error`
  reply `Unimplemented`).

Checks: `cargo test`, `cargo clippy --all-targets -- -D warnings`, and from
`lm15-contract`:
`lm15-python2/.venv/bin/python harness/check.py --shim rust --direction serde`.

# lm15-rs

Rust port of lm15, rebuilt module-by-module against the
[lm15-contract](https://github.com/lm15-dev/lm15-contract) corpus after the
stale v1 implementation was removed (2026-08-31).

## Status

| Module | Contract surface | State |
|---|---|---|
| `auth` | spec/auth.md AUTH-1/2/5/7 + AUTH-8 read side | fixture-verified (`conformance/auth_resolution.json`) |
| everything else | — | not yet rebuilt |

The `auth` module ships: credential providers (`CredentialProvider`,
`StaticCredential`, `FnCredential`), the resolution chain + `explain_auth`
doctor report, and read-only borrowed-credential loaders for the Claude Code
and Codex CLI files. Not yet implemented (stated, not absorbed): the
AUTH-3/4 write side (locked double-checked refresh, atomic 0600 writes) and
the AUTH-9 login primitives.

```bash
cargo test
```

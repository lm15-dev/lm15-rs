# README rows: auth (module 3a / 3b)

Rows for the parent to merge into `README.md`. Contract at `CONTRACT_PIN`.

## Status table rows

| Module | Contract surface | State |
|---|---|---|
| 3a core auth | spec/auth.md AUTH-1 (`key`, `oauth`, `oauth-unless-explicit`), AUTH-2 credential values + D1 scheme selection, AUTH-5, AUTH-7 doctor, AUTH-8 read side, AUTH-10 policy table | fixture-verified: 26/26 core cases of `conformance/auth_resolution.json` (`tests/auth_resolution_contract.rs`, same core/cloud split as `harness/check.py --auth-scope core`) |
| 3b cloud chains | AUTH-1 `aws-chain`/`azure-chain`/`gcp-chain`, AUTH-11 rung kinds, SigV4, RS256 | not implemented; cloud-chain providers are in the policy table as data, and `explain_auth` answers `AuthError::NotImplemented` (class `NotConfiguredError`, code `not_configured`) naming module 3b. The 11 cloud cases are asserted to answer that error and counted, not skipped |

## What the `auth` module ships

- `Credential` (`ApiKey`, `BearerToken`, `AwsCredentials`) with canonical JSON
  (`to_json`/`from_json`; the `credential` vectors of `serde/canonical.json`
  round-trip exactly), redacting `Debug`/`Display`, AUTH-3 expiry skew.
- `CredentialProvider` (one method, `credential()`, returning the value);
  `StaticCredential`, `FnCredential`; `&str`/`String` are the `ApiKey`
  shorthand. Never cached by the caller.
- `select_scheme(policy_schemes, &credential)` per AUTH-2 / D1.
- `ACCESS_POLICIES` / `access_policy(provider)`: the auth columns of
  `lm15/access.py` (provider, credential policy, env keys in declared order,
  auth schemes, placeholder key, login hint) copied as data.
- `explain_auth(provider, &ExplainOptions) -> Result<Report, AuthError>`:
  the AUTH-7 walk; every rung as `selected`/`shadowed`/`absent`
  (`unprobed` exists in the vocabulary; only module 3b produces it).
- Read-only loaders for the Claude Code file, the Codex CLI file, and the
  xAI entry of the lm15-owned store / Pi agent store (AUTH-8).

## Not implemented (stated, not absorbed)

- Module 3b, as above.
- AUTH-3/4 write side (locked double-checked refresh, atomic 0600 writes):
  this port reads credentials only.
- AUTH-9 `login(provider)`: not shipped. The xAI login hint therefore names
  the AUTH-9 door, not a command this port runs.

## Stated deviations

- AUTH-2 says `expires_at` "is RFC 3339". `Credential` stores it as Unix
  seconds (`Option<i64>`) and converts at the JSON boundary; the wire form
  is unchanged (whole seconds, UTC, `Z`), which is what the reference
  emits (`lm15/credentials.py` `format_rfc3339`). Reason: no date crate.
- The provider table names the OpenAI Chat Completions door `openai-chat`
  (the registry id in `lm15/registry.py`); `lm15/access.py` spells the same
  policy `openai_chat`. `canonical_provider` maps the underscore alias.
- `AccessPolicy` carries the auth columns only (`provider`,
  `credential_policy`, `env_keys`, `auth_scheme`, `placeholder_key`,
  `login_hint`). The wire columns of AUTH-10 (`supports`, `headers`,
  `backend`, `host`, `settings`, `base_url`) belong to the dialect modules
  and are not in this module.
- Errors: `AuthError` is a module-local enum (`code()`, `class_name()`)
  until the port-wide `Lm15Error` exists; every variant maps to
  `NotConfiguredError` / `not_configured`.

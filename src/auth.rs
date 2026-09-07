//! lm15-contract auth surface, module 3a (playbooks/port.md; spec/auth.md,
//! ratified 2026-08-31, amendments through 2026-09-06):
//!
//! - AUTH-2 credential values ([`Credential`]) and providers
//!   ([`CredentialProvider`]), with the D1 scheme selection ([`select_scheme`]);
//! - AUTH-10 access policies as data ([`AccessPolicy`], [`access_policy`]);
//! - AUTH-1 resolution for the `key`, `oauth` and `oauth-unless-explicit`
//!   policies, explained rung by rung by [`explain_auth`] (AUTH-7);
//! - AUTH-8 read side of the borrowed CLI files and the lm15-owned store.
//!
//! Secrecy invariant (AUTH-5): no secret value is stored on a [`Report`],
//! rendered by `describe`, or emitted by any `Debug`/`Display` impl here.
//!
//! Module 3b (cloud chains: `aws-chain`, `azure-chain`, `gcp-chain`; AUTH-11
//! rung kinds; SigV4; RS256) is not implemented. Their policies are present
//! in the table; [`explain_auth`] answers [`AuthError::NotImplemented`]
//! (class `NotConfiguredError`) naming module 3b. Also not implemented
//! (stated, not absorbed): the AUTH-3/4 write side and the AUTH-9 login
//! primitives. This port reads credentials only.

mod credential;
mod doctor;
mod error;
mod policy;
mod stores;
mod time;

pub use credential::{
    select_scheme, AuthScheme, Credential, CredentialKind, CredentialProvider, FnCredential,
    StaticCredential, EXPIRY_SKEW_SECONDS,
};
pub use doctor::{explain_auth, ExplainOptions, Report, Step, StepState};
pub use error::AuthError;
pub use policy::{
    access_policy, canonical_provider, known_providers, AccessPolicy, CredentialPolicy,
    ACCESS_POLICIES,
};
pub use stores::{
    read_claude_code_credential, read_codex_cli_credential, read_xai_credential,
    LocalOAuthCredential, CLAUDE_CODE_LOGIN_HINT, OPENAI_CODEX_LOGIN_HINT, XAI_LOGIN_HINT,
};
pub use time::{format_rfc3339, parse_rfc3339};

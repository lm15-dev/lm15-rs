//! lm15 — Rust port, rebuilt module-by-module against the lm15-contract
//! corpus after the stale v1 implementation was removed (2026-08-31).
//!
//! Implemented modules (playbooks/port.md order of work):
//!
//! 1. canonical types + serde (`spec/types.md`, `spec/vocabularies.md`,
//!    `spec/invariants.md`, `docs/serde-rules.md`) — [`types`], [`serde`].
//! 2. errors (`spec/vocabularies.md` ErrorCode + hierarchy shape) and
//!    provider error normalization — [`errors`], [`registry`].
//!
//! 3a. core auth (`spec/auth.md` AUTH-1/2/5/7/8/10; module 3b cloud chains
//!    are not implemented) — [`auth`]. [`Credential`] is the one AUTH-2
//!    value type, exported here and as `auth::Credential`.
//!
//! 4. dialects, request side — the skeleton: [`wire`] (the emit path),
//!    the full AUTH-10 policy table ([`auth`]), [`cloud`] (host settings,
//!    URL rendering, rewrites, SigV4), [`compat`] (preset tables),
//!    [`adapter`] (`ProviderLM` and the named constructors),
//!    [`registry::adapter_for`]. The four [`dialects`] are stubs until
//!    their workers land.

// `Lm15Error` is the family's one error enum (api-family § Errors); its
// `ErrorMeta` payload is 136 bytes, over clippy's 128-byte `Result` limit.
// Boxing every error would put a `Box` on the user's `?`; the error path is
// not the hot path.
#![allow(clippy::result_large_err)]

pub mod adapter;
pub mod auth;
pub mod cloud;
pub mod compat;
pub mod dialects;
pub mod errors;
pub mod registry;
pub mod serde;
pub mod types;
pub mod wire;

pub use adapter::{
    AnthropicLM, ClaudeCodeLM, GeminiLM, LmBuilder, OpenAIChatLM, OpenAICodexLM, OpenAILM,
    ProviderLM, XaiLM,
};
pub use auth::{AccessPolicy, Credential, CredentialProvider, HostSpec};
pub use cloud::hosts::HostSettings;
pub use compat::{AnthropicCompat, OpenAIChatCompat, OpenAIResponsesCompat};
pub use errors::{normalize_error, ErrorClass, ErrorCode, ErrorMeta, Lm15Error};
pub use serde::Canonical;
pub use types::*;
pub use wire::TransportRequest;

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
//! 4. dialects, request side: [`wire`] (the emit path), the full AUTH-10
//!    policy table ([`auth`]), [`cloud`] (host settings, URL rendering,
//!    rewrites, SigV4), [`compat`] (preset tables), [`adapter`]
//!    (`ProviderLM` and the named constructors), [`registry::adapter_for`],
//!    the four [`dialects`].
//!
//! 5. dialects, response side and stream assembly (MAP-1..4, MAP-9):
//!    `parse_response` / `parse_stream_event` on each dialect, [`sse`]
//!    (the SSE parser), [`stream`] (the MAP-3/4 coalescer, the MAP-9
//!    accumulator, `materialize_response`), `ProviderLM::parse_response`,
//!    `ProviderLM::stream_decoder` / `replay_stream`.

// `Lm15Error` is the family's one error enum (api-family § Errors); its
// `ErrorMeta` payload is 136 bytes, over clippy's 128-byte `Result` limit.
// Boxing every error would put a `Box` on the user's `?`; the error path is
// not the hot path.
#![allow(clippy::result_large_err)]

pub mod adapter;
pub mod auth;
#[cfg(feature = "blocking")]
pub mod blocking;
pub mod cloud;
pub mod compat;
pub mod dialects;
pub mod errors;
pub mod registry;
pub mod response_stream;
pub mod router;
pub mod serde;
pub mod sse;
pub mod stream;
pub mod surfaces;
pub mod transport;
pub mod types;
pub mod wire;

pub use adapter::{
    AnthropicLM, ClaudeCodeLM, EventStream, GeminiLM, LmBuilder, OpenAIChatLM, OpenAICodexLM,
    OpenAILM, ProviderLM, XaiLM,
};
pub use auth::{AccessPolicy, Credential, CredentialProvider, HostSpec};
pub use cloud::hosts::HostSettings;
pub use compat::{AnthropicCompat, OpenAIChatCompat, OpenAIResponsesCompat};
pub use errors::{normalize_error, ErrorClass, ErrorCode, ErrorMeta, Lm15Error};
pub use response_stream::ResponseStream;
pub use router::{LMRouter, Resolution, RouteRule, RouteSource, RouterConfig, DEFAULT_RULES};
pub use serde::Canonical;
pub use transport::{HttpTransport, Transport, TransportResponse};
pub use types::*;
pub use wire::TransportRequest;

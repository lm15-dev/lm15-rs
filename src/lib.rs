//! lm15 — the Rust port of lm15: one canonical request/response model
//! over every provider the lm15-contract names, byte-exact against its
//! corpus (the pinned commit is in `CONTRACT_PIN`).
//!
//! The map, in playbooks/port.md module order:
//!
//! 1. canonical types + serde — [`types`], [`serde`] ([`Canonical`]).
//! 2. errors and provider error normalization — [`errors`], [`registry`].
//! 3. auth — [`auth`] (AUTH-1..10: credentials, the policy table, the
//!    doctor, stored logins with refresh, the login door) and [`cloud`]
//!    (the cloud chains, AUTH-11, SigV4, RS256, host settings).
//! 4. dialects, request side — [`wire`] (the emit path), [`compat`] (the
//!    preset tables), the four [`dialects`], [`adapter`] (`ProviderLM`,
//!    the named constructors), [`registry::adapter_for`].
//! 5. response side and stream assembly — [`sse`], [`stream`]; the
//!    network — [`transport`], [`response_stream`]; the router —
//!    [`router`] ([`LMRouter`]); the `blocking` feature mirrors the names.
//! 6. model listing, the files / batch / cache surfaces ([`surfaces`]),
//!    image / speech / video generation, and live sessions ([`live`]) —
//!    all methods on [`ProviderLM`].

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
pub mod jobs;
pub mod live;
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
pub use dialects::openai_chat::ingest::request_from_openai_chat;
pub use dialects::openai_chat::response_from_openai_chat;
pub use errors::{normalize_error, ErrorClass, ErrorCode, ErrorMeta, Lm15Error};
pub use jobs::{BatchJob, VideoJob, WaitError, WaitOptions};
pub use live::{LiveSession, Turn, TurnEnd, TurnView};
pub use response_stream::ResponseStream;
pub use router::{
    openai_chat_model_string, LMRouter, Resolution, RouteRule, RouteSource, RouterConfig,
    DEFAULT_RULES, LITELLM_PROVIDER_PREFIXES,
};
pub use serde::Canonical;
pub use transport::{HttpTransport, Transport, TransportResponse};
pub use types::*;
pub use wire::TransportRequest;

//! lm15 — the Rust port of lm15: one canonical request/response model
//! over the providers declared by lm15-contract. `CONTRACT_PIN` names the
//! implementation target, not proof of conformance; see docs/catchup.md.
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

pub mod adaptation;
pub mod adapter;
pub mod auth;
#[cfg(feature = "blocking")]
pub mod blocking;
pub mod cloud;
pub mod compat;
pub mod dialects;
pub mod errors;
pub mod jobs;
pub mod judgments;
#[cfg(feature = "native")]
pub mod live;
#[cfg(feature = "native")]
pub mod login;
pub mod registry;
pub mod response_stream;
pub mod router;
pub mod scoring;
pub mod serde;
pub mod sse;
pub mod stop;
pub mod stream;
pub mod surfaces;
pub mod testing;
pub mod tooling;
pub mod transport;
pub mod types;
#[cfg(feature = "wasm")]
pub mod wasm;
pub mod wire;

pub use adaptation::{Adaptation, AdaptationAction, AdaptationPolicy};
pub use adapter::{
    AnthropicLM, ClaudeCodeLM, EventStream, GeminiLM, LmBuilder, OpenAIChatLM, OpenAICodexLM,
    OpenAILM, ProviderLM, TypeSafeLM, XaiLM,
};
pub use auth::{
    default_credentials_path, AccessPolicy, Credential, CredentialFileStore, CredentialProvider,
    CredentialSource, HostSpec, SourcedCredential,
};
pub use cloud::hosts::HostSettings;
pub use compat::{AnthropicCompat, OpenAIChatCompat, OpenAIResponsesCompat};
pub use dialects::openai_chat::ingest::request_from_openai_chat;
pub use dialects::openai_chat::response_from_openai_chat;
pub use errors::{
    normalize_error, CollectionLimit, DiagnosticHeaders, ErrorClass, ErrorCode, ErrorMeta,
    Lm15Error,
};
pub use jobs::{BatchJob, VideoJob, WaitError, WaitOptions, WaitSnapshot};
pub use judgments::{
    choice, choice_described, judgments, judgments_named, score, score_named, yes_no, Judgment,
    JudgmentKind,
};
#[cfg(feature = "native")]
pub use live::{
    LiveEventSource, LiveSession, Turn, TurnEnd, TurnLimits, TurnView, DEFAULT_TURN_MAX_BYTES,
    DEFAULT_TURN_MAX_EVENTS,
};
pub use response_stream::ResponseStream;
pub use router::{
    openai_chat_model_string, DeclaredProvider, LMRouter, Resolution, RouteRule, RouteSource,
    RouterConfig, DEFAULT_RULES, LITELLM_PROVIDER_PREFIXES,
};
pub use serde::Canonical;
pub use testing::LanguageModel;
#[cfg(feature = "native")]
pub use transport::HttpTransport;
pub use transport::{Timeouts, Transport, TransportResponse, DEFAULT_MAX_CONNECTIONS};
pub use types::*;
pub use wire::TransportRequest;

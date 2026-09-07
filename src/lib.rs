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

pub mod auth;
pub mod errors;
pub mod registry;
pub mod serde;
pub mod types;

pub use auth::Credential;
pub use errors::{normalize_error, ErrorClass, ErrorCode, ErrorMeta, Lm15Error};
pub use serde::Canonical;
pub use types::*;

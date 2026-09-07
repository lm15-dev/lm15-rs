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
//! Module 3a (auth surface, spec/auth.md) is in progress in [`auth`].

pub mod auth;
pub mod errors;
pub mod registry;
pub mod serde;
pub mod types;

pub use errors::{normalize_error, ErrorClass, ErrorCode, ErrorMeta, Lm15Error};
pub use serde::Canonical;
pub use types::*;

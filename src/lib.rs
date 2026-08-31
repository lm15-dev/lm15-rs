//! lm15 — Rust port, rebuilt module-by-module against the lm15-contract
//! corpus after the stale v1 implementation was removed (2026-08-31).
//!
//! Currently implemented: the auth surface (spec/auth.md AUTH-1/2/5/7 and
//! the AUTH-8 read side), fixture-verified against
//! `conformance/auth_resolution.json`.

pub mod auth;

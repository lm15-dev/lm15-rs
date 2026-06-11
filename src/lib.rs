//! lm15 — one canonical model for LLM providers. Rust port, implemented
//! from the lm15-contract spec (spec/types.md, spec/vocabularies.md,
//! spec/invariants.md, docs/serde-rules.md).

pub mod errors;
pub mod providers;
pub mod stream;
pub mod surface;
pub mod types;
pub mod vet;

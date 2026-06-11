//! lm15 — one canonical model for LLM providers. Rust port, implemented
//! from the lm15-contract spec (spec/types.md, spec/vocabularies.md,
//! spec/invariants.md, docs/serde-rules.md).

pub mod client;
pub mod errors;
pub mod providers;
pub mod stream;
pub mod surface;
pub mod transport;
pub mod types;
pub mod vet;

pub use client::{AnthropicLM, ChatPreset, GeminiLM, OpenAIChatLM, OpenAILM};
pub use errors::Lm15Error;
pub use stream::materialize_response;
pub use types::{Config, Message, Part, Request, Response, StreamEvent, Tool, ToolCallView, Usage};

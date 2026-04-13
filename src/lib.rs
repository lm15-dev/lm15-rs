//! # lm15
//!
//! One interface for OpenAI, Anthropic, and Gemini. Minimal dependencies.
//!
//! ```rust,no_run
//! use lm15::types::{LMRequest, Message};
//! use lm15::factory::build_default;
//!
//! let client = build_default(None);
//! let request = LMRequest {
//!     model: "gpt-4.1-mini".into(),
//!     messages: vec![Message::user("Hello!")],
//!     system: None,
//!     tools: vec![],
//!     config: Default::default(),
//! };
//! let response = client.complete(&request, "").unwrap();
//! println!("{}", response.text().unwrap_or_default());
//! ```

pub mod types;
pub mod errors;
pub mod transport;
pub mod capabilities;
pub mod provider;
pub mod client;
pub mod conversation;
pub mod factory;

// Re-export key types at crate root
pub use types::{
    Part, PartType, Message, Role, LMRequest, LMResponse, Config, Usage,
    FinishReason, Tool, ToolCallInfo, DataSource, StreamEvent, StreamChunk,
    ErrorInfo, PartDelta, JsonObject,
    EmbeddingRequest, EmbeddingResponse,
    FileUploadRequest, FileUploadResponse,
    ImageGenerationRequest, ImageGenerationResponse,
};
pub use errors::LM15Error;
pub use client::UniversalLM;
pub use conversation::Conversation;
pub use factory::{build_default, BuildOpts, providers};
pub use capabilities::resolve_provider;

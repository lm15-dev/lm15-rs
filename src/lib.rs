//! # lm15
//!
//! One interface for OpenAI, Anthropic, and Gemini. Minimal dependencies.
//!
//! ```rust,no_run
//! let mut result = lm15::call("gpt-4.1-mini", "Hello.", None);
//! println!("{}", result.text().unwrap());
//! ```

pub mod types;
pub mod errors;
pub mod transport;
pub mod capabilities;
pub mod provider;
pub mod client;
pub mod conversation;
pub mod factory;
pub mod result;
pub mod model;
pub mod middleware;
pub mod model_catalog;
pub mod cost;
pub mod api;
pub mod curl;

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
pub use result::LMResult;
pub use model::{Model, ModelOpts, CallOpts, HistoryEntry, Reasoning};
pub use model_catalog::{ModelSpec, fetch_models_dev, build_provider_model_index};
pub use cost::{
    CostBreakdown, estimate_cost, estimate_cost_for_spec,
    enable_cost_tracking, disable_cost_tracking,
    get_cost_index, set_cost_index, lookup_cost, sum_costs,
};
pub use middleware::{with_retries, with_cache, with_history, MiddlewareHistoryEntry};
pub use api::{call, model, prepare, send, configure, configure_with_tracking, CallOptions};
pub use curl::{
    CurlOptions, HttpRequestDump,
    build_http_request, dump_curl, dump_http,
    http_request_to_curl, http_request_to_dict,
};

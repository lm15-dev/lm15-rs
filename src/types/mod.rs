//! Canonical types (spec/types.md) with construction-time invariants
//! (spec/invariants.md). Every type carries public fields, a `Default`
//! where the spec gives one, and a `validate()` that enforces the INV-*
//! rules; factory constructors validate and return `Result`.
//!
//! Number rule (docs/serde-rules.md): int fields are `u64`/`i64`, float
//! fields are `f64`. The cross-type coercions (INV-007/INV-008) happen at
//! the JSON boundary in [`crate::serde`]; the Rust types cannot hold the
//! wrong number kind.

mod config;
mod continuation;
mod delta;
mod endpoints;
mod json;
mod live;
mod message;
mod model_info;
mod parts;
mod request;
mod stream;
mod tools;
mod usage;
mod vocab;

pub use config::{CacheConfig, Config, Reasoning, ToolChoice};
pub use continuation::{continuation_data, ContinuationState};
pub use delta::{
    AudioDelta, CitationDelta, ContinuationDelta, Delta, ImageDelta, TextDelta, ThinkingDelta,
    ToolCallDelta,
};
pub use endpoints::{
    BatchEntry, BatchJobInfo, BatchRequest, CacheInfo, CachePage, CachedPrefix, FileInfo, FilePage,
    FileUploadRequest, ImageGenerationRequest, ImageGenerationResponse, SpeechGenerationRequest,
    SpeechGenerationResponse, VideoGenerationRequest, VideoJobInfo,
};
pub(crate) use json::{base64_decode, base64_encode};
pub use json::{
    base64_payload, is_base64_shaped, normalize_rfc3339, parse_rfc3339_lenient, JsonObject,
    ValidationError, ValidationKind,
};
pub use live::{
    AudioFormat, LiveClientAudioEvent, LiveClientEndAudioEvent, LiveClientEvent,
    LiveClientImageEvent, LiveClientInterruptEvent, LiveClientTextEvent, LiveClientToolResultEvent,
    LiveClientTurnEvent, LiveConfig, LiveServerAudioEvent, LiveServerErrorEvent, LiveServerEvent,
    LiveServerInterruptedEvent, LiveServerTextEvent, LiveServerToolCallDeltaEvent,
    LiveServerToolCallEvent, LiveServerTurnEndEvent, LiveServerUsageEvent,
};
pub use message::{Message, SystemContent};
pub use model_info::{InferenceModelInfo, InferencePricing, ModelInfo, ModelOrigin};
pub use parts::{
    AudioPart, BinaryPart, CitationPart, ContentInput, DocumentPart, ImagePart, Part, RefusalPart,
    TextPart, ThinkingPart, ToolCallInfo, ToolCallPart, ToolResultPart, VideoPart,
};
pub use request::{Request, Response};
pub use stream::{
    ErrorDetail, StreamDeltaEvent, StreamEndEvent, StreamErrorEvent, StreamEvent, StreamStartEvent,
};
pub use tools::{BuiltinTool, FunctionTool, Tool};
pub use usage::{TokenLogprob, TopLogprob, Usage};
pub use vocab::{
    AudioEncoding, BatchOutcome, BatchStatus, CacheMode, CachePrefix, CacheRetention,
    FileReadiness, FinishReason, ImageDetail, ReasoningEffort, ReasoningSummary, Role,
    ToolChoiceMode, VideoStatus,
};

//! Stream events and ErrorDetail (types.md § Stream events).

use crate::errors::ErrorCode;

use super::delta::Delta;
use super::json::{opt_non_empty, JsonObject, VResult};
use super::usage::Usage;
use super::vocab::FinishReason;

/// Structured error information carried by `error` events and batch entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorDetail {
    pub code: ErrorCode,
    pub message: String,
    pub provider_code: Option<String>,
}

impl ErrorDetail {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        ErrorDetail {
            code,
            message: message.into(),
            provider_code: None,
        }
    }

    pub fn validate(&self) -> VResult<()> {
        opt_non_empty(self.provider_code.as_ref(), "ErrorDetail.provider_code")
    }
}

/// The response stream has started.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StreamStartEvent {
    pub id: Option<String>,
    pub model: Option<String>,
}

/// A typed content delta arrived.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamDeltaEvent {
    pub delta: Delta,
}

/// The stream completed. Exactly one per stream, final (MAP-3).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StreamEndEvent {
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<Usage>,
    pub provider_data: Option<JsonObject>,
}

/// The stream failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamErrorEvent {
    pub error: ErrorDetail,
}

/// vocabularies.md § StreamEventType.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    Start(StreamStartEvent),
    Delta(StreamDeltaEvent),
    End(StreamEndEvent),
    Error(StreamErrorEvent),
}

impl StreamEvent {
    pub const TYPES: &'static [&'static str] = &["start", "delta", "end", "error"];

    pub fn type_name(&self) -> &'static str {
        match self {
            StreamEvent::Start(_) => "start",
            StreamEvent::Delta(_) => "delta",
            StreamEvent::End(_) => "end",
            StreamEvent::Error(_) => "error",
        }
    }

    pub fn validate(&self) -> VResult<()> {
        match self {
            StreamEvent::Start(e) => {
                opt_non_empty(e.id.as_ref(), "StreamStartEvent.id")?;
                opt_non_empty(e.model.as_ref(), "StreamStartEvent.model")
            }
            StreamEvent::Delta(e) => e.delta.validate(),
            StreamEvent::End(e) => e.usage.as_ref().map_or(Ok(()), Usage::validate),
            StreamEvent::Error(e) => e.error.validate(),
        }
    }
}

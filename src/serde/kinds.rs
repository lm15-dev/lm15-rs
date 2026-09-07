//! The closed set of serde kinds (harness/PROTOCOL.md § Serde kinds, 36).

use serde_json::Value;

use super::Canonical;
use crate::types::*;

/// Every kind string accepted by `serde_roundtrip` / `validate`, in the
/// order PROTOCOL.md lists them.
pub const KINDS: &[&str] = &[
    "part",
    "message",
    "tool",
    "tool_choice",
    "reasoning",
    "config",
    "cache_config",
    "cache_info",
    "cache_page",
    "cached_prefix",
    "token_logprob",
    "continuation_state",
    "error_detail",
    "delta",
    "usage",
    "stream_event",
    "request",
    "response",
    "model_info",
    "batch_request",
    "batch_job",
    "batch_entry",
    "file_upload_request",
    "file_info",
    "file_page",
    "image_generation_request",
    "image_generation_response",
    "speech_generation_request",
    "speech_generation_response",
    "video_generation_request",
    "video_job",
    "audio_format",
    "live_config",
    "live_client_event",
    "live_server_event",
    "credential",
];

fn rt<T: Canonical>(value: &Value) -> Result<Value, ValidationError> {
    Ok(T::from_json(value)?.to_json())
}

/// `to_json(from_json(value))` for `kind`: no cleaning, no help. An
/// unknown kind is a `ValueError` (INV-044).
pub fn roundtrip(kind: &str, value: &Value) -> Result<Value, ValidationError> {
    match kind {
        "part" => rt::<Part>(value),
        "message" => rt::<Message>(value),
        "tool" => rt::<Tool>(value),
        "tool_choice" => rt::<ToolChoice>(value),
        "reasoning" => rt::<Reasoning>(value),
        "config" => rt::<Config>(value),
        "cache_config" => rt::<CacheConfig>(value),
        "cache_info" => rt::<CacheInfo>(value),
        "cache_page" => rt::<CachePage>(value),
        "cached_prefix" => rt::<CachedPrefix>(value),
        "token_logprob" => rt::<TokenLogprob>(value),
        "continuation_state" => rt::<ContinuationState>(value),
        "error_detail" => rt::<ErrorDetail>(value),
        "delta" => rt::<Delta>(value),
        "usage" => rt::<Usage>(value),
        "stream_event" => rt::<StreamEvent>(value),
        "request" => rt::<Request>(value),
        "response" => rt::<Response>(value),
        "model_info" => rt::<ModelInfo>(value),
        "batch_request" => rt::<BatchRequest>(value),
        "batch_job" => rt::<BatchJobInfo>(value),
        "batch_entry" => rt::<BatchEntry>(value),
        "file_upload_request" => rt::<FileUploadRequest>(value),
        "file_info" => rt::<FileInfo>(value),
        "file_page" => rt::<FilePage>(value),
        "image_generation_request" => rt::<ImageGenerationRequest>(value),
        "image_generation_response" => rt::<ImageGenerationResponse>(value),
        "speech_generation_request" => rt::<SpeechGenerationRequest>(value),
        "speech_generation_response" => rt::<SpeechGenerationResponse>(value),
        "video_generation_request" => rt::<VideoGenerationRequest>(value),
        "video_job" => rt::<VideoJobInfo>(value),
        "audio_format" => rt::<AudioFormat>(value),
        "live_config" => rt::<LiveConfig>(value),
        "live_client_event" => rt::<LiveClientEvent>(value),
        "live_server_event" => rt::<LiveServerEvent>(value),
        "credential" => rt::<Credential>(value),
        other => Err(ValidationError::value(format!("unknown kind: {other}"))),
    }
}

/// The `validate` op: the same read, returning the normalized form.
pub fn validate(kind: &str, value: &Value) -> Result<Value, ValidationError> {
    roundtrip(kind, value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn kinds_count_matches_protocol() {
        assert_eq!(KINDS.len(), 36);
        for kind in KINDS {
            // Every kind dispatches (an empty object fails validation, not
            // dispatch).
            let err = roundtrip(kind, &json!({})).err().map(|e| e.message);
            assert_ne!(err, Some(format!("unknown kind: {kind}")), "{kind}");
        }
        assert!(roundtrip("nope", &json!({})).is_err());
    }
}

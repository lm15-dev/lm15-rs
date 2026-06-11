//! Surface dump for the coverage ratchet (vet op `surface_dump`).
//!
//! DEVIATION (allowed by the port work-order): Rust has no runtime
//! reflection over struct fields, so this is an explicit type registry kept
//! in lockstep with `types.rs`. Each Part/Delta/StreamEvent/Tool/Live event
//! variant is reported under its canonical per-type name (TextPart, …) with
//! its wire fields including the `type` discriminator.

use serde_json::{json, Map, Value};

fn fields(names: &[&str]) -> Value {
    json!({ "fields": names })
}

pub fn surface_dump() -> Value {
    let mut types = Map::new();
    let mut add = |name: &str, f: &[&str]| {
        types.insert(name.to_string(), fields(f));
    };

    // Parts
    add("TextPart", &["type", "text", "continuation"]);
    add(
        "ThinkingPart",
        &["type", "text", "redacted", "continuation"],
    );
    add("RefusalPart", &["type", "text", "continuation"]);
    add(
        "CitationPart",
        &["type", "url", "title", "text", "continuation"],
    );
    add(
        "ImagePart",
        &[
            "type",
            "media_type",
            "data",
            "url",
            "file_id",
            "path",
            "detail",
            "continuation",
        ],
    );
    add(
        "AudioPart",
        &[
            "type",
            "media_type",
            "data",
            "url",
            "file_id",
            "path",
            "continuation",
        ],
    );
    add(
        "VideoPart",
        &[
            "type",
            "media_type",
            "data",
            "url",
            "file_id",
            "path",
            "continuation",
        ],
    );
    add(
        "DocumentPart",
        &[
            "type",
            "media_type",
            "data",
            "url",
            "file_id",
            "path",
            "continuation",
        ],
    );
    add(
        "BinaryPart",
        &[
            "type",
            "media_type",
            "data",
            "url",
            "file_id",
            "path",
            "continuation",
        ],
    );
    add(
        "ToolCallPart",
        &["type", "id", "name", "input", "continuation"],
    );
    add(
        "ToolResultPart",
        &["type", "id", "content", "name", "is_error", "continuation"],
    );

    // Message / continuation
    add("Message", &["role", "parts", "continuation"]);
    add("ContinuationState", &["provider", "kind", "data"]);

    // Tools
    add(
        "FunctionTool",
        &["type", "name", "description", "parameters"],
    );
    add("BuiltinTool", &["type", "name", "config"]);

    // Configuration
    add("ToolChoice", &["mode", "allowed", "parallel"]);
    add(
        "Reasoning",
        &["effort", "thinking_budget", "total_budget", "summary"],
    );
    add(
        "CacheConfig",
        &["mode", "retention", "key", "prefix_until_index"],
    );
    add(
        "Config",
        &[
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "stop",
            "response_format",
            "tool_choice",
            "reasoning",
            "cache",
            "extensions",
        ],
    );

    // Deltas
    add("TextDelta", &["type", "text", "part_index"]);
    add("ThinkingDelta", &["type", "text", "part_index"]);
    add(
        "AudioDelta",
        &["type", "data", "url", "file_id", "part_index", "media_type"],
    );
    add(
        "ImageDelta",
        &["type", "data", "url", "file_id", "part_index", "media_type"],
    );
    add(
        "ToolCallDelta",
        &["type", "input", "part_index", "id", "name"],
    );
    add(
        "CitationDelta",
        &["type", "text", "url", "title", "part_index"],
    );
    add(
        "ContinuationDelta",
        &["type", "provider", "kind", "data", "part_index"],
    );

    // Stream events
    add("StreamStartEvent", &["type", "id", "model"]);
    add("StreamDeltaEvent", &["type", "delta"]);
    add(
        "StreamEndEvent",
        &["type", "finish_reason", "usage", "provider_data"],
    );
    add("StreamErrorEvent", &["type", "error"]);
    add("ErrorDetail", &["code", "message", "provider_code"]);

    // Request / Response
    add(
        "Request",
        &["model", "messages", "system", "tools", "config"],
    );
    add(
        "Usage",
        &[
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "input_audio_tokens",
            "output_audio_tokens",
        ],
    );
    add(
        "Response",
        &[
            "id",
            "model",
            "message",
            "finish_reason",
            "usage",
            "provider_data",
        ],
    );

    // ModelInfo family
    add(
        "InferencePricing",
        &[
            "input_per_million",
            "output_per_million",
            "cache_read_per_million",
            "cache_write_per_million",
            "currency",
            "dimensions",
        ],
    );
    add(
        "TrainingPricing",
        &[
            "training_tokens_per_million",
            "gpu_second",
            "currency",
            "dimensions",
        ],
    );
    add(
        "InferenceModelInfo",
        &[
            "input_modalities",
            "output_modalities",
            "context_window",
            "max_output_tokens",
            "supports_reasoning",
            "reasoning_efforts",
            "pricing",
            "extensions",
        ],
    );
    add(
        "TrainingModelInfo",
        &[
            "supports_lora",
            "supports_full_finetune",
            "trainable_modalities",
            "pricing",
            "extensions",
        ],
    );
    add(
        "ModelOrigin",
        &["type", "id", "base_model", "provider_data"],
    );
    add(
        "ModelInfo",
        &[
            "id",
            "provider",
            "api_family",
            "aliases",
            "origin",
            "inference",
            "training",
            "extensions",
        ],
    );

    // Audio / Live
    add("AudioFormat", &["encoding", "sample_rate", "channels"]);
    add(
        "LiveConfig",
        &[
            "model",
            "system",
            "tools",
            "voice",
            "input_format",
            "output_format",
            "extensions",
        ],
    );
    add("LiveClientTurnEvent", &["type", "parts", "turn_complete"]);
    add("LiveClientAudioEvent", &["type", "data", "media_type"]);
    add("LiveClientImageEvent", &["type", "data", "media_type"]);
    add("LiveClientTextEvent", &["type", "text"]);
    add("LiveClientToolResultEvent", &["type", "id", "content"]);
    add("LiveClientInterruptEvent", &["type"]);
    add("LiveClientEndAudioEvent", &["type"]);
    add("LiveServerAudioEvent", &["type", "data", "media_type"]);
    add("LiveServerTextEvent", &["type", "text"]);
    add("LiveServerToolCallEvent", &["type", "id", "name", "input"]);
    add(
        "LiveServerToolCallDeltaEvent",
        &["type", "input_delta", "id", "name"],
    );
    add("LiveServerInterruptedEvent", &["type"]);
    add("LiveServerTurnEndEvent", &["type", "usage"]);
    add("LiveServerErrorEvent", &["type", "error"]);

    let enums = json!({
        "Role": ["user", "assistant", "tool", "developer"],
        "PartType": [
            "text", "image", "audio", "video", "document", "binary",
            "tool_call", "tool_result", "thinking", "refusal", "citation"
        ],
        "DeltaType": [
            "text", "thinking", "audio", "image", "tool_call", "citation",
            "continuation"
        ],
        "FinishReason": ["stop", "length", "tool_call", "content_filter", "error"],
        "ReasoningEffort": ["off", "adaptive", "minimal", "low", "medium", "high", "xhigh"],
        "ReasoningSummary": ["auto", "concise", "detailed"],
        "ErrorCode": [
            "auth", "billing", "rate_limit", "invalid_request", "context_length",
            "timeout", "server", "unsupported_model", "unsupported_feature",
            "not_configured", "transport", "provider"
        ],
        "StreamEventType": ["start", "delta", "end", "error"],
        "BatchStatus": ["submitted", "queued", "running", "completed", "failed", "cancelled"],
        "AudioEncoding": ["pcm16", "opus", "mp3", "aac"],
        "ToolChoiceMode": ["auto", "required", "none"],
        "CacheMode": ["auto", "off"],
        "CacheRetention": ["short", "long"],
        "LiveClientEventType": [
            "turn", "audio", "image", "text", "tool_result", "interrupt", "end_audio"
        ],
        "LiveServerEventType": [
            "audio", "text", "tool_call", "tool_call_delta", "interrupted",
            "turn_end", "error"
        ],
    });

    json!({ "types": Value::Object(types), "enums": enums })
}

//! The Anthropic Messages dialect, response side (module 5;
//! `lm15/providers/anthropic.py` `parse_response` / `parse_stream_events`).

use serde_json::Value;

use super::super::wire_json::{
    array_or_empty, body_object, count_of, error_detail, first_str, frame_object, id_or_none,
    index_of, object_or_empty, provider_error, str_or_empty, str_or_none, usage, Unmapped,
};
use crate::errors::{ErrorClass, Lm15Error};
use crate::sse::SseEvent;
use crate::types::{
    CitationDelta, CitationPart, ContinuationDelta, ContinuationState, Delta, FinishReason,
    JsonObject, Message, Part, Request, Response, Role, StreamDeltaEvent, StreamEndEvent,
    StreamErrorEvent, StreamEvent, StreamStartEvent, TextDelta, TextPart, ThinkingDelta,
    ThinkingPart, ToolCallDelta, ToolCallPart, Usage,
};

/// Blocks the provider executed itself (MAP-1): never parts.
pub const ANTHROPIC_PROVIDER_EXECUTED_BLOCKS: &[&str] = &[
    "server_tool_use",
    "web_search_tool_result",
    "code_execution_tool_result",
];

/// `_finish_reason`.
pub fn finish_reason_of(stop_reason: Option<&Value>, has_tool_call: bool) -> FinishReason {
    if has_tool_call {
        return FinishReason::ToolCall;
    }
    let reason = str_or_empty(stop_reason).to_lowercase();
    match reason.as_str() {
        "max_tokens" | "model_context_window_exceeded" => FinishReason::Length,
        "tool_use" | "pause_turn" => FinishReason::ToolCall,
        "refusal" | "safety" | "content_filter" => FinishReason::ContentFilter,
        _ => FinishReason::Stop,
    }
}

/// `Usage` from a Messages `usage` object. INV-029: absent counters stay
/// `None`; the total is summed from the primaries (Anthropic reports none).
pub fn usage_from_messages(provider: &str, data: &JsonObject) -> Result<Usage, Lm15Error> {
    let details = object_or_empty(data.get("output_tokens_details"));
    let c = |field: &str, value: Option<&Value>| count_of(provider, field, value);
    Ok(usage(Usage {
        input_tokens: c("input_tokens", data.get("input_tokens"))?,
        output_tokens: c("output_tokens", data.get("output_tokens"))?,
        total_tokens: None,
        cache_read_tokens: c(
            "cache_read_input_tokens",
            data.get("cache_read_input_tokens"),
        )?,
        cache_write_tokens: c(
            "cache_creation_input_tokens",
            data.get("cache_creation_input_tokens"),
        )?,
        reasoning_tokens: c("thinking_tokens", details.get("thinking_tokens"))?,
        input_audio_tokens: None,
        output_audio_tokens: None,
    }))
}

fn citation_from_block(citation: &JsonObject) -> Option<CitationPart> {
    let url = first_str(citation, &["url", "uri"]);
    let title = first_str(citation, &["title", "document_title", "source_title"]);
    let text = first_str(citation, &["cited_text", "text", "quote"]);
    if url.is_none() && title.is_none() && text.is_none() {
        return None;
    }
    Some(CitationPart {
        url,
        title,
        text,
        continuation: Vec::new(),
    })
}

fn signature_state(signature: String) -> ContinuationState {
    let mut data = JsonObject::new();
    data.insert("signature".into(), Value::String(signature));
    ContinuationState {
        provider: "anthropic".into(),
        kind: "thinking_signature".into(),
        data,
    }
}

fn redacted_state(blob: Value) -> ContinuationState {
    let mut data = JsonObject::new();
    data.insert("data".into(), blob);
    ContinuationState {
        provider: "anthropic".into(),
        kind: "redacted_thinking".into(),
        data,
    }
}

/// The class of a stream error frame's `type` (`_error_detail`).
pub fn stream_error_class(provider_code: &str, message: &str) -> ErrorClass {
    if crate::errors::anthropic_is_context_length(message) {
        return ErrorClass::ContextLengthError;
    }
    if provider_code == "not_found_error" && crate::errors::is_model_error(message) {
        return ErrorClass::UnsupportedModelError;
    }
    crate::errors::ANTHROPIC_ERROR_TYPES
        .iter()
        .find(|(k, _)| *k == provider_code)
        .map(|(_, c)| *c)
        .unwrap_or(ErrorClass::ProviderError)
}

pub fn parse_response(
    provider: &str,
    request: &Request,
    body: &[u8],
) -> Result<Response, Lm15Error> {
    let data = body_object(provider, body)?;
    let mut parts: Vec<Part> = Vec::new();
    let mut unmapped = Unmapped::default();
    for (block_index, block) in array_or_empty(data.get("content")).iter().enumerate() {
        let path = format!("content[{block_index}]");
        let Value::Object(block) = block else {
            unmapped.record_shape(path, block);
            continue;
        };
        let block_type = str_or_empty(block.get("type"));
        match block_type.as_str() {
            "text" => {
                parts.push(Part::Text(TextPart::new(str_or_empty(block.get("text")))));
                for citation in array_or_empty(block.get("citations")) {
                    let Value::Object(citation) = citation else {
                        continue;
                    };
                    if let Some(citation) = citation_from_block(citation) {
                        parts.push(Part::Citation(citation));
                    }
                }
            }
            "tool_use" => {
                let name = str_or_none(block.get("name")).ok_or_else(|| {
                    provider_error(
                        ErrorClass::ProviderError,
                        provider,
                        format!("{provider}: {path} is a tool_use block with no name; lm15 does not guess which tool the model meant"),
                        None,
                    )
                })?;
                parts.push(Part::ToolCall(ToolCallPart {
                    id: str_or_none(block.get("id"))
                        .unwrap_or_else(|| format!("tool_{}", parts.len())),
                    name,
                    input: match block.get("input") {
                        Some(Value::Object(o)) => o.clone(),
                        _ => JsonObject::new(),
                    },
                    continuation: Vec::new(),
                }));
            }
            "thinking" => {
                let continuation = match str_or_none(block.get("signature")) {
                    Some(signature) => vec![signature_state(signature)],
                    None => Vec::new(),
                };
                parts.push(Part::Thinking(ThinkingPart {
                    text: first_str(block, &["thinking", "text"]).unwrap_or_default(),
                    continuation,
                }));
            }
            "redacted_thinking" => {
                // MAP-7 rule 11: hidden thinking is empty text plus replay
                // state; the blob goes back verbatim.
                let continuation = match block.get("data") {
                    Some(Value::Null) | None => Vec::new(),
                    Some(blob) => vec![redacted_state(blob.clone())],
                };
                parts.push(Part::Thinking(ThinkingPart {
                    text: String::new(),
                    continuation,
                }));
            }
            t if ANTHROPIC_PROVIDER_EXECUTED_BLOCKS.contains(&t) => {}
            _ => unmapped.record(path, block.get("type")),
        }
    }

    if parts.is_empty() {
        parts.push(Part::Text(TextPart::new("")));
    }

    let usage = usage_from_messages(provider, object_or_empty(data.get("usage")))?;
    let has_tool = parts.iter().any(|p| matches!(p, Part::ToolCall(_)));
    Ok(Response {
        id: id_or_none(data.get("id")),
        model: str_or_none(data.get("model")).unwrap_or_else(|| request.model.clone()),
        message: Message {
            role: Role::Assistant,
            parts,
            continuation: Vec::new(),
        },
        finish_reason: finish_reason_of(data.get("stop_reason"), has_tool),
        usage,
        logprobs: None,
        provider_data: Some(unmapped.attach(data)),
    })
}

fn delta(delta: Delta) -> StreamEvent {
    StreamEvent::Delta(StreamDeltaEvent { delta })
}

pub fn parse_stream_event(
    provider: &str,
    request: &Request,
    raw: &SseEvent,
    out: &mut Vec<StreamEvent>,
) -> Result<(), Lm15Error> {
    if raw.data.is_empty() {
        return Ok(());
    }
    let Some(payload) = frame_object(&raw.data) else {
        return Err(provider_error(
            ErrorClass::ProviderError,
            provider,
            format!("{provider}: stream frame is not a JSON object"),
            None,
        ));
    };
    let et = str_or_empty(payload.get("type"));
    match et.as_str() {
        "message_start" => {
            let msg = object_or_empty(payload.get("message"));
            out.push(StreamEvent::Start(StreamStartEvent {
                id: id_or_none(msg.get("id")),
                model: Some(str_or_none(msg.get("model")).unwrap_or_else(|| request.model.clone())),
            }));
        }
        "content_block_start" => {
            let block = object_or_empty(payload.get("content_block"));
            let idx = index_of(payload.get("index"));
            match str_or_empty(block.get("type")).as_str() {
                "tool_use" => {
                    // A streamed tool_use opens with `input: {}`, a
                    // placeholder; the arguments arrive as input_json_delta
                    // fragments. A non-empty start input is kept verbatim.
                    let input = match block.get("input") {
                        Some(Value::Object(o)) if !o.is_empty() => {
                            Value::Object(o.clone()).to_string()
                        }
                        Some(Value::Object(_)) => String::new(),
                        other => str_or_empty(other),
                    };
                    out.push(delta(Delta::ToolCall(ToolCallDelta {
                        input,
                        part_index: idx,
                        id: str_or_none(block.get("id")),
                        name: str_or_none(block.get("name")),
                    })));
                }
                "redacted_thinking" => {
                    // MAP-7 rule 11: an empty thinking delta opens the hidden
                    // block; its replay state is the block's only content.
                    if let Some(blob) = block.get("data").filter(|v| !v.is_null()) {
                        out.push(delta(Delta::Thinking(ThinkingDelta {
                            text: String::new(),
                            part_index: idx,
                        })));
                        let state = redacted_state(blob.clone());
                        out.push(delta(Delta::Continuation(ContinuationDelta {
                            provider: state.provider,
                            kind: state.kind,
                            data: state.data,
                            part_index: Some(idx),
                        })));
                    }
                }
                _ => {}
            }
        }
        "content_block_delta" => {
            let d = object_or_empty(payload.get("delta"));
            let idx = index_of(payload.get("index"));
            match str_or_empty(d.get("type")).as_str() {
                "text_delta" => out.push(delta(Delta::Text(TextDelta {
                    text: str_or_empty(d.get("text")),
                    part_index: idx,
                    logprobs: Vec::new(),
                }))),
                "input_json_delta" => out.push(delta(Delta::ToolCall(ToolCallDelta {
                    input: str_or_empty(d.get("partial_json")),
                    part_index: idx,
                    id: None,
                    name: None,
                }))),
                "thinking_delta" => out.push(delta(Delta::Thinking(ThinkingDelta {
                    text: str_or_empty(d.get("thinking")),
                    part_index: idx,
                }))),
                "signature_delta" => {
                    if let Some(signature) = str_or_none(d.get("signature")) {
                        let state = signature_state(signature);
                        out.push(delta(Delta::Continuation(ContinuationDelta {
                            provider: state.provider,
                            kind: state.kind,
                            data: state.data,
                            part_index: Some(idx),
                        })));
                    }
                }
                "citation_delta" | "citations_delta" => {
                    let citation = match d.get("citation") {
                        Some(Value::Object(c)) => c,
                        _ => d,
                    };
                    out.push(delta(Delta::Citation(CitationDelta {
                        text: first_str(citation, &["cited_text", "text"]),
                        url: str_or_none(citation.get("url")),
                        title: str_or_none(citation.get("title")),
                        part_index: idx,
                    })));
                }
                _ => {}
            }
        }
        "message_delta" => {
            // The authoritative stop_reason and final usage; message_stop is
            // the bare terminator (MAP-3, D9).
            let d = object_or_empty(payload.get("delta"));
            let usage = match payload.get("usage") {
                Some(Value::Object(u)) if !u.is_empty() => Some(usage_from_messages(provider, u)?),
                _ => None,
            };
            let stop_reason = d.get("stop_reason").filter(|v| !v.is_null());
            if stop_reason.is_some() || usage.is_some() {
                out.push(StreamEvent::End(StreamEndEvent {
                    finish_reason: stop_reason.map(|r| finish_reason_of(Some(r), false)),
                    usage,
                    provider_data: Some(payload.clone()),
                }));
            }
        }
        "message_stop" => out.push(StreamEvent::End(StreamEndEvent::default())),
        "error" => {
            let (provider_code, message) = match payload.get("error") {
                Some(Value::Object(err)) => (
                    first_str(err, &["type", "code"])
                        .or_else(|| str_or_none(payload.get("code")))
                        .unwrap_or_else(|| "provider".to_string()),
                    str_or_none(err.get("message"))
                        .or_else(|| str_or_none(payload.get("message")))
                        .unwrap_or_default(),
                ),
                _ => (
                    first_str(&payload, &["code", "error_type"])
                        .unwrap_or_else(|| "provider".to_string()),
                    str_or_empty(payload.get("message")),
                ),
            };
            out.push(StreamEvent::Error(StreamErrorEvent {
                error: error_detail(
                    stream_error_class(&provider_code, &message),
                    &provider_code,
                    &message,
                ),
            }));
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    fn request() -> Request {
        Request::new("claude-x", vec![Message::user("hi").unwrap()]).unwrap()
    }

    #[test]
    fn complete_body() {
        let body = serde_json::json!({
            "id": "msg_1", "model": "claude-x-2026", "stop_reason": "tool_use",
            "content": [
                {"type": "thinking", "thinking": "hm", "signature": "sig"},
                {"type": "redacted_thinking", "data": "blob"},
                {"type": "text", "text": "Hi", "citations": [{"cited_text": "q", "url": "https://a"}]},
                {"type": "server_tool_use"},
                {"type": "tool_use", "id": "t1", "name": "f", "input": {"a": 1}},
                {"type": "odd"}
            ],
            "usage": {"input_tokens": 3, "output_tokens": 5, "cache_read_input_tokens": 0,
                      "output_tokens_details": {"thinking_tokens": 2}}
        });
        let r = parse_response("anthropic", &request(), body.to_string().as_bytes()).unwrap();
        assert_eq!(r.message.parts.len(), 5);
        assert_eq!(r.finish_reason, FinishReason::ToolCall);
        assert_eq!(r.usage.total_tokens, Some(8));
        assert_eq!(r.usage.cache_read_tokens, Some(0));
        assert_eq!(r.usage.reasoning_tokens, Some(2));
        match &r.message.parts[1] {
            Part::Thinking(t) => assert_eq!(t.continuation[0].kind, "redacted_thinking"),
            other => panic!("{other:?}"),
        }
        assert_eq!(
            r.provider_data.unwrap()["_lm15_unmapped"][0]["path"],
            "content[5]"
        );
    }

    #[test]
    fn stream_frames() {
        let mut out = Vec::new();
        for data in [
            r#"{"type": "message_start", "message": {"id": "msg_1", "model": "claude-x"}}"#,
            r#"{"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "t", "name": "f", "input": {}}}"#,
            r#"{"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{}"}}"#,
            r#"{"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 4}}"#,
            r#"{"type": "message_stop"}"#,
        ] {
            parse_stream_event(
                "anthropic",
                &request(),
                &SseEvent {
                    event: None,
                    data: data.into(),
                },
                &mut out,
            )
            .unwrap();
        }
        assert_eq!(out.len(), 5);
        assert!(
            matches!(&out[1], StreamEvent::Delta(d) if matches!(&d.delta, Delta::ToolCall(t) if t.input.is_empty() && t.name.as_deref() == Some("f")))
        );
        assert!(
            matches!(&out[3], StreamEvent::End(e) if e.finish_reason == Some(FinishReason::ToolCall) && e.usage.unwrap().output_tokens == Some(4))
        );
    }
}

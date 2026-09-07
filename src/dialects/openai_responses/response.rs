//! The Responses API, response side (module 5): a complete body to a
//! `Response` (`lm15/providers/openai.py` `parse_response`) and one SSE
//! frame to canonical events (`parse_stream_events`). Tables are the
//! reference's, copied as data.

use serde_json::Value;

use super::super::wire_json::{
    array_or_empty, body_object, count_of, error_detail, first_str, frame_object, id_or_none,
    index_of, object_or_empty, openai_token_logprobs, parse_json_object, provider_error,
    str_or_empty, str_or_none, truthy, usage, Unmapped,
};
use crate::errors::{ErrorClass, Lm15Error};
use crate::sse::SseEvent;
use crate::types::{
    AudioDelta, AudioPart, CitationDelta, CitationPart, ContinuationDelta, ContinuationState,
    Delta, FinishReason, ImageDelta, ImagePart, JsonObject, Message, Part, RefusalPart, Request,
    Response, Role, StreamDeltaEvent, StreamEndEvent, StreamErrorEvent, StreamEvent,
    StreamStartEvent, TextDelta, TextPart, ThinkingDelta, ThinkingPart, TokenLogprob,
    ToolCallDelta, ToolCallPart, Usage,
};

/// Output items the provider executed itself (MAP-1): never parts.
pub const OPENAI_PROVIDER_EXECUTED_ITEMS: &[&str] = &[
    "web_search_call",
    "file_search_call",
    "code_interpreter_call",
    "computer_call",
    "computer_use_call",
];

/// `OpenAILM._response_error_code_map` (`lm15/providers/openai.py`): an
/// in-band `{"error": {"code"}}` envelope on a 2xx body.
pub const RESPONSE_ERROR_CODES: &[(&str, ErrorClass)] = &[
    ("server_error", ErrorClass::ServerError),
    ("rate_limit_exceeded", ErrorClass::RateLimitError),
    ("invalid_prompt", ErrorClass::InvalidRequestError),
    ("vector_store_timeout", ErrorClass::TimeoutError),
    ("invalid_image", ErrorClass::InvalidRequestError),
    ("invalid_image_format", ErrorClass::InvalidRequestError),
    ("invalid_base64_image", ErrorClass::InvalidRequestError),
    ("invalid_image_url", ErrorClass::InvalidRequestError),
    ("image_too_large", ErrorClass::InvalidRequestError),
    ("image_too_small", ErrorClass::InvalidRequestError),
    ("image_parse_error", ErrorClass::InvalidRequestError),
    (
        "image_content_policy_violation",
        ErrorClass::InvalidRequestError,
    ),
    ("invalid_image_mode", ErrorClass::InvalidRequestError),
    ("image_file_too_large", ErrorClass::InvalidRequestError),
    (
        "unsupported_image_media_type",
        ErrorClass::InvalidRequestError,
    ),
    ("empty_image_file", ErrorClass::InvalidRequestError),
    ("failed_to_download_image", ErrorClass::InvalidRequestError),
    ("image_file_not_found", ErrorClass::InvalidRequestError),
    ("model_not_found", ErrorClass::UnsupportedModelError),
    ("model_not_available", ErrorClass::UnsupportedModelError),
    ("unsupported_model", ErrorClass::UnsupportedModelError),
    ("DeploymentNotFound", ErrorClass::UnsupportedModelError),
];

/// `_stream_error_code_map`: the response table plus the HTTP-level codes
/// a stream error frame may carry.
pub const STREAM_ERROR_CODES: &[(&str, ErrorClass)] = &[
    ("context_length_exceeded", ErrorClass::ContextLengthError),
    ("invalid_api_key", ErrorClass::AuthError),
    ("insufficient_quota", ErrorClass::BillingError),
    ("1113", ErrorClass::BillingError),
    ("exceeded_current_quota_error", ErrorClass::BillingError),
    ("authentication_error", ErrorClass::AuthError),
    ("rate_limit_error", ErrorClass::RateLimitError),
];

fn lookup(table: &[(&str, ErrorClass)], code: &str) -> Option<ErrorClass> {
    table.iter().find(|(k, _)| *k == code).map(|(_, c)| *c)
}

/// The in-band error of a 2xx body (`_response_error`): the class from
/// the response table, else `ServerError`.
pub fn response_error(provider: &str, error: &JsonObject) -> Lm15Error {
    let code = str_or_empty(error.get("code"));
    let message = match error.get("message") {
        Some(m) if truthy(Some(m)) => super::super::wire_json::py_str(m),
        _ => Value::Object(error.clone()).to_string(),
    };
    let class = lookup(RESPONSE_ERROR_CODES, &code).unwrap_or(ErrorClass::ServerError);
    let msg = if !message.is_empty() {
        message
    } else if !code.is_empty() {
        code.clone()
    } else {
        "provider error".to_string()
    };
    provider_error(class, provider, msg, Some(code))
}

/// The class of a stream error frame's code (`_error_detail`).
pub fn stream_error_class(provider_code: &str) -> ErrorClass {
    lookup(STREAM_ERROR_CODES, provider_code)
        .or_else(|| lookup(RESPONSE_ERROR_CODES, provider_code))
        .unwrap_or(ErrorClass::ProviderError)
}

/// `Usage` from a Responses `usage` object (both the complete body and
/// `response.completed`).
pub fn usage_from_responses(provider: &str, data: &JsonObject) -> Result<Usage, Lm15Error> {
    let input = object_or_empty(data.get("input_tokens_details"));
    let output = object_or_empty(data.get("output_tokens_details"));
    let c = |field: &str, value: Option<&Value>| count_of(provider, field, value);
    Ok(usage(Usage {
        input_tokens: c("input_tokens", data.get("input_tokens"))?,
        output_tokens: c("output_tokens", data.get("output_tokens"))?,
        total_tokens: c("total_tokens", data.get("total_tokens"))?,
        cache_read_tokens: c("cached_tokens", input.get("cached_tokens"))?,
        cache_write_tokens: c("cache_write_tokens", input.get("cache_write_tokens"))?,
        reasoning_tokens: c("reasoning_tokens", output.get("reasoning_tokens"))?,
        input_audio_tokens: c("input audio_tokens", input.get("audio_tokens"))?,
        output_audio_tokens: c("output audio_tokens", output.get("audio_tokens"))?,
    }))
}

/// `_finish_from_status`.
fn finish_from_status(data: &JsonObject, has_tool_call: bool) -> FinishReason {
    if has_tool_call {
        return FinishReason::ToolCall;
    }
    let status = str_or_empty(data.get("status")).to_lowercase();
    let reason = match data.get("incomplete_details") {
        Some(Value::Object(incomplete)) => str_or_empty(incomplete.get("reason")).to_lowercase(),
        _ => String::new(),
    };
    if status == "incomplete" && reason.contains("token") {
        return FinishReason::Length;
    }
    if reason.contains("content_filter") || reason.contains("safety") {
        return FinishReason::ContentFilter;
    }
    FinishReason::Stop
}

// ─── Citations (`_annotation_text`, `_citation_from_openai_annotation`) ──

fn annotation_text(annotation: &JsonObject, source_text: Option<&str>) -> Option<String> {
    for key in ["text", "snippet", "cited_text", "quote"] {
        if let Some(text) = str_or_none(annotation.get(key)) {
            return Some(text);
        }
    }
    let start = int_or_none(annotation.get("start_index"))?;
    let end = int_or_none(annotation.get("end_index"))?;
    let source = source_text?;
    if start < end && end <= source.chars().count() {
        let cited: String = source.chars().skip(start).take(end - start).collect();
        return Some(cited);
    }
    None
}

fn int_or_none(value: Option<&Value>) -> Option<usize> {
    match value {
        Some(Value::Bool(_)) | None => None,
        Some(Value::Number(n)) => n
            .as_i64()
            .or_else(|| n.as_f64().map(|f| f as i64))
            .and_then(|i| usize::try_from(i).ok()),
        Some(Value::String(s)) => s
            .trim()
            .parse::<i64>()
            .ok()
            .and_then(|i| usize::try_from(i).ok()),
        _ => None,
    }
}

fn citation_from_annotation(
    annotation: &JsonObject,
    source_text: Option<&str>,
) -> Option<CitationPart> {
    let url = first_str(annotation, &["url", "uri"]);
    let title = first_str(annotation, &["title", "filename", "file_id"]);
    let text = annotation_text(annotation, source_text);
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

// ─── Complete body ───────────────────────────────────────────────────

pub fn parse_response(
    provider: &str,
    request: &Request,
    body: &[u8],
) -> Result<Response, Lm15Error> {
    let data = body_object(provider, body)?;
    if let Some(Value::Object(error)) = data.get("error") {
        return Err(response_error(provider, error));
    }

    let mut parts: Vec<Part> = Vec::new();
    let mut unmapped = Unmapped::default();
    let mut logprobs: Vec<TokenLogprob> = Vec::new();
    for (item_index, item) in array_or_empty(data.get("output")).iter().enumerate() {
        let Value::Object(item) = item else {
            unmapped.record_shape(format!("output[{item_index}]"), item);
            continue;
        };
        let item_type = str_or_empty(item.get("type"));
        match item_type.as_str() {
            "message" => {
                for (content_index, content) in
                    array_or_empty(item.get("content")).iter().enumerate()
                {
                    let path = format!("output[{item_index}].content[{content_index}]");
                    let Value::Object(content) = content else {
                        unmapped.record_shape(path, content);
                        continue;
                    };
                    let ctype = str_or_empty(content.get("type"));
                    match ctype.as_str() {
                        "output_text" | "text" => {
                            let text = str_or_empty(content.get("text"));
                            parts.push(Part::Text(TextPart::new(text.clone())));
                            // Per-block wire lists concatenate in document
                            // order into the message-level sequence.
                            logprobs.extend(openai_token_logprobs(content.get("logprobs")));
                            for annotation in array_or_empty(content.get("annotations")) {
                                let Value::Object(annotation) = annotation else {
                                    continue;
                                };
                                if let Some(citation) =
                                    citation_from_annotation(annotation, Some(&text))
                                {
                                    parts.push(Part::Citation(citation));
                                }
                            }
                        }
                        "refusal" => {
                            let text = first_str(content, &["refusal", "text"]);
                            parts.push(match text {
                                Some(text) => Part::Refusal(RefusalPart {
                                    text,
                                    continuation: Vec::new(),
                                }),
                                None => Part::Text(TextPart::new("")),
                            });
                        }
                        "output_image" => {
                            if let Some(b64) = first_str(content, &["b64_json", "image_base64"]) {
                                parts.push(Part::Image(ImagePart {
                                    media_type: "image/png".into(),
                                    data: Some(b64),
                                    ..Default::default()
                                }));
                            }
                        }
                        "output_audio" => {
                            let audio = object_or_empty(content.get("audio"));
                            let b64 = str_or_none(audio.get("data"))
                                .or_else(|| str_or_none(content.get("b64_json")));
                            if let Some(b64) = b64 {
                                parts.push(Part::Audio(AudioPart {
                                    media_type: "audio/wav".into(),
                                    data: Some(b64),
                                    ..Default::default()
                                }));
                            }
                        }
                        _ => unmapped.record(path, content.get("type")),
                    }
                }
            }
            "function_call" => {
                let id = first_str(item, &["call_id", "id"])
                    .unwrap_or_else(|| format!("call_{}", parts.len()));
                let name = str_or_none(item.get("name"))
                    .ok_or_else(|| unnamed_call(provider, &format!("output[{item_index}]")))?;
                parts.push(Part::ToolCall(ToolCallPart {
                    id,
                    name,
                    input: parse_json_object(item.get("arguments")),
                    continuation: Vec::new(),
                }));
            }
            "reasoning" => {
                // MAP-7 rule 8: the summary is the visible text; id and
                // encrypted_content are replay state.
                let text = match item.get("summary") {
                    Some(Value::Array(entries)) => entries
                        .iter()
                        .map(|entry| match entry {
                            Value::Object(o) => {
                                o.get("text").map(str_or_empty_value).unwrap_or_default()
                            }
                            other => super::super::wire_json::py_str(other),
                        })
                        .collect::<Vec<_>>()
                        .join("\n"),
                    other => str_or_none(other)
                        .or_else(|| str_or_none(item.get("text")))
                        .unwrap_or_default(),
                };
                let mut state = JsonObject::new();
                if let Some(id) = str_or_none(item.get("id")) {
                    state.insert("id".into(), Value::String(id));
                }
                if let Some(enc) = str_or_none(item.get("encrypted_content")) {
                    state.insert("encrypted_content".into(), Value::String(enc));
                }
                let continuation = if state.is_empty() {
                    Vec::new()
                } else {
                    vec![reasoning_state(state)]
                };
                if !text.is_empty() || !continuation.is_empty() {
                    parts.push(Part::Thinking(ThinkingPart { text, continuation }));
                }
            }
            t if OPENAI_PROVIDER_EXECUTED_ITEMS.contains(&t) => {}
            _ => unmapped.record(format!("output[{item_index}]"), item.get("type")),
        }
    }

    if parts.is_empty() {
        parts.push(Part::Text(TextPart::new(str_or_empty(
            data.get("output_text"),
        ))));
    }

    let usage = usage_from_responses(provider, object_or_empty(data.get("usage")))?;
    let has_tool = parts.iter().any(|p| matches!(p, Part::ToolCall(_)));
    let finish_reason = finish_from_status(&data, has_tool);
    Ok(Response {
        id: id_or_none(data.get("id")),
        model: str_or_none(data.get("model")).unwrap_or_else(|| request.model.clone()),
        message: Message {
            role: Role::Assistant,
            parts,
            continuation: Vec::new(),
        },
        finish_reason,
        usage,
        logprobs: if logprobs.is_empty() {
            None
        } else {
            Some(logprobs)
        },
        provider_data: Some(unmapped.attach(data)),
    })
}

fn str_or_empty_value(value: &Value) -> String {
    str_or_empty(Some(value))
}

fn reasoning_state(data: JsonObject) -> ContinuationState {
    ContinuationState {
        provider: "openai".into(),
        kind: "reasoning_item".into(),
        data,
    }
}

/// A complete body's `function_call` without a name is not actionable
/// (MAP-1) and lm15 never guesses a tool identity (MAP-9 applied to the
/// complete path; the reference substitutes `"tool"`, stated in README).
fn unnamed_call(provider: &str, path: &str) -> Lm15Error {
    provider_error(
        ErrorClass::ProviderError,
        provider,
        format!("{provider}: {path} is a function_call with no name; lm15 does not guess which tool the model meant"),
        None,
    )
}

// ─── Stream frames ───────────────────────────────────────────────────

pub fn parse_stream_event(
    provider: &str,
    request: &Request,
    raw: &SseEvent,
    out: &mut Vec<StreamEvent>,
) -> Result<(), Lm15Error> {
    if raw.data.is_empty() {
        return Ok(());
    }
    if raw.data == "[DONE]" {
        // A bare terminator: no finish reason, no usage (MAP-3).
        out.push(StreamEvent::End(StreamEndEvent::default()));
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
    let index = || index_of(payload.get("output_index"));

    // Reasoning items (MAP-7.9): the added frame opens the thinking slot;
    // the done frame carries the replay state after all summary fragments.
    if et == "response.output_item.added" || et == "response.output_item.done" {
        if let Some(Value::Object(item)) = payload.get("item") {
            if str_or_empty(item.get("type")) == "reasoning" {
                if et == "response.output_item.added" {
                    out.push(delta(Delta::Thinking(ThinkingDelta {
                        text: String::new(),
                        part_index: index(),
                    })));
                } else {
                    let mut state = JsonObject::new();
                    for key in ["id", "encrypted_content"] {
                        if truthy(item.get(key)) {
                            state.insert(key.into(), item[key].clone());
                        }
                    }
                    if !state.is_empty() {
                        out.push(delta(Delta::Continuation(ContinuationDelta {
                            provider: "openai".into(),
                            kind: "reasoning_item".into(),
                            data: state,
                            part_index: Some(index()),
                        })));
                    }
                }
                return Ok(());
            }
        }
    }

    match et.as_str() {
        "response.created" => {
            let response = object_or_empty(payload.get("response"));
            out.push(StreamEvent::Start(StreamStartEvent {
                id: id_or_none(response.get("id")),
                model: Some(
                    str_or_none(response.get("model")).unwrap_or_else(|| request.model.clone()),
                ),
            }));
        }
        "response.output_text.delta" | "response.refusal.delta" => {
            out.push(delta(Delta::Text(TextDelta {
                text: str_or_empty(payload.get("delta")),
                part_index: index(),
                logprobs: openai_token_logprobs(payload.get("logprobs")),
            })));
        }
        "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
            out.push(delta(Delta::Thinking(ThinkingDelta {
                text: str_or_empty(payload.get("delta")),
                part_index: index(),
            })));
        }
        "response.output_text.annotation.added" => {
            if let Some(Value::Object(annotation)) = payload.get("annotation") {
                if let Some(citation) = citation_from_annotation(annotation, None) {
                    out.push(delta(Delta::Citation(CitationDelta {
                        text: citation.text,
                        url: citation.url,
                        title: citation.title,
                        part_index: index(),
                    })));
                }
            }
        }
        "response.output_audio.delta" => {
            out.push(delta(Delta::Audio(AudioDelta {
                data: Some(str_or_empty(payload.get("delta"))),
                part_index: index(),
                media_type: Some("audio/wav".into()),
                ..Default::default()
            })));
        }
        "response.output_image.delta" | "response.image.delta" => {
            out.push(delta(Delta::Image(ImageDelta {
                data: Some(str_or_empty(payload.get("delta"))),
                part_index: index(),
                media_type: Some("image/png".into()),
                ..Default::default()
            })));
        }
        "response.output_item.added" => {
            let item = object_or_empty(payload.get("item"));
            if str_or_empty(item.get("type")) == "function_call" {
                out.push(delta(Delta::ToolCall(ToolCallDelta {
                    input: str_or_empty(item.get("arguments")),
                    part_index: index(),
                    id: first_str(item, &["call_id", "id"]),
                    name: str_or_none(item.get("name")),
                })));
            }
        }
        "response.function_call_arguments.delta" => {
            out.push(delta(Delta::ToolCall(ToolCallDelta {
                input: str_or_empty(payload.get("delta")),
                part_index: index(),
                id: first_str(&payload, &["call_id", "id"]),
                name: str_or_none(payload.get("name")),
            })));
        }
        "response.completed" => {
            let response = object_or_empty(payload.get("response"));
            let usage = usage_from_responses(provider, object_or_empty(response.get("usage")))?;
            let has_tool = array_or_empty(response.get("output")).iter().any(|item| {
                matches!(item, Value::Object(o) if str_or_empty(o.get("type")) == "function_call")
            });
            out.push(StreamEvent::End(StreamEndEvent {
                finish_reason: Some(if has_tool {
                    FinishReason::ToolCall
                } else {
                    FinishReason::Stop
                }),
                usage: Some(usage),
                provider_data: match payload.get("response") {
                    Some(Value::Object(o)) => Some(o.clone()),
                    _ => None,
                },
            }));
        }
        "response.error" | "error" => {
            let (provider_code, message) = match payload.get("error") {
                Some(Value::Object(err)) => (
                    first_str(err, &["code", "type"])
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
                error: error_detail(stream_error_class(&provider_code), &provider_code, &message),
            }));
        }
        _ => {}
    }
    Ok(())
}

fn delta(delta: Delta) -> StreamEvent {
    StreamEvent::Delta(StreamDeltaEvent { delta })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    fn request() -> Request {
        Request::new("gpt-x", vec![Message::user("hi").unwrap()]).unwrap()
    }

    #[test]
    fn complete_body_maps_text_calls_reasoning_and_usage() {
        let body = serde_json::json!({
            "id": "resp_1", "model": "gpt-x-2026", "status": "completed",
            "output": [
                {"type": "reasoning", "id": "rs_1", "encrypted_content": "enc", "summary": [{"type": "summary_text", "text": "plan"}]},
                {"type": "message", "content": [
                    {"type": "output_text", "text": "Hello", "annotations": [{"type": "url_citation", "url": "https://a", "title": "A", "start_index": 0, "end_index": 2}]},
                    {"type": "refusal", "refusal": "no"}
                ]},
                {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{\"a\": 1}"},
                {"type": "web_search_call"},
                {"type": "mystery"}
            ],
            "usage": {"input_tokens": 3, "output_tokens": 4.0, "total_tokens": 7, "output_tokens_details": {"reasoning_tokens": 2}}
        });
        let response = parse_response("openai", &request(), body.to_string().as_bytes()).unwrap();
        assert_eq!(response.id.as_deref(), Some("resp_1"));
        assert_eq!(response.model, "gpt-x-2026");
        assert_eq!(response.finish_reason, FinishReason::ToolCall);
        assert_eq!(response.message.parts.len(), 5);
        match &response.message.parts[0] {
            Part::Thinking(t) => {
                assert_eq!(t.text, "plan");
                assert_eq!(t.continuation[0].kind, "reasoning_item");
            }
            other => panic!("{other:?}"),
        }
        match &response.message.parts[2] {
            Part::Citation(c) => assert_eq!(c.text.as_deref(), Some("He")),
            other => panic!("{other:?}"),
        }
        assert_eq!(response.usage.output_tokens, Some(4));
        assert_eq!(response.usage.reasoning_tokens, Some(2));
        let unmapped = &response.provider_data.unwrap()["_lm15_unmapped"];
        assert_eq!(unmapped[0]["path"], "output[4]");
        assert_eq!(unmapped[0]["type"], "mystery");
    }

    #[test]
    fn in_band_error_and_empty_output() {
        let err = parse_response(
            "openai",
            &request(),
            br#"{"error": {"code": "rate_limit_exceeded", "message": "slow"}}"#,
        )
        .unwrap_err();
        assert_eq!(err.class_name(), "RateLimitError");
        assert_eq!(err.provider_code(), Some("rate_limit_exceeded"));
        let response = parse_response(
            "openai",
            &request(),
            br#"{"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": []}"#,
        )
        .unwrap();
        assert_eq!(response.message.parts, vec![Part::text("")]);
        assert_eq!(response.finish_reason, FinishReason::Length);
        assert_eq!(response.model, "gpt-x");
    }

    #[test]
    fn unnamed_function_call_refuses() {
        let err = parse_response(
            "openai",
            &request(),
            br#"{"output": [{"type": "function_call", "call_id": "c", "arguments": "{}"}]}"#,
        )
        .unwrap_err();
        assert_eq!(err.class_name(), "ProviderError");
    }

    #[test]
    fn stream_frames() {
        let mut out = Vec::new();
        let frames = [
            r#"{"type": "response.created", "response": {"id": "r1", "model": "m"}}"#,
            r#"{"type": "response.output_item.added", "output_index": 0, "item": {"type": "reasoning"}}"#,
            r#"{"type": "response.reasoning_summary_text.delta", "output_index": 0, "delta": "think"}"#,
            r#"{"type": "response.output_item.done", "output_index": 0, "item": {"type": "reasoning", "id": "rs"}}"#,
            r#"{"type": "response.output_item.added", "output_index": 1, "item": {"type": "function_call", "call_id": "c", "name": "f", "arguments": ""}}"#,
            r#"{"type": "response.function_call_arguments.delta", "output_index": 1, "delta": "{}"}"#,
            r#"{"type": "response.completed", "response": {"output": [{"type": "function_call"}], "usage": {"input_tokens": 1, "output_tokens": 1}}}"#,
            "[DONE]",
        ];
        for data in frames {
            parse_stream_event(
                "openai",
                &request(),
                &SseEvent {
                    event: None,
                    data: data.into(),
                },
                &mut out,
            )
            .unwrap();
        }
        assert_eq!(out.len(), 8);
        assert!(matches!(&out[0], StreamEvent::Start(s) if s.id.as_deref() == Some("r1")));
        assert!(
            matches!(&out[3], StreamEvent::Delta(d) if matches!(&d.delta, Delta::Continuation(c) if c.part_index == Some(0)))
        );
        assert!(
            matches!(&out[6], StreamEvent::End(e) if e.finish_reason == Some(FinishReason::ToolCall) && e.usage.unwrap().total_tokens == Some(2))
        );
        assert!(
            matches!(&out[7], StreamEvent::End(e) if e.finish_reason.is_none() && e.usage.is_none())
        );
    }
}

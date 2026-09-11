//! The Chat Completions dialect, response side (module 5;
//! `lm15/providers/openai_chat.py` `parse_response` /
//! `parse_stream_events`). The error tables are the Responses dialect's
//! (the envelope family is shared).

use serde_json::Value;

use super::super::openai_responses::response::{response_error, stream_error_class};
use super::super::wire_json::{
    array_or_empty, body_object, count_of, error_detail, first_str, frame_object, id_or_none,
    index_of, object_or_empty, openai_token_logprobs, parse_json_object, provider_error,
    str_or_empty, str_or_none, truthy, usage, Unmapped,
};
use crate::errors::{ErrorClass, Lm15Error};
use crate::sse::SseEvent;
use crate::types::{
    Delta, FinishReason, JsonObject, Message, Part, RefusalPart, Request, Response, Role,
    StreamDeltaEvent, StreamEndEvent, StreamErrorEvent, StreamEvent, TextDelta, TextPart,
    ThinkingDelta, ThinkingPart, ToolCallDelta, ToolCallPart, Usage,
};

/// `_FINISH_REASON_MAP` (`lm15/providers/openai_chat.py:83-89`).
pub const FINISH_REASONS: &[(&str, FinishReason)] = &[
    ("stop", FinishReason::Stop),
    ("length", FinishReason::Length),
    ("tool_calls", FinishReason::ToolCall),
    ("function_call", FinishReason::ToolCall),
    ("content_filter", FinishReason::ContentFilter),
];

fn finish_reason_of(raw: &str) -> Option<FinishReason> {
    FINISH_REASONS
        .iter()
        .find(|(k, _)| *k == raw)
        .map(|(_, r)| *r)
}

/// `_usage_from_chat`.
pub fn usage_from_chat(provider: &str, data: &JsonObject) -> Result<Usage, Lm15Error> {
    let prompt = object_or_empty(data.get("prompt_tokens_details"));
    let completion = object_or_empty(data.get("completion_tokens_details"));
    let c = |field: &str, value: Option<&Value>| count_of(provider, field, value);
    Ok(usage(Usage {
        input_tokens: c("prompt_tokens", data.get("prompt_tokens"))?,
        output_tokens: c("completion_tokens", data.get("completion_tokens"))?,
        total_tokens: c("total_tokens", data.get("total_tokens"))?,
        cache_read_tokens: c("cached_tokens", prompt.get("cached_tokens"))?,
        cache_write_tokens: c("cache_write_tokens", prompt.get("cache_write_tokens"))?,
        reasoning_tokens: c("reasoning_tokens", completion.get("reasoning_tokens"))?,
        input_audio_tokens: c("prompt audio_tokens", prompt.get("audio_tokens"))?,
        output_audio_tokens: c("completion audio_tokens", completion.get("audio_tokens"))?,
    }))
}

pub fn parse_response(
    provider: &str,
    request: &Request,
    body: &[u8],
) -> Result<Response, Lm15Error> {
    let data = body_object(provider, body)?;
    response_from_chat_body(provider, data, Some(&request.model), None)
}

/// A Chat Completions response body → the canonical `Response` (MAP-12 rule
/// 9; `lm15/providers/openai_chat.py` `response_from_openai_chat`). The
/// reading-side twin of `request_from_openai_chat`: `body` is the JSON object
/// a Chat Completions server (or a client library imitating one — litellm's
/// `ModelResponse.model_dump()`) returned. It is the same reader
/// `parse_response` runs on provider traffic. `model` fills `Response.model`
/// when the body carries none; `choice` names the choice to read — unset, a
/// body with several choices is refused. Keys the reader does not know are
/// neither refused nor lost: the whole body is `provider_data`. No compat is
/// taken: the response shape does not vary by server.
pub fn response_from_openai_chat(
    provider: &str,
    body: &Value,
    model: Option<&str>,
    choice: Option<usize>,
) -> Result<Response, Lm15Error> {
    let Value::Object(data) = body else {
        return Err(malformed_body(
            provider,
            "a Chat Completions response body is a JSON object",
        ));
    };
    response_from_chat_body(provider, data.clone(), model, choice)
}

fn malformed_body(provider: &str, message: &str) -> Lm15Error {
    let mut meta = crate::errors::ErrorMeta::new(format!("{provider}: {message}"));
    meta.provider = Some(provider.to_string());
    Lm15Error::InvalidRequestError(meta)
}

/// The one Chat Completions response reader: `parse_response` for provider
/// traffic and `response_from_openai_chat` for a foreign body share it.
/// `choice` names the choice to read; `None` means "the only one", and a
/// body with several choices is then refused rather than silently reduced
/// to its first (the reading-side twin of MAP-12's refusal of `n`).
fn response_from_chat_body(
    provider: &str,
    data: JsonObject,
    model: Option<&str>,
    choice: Option<usize>,
) -> Result<Response, Lm15Error> {
    if let Some(Value::Object(error)) = data.get("error") {
        return Err(response_error(provider, error));
    }

    let mut parts: Vec<Part> = Vec::new();
    let mut unmapped = Unmapped::default();
    let choices = match data.get("choices") {
        None | Some(Value::Null) => &[][..],
        Some(Value::Array(items)) => items.as_slice(),
        Some(_) => return Err(malformed_body(provider, "choices must be an array")),
    };
    let index = match choice {
        None => {
            if choices.len() > 1 {
                return Err(super::text::unsupported(
                    provider,
                    format!(
                        "the body carries {} choices; a canonical Response is one message — \
                         name the choice to read (choice=i) and read each one, or send no n",
                        choices.len()
                    ),
                ));
            }
            0
        }
        Some(i) => {
            if i >= choices.len() {
                return Err(malformed_body(
                    provider,
                    &format!(
                        "choice={i} but the body carries {} choice(s)",
                        choices.len()
                    ),
                ));
            }
            i
        }
    };
    let path = format!("choices[{index}]");
    let choice = match choices.get(index) {
        Some(Value::Object(choice)) => choice,
        Some(other) => {
            unmapped.record_shape(path.clone(), other);
            object_or_empty(None)
        }
        None => object_or_empty(None),
    };
    let message = object_or_empty(choice.get("message"));

    if let Some(reasoning) = first_str(message, &["reasoning_content", "reasoning"]) {
        parts.push(Part::Thinking(ThinkingPart::new(reasoning)));
    }

    match message.get("content") {
        Some(Value::String(content)) => {
            if !content.is_empty() {
                parts.push(Part::Text(TextPart::new(content.clone())));
            }
        }
        Some(Value::Array(items)) => {
            for (content_index, item) in items.iter().enumerate() {
                match item {
                    Value::Object(o) if str_or_empty(o.get("type")) == "text" => {
                        parts.push(Part::Text(TextPart::new(str_or_empty(o.get("text")))));
                    }
                    Value::Object(o) => unmapped.record(
                        format!("{path}.message.content[{content_index}]"),
                        o.get("type"),
                    ),
                    other => unmapped
                        .record_shape(format!("{path}.message.content[{content_index}]"), other),
                }
            }
        }
        Some(Value::Null) | None => {}
        Some(other) => unmapped.record_shape(format!("{path}.message.content"), other),
    }

    if let Some(refusal) = str_or_none(message.get("refusal")) {
        parts.push(Part::Refusal(RefusalPart {
            text: refusal,
            continuation: Vec::new(),
        }));
    }

    for (call_index, call) in array_or_empty(message.get("tool_calls")).iter().enumerate() {
        let path = format!("{path}.message.tool_calls[{call_index}]");
        let Value::Object(call) = call else {
            unmapped.record_shape(path, call);
            continue;
        };
        let call_type = str_or_none(call.get("type")).unwrap_or_else(|| "function".to_string());
        if call_type != "function" {
            unmapped.record_text(path, &call_type);
            continue;
        }
        let function = object_or_empty(call.get("function"));
        let name = str_or_none(function.get("name")).ok_or_else(|| {
            provider_error(
                ErrorClass::ProviderError,
                provider,
                format!("{provider}: {path} is a tool call with no function.name; lm15 does not guess which tool the model meant"),
                None,
            )
        })?;
        parts.push(Part::ToolCall(ToolCallPart {
            id: str_or_none(call.get("id")).unwrap_or_else(|| format!("call_{}", parts.len())),
            name,
            input: parse_json_object(function.get("arguments")),
            continuation: Vec::new(),
        }));
    }

    if parts.is_empty() {
        // MAP-2: a response message is never empty.
        parts.push(Part::Text(TextPart::new("")));
    }

    let has_tool = parts.iter().any(|p| matches!(p, Part::ToolCall(_)));
    let finish_reason = if has_tool {
        FinishReason::ToolCall
    } else {
        match choice.get("finish_reason") {
            None | Some(Value::Null) => FinishReason::Stop,
            Some(Value::String(s)) if s.is_empty() => FinishReason::Stop,
            Some(raw) => {
                let text = super::super::wire_json::py_str(raw);
                match finish_reason_of(&text) {
                    Some(reason) => reason,
                    None => {
                        unmapped.record_text(format!("{path}.finish_reason"), &text);
                        FinishReason::Stop
                    }
                }
            }
        }
    };
    let usage = usage_from_chat(provider, object_or_empty(data.get("usage")))?;
    // choices[0].logprobs.content is the message-level token sequence;
    // refusal logprobs stay in provider_data.
    let logprobs = openai_token_logprobs(object_or_empty(choice.get("logprobs")).get("content"));
    Ok(Response {
        id: id_or_none(data.get("id")),
        model: match str_or_none(data.get("model")).or_else(|| model.map(str::to_string)) {
            Some(m) => m,
            None => {
                return Err(malformed_body(
                    provider,
                    "the body carries no model; pass model=",
                ))
            }
        },
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

pub fn parse_stream_event(
    provider: &str,
    _request: &Request,
    raw: &SseEvent,
    out: &mut Vec<StreamEvent>,
) -> Result<(), Lm15Error> {
    if raw.data.is_empty() {
        return Ok(());
    }
    if raw.data == "[DONE]" {
        out.push(StreamEvent::End(StreamEndEvent::default()));
        return Ok(());
    }
    let Some(payload) = frame_object(&raw.data) else {
        // A non-object JSON frame is ignored (the reference's
        // `isinstance` guard); text that is not JSON is a broken stream.
        return if serde_json::from_str::<Value>(&raw.data).is_ok() {
            Ok(())
        } else {
            Err(provider_error(
                ErrorClass::ProviderError,
                provider,
                format!("{provider}: stream frame is not JSON"),
                None,
            ))
        };
    };

    if let Some(Value::Object(err)) = payload.get("error") {
        let provider_code =
            first_str(err, &["code", "type"]).unwrap_or_else(|| "provider".to_string());
        let message = str_or_empty(err.get("message"));
        out.push(StreamEvent::Error(StreamErrorEvent {
            error: error_detail(stream_error_class(&provider_code), &provider_code, &message),
        }));
        return Ok(());
    }

    let choices = array_or_empty(payload.get("choices"));
    let choice = match choices.first() {
        Some(Value::Object(c)) => c,
        _ => object_or_empty(None),
    };
    let delta = object_or_empty(choice.get("delta"));

    if let Some(reasoning) = first_str(delta, &["reasoning_content", "reasoning"]) {
        out.push(StreamEvent::Delta(StreamDeltaEvent {
            delta: Delta::Thinking(ThinkingDelta {
                text: reasoning,
                part_index: 0,
            }),
        }));
    }

    if let Some(Value::String(content)) = delta.get("content") {
        if !content.is_empty() {
            let logprobs = object_or_empty(choice.get("logprobs"));
            out.push(StreamEvent::Delta(StreamDeltaEvent {
                delta: Delta::Text(TextDelta {
                    text: content.clone(),
                    part_index: 0,
                    logprobs: openai_token_logprobs(logprobs.get("content")),
                }),
            }));
        }
    }

    for call in array_or_empty(delta.get("tool_calls")) {
        let Value::Object(call) = call else { continue };
        let function = object_or_empty(call.get("function"));
        out.push(StreamEvent::Delta(StreamDeltaEvent {
            delta: Delta::ToolCall(ToolCallDelta {
                input: str_or_empty(function.get("arguments")),
                part_index: index_of(call.get("index")),
                id: str_or_none(call.get("id")),
                name: str_or_none(function.get("name")),
            }),
        }));
    }

    // MAP-3 (D9): the end event's provider_data is the frame that supplied
    // usage, verbatim, else the frame that supplied finish_reason; [DONE]
    // contributes nothing.
    let usage_frame = match payload.get("usage") {
        Some(Value::Object(u)) => Some(usage_from_chat(provider, u)?),
        _ => None,
    };
    if truthy(choice.get("finish_reason")) {
        let raw = super::super::wire_json::py_str(&choice["finish_reason"]);
        out.push(StreamEvent::End(StreamEndEvent {
            finish_reason: Some(finish_reason_of(&raw).unwrap_or(FinishReason::Stop)),
            usage: usage_frame,
            provider_data: Some(payload.clone()),
        }));
    } else if let Some(usage) = usage_frame {
        // Final usage-only chunk (stream_options.include_usage).
        out.push(StreamEvent::End(StreamEndEvent {
            finish_reason: None,
            usage: Some(usage),
            provider_data: Some(payload.clone()),
        }));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    fn request() -> Request {
        Request::new("m", vec![Message::user("hi").unwrap()]).unwrap()
    }

    #[test]
    fn complete_body() {
        let body = serde_json::json!({
            "id": "chatcmpl-1", "model": "m-2026",
            "choices": [{"message": {"role": "assistant", "content": "Hi", "reasoning_content": "hm",
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"x\": 2}"}}]},
                "finish_reason": "tool_calls",
                "logprobs": {"content": [{"token": "Hi", "logprob": -0.1}]}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7,
                      "prompt_tokens_details": {"cached_tokens": 1}}
        });
        let r = parse_response("openai_chat", &request(), body.to_string().as_bytes()).unwrap();
        assert_eq!(r.message.parts.len(), 3);
        assert_eq!(r.finish_reason, FinishReason::ToolCall);
        assert_eq!(r.usage.cache_read_tokens, Some(1));
        assert_eq!(r.logprobs.unwrap().len(), 1);
        assert!(!r.provider_data.unwrap().contains_key("_lm15_unmapped"));
    }

    #[test]
    fn unknown_finish_reason_is_recorded() {
        let body = br#"{"choices": [{"message": {"content": ""}, "finish_reason": "weird"}]}"#;
        let r = parse_response("openai_chat", &request(), body).unwrap();
        assert_eq!(r.message.parts, vec![Part::text("")]);
        assert_eq!(r.finish_reason, FinishReason::Stop);
        assert_eq!(
            r.provider_data.unwrap()["_lm15_unmapped"][0]["type"],
            "weird"
        );
    }

    #[test]
    fn stream_frames_end_on_finish_and_usage() {
        let mut out = Vec::new();
        for data in [
            r#"{"choices": [{"delta": {"content": "Hi"}}]}"#,
            r#"{"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c", "function": {"name": "f", "arguments": ""}}]}}]}"#,
            r#"{"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}"#,
            r#"{"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}"#,
            "[DONE]",
        ] {
            parse_stream_event(
                "openai_chat",
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
            matches!(&out[2], StreamEvent::End(e) if e.finish_reason == Some(FinishReason::ToolCall) && e.usage.is_none())
        );
        assert!(
            matches!(&out[3], StreamEvent::End(e) if e.finish_reason.is_none() && e.usage.is_some())
        );
    }
}

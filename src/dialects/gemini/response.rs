//! The Gemini dialect, response side (module 5; `lm15/providers/gemini.py`
//! `_parse_candidate_parts`, `parse_response`, `parse_stream_events`,
//! `_gemini_usage`, `_gemini_citations`, `_gemini_token_logprobs`).

use serde_json::Value;

use super::super::wire_json::{
    array_or_empty, body_object, count_of, error_detail, first_str, frame_object, id_or_none,
    object_or_empty, provider_error, py_str, str_or_empty, str_or_none, truthy, usage, Unmapped,
};
use crate::errors::{ErrorClass, Lm15Error};
use crate::sse::SseEvent;
use crate::types::{
    AudioDelta, AudioPart, CitationPart, ContinuationDelta, ContinuationState, Delta, DocumentPart,
    FinishReason, ImageDelta, ImagePart, JsonObject, Message, Part, Request, Response, Role,
    StreamDeltaEvent, StreamEndEvent, StreamErrorEvent, StreamEvent, TextDelta, TextPart,
    ThinkingDelta, ThinkingPart, TokenLogprob, ToolCallDelta, ToolCallPart, TopLogprob, Usage,
};

/// Part keys the provider executed itself (MAP-1): never parts.
pub const GEMINI_PROVIDER_EXECUTED_PART_KEYS: &[&str] = &["executableCode", "codeExecutionResult"];

/// `_is_candidate_finish_error`: a finish reason that is an in-band error.
pub const CANDIDATE_FINISH_ERRORS: &[&str] = &[
    "SAFETY",
    "RECITATION",
    "LANGUAGE",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "MALFORMED_FUNCTION_CALL",
    "IMAGE_SAFETY",
    "IMAGE_PROHIBITED_CONTENT",
    "IMAGE_OTHER",
    "NO_IMAGE",
    "IMAGE_RECITATION",
    "UNEXPECTED_TOOL_CALL",
    "TOO_MANY_TOOL_CALLS",
    "MISSING_THOUGHT_SIGNATURE",
    "MALFORMED_RESPONSE",
];

/// `_finish_reason`.
pub fn finish_reason_of(reason: Option<&Value>, has_tool_call: bool) -> FinishReason {
    if has_tool_call {
        return FinishReason::ToolCall;
    }
    let r = str_or_empty(reason).to_uppercase();
    match r.as_str() {
        "MAX_TOKENS" => FinishReason::Length,
        "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" => {
            FinishReason::ContentFilter
        }
        _ => FinishReason::Stop,
    }
}

/// `_gemini_usage` (INV-029 with the proto3-JSON exception, MAP-3): inside a
/// present `usageMetadata` an absent primary counter is a reported `0`;
/// an absent `usageMetadata` reports nothing.
pub fn usage_from_metadata(
    provider: &str,
    metadata: Option<&Value>,
    output_keys: &[&str],
) -> Result<Usage, Lm15Error> {
    let Some(Value::Object(data)) = metadata else {
        return Ok(Usage::default());
    };
    if data.is_empty() {
        return Ok(Usage::default());
    }
    let c = |field: &str, value: Option<&Value>| count_of(provider, field, value);
    // `usage_payload.get(key, 0)`: an absent key is 0, a present one is
    // read as it is.
    let present_or_zero = |field: &str, value: Option<&Value>| match value {
        Some(v) => c(field, Some(v)),
        None => Ok(Some(0)),
    };
    let output_tokens = present_or_zero(
        "candidatesTokenCount",
        output_keys.iter().find_map(|key| data.get(*key)),
    )?;
    Ok(usage(Usage {
        input_tokens: present_or_zero("promptTokenCount", data.get("promptTokenCount"))?,
        output_tokens,
        total_tokens: c("totalTokenCount", data.get("totalTokenCount"))?,
        cache_read_tokens: c(
            "cachedContentTokenCount",
            data.get("cachedContentTokenCount"),
        )?,
        cache_write_tokens: None,
        reasoning_tokens: c("thoughtsTokenCount", data.get("thoughtsTokenCount"))?,
        input_audio_tokens: modality_tokens(provider, data.get("promptTokensDetails"), "AUDIO")?,
        output_audio_tokens: modality_tokens(
            provider,
            data.get("candidatesTokensDetails")
                .filter(|v| truthy(Some(v)))
                .or_else(|| data.get("responseTokensDetails")),
            "AUDIO",
        )?,
    }))
}

/// Sum of `tokenCount` over the entries of `modality`; `None` when the
/// breakdown has no such entry.
fn modality_tokens(
    provider: &str,
    details: Option<&Value>,
    modality: &str,
) -> Result<Option<u64>, Lm15Error> {
    let Some(Value::Array(entries)) = details else {
        return Ok(None);
    };
    let mut total = 0u64;
    let mut seen = false;
    for entry in entries {
        let Value::Object(entry) = entry else {
            continue;
        };
        if str_or_empty(entry.get("modality")) != modality {
            continue;
        }
        seen = true;
        // `e.get("tokenCount", 0)`: proto3 omits a zero count.
        total += count_of(provider, "tokenCount", entry.get("tokenCount"))?.unwrap_or(0);
    }
    Ok(if seen { Some(total) } else { None })
}

fn signature_state(signature: &Value) -> ContinuationState {
    let mut data = JsonObject::new();
    data.insert("value".into(), Value::String(py_str(signature)));
    ContinuationState {
        provider: "gemini".into(),
        kind: "thought_signature".into(),
        data,
    }
}

fn thought_signature(part: &JsonObject) -> Vec<ContinuationState> {
    match part.get("thoughtSignature") {
        Some(Value::Null) | None => Vec::new(),
        Some(signature) => vec![signature_state(signature)],
    }
}

/// `_gemini_token_logprobs` (doc-based; no live capture).
pub fn token_logprobs(result: Option<&Value>) -> Vec<TokenLogprob> {
    let Some(Value::Object(result)) = result else {
        return Vec::new();
    };
    let chosen = array_or_empty(result.get("chosenCandidates"));
    let top_steps = array_or_empty(result.get("topCandidates"));
    let mut out = Vec::new();
    for (i, cand) in chosen.iter().enumerate() {
        let Value::Object(cand) = cand else { continue };
        let step = match top_steps.get(i) {
            Some(Value::Object(step)) => step,
            _ => object_or_empty(None),
        };
        let mut top = Vec::new();
        for alt in array_or_empty(step.get("candidates")) {
            let Value::Object(alt) = alt else { continue };
            top.push(TopLogprob {
                token: str_or_empty(alt.get("token")),
                logprob: alt
                    .get("logProbability")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0),
                bytes: None,
                token_id: alt.get("tokenId").and_then(Value::as_i64),
            });
        }
        out.push(TokenLogprob {
            token: str_or_empty(cand.get("token")),
            logprob: cand
                .get("logProbability")
                .and_then(Value::as_f64)
                .unwrap_or(0.0),
            bytes: None,
            token_id: cand.get("tokenId").and_then(Value::as_i64),
            top,
        });
    }
    out
}

// ─── Grounding citations ─────────────────────────────────────────────

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

fn segment_text(segment: &JsonObject, full_text: &str) -> Option<String> {
    if let Some(Value::String(text)) = segment.get("text") {
        if !text.is_empty() {
            return Some(text.clone());
        }
    }
    let start = int_or_none(segment.get("startIndex"))?;
    let end = int_or_none(segment.get("endIndex"))?;
    if start < end && end <= full_text.chars().count() {
        return Some(full_text.chars().skip(start).take(end - start).collect());
    }
    None
}

fn citations(candidate: &JsonObject, full_text: &str) -> Vec<CitationPart> {
    let Some(Value::Object(grounding)) = candidate.get("groundingMetadata") else {
        return Vec::new();
    };
    let chunks = array_or_empty(grounding.get("groundingChunks"));
    let Some(Value::Array(supports)) = grounding.get("groundingSupports") else {
        return Vec::new();
    };
    let mut out: Vec<CitationPart> = Vec::new();
    let mut seen: Vec<(Option<String>, Option<String>, Option<String>)> = Vec::new();
    for support in supports {
        let Value::Object(support) = support else {
            continue;
        };
        let segment = object_or_empty(support.get("segment"));
        let cited_text = segment_text(segment, full_text);
        let Some(Value::Array(indices)) = support.get("groundingChunkIndices") else {
            continue;
        };
        for index in indices {
            let chunk = int_or_none(Some(index))
                .and_then(|i| chunks.get(i))
                .and_then(Value::as_object);
            let source = chunk
                .and_then(|c| {
                    ["web", "retrievedContext", "googleSearch"]
                        .iter()
                        .map(|k| c.get(*k))
                        .find(|v| truthy(*v))
                        .flatten()
                })
                .and_then(Value::as_object);
            let source = source.unwrap_or_else(|| object_or_empty(None));
            let url = first_str(source, &["uri", "url"]);
            let title = first_str(source, &["title", "name"]);
            let key = (url.clone(), title.clone(), cited_text.clone());
            if seen.contains(&key) || (url.is_none() && title.is_none() && cited_text.is_none()) {
                continue;
            }
            seen.push(key);
            out.push(CitationPart {
                url,
                title,
                text: cited_text.clone(),
                continuation: Vec::new(),
            });
        }
    }
    out
}

// ─── In-band errors ──────────────────────────────────────────────────

/// `_inband_error`: a blocked prompt or a candidate whose finish reason is
/// an error, as `InvalidRequestError`.
pub fn inband_error(provider: &str, data: &JsonObject) -> Option<Lm15Error> {
    if let Some(Value::Object(feedback)) = data.get("promptFeedback") {
        let block_reason = str_or_empty(feedback.get("blockReason"));
        if !block_reason.is_empty() && block_reason != "BLOCK_REASON_UNSPECIFIED" {
            return Some(provider_error(
                ErrorClass::InvalidRequestError,
                provider,
                format!("Prompt blocked: {block_reason}"),
                Some("promptFeedback".into()),
            ));
        }
    }
    if let Some(Value::Object(candidate)) = array_or_empty(data.get("candidates")).first() {
        let finish_reason = str_or_empty(candidate.get("finishReason"));
        if CANDIDATE_FINISH_ERRORS.contains(&finish_reason.as_str()) {
            let finish_message = str_or_empty(candidate.get("finishMessage"));
            let message = if finish_message.is_empty() {
                format!("Candidate blocked: {finish_reason}")
            } else {
                finish_message
            };
            return Some(provider_error(
                ErrorClass::InvalidRequestError,
                provider,
                message,
                Some(if finish_reason.is_empty() {
                    "finishReason".into()
                } else {
                    finish_reason
                }),
            ));
        }
    }
    None
}

/// The class of a stream error frame's `status` (`_error_detail`).
pub fn stream_error_class(provider_code: &str, message: &str) -> ErrorClass {
    if crate::errors::gemini_is_context_length(message) {
        return ErrorClass::ContextLengthError;
    }
    if provider_code == "NOT_FOUND" && crate::errors::is_model_error(message) {
        return ErrorClass::UnsupportedModelError;
    }
    crate::errors::GEMINI_ERROR_STATUSES
        .iter()
        .find(|(k, _)| *k == provider_code)
        .map(|(_, c)| *c)
        .unwrap_or(ErrorClass::ProviderError)
}

// ─── Parts ───────────────────────────────────────────────────────────

fn media_part(mime: String, data: Option<String>, url: Option<String>) -> Part {
    if mime.starts_with("image/") {
        Part::Image(ImagePart {
            media_type: mime,
            data,
            url,
            ..Default::default()
        })
    } else if mime.starts_with("audio/") {
        Part::Audio(AudioPart {
            media_type: mime,
            data,
            url,
            ..Default::default()
        })
    } else {
        Part::Document(DocumentPart {
            media_type: mime,
            data,
            url,
            ..Default::default()
        })
    }
}

fn candidate_parts(
    provider: &str,
    parts_payload: &[Value],
    unmapped: &mut Unmapped,
    path_prefix: &str,
) -> Result<Vec<Part>, Lm15Error> {
    let mut parts: Vec<Part> = Vec::new();
    for (part_index, part) in parts_payload.iter().enumerate() {
        let path = format!("{path_prefix}[{part_index}]");
        let Value::Object(part) = part else {
            unmapped.record_shape(path, part);
            continue;
        };
        if truthy(part.get("thought")) && part.contains_key("text") {
            // Classified by the flag, not by non-empty text: 3.x can send
            // {thought: true, text: "", thoughtSignature} (MAP-7 rule 8).
            parts.push(Part::Thinking(ThinkingPart {
                text: str_or_empty(part.get("text")),
                continuation: thought_signature(part),
            }));
        } else if part.contains_key("text") {
            // On 3.x the answer text carries the turn's thoughtSignature.
            parts.push(Part::Text(TextPart {
                text: str_or_empty(part.get("text")),
                continuation: thought_signature(part),
            }));
        } else if let Some(Value::Object(fc)) = part.get("functionCall") {
            let signature = part
                .get("thoughtSignature")
                .filter(|v| truthy(Some(v)))
                .or_else(|| fc.get("thoughtSignature"))
                .filter(|v| !v.is_null());
            let name = str_or_none(fc.get("name")).ok_or_else(|| {
                provider_error(
                    ErrorClass::ProviderError,
                    provider,
                    format!("{provider}: {path} is a functionCall with no name; lm15 does not guess which tool the model meant"),
                    None,
                )
            })?;
            parts.push(Part::ToolCall(ToolCallPart {
                // MAP-9: a missing id is the lm15 correlator tool_call_<index>,
                // the same one the stream assembler mints (INV-051 parity).
                id: str_or_none(fc.get("id"))
                    .unwrap_or_else(|| format!("tool_call_{}", parts.len())),
                name,
                input: match fc.get("args") {
                    Some(Value::Object(o)) => o.clone(),
                    _ => JsonObject::new(),
                },
                continuation: signature
                    .map(|s| vec![signature_state(s)])
                    .unwrap_or_default(),
            }));
        } else if let Some(Value::Object(inline)) = part.get("inlineData") {
            let mime = str_or_none(inline.get("mimeType"))
                .unwrap_or_else(|| "application/octet-stream".to_string());
            let Some(data) = str_or_none(inline.get("data")) else {
                continue;
            };
            parts.push(media_part(mime, Some(data), None));
        } else if let Some(Value::Object(fd)) = part.get("fileData") {
            let Some(uri) = str_or_none(fd.get("fileUri")) else {
                continue;
            };
            let mime = str_or_none(fd.get("mimeType"))
                .unwrap_or_else(|| "application/octet-stream".to_string());
            parts.push(media_part(mime, None, Some(uri)));
        } else if GEMINI_PROVIDER_EXECUTED_PART_KEYS
            .iter()
            .any(|k| part.contains_key(*k))
        {
        } else {
            let mut keys: Vec<&str> = part.keys().map(String::as_str).collect();
            keys.sort_unstable();
            let joined = keys.join("+");
            unmapped.record_text(
                path,
                if joined.is_empty() {
                    "<empty>"
                } else {
                    &joined
                },
            );
        }
    }
    Ok(parts)
}

pub fn parse_response(
    provider: &str,
    request: &Request,
    body: &[u8],
) -> Result<Response, Lm15Error> {
    let data = body_object(provider, body)?;
    if let Some(err) = inband_error(provider, &data) {
        return Err(err);
    }
    let candidate = match array_or_empty(data.get("candidates")).first() {
        Some(Value::Object(c)) => c,
        _ => object_or_empty(None),
    };
    let content = object_or_empty(candidate.get("content"));
    let mut unmapped = Unmapped::default();
    let mut parts = candidate_parts(
        provider,
        array_or_empty(content.get("parts")),
        &mut unmapped,
        "candidates[0].content.parts",
    )?;
    let full_text: String = parts
        .iter()
        .filter_map(|p| match p {
            Part::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect();
    parts.extend(
        citations(candidate, &full_text)
            .into_iter()
            .map(Part::Citation),
    );
    if parts.is_empty() {
        parts.push(Part::Text(TextPart::new("")));
    }
    let usage = usage_from_metadata(
        provider,
        data.get("usageMetadata"),
        &["candidatesTokenCount", "responseTokenCount"],
    )?;
    let has_tool = parts.iter().any(|p| matches!(p, Part::ToolCall(_)));
    let logprobs = token_logprobs(candidate.get("logprobsResult"));
    Ok(Response {
        id: id_or_none(data.get("responseId")),
        model: request.model.clone(),
        message: Message {
            role: Role::Assistant,
            parts,
            continuation: Vec::new(),
        },
        finish_reason: finish_reason_of(candidate.get("finishReason"), has_tool),
        usage,
        logprobs: if logprobs.is_empty() {
            None
        } else {
            Some(logprobs)
        },
        provider_data: Some(unmapped.attach(data)),
    })
}

fn delta(delta: Delta) -> StreamEvent {
    StreamEvent::Delta(StreamDeltaEvent { delta })
}

fn signature_delta(signature: &Value, idx: u64) -> StreamEvent {
    let state = signature_state(signature);
    delta(Delta::Continuation(ContinuationDelta {
        provider: state.provider,
        kind: state.kind,
        data: state.data,
        part_index: Some(idx),
    }))
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
    let Some(payload) = frame_object(&raw.data) else {
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
    if let Some(err) = payload.get("error") {
        let (provider_code, message) = match err {
            Value::Object(err) => (
                first_str(err, &["status", "code"]).unwrap_or_else(|| "provider".to_string()),
                str_or_empty(err.get("message")),
            ),
            _ => ("provider".to_string(), String::new()),
        };
        out.push(StreamEvent::Error(StreamErrorEvent {
            error: error_detail(
                stream_error_class(&provider_code, &message),
                &provider_code,
                &message,
            ),
        }));
        return Ok(());
    }
    if let Some(inband) = inband_error(provider, &payload) {
        out.push(StreamEvent::Error(StreamErrorEvent {
            error: crate::types::ErrorDetail {
                code: inband.code(),
                message: inband.to_string(),
                provider_code: Some("inband_finish_reason".into()),
            },
        }));
        return Ok(());
    }

    let candidate = match array_or_empty(payload.get("candidates")).first() {
        Some(Value::Object(c)) => Some(c),
        _ => None,
    };
    let mut yielded_delta = false;
    let mut saw_tool = false;
    let mut finish: Option<&Value> = None;
    if let Some(candidate) = candidate {
        let content = object_or_empty(candidate.get("content"));
        // Chunk-level decoding telemetry rides the chunk's first text delta.
        let mut chunk_logprobs = token_logprobs(candidate.get("logprobsResult"));
        for (idx, part) in array_or_empty(content.get("parts")).iter().enumerate() {
            let Value::Object(part) = part else { continue };
            let idx = idx as u64;
            if truthy(part.get("thought")) && part.contains_key("text") {
                yielded_delta = true;
                out.push(delta(Delta::Thinking(ThinkingDelta {
                    text: str_or_empty(part.get("text")),
                    part_index: idx,
                })));
                if let Some(signature) = part.get("thoughtSignature").filter(|v| !v.is_null()) {
                    out.push(signature_delta(signature, idx));
                }
            } else if part.contains_key("text") {
                yielded_delta = true;
                out.push(delta(Delta::Text(TextDelta {
                    text: str_or_empty(part.get("text")),
                    part_index: idx,
                    logprobs: std::mem::take(&mut chunk_logprobs),
                })));
                if let Some(signature) = part.get("thoughtSignature").filter(|v| !v.is_null()) {
                    out.push(signature_delta(signature, idx));
                }
            } else if let Some(Value::Object(fc)) = part.get("functionCall") {
                saw_tool = true;
                yielded_delta = true;
                let args = fc
                    .get("args")
                    .cloned()
                    .unwrap_or_else(|| Value::Object(JsonObject::new()));
                out.push(delta(Delta::ToolCall(ToolCallDelta {
                    input: args.to_string(),
                    part_index: idx,
                    id: str_or_none(fc.get("id")),
                    name: str_or_none(fc.get("name")),
                })));
                let signature = part
                    .get("thoughtSignature")
                    .filter(|v| truthy(Some(v)))
                    .or_else(|| fc.get("thoughtSignature"))
                    .filter(|v| !v.is_null());
                if let Some(signature) = signature {
                    out.push(signature_delta(signature, idx));
                }
            } else if let Some(Value::Object(inline)) = part.get("inlineData") {
                let mime = str_or_none(inline.get("mimeType"))
                    .unwrap_or_else(|| "application/octet-stream".to_string());
                let data = str_or_empty(inline.get("data"));
                if mime.starts_with("audio/") {
                    yielded_delta = true;
                    out.push(delta(Delta::Audio(AudioDelta {
                        data: Some(data),
                        part_index: idx,
                        media_type: Some(mime),
                        ..Default::default()
                    })));
                } else if mime.starts_with("image/") {
                    yielded_delta = true;
                    out.push(delta(Delta::Image(ImageDelta {
                        data: Some(data),
                        part_index: idx,
                        media_type: Some(mime),
                        ..Default::default()
                    })));
                }
            }
        }
        finish = candidate.get("finishReason").filter(|v| truthy(Some(v)));
    }

    let usage_of = || {
        usage_from_metadata(
            provider,
            payload.get("usageMetadata"),
            &["candidatesTokenCount", "responseTokenCount"],
        )
    };
    if finish.is_some() {
        out.push(StreamEvent::End(StreamEndEvent {
            finish_reason: Some(finish_reason_of(finish, saw_tool)),
            usage: Some(usage_of()?),
            provider_data: Some(payload.clone()),
        }));
    } else if !yielded_delta && payload.contains_key("usageMetadata") {
        out.push(StreamEvent::End(StreamEndEvent {
            finish_reason: Some(FinishReason::Stop),
            usage: Some(usage_of()?),
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
        Request::new("gemini-x", vec![Message::user("hi").unwrap()]).unwrap()
    }

    #[test]
    fn complete_body_with_grounding_and_zero_usage() {
        let body = serde_json::json!({
            "responseId": "r1",
            "candidates": [{
                "content": {"parts": [
                    {"thought": true, "text": "", "thoughtSignature": "sig"},
                    {"text": "Paris is the capital."},
                    {"functionCall": {"name": "f", "args": {"a": 1}}},
                    {"inlineData": {"mimeType": "image/png", "data": "aGk="}},
                    {"executableCode": {}},
                    {"weird": 1, "alpha": 2}
                ]},
                "finishReason": "STOP",
                "groundingMetadata": {
                    "groundingChunks": [{"web": {"uri": "https://a", "title": "A"}}],
                    "groundingSupports": [{"segment": {"startIndex": 0, "endIndex": 5}, "groundingChunkIndices": [0, 0]}]
                }
            }],
            "usageMetadata": {"promptTokenCount": 2, "thoughtsTokenCount": 3, "totalTokenCount": 5}
        });
        let r = parse_response("gemini", &request(), body.to_string().as_bytes()).unwrap();
        assert_eq!(r.id.as_deref(), Some("r1"));
        assert_eq!(r.model, "gemini-x");
        assert_eq!(r.finish_reason, FinishReason::ToolCall);
        assert_eq!(r.message.parts.len(), 5);
        match &r.message.parts[2] {
            Part::ToolCall(t) => assert_eq!(t.id, "tool_call_2"),
            other => panic!("{other:?}"),
        }
        match &r.message.parts[4] {
            Part::Citation(c) => assert_eq!(c.text.as_deref(), Some("Paris")),
            other => panic!("{other:?}"),
        }
        assert_eq!(r.usage.output_tokens, Some(0));
        assert_eq!(r.usage.input_tokens, Some(2));
        assert_eq!(r.usage.total_tokens, Some(5));
        assert_eq!(
            r.provider_data.unwrap()["_lm15_unmapped"][0]["type"],
            "alpha+weird"
        );
    }

    #[test]
    fn blocked_prompt_is_an_in_band_error() {
        let err = parse_response(
            "gemini",
            &request(),
            br#"{"promptFeedback": {"blockReason": "SAFETY"}}"#,
        )
        .unwrap_err();
        assert_eq!(err.class_name(), "InvalidRequestError");
        assert_eq!(err.provider_code(), Some("promptFeedback"));
        let mut out = Vec::new();
        parse_stream_event(
            "gemini",
            &request(),
            &SseEvent {
                event: None,
                data: r#"{"candidates": [{"finishReason": "SAFETY"}]}"#.into(),
            },
            &mut out,
        )
        .unwrap();
        assert!(
            matches!(&out[0], StreamEvent::Error(e) if e.error.provider_code.as_deref() == Some("inband_finish_reason"))
        );
    }

    #[test]
    fn stream_chunks() {
        let mut out = Vec::new();
        for data in [
            r#"{"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]}"#,
            r#"{"candidates": [{"content": {"parts": [{"functionCall": {"name": "f", "args": {}}, "thoughtSignature": "s"}]}, "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 2}}"#,
        ] {
            parse_stream_event(
                "gemini",
                &request(),
                &SseEvent {
                    event: None,
                    data: data.into(),
                },
                &mut out,
            )
            .unwrap();
        }
        assert_eq!(out.len(), 4);
        assert!(
            matches!(&out[1], StreamEvent::Delta(d) if matches!(&d.delta, Delta::ToolCall(t) if t.input == "{}"))
        );
        assert!(
            matches!(&out[3], StreamEvent::End(e) if e.finish_reason == Some(FinishReason::ToolCall))
        );
    }
}

//! SSE parsing, MAP-3 stream coalescing, and stream materialization
//! (reference: lm15.sse, lm15.result; docs/mapping-rules.md MAP-3).

use serde_json::{Map, Value};

use crate::types::{
    ContinuationState, Delta, JsonObject, Message, Part, Request, Response, StreamEvent, Usage,
};

// ─── SSE parsing ─────────────────────────────────────────────────────

/// One parsed Server-Sent Event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

/// Parse an SSE byte stream into events (reference: lm15.sse.parse_sse).
/// Lossy UTF-8 decode per line; a blank line flushes a pending event; `:`
/// comment lines are skipped; multiple `data:` lines join with `\n`.
pub fn parse_sse(body: &[u8]) -> Vec<SseEvent> {
    let mut events = Vec::new();
    let mut event_name: Option<String> = None;
    let mut data_lines: Vec<String> = Vec::new();

    fn flush(
        events: &mut Vec<SseEvent>,
        event_name: &mut Option<String>,
        data_lines: &mut Vec<String>,
    ) {
        if !data_lines.is_empty() {
            events.push(SseEvent {
                event: event_name.clone(),
                data: data_lines.join("\n"),
            });
        }
        *event_name = None;
        data_lines.clear();
    }

    for raw in body.split_inclusive(|&b| b == b'\n') {
        let line = String::from_utf8_lossy(raw);
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            flush(&mut events, &mut event_name, &mut data_lines);
            continue;
        }
        if line.starts_with(':') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("event:") {
            event_name = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("data:") {
            data_lines.push(rest.trim_start().to_string());
        }
    }
    flush(&mut events, &mut event_name, &mut data_lines);
    events
}

// ─── MAP-3 coalescer ─────────────────────────────────────────────────

/// Coalesce a raw event trace into the canonical post-MAP-3 trace: start,
/// delta and error events pass through; every end event is absorbed (later
/// non-`None` fields replace, `None` never erases) and exactly one merged
/// final StreamEndEvent is emitted. No end event is fabricated if none was
/// seen.
pub fn coalesce_stream(events: Vec<StreamEvent>) -> Vec<StreamEvent> {
    let mut out = Vec::with_capacity(events.len());
    let mut saw_end = false;
    let mut finish_reason: Option<String> = None;
    let mut usage: Option<Usage> = None;
    let mut provider_data: Option<JsonObject> = None;
    for event in events {
        match event {
            StreamEvent::End {
                finish_reason: fr,
                usage: u,
                provider_data: pd,
            } => {
                saw_end = true;
                if fr.is_some() {
                    finish_reason = fr;
                }
                if u.is_some() {
                    usage = u;
                }
                if pd.is_some() {
                    provider_data = pd;
                }
            }
            other => out.push(other),
        }
    }
    if saw_end {
        out.push(StreamEvent::End {
            finish_reason,
            usage,
            provider_data,
        });
    }
    out
}

// ─── Materialization (reference: lm15.result._RoundState) ───────────

/// `_parse_json_best_effort`: `{}` for empty, dict as-is, scalar wrapped as
/// `{"value": ...}`, unparseable as `{"partial_json": raw}`. serde_json
/// rejects NaN/Infinity natively, matching the reference's constant guard.
fn parse_json_best_effort(raw: &str) -> JsonObject {
    if raw.is_empty() {
        return Map::new();
    }
    match serde_json::from_str::<Value>(raw) {
        Ok(Value::Object(map)) => map,
        Ok(other) => {
            let mut m = Map::new();
            m.insert("value".into(), other);
            m
        }
        Err(_) => {
            let mut m = Map::new();
            m.insert("partial_json".into(), Value::String(raw.to_string()));
            m
        }
    }
}

/// Best-effort base64 decode of one chunk (reference `_concat_b64_chunks`:
/// undecodable chunks contribute nothing).
fn b64_decode_loose(chunk: &str) -> Vec<u8> {
    fn val(c: u8) -> Option<u32> {
        match c {
            b'A'..=b'Z' => Some((c - b'A') as u32),
            b'a'..=b'z' => Some((c - b'a' + 26) as u32),
            b'0'..=b'9' => Some((c - b'0' + 52) as u32),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let mut out = Vec::new();
    let mut acc = 0u32;
    let mut bits = 0u32;
    for &c in chunk.as_bytes() {
        if c == b'=' || c.is_ascii_whitespace() {
            continue;
        }
        let Some(v) = val(c) else {
            return Vec::new();
        };
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    out
}

fn b64_encode(data: &[u8]) -> String {
    const TBL: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(TBL[(n >> 18) as usize & 63] as char);
        out.push(TBL[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            TBL[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TBL[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// `_pcm_to_wav`: wrap raw PCM in a 16-bit mono 24kHz WAV header.
fn pcm_to_wav(pcm: &[u8]) -> Vec<u8> {
    let (sample_rate, channels, bits): (u32, u16, u16) = (24000, 1, 16);
    let byte_rate = sample_rate * u32::from(channels) * u32::from(bits) / 8;
    let block_align = channels * bits / 8;
    let data_size = pcm.len() as u32;
    let mut out = Vec::with_capacity(44 + pcm.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_size).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());
    out.extend_from_slice(pcm);
    out
}

/// Per-part-index accumulator (reference `_RoundState` maps keyed by index).
#[derive(Default)]
struct PartSlot {
    thinking: Option<String>,
    text: Option<String>,
    image: Option<Part>,
    audio_chunks: Vec<String>,
    audio_media_type: Option<Option<String>>,
    citations: Vec<(Option<String>, Option<String>, Option<String>)>,
    tool_raw: Option<String>,
    tool_id: Option<String>,
    tool_name: Option<String>,
    continuation: Vec<ContinuationState>,
}

impl PartSlot {
    fn has_content(&self) -> bool {
        self.thinking.is_some()
            || self.text.is_some()
            || self.image.is_some()
            || !self.audio_chunks.is_empty()
            || !self.citations.is_empty()
    }

    fn has_tool_call(&self) -> bool {
        self.tool_raw.is_some() || self.tool_id.is_some() || self.tool_name.is_some()
    }
}

/// Build a complete Response from a (coalesced) event trace
/// (reference: lm15.result._RoundState.apply + materialize).
pub fn materialize_response(events: &[StreamEvent], request: &Request) -> Response {
    use std::collections::BTreeMap;
    let mut slots: BTreeMap<u64, PartSlot> = BTreeMap::new();
    let mut started_id: Option<String> = None;
    let mut started_model: Option<String> = None;
    let mut finish_reason: Option<String> = None;
    let mut usage: Option<Usage> = None;
    let mut provider_data: Option<JsonObject> = None;
    let mut message_continuation: Vec<ContinuationState> = Vec::new();

    for event in events {
        match event {
            StreamEvent::Start { id, model } => {
                if id.is_some() {
                    started_id = id.clone();
                }
                if model.is_some() {
                    started_model = model.clone();
                }
            }
            StreamEvent::End {
                finish_reason: fr,
                usage: u,
                provider_data: pd,
            } => {
                if fr.is_some() {
                    finish_reason = fr.clone();
                }
                if u.is_some() {
                    usage = u.clone();
                }
                if pd.is_some() {
                    provider_data = pd.clone();
                }
            }
            StreamEvent::Error { .. } => {}
            StreamEvent::Delta { delta } => match delta {
                Delta::Text { text, part_index } => {
                    slots
                        .entry(*part_index)
                        .or_default()
                        .text
                        .get_or_insert_with(String::new)
                        .push_str(text);
                }
                Delta::Thinking { text, part_index } => {
                    slots
                        .entry(*part_index)
                        .or_default()
                        .thinking
                        .get_or_insert_with(String::new)
                        .push_str(text);
                }
                Delta::Audio {
                    data,
                    part_index,
                    media_type,
                    ..
                } => {
                    let slot = slots.entry(*part_index).or_default();
                    slot.audio_chunks.push(data.clone().unwrap_or_default());
                    if slot.audio_media_type.is_none() {
                        slot.audio_media_type = Some(media_type.clone());
                    }
                }
                Delta::Image {
                    data,
                    url,
                    file_id,
                    part_index,
                    media_type,
                } => {
                    if data.is_none() && url.is_none() && file_id.is_none() {
                        continue;
                    }
                    let media_type = media_type
                        .clone()
                        .unwrap_or_else(|| "image/png".to_string());
                    // Reference precedence: data, then url, then file_id.
                    let (data, url, file_id) = if data.is_some() {
                        (data.clone(), None, None)
                    } else if url.is_some() {
                        (None, url.clone(), None)
                    } else {
                        (None, None, file_id.clone())
                    };
                    slots.entry(*part_index).or_default().image = Some(Part::Image {
                        media_type,
                        data,
                        url,
                        file_id,
                        path: None,
                        detail: None,
                        continuation: Vec::new(),
                    });
                }
                Delta::Citation {
                    text,
                    url,
                    title,
                    part_index,
                } => {
                    slots.entry(*part_index).or_default().citations.push((
                        text.clone(),
                        url.clone(),
                        title.clone(),
                    ));
                }
                Delta::ToolCall {
                    input,
                    part_index,
                    id,
                    name,
                } => {
                    let slot = slots.entry(*part_index).or_default();
                    if let Some(id) = id {
                        slot.tool_id = Some(id.clone());
                    }
                    if let Some(name) = name {
                        slot.tool_name = Some(name.clone());
                    }
                    slot.tool_raw
                        .get_or_insert_with(String::new)
                        .push_str(input);
                }
                Delta::Continuation {
                    provider,
                    kind,
                    data,
                    part_index,
                } => {
                    let state = ContinuationState {
                        provider: provider.clone(),
                        kind: kind.clone(),
                        data: data.clone(),
                    };
                    match part_index {
                        None => message_continuation.push(state),
                        Some(idx) => slots.entry(*idx).or_default().continuation.push(state),
                    }
                }
            },
        }
    }

    let tool_names: Vec<&str> = request
        .tools
        .iter()
        .filter_map(|t| match t {
            crate::types::Tool::Function { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();

    let mut parts: Vec<Part> = Vec::new();
    for (pos, (idx, slot)) in slots.iter().enumerate() {
        let continuation = slot.continuation.clone();
        if let Some(text) = &slot.thinking {
            parts.push(Part::Thinking {
                text: text.clone(),
                redacted: false,
                continuation: continuation.clone(),
            });
        }
        if let Some(text) = &slot.text {
            parts.push(Part::Text {
                text: text.clone(),
                continuation: continuation.clone(),
            });
        }
        if let Some(Part::Image {
            media_type,
            data,
            url,
            file_id,
            path,
            detail,
            ..
        }) = &slot.image
        {
            parts.push(Part::Image {
                media_type: media_type.clone(),
                data: data.clone(),
                url: url.clone(),
                file_id: file_id.clone(),
                path: path.clone(),
                detail: detail.clone(),
                continuation: continuation.clone(),
            });
        }
        if !slot.audio_chunks.is_empty() {
            let mut raw = Vec::new();
            for chunk in &slot.audio_chunks {
                raw.extend_from_slice(&b64_decode_loose(chunk));
            }
            let media_type = slot.audio_media_type.clone().flatten();
            let (data, media_type) = match media_type.as_deref() {
                None | Some("audio/pcm") | Some("audio/pcm16") => {
                    (b64_encode(&pcm_to_wav(&raw)), "audio/wav".to_string())
                }
                Some(mt) => (b64_encode(&raw), mt.to_string()),
            };
            parts.push(Part::Audio {
                media_type,
                data: Some(data),
                url: None,
                file_id: None,
                path: None,
                continuation: continuation.clone(),
            });
        }
        for (text, url, title) in &slot.citations {
            parts.push(Part::Citation {
                text: text.clone(),
                url: url.clone(),
                title: title.clone(),
                continuation: continuation.clone(),
            });
        }
        if slot.has_tool_call() {
            let raw = slot.tool_raw.clone().unwrap_or_default();
            let input = parse_json_best_effort(&raw);
            let name = match &slot.tool_name {
                Some(n) if !n.is_empty() => n.clone(),
                _ => {
                    if tool_names.len() == 1 {
                        tool_names[0].to_string()
                    } else if pos < tool_names.len() {
                        tool_names[pos].to_string()
                    } else {
                        "tool".to_string()
                    }
                }
            };
            let id = match &slot.tool_id {
                Some(id) if !id.is_empty() => id.clone(),
                _ => format!("tool_call_{idx}"),
            };
            parts.push(Part::ToolCall {
                id,
                name,
                input,
                continuation,
            });
        } else if !slot.has_content() {
            // Continuation-only slot: anchor it on an empty text part.
            parts.push(Part::Text {
                text: String::new(),
                continuation,
            });
        }
    }

    if parts.is_empty() {
        parts.push(Part::Text {
            text: String::new(),
            continuation: Vec::new(),
        });
    }

    let has_tool_calls = parts.iter().any(|p| matches!(p, Part::ToolCall { .. }));
    let finish = match finish_reason {
        None => {
            if has_tool_calls {
                "tool_call".to_string()
            } else {
                "stop".to_string()
            }
        }
        Some(f) if f == "stop" && has_tool_calls => "tool_call".to_string(),
        Some(f) => f,
    };

    Response {
        id: started_id,
        model: started_model.unwrap_or_else(|| request.model.clone()),
        message: Message {
            role: "assistant".to_string(),
            parts,
            continuation: message_continuation,
        },
        finish_reason: finish,
        usage: usage.unwrap_or_default(),
        provider_data,
    }
}

/// Full replay pipeline for one provider stream body: SSE parse, per-frame
/// adapter mapping, MAP-3 coalesce. Returns the canonical event trace.
pub fn parse_stream_body(
    provider: &str,
    request: &Request,
    body: &[u8],
) -> Result<Vec<StreamEvent>, String> {
    let mut raw_events = Vec::new();
    for sse in parse_sse(body) {
        let mapped = match provider {
            "openai" => crate::providers::openai::parse_stream_events(request, &sse.data),
            "openai_chat" => crate::providers::openai_chat::parse_stream_events(request, &sse.data),
            "anthropic" => crate::providers::anthropic::parse_stream_events(request, &sse.data),
            "gemini" => crate::providers::gemini::parse_stream_events(request, &sse.data),
            other => return Err(format!("unknown provider: {other}")),
        }?;
        raw_events.extend(mapped);
    }
    Ok(coalesce_stream(raw_events))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Delta;

    fn end(fr: Option<&str>, usage: Option<Usage>) -> StreamEvent {
        StreamEvent::End {
            finish_reason: fr.map(str::to_string),
            usage,
            provider_data: None,
        }
    }

    #[test]
    fn sse_basic() {
        let body = b"event: ping\ndata: one\ndata: two\n\ndata: [DONE]\n\n";
        let events = parse_sse(body);
        assert_eq!(
            events,
            vec![
                SseEvent {
                    event: Some("ping".to_string()),
                    data: "one\ntwo".to_string()
                },
                SseEvent {
                    event: None,
                    data: "[DONE]".to_string()
                },
            ]
        );
    }

    #[test]
    fn sse_comments_and_trailing_event() {
        let events = parse_sse(b": comment\ndata: tail");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "tail");
    }

    #[test]
    fn coalesce_merges_post_finish_usage_chunk() {
        // vLLM/SGLang/Groq shape: finish_reason chunk, then usage-only chunk.
        let usage = Usage {
            input_tokens: Some(14),
            output_tokens: Some(22),
            total_tokens: Some(36),
            ..Usage::default()
        };
        let events = vec![
            StreamEvent::Delta {
                delta: Delta::Text {
                    text: "hi".to_string(),
                    part_index: 0,
                },
            },
            end(Some("stop"), None),
            end(None, Some(usage.clone())),
        ];
        let out = coalesce_stream(events);
        assert_eq!(out.len(), 2);
        assert_eq!(
            out[1],
            StreamEvent::End {
                finish_reason: Some("stop".to_string()),
                usage: Some(usage),
                provider_data: None,
            }
        );
    }

    #[test]
    fn coalesce_none_never_erases() {
        let usage = Usage {
            input_tokens: Some(1),
            ..Usage::default()
        };
        let out = coalesce_stream(vec![
            end(Some("stop"), Some(usage.clone())),
            end(None, None),
        ]);
        assert_eq!(out, vec![end(Some("stop"), Some(usage))]);
    }

    #[test]
    fn coalesce_no_end_fabricated() {
        let events = vec![StreamEvent::Delta {
            delta: Delta::Text {
                text: "x".to_string(),
                part_index: 0,
            },
        }];
        assert_eq!(coalesce_stream(events.clone()), events);
    }

    #[test]
    fn parse_json_best_effort_shapes() {
        assert!(parse_json_best_effort("").is_empty());
        assert_eq!(
            serde_json::Value::Object(parse_json_best_effort("{\"a\":1}")),
            serde_json::json!({"a": 1})
        );
        assert_eq!(
            serde_json::Value::Object(parse_json_best_effort("3")),
            serde_json::json!({"value": 3})
        );
        assert_eq!(
            serde_json::Value::Object(parse_json_best_effort("{\"a\":")),
            serde_json::json!({"partial_json": "{\"a\":"})
        );
    }
}

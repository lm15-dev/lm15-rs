//! Stream materialization (the reference's `lm15/result.py`): one engine
//! every port shares, push-based because Rust has no generators.
//!
//! - [`Coalescer`] enforces MAP-3 and MAP-4 over a dialect's raw events:
//!   exactly one leading start event, exactly one final end event carrying
//!   the merged finish reason, usage and (D9) provider data.
//! - [`StreamAccumulator`] folds canonical events into a `Response` by the
//!   MAP-9 assembly algorithm and refuses to invent a tool-call name.
//! - [`materialize_response`] is the one-shot form; [`response_to_events`]
//!   is the lossless inverse for the streamable part kinds.

use std::collections::BTreeMap;

use crate::errors::{ErrorMeta, Lm15Error, StreamAssembly};
use crate::types::{
    base64_decode, base64_encode, AudioDelta, AudioPart, CitationDelta, CitationPart,
    ContinuationDelta, ContinuationState, Delta, ErrorDetail, FinishReason, ImageDelta, ImagePart,
    JsonObject, Message, Part, Request, Response, Role, StreamDeltaEvent, StreamEndEvent,
    StreamEvent, StreamStartEvent, TextDelta, TextPart, ThinkingDelta, ThinkingPart, TokenLogprob,
    ToolCallDelta, ToolCallPart, Usage,
};

// ─── Coalescer (MAP-3, MAP-4) ────────────────────────────────────────

/// D9: which adapter end event's `provider_data` the merged end carries.
/// Rank 2: a frame that supplied usage. Rank 1: a frame that supplied
/// finish_reason. Rank 0: anything else that carried data. A later frame
/// replaces an earlier one only at the same or a higher rank.
#[derive(Debug, Default)]
struct EndProviderData {
    value: Option<JsonObject>,
    rank: i8,
}

impl EndProviderData {
    fn new() -> Self {
        EndProviderData {
            value: None,
            rank: -1,
        }
    }

    fn absorb(&mut self, event: &StreamEndEvent) {
        let Some(data) = &event.provider_data else {
            return;
        };
        let rank = if event.usage.is_some() {
            2
        } else if event.finish_reason.is_some() {
            1
        } else {
            0
        };
        if rank >= self.rank {
            self.value = Some(data.clone());
            self.rank = rank;
        }
    }
}

/// Enforces MAP-3 and MAP-4 over the raw events a dialect emits. Delta and
/// error events pass through; end events are absorbed (a later non-`None`
/// field replaces, `None` never erases) and one merged end is emitted by
/// [`Coalescer::finish`]; a synthesized start (with the request's model)
/// precedes the first delta of a dialect without a start frame; duplicate
/// starts collapse to the first; error events never force a start.
#[derive(Debug)]
pub struct Coalescer {
    model: Option<String>,
    started: bool,
    saw_end: bool,
    finish_reason: Option<FinishReason>,
    usage: Option<Usage>,
    end_data: EndProviderData,
}

impl Coalescer {
    /// `model` is the request's model, used for a synthesized start.
    pub fn new(model: Option<String>) -> Self {
        Coalescer {
            model,
            started: false,
            saw_end: false,
            finish_reason: None,
            usage: None,
            end_data: EndProviderData::new(),
        }
    }

    /// Fold one raw event; returns the events to yield now.
    pub fn push(&mut self, event: StreamEvent) -> Vec<StreamEvent> {
        match event {
            StreamEvent::Start(_) => {
                if self.started {
                    return Vec::new();
                }
                self.started = true;
                vec![event]
            }
            StreamEvent::End(end) => {
                self.saw_end = true;
                if end.finish_reason.is_some() {
                    self.finish_reason = end.finish_reason;
                }
                if end.usage.is_some() {
                    self.usage = end.usage;
                }
                self.end_data.absorb(&end);
                Vec::new()
            }
            StreamEvent::Delta(_) if !self.started => {
                self.started = true;
                vec![
                    StreamEvent::Start(StreamStartEvent {
                        id: None,
                        model: self.model.clone(),
                    }),
                    event,
                ]
            }
            other => vec![other],
        }
    }

    /// The source is exhausted: the single merged end event (preceded by a
    /// start when nothing started the stream), or nothing when no end was
    /// seen (a truncated stream gets no fabricated end).
    pub fn finish(self) -> Vec<StreamEvent> {
        if !self.saw_end {
            return Vec::new();
        }
        let mut out = Vec::new();
        if !self.started {
            out.push(StreamEvent::Start(StreamStartEvent {
                id: None,
                model: self.model,
            }));
        }
        out.push(StreamEvent::End(StreamEndEvent {
            finish_reason: self.finish_reason,
            usage: self.usage,
            provider_data: self.end_data.value,
        }));
        out
    }
}

/// The post-coalesce trace of a raw event sequence.
pub fn coalesce_stream(events: Vec<StreamEvent>, model: Option<String>) -> Vec<StreamEvent> {
    let mut coalescer = Coalescer::new(model);
    let mut out = Vec::with_capacity(events.len() + 2);
    for event in events {
        out.extend(coalescer.push(event));
    }
    out.extend(coalescer.finish());
    out
}

// ─── Accumulator (MAP-9 assembly) ────────────────────────────────────

#[derive(Debug, Default)]
struct ToolCallMeta {
    id: Option<String>,
    name: Option<String>,
    raw: String,
}

/// Accumulates canonical stream events into a complete `Response`.
///
/// `push` ignores error events — deciding whether to fail is the caller's
/// job. `response()` materializes whatever has been accumulated; callers
/// normally push through the end event first.
#[derive(Debug)]
pub struct StreamAccumulator {
    request_model: String,
    started_id: Option<String>,
    started_model: Option<String>,
    finish_reason: Option<FinishReason>,
    usage: Option<Usage>,
    text_parts: BTreeMap<u64, String>,
    thinking_parts: BTreeMap<u64, String>,
    audio_chunks: BTreeMap<u64, Vec<String>>,
    audio_media_types: BTreeMap<u64, Option<String>>,
    image_parts: BTreeMap<u64, ImagePart>,
    citation_parts: BTreeMap<u64, Vec<CitationPart>>,
    tool_calls: BTreeMap<u64, ToolCallMeta>,
    message_continuation: Vec<ContinuationState>,
    part_continuation: BTreeMap<u64, Vec<ContinuationState>>,
    logprobs: Vec<TokenLogprob>,
    provider_data: Option<JsonObject>,
}

impl StreamAccumulator {
    pub fn new(request: &Request) -> Self {
        StreamAccumulator::for_model(request.model.clone())
    }

    /// An accumulator knowing only the request's model (the fallback
    /// `Response.model` when no start event names one).
    pub fn for_model(model: impl Into<String>) -> Self {
        StreamAccumulator {
            request_model: model.into(),
            started_id: None,
            started_model: None,
            finish_reason: None,
            usage: None,
            text_parts: BTreeMap::new(),
            thinking_parts: BTreeMap::new(),
            audio_chunks: BTreeMap::new(),
            audio_media_types: BTreeMap::new(),
            image_parts: BTreeMap::new(),
            citation_parts: BTreeMap::new(),
            tool_calls: BTreeMap::new(),
            message_continuation: Vec::new(),
            part_continuation: BTreeMap::new(),
            logprobs: Vec::new(),
            provider_data: None,
        }
    }

    /// Fold one canonical stream event into the accumulated state.
    pub fn push(&mut self, event: &StreamEvent) {
        match event {
            StreamEvent::Start(start) => {
                if start.id.is_some() {
                    self.started_id = start.id.clone();
                }
                if start.model.is_some() {
                    self.started_model = start.model.clone();
                }
            }
            StreamEvent::End(end) => {
                if end.finish_reason.is_some() {
                    self.finish_reason = end.finish_reason;
                }
                if end.usage.is_some() {
                    self.usage = end.usage;
                }
                if end.provider_data.is_some() {
                    self.provider_data = end.provider_data.clone();
                }
            }
            StreamEvent::Error(_) => {}
            StreamEvent::Delta(StreamDeltaEvent { delta }) => self.push_delta(delta),
        }
    }

    fn push_delta(&mut self, delta: &Delta) {
        match delta {
            Delta::Text(d) => {
                self.text_parts
                    .entry(d.part_index)
                    .or_default()
                    .push_str(&d.text);
                self.logprobs.extend(d.logprobs.iter().cloned());
            }
            Delta::Thinking(d) => {
                self.thinking_parts
                    .entry(d.part_index)
                    .or_default()
                    .push_str(&d.text);
            }
            Delta::Audio(d) => {
                self.audio_chunks
                    .entry(d.part_index)
                    .or_default()
                    .push(d.data.clone().unwrap_or_default());
                self.audio_media_types
                    .entry(d.part_index)
                    .or_insert_with(|| d.media_type.clone());
            }
            Delta::ToolCall(d) => {
                let meta = self.tool_calls.entry(d.part_index).or_default();
                if let Some(id) = &d.id {
                    meta.id = Some(id.clone());
                }
                if let Some(name) = &d.name {
                    meta.name = Some(name.clone());
                }
                meta.raw.push_str(&d.input);
            }
            Delta::Image(d) => {
                let media_type = d
                    .media_type
                    .clone()
                    .unwrap_or_else(|| "image/png".to_string());
                let part = if let Some(data) = &d.data {
                    ImagePart {
                        media_type,
                        data: Some(data.clone()),
                        ..Default::default()
                    }
                } else if let Some(url) = &d.url {
                    ImagePart {
                        media_type,
                        url: Some(url.clone()),
                        ..Default::default()
                    }
                } else if let Some(file_id) = &d.file_id {
                    ImagePart {
                        media_type,
                        file_id: Some(file_id.clone()),
                        ..Default::default()
                    }
                } else {
                    return;
                };
                self.image_parts.insert(d.part_index, part);
            }
            Delta::Citation(d) => {
                self.citation_parts
                    .entry(d.part_index)
                    .or_default()
                    .push(CitationPart {
                        url: d.url.clone(),
                        title: d.title.clone(),
                        text: d.text.clone(),
                        continuation: Vec::new(),
                    });
            }
            Delta::Continuation(d) => {
                let state = d.to_state();
                match d.part_index {
                    None => self.message_continuation.push(state),
                    Some(index) => self.part_continuation.entry(index).or_default().push(state),
                }
            }
        }
    }

    /// The complete `Response`. A tool call whose fragments never carried a
    /// name is a `StreamAssemblyError` (MAP-9) carrying everything else as
    /// `partial`.
    pub fn response(&self) -> Result<Response, Lm15Error> {
        let unnamed: Vec<u64> = self
            .tool_calls
            .iter()
            .filter(|(_, meta)| meta.name.as_deref().is_none_or(str::is_empty))
            .map(|(index, _)| *index)
            .collect();
        if let Some(first) = unnamed.first() {
            let partial = self.assemble(&unnamed);
            let mut meta = ErrorMeta::new(format!(
                "tool call at part {first} arrived without a name; the adapter that produced \
                 this stream must set ToolCallDelta.name on the call's first fragment (MAP-9: \
                 lm15 does not guess which tool the model meant)"
            ));
            meta.provider_code = None;
            return Err(Lm15Error::StreamAssemblyError(StreamAssembly {
                meta,
                partial: Some(Box::new(partial)),
                part_index: Some(*first),
            }));
        }
        Ok(self.assemble(&[]))
    }

    /// MAP-9 assembly: slots in ascending index; within a slot the fixed
    /// kind order thinking, text, image, audio, citations, tool call.
    fn assemble(&self, skip: &[u64]) -> Response {
        let mut indexes: Vec<u64> = self
            .thinking_parts
            .keys()
            .chain(self.text_parts.keys())
            .chain(self.image_parts.keys())
            .chain(self.audio_chunks.keys())
            .chain(self.citation_parts.keys())
            .chain(self.tool_calls.keys())
            .chain(self.part_continuation.keys())
            .copied()
            .collect();
        indexes.sort_unstable();
        indexes.dedup();

        let mut parts: Vec<Part> = Vec::new();
        for idx in indexes {
            let continuation = self
                .part_continuation
                .get(&idx)
                .cloned()
                .unwrap_or_default();
            let skipped = skip.contains(&idx);
            let has_tool = self.tool_calls.contains_key(&idx) && !skipped;
            let mut emitted = false;
            if let Some(text) = self.thinking_parts.get(&idx) {
                emitted = true;
                parts.push(Part::Thinking(ThinkingPart {
                    text: text.clone(),
                    continuation: continuation.clone(),
                }));
            }
            if let Some(text) = self.text_parts.get(&idx) {
                emitted = true;
                parts.push(Part::Text(TextPart {
                    text: text.clone(),
                    continuation: continuation.clone(),
                }));
            }
            if let Some(image) = self.image_parts.get(&idx) {
                emitted = true;
                parts.push(Part::Image(ImagePart {
                    continuation: continuation.clone(),
                    ..image.clone()
                }));
            }
            if let Some(chunks) = self.audio_chunks.get(&idx) {
                emitted = true;
                let raw = concat_b64_chunks(chunks);
                let media_type = self.audio_media_types.get(&idx).cloned().flatten();
                let (media_type, bytes) = match media_type.as_deref() {
                    None | Some("audio/pcm") | Some("audio/pcm16") => {
                        ("audio/wav".to_string(), pcm_to_wav(&raw, 24000, 1, 16))
                    }
                    Some(other) => (other.to_string(), raw),
                };
                parts.push(Part::Audio(AudioPart {
                    media_type,
                    data: Some(base64_encode(&bytes)),
                    continuation: continuation.clone(),
                    ..Default::default()
                }));
            }
            if let Some(citations) = self.citation_parts.get(&idx) {
                emitted = true;
                parts.extend(citations.iter().map(|c| {
                    Part::Citation(CitationPart {
                        continuation: continuation.clone(),
                        ..c.clone()
                    })
                }));
            }
            if has_tool {
                let meta = &self.tool_calls[&idx];
                let id = meta
                    .id
                    .clone()
                    .filter(|s| !s.is_empty())
                    // A missing id gets an lm15-minted correlator (Gemini
                    // sends none); a missing name is never minted.
                    .unwrap_or_else(|| format!("tool_call_{idx}"));
                parts.push(Part::ToolCall(ToolCallPart {
                    id,
                    name: meta.name.clone().unwrap_or_default(),
                    input: crate::dialects::wire_json::parse_json_text(&meta.raw),
                    continuation,
                }));
            } else if !skipped && !emitted {
                // A slot that received only continuation state keeps it on
                // an empty text part.
                parts.push(Part::Text(TextPart {
                    text: String::new(),
                    continuation,
                }));
            }
        }
        if parts.is_empty() {
            parts.push(Part::text(""));
        }

        let has_tool_calls = parts.iter().any(|p| matches!(p, Part::ToolCall(_)));
        let finish_reason = match self.finish_reason {
            None if has_tool_calls => FinishReason::ToolCall,
            None => FinishReason::Stop,
            Some(FinishReason::Stop) if has_tool_calls => FinishReason::ToolCall,
            Some(reason) => reason,
        };
        Response {
            id: self.started_id.clone(),
            model: self
                .started_model
                .clone()
                .unwrap_or_else(|| self.request_model.clone()),
            message: Message {
                role: Role::Assistant,
                parts,
                continuation: self.message_continuation.clone(),
            },
            finish_reason,
            usage: self.usage.unwrap_or_default(),
            logprobs: if self.logprobs.is_empty() {
                None
            } else {
                Some(self.logprobs.clone())
            },
            provider_data: self.provider_data.clone(),
        }
    }
}

/// Consume canonical events and build the complete `Response`. An error
/// event is returned as the typed error; iteration stops at the end event.
pub fn materialize_response<'a>(
    events: impl IntoIterator<Item = &'a StreamEvent>,
    request: &Request,
) -> Result<Response, Lm15Error> {
    let mut accumulator = StreamAccumulator::new(request);
    let mut response: Option<Response> = None;
    for event in events {
        if let Some(response) = response {
            return Err(crate::response_stream::trailing(response));
        }
        if let StreamEvent::Error(error) = event {
            return Err(error_from_detail(&error.error));
        }
        accumulator.push(event);
        if matches!(event, StreamEvent::End(_)) {
            response = Some(accumulator.response()?);
        }
    }
    response.ok_or_else(|| crate::response_stream::incomplete(&accumulator))
}

/// The typed error of a stream error event.
pub fn error_from_detail(detail: &ErrorDetail) -> Lm15Error {
    let mut meta = ErrorMeta::new(detail.message.clone());
    meta.provider_code = detail.provider_code.clone();
    Lm15Error::of_class(detail.code.class(), meta)
}

/// A complete `Response` as stream events, losslessly for the streamable
/// part kinds (INV-035); a part with no delta form is a `ValidationError`
/// (the reference's `TypeError`).
pub fn response_to_events(
    response: &Response,
) -> Result<Vec<StreamEvent>, crate::types::ValidationError> {
    let mut out = vec![StreamEvent::Start(StreamStartEvent {
        id: response.id.clone(),
        model: Some(response.model.clone()),
    })];
    // Response.logprobs is message-level; the whole sequence rides the
    // first text delta so Response -> events -> Response is lossless.
    let mut pending_logprobs = response.logprobs.clone().unwrap_or_default();
    let push = |out: &mut Vec<StreamEvent>, delta: Delta| {
        out.push(StreamEvent::Delta(StreamDeltaEvent { delta }));
    };
    for (idx, part) in response.message.parts.iter().enumerate() {
        let idx = idx as u64;
        match part {
            Part::Text(p) => {
                push(
                    &mut out,
                    Delta::Text(TextDelta {
                        text: p.text.clone(),
                        part_index: idx,
                        logprobs: std::mem::take(&mut pending_logprobs),
                    }),
                );
            }
            Part::Thinking(p) => push(
                &mut out,
                Delta::Thinking(ThinkingDelta {
                    text: p.text.clone(),
                    part_index: idx,
                }),
            ),
            Part::ToolCall(p) => push(
                &mut out,
                Delta::ToolCall(ToolCallDelta {
                    input: serde_json::Value::Object(p.input.clone()).to_string(),
                    part_index: idx,
                    id: Some(p.id.clone()),
                    name: Some(p.name.clone()),
                }),
            ),
            Part::Image(p) => push(
                &mut out,
                Delta::Image(ImageDelta {
                    data: p.data.clone(),
                    url: p.url.clone(),
                    file_id: p.file_id.clone(),
                    part_index: idx,
                    media_type: Some(p.media_type.clone()),
                }),
            ),
            Part::Audio(p) => {
                let Some(data) = &p.data else {
                    return Err(crate::types::ValidationError::type_error(
                        "Cannot convert AudioPart to StreamEvent: AudioDelta only supports inline data",
                    ));
                };
                push(
                    &mut out,
                    Delta::Audio(AudioDelta {
                        data: Some(data.clone()),
                        part_index: idx,
                        media_type: Some(p.media_type.clone()),
                        ..Default::default()
                    }),
                );
            }
            Part::Citation(p) => push(
                &mut out,
                Delta::Citation(CitationDelta {
                    text: p.text.clone(),
                    url: p.url.clone(),
                    title: p.title.clone(),
                    part_index: idx,
                }),
            ),
            other => {
                return Err(crate::types::ValidationError::type_error(format!(
                    "Cannot convert {} part to StreamEvent: no Delta variant exists",
                    other.type_name()
                )));
            }
        }
        for state in part.continuation() {
            push(
                &mut out,
                Delta::Continuation(ContinuationDelta {
                    provider: state.provider.clone(),
                    kind: state.kind.clone(),
                    data: state.data.clone(),
                    part_index: Some(idx),
                }),
            );
        }
    }
    for state in &response.message.continuation {
        push(
            &mut out,
            Delta::Continuation(ContinuationDelta {
                provider: state.provider.clone(),
                kind: state.kind.clone(),
                data: state.data.clone(),
                part_index: None,
            }),
        );
    }
    out.push(StreamEvent::End(StreamEndEvent {
        finish_reason: Some(response.finish_reason),
        usage: Some(response.usage),
        provider_data: response.provider_data.clone(),
    }));
    Ok(out)
}

// ─── Audio helpers ───────────────────────────────────────────────────

/// Decode each base64 chunk and concatenate the raw bytes; a chunk that
/// does not decode is dropped (the reference's fallback after padding).
fn concat_b64_chunks(chunks: &[String]) -> Vec<u8> {
    let mut raw = Vec::new();
    for chunk in chunks {
        if chunk.is_empty() {
            continue;
        }
        if let Ok(bytes) = base64_decode(chunk) {
            raw.extend(bytes);
        }
    }
    raw
}

/// Wrap raw PCM bytes in a WAV header.
fn pcm_to_wav(pcm: &[u8], sample_rate: u32, channels: u16, bits: u16) -> Vec<u8> {
    let byte_rate = sample_rate * u32::from(channels) * u32::from(bits) / 8;
    let block_align = channels * bits / 8;
    let data_size = pcm.len() as u32;
    let mut out = Vec::with_capacity(44 + pcm.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_size).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    fn request() -> Request {
        Request::new("m", vec![Message::user("hi").unwrap()]).unwrap()
    }

    fn text(index: u64, text: &str) -> StreamEvent {
        StreamEvent::Delta(StreamDeltaEvent {
            delta: Delta::Text(TextDelta {
                text: text.into(),
                part_index: index,
                logprobs: Vec::new(),
            }),
        })
    }

    fn end(finish: Option<FinishReason>, usage: Option<Usage>, data: bool) -> StreamEvent {
        StreamEvent::End(StreamEndEvent {
            finish_reason: finish,
            usage,
            provider_data: data.then(JsonObject::new),
        })
    }

    #[test]
    fn map_3_and_4_one_start_one_final_end() {
        let usage = Usage {
            input_tokens: Some(1),
            output_tokens: Some(2),
            total_tokens: Some(3),
            ..Default::default()
        };
        let raw = vec![
            text(0, "a"),
            end(Some(FinishReason::ToolCall), None, true),
            end(None, Some(usage), true),
            end(None, None, false),
        ];
        let out = coalesce_stream(raw, Some("m".into()));
        assert_eq!(out.len(), 3);
        assert!(matches!(&out[0], StreamEvent::Start(s) if s.model.as_deref() == Some("m")));
        match &out[2] {
            StreamEvent::End(e) => {
                assert_eq!(e.finish_reason, Some(FinishReason::ToolCall));
                assert_eq!(e.usage, Some(usage));
                assert!(e.provider_data.is_some());
            }
            other => panic!("{other:?}"),
        }
        // No end seen: nothing fabricated. Duplicate starts collapse.
        let out = coalesce_stream(
            vec![
                StreamEvent::Start(StreamStartEvent::default()),
                StreamEvent::Start(StreamStartEvent::default()),
                text(0, "x"),
            ],
            None,
        );
        assert_eq!(out.len(), 2);
        // An end without deltas still gets a start.
        let out = coalesce_stream(
            vec![end(Some(FinishReason::Stop), None, false)],
            Some("m".into()),
        );
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn map_9_slots_kind_order_and_unnamed_refusal() {
        let mut acc = StreamAccumulator::new(&request());
        acc.push(&StreamEvent::Start(StreamStartEvent {
            id: Some("id".into()),
            model: Some("served".into()),
        }));
        acc.push(&text(0, "Check"));
        acc.push(&text(0, "ing."));
        acc.push(&StreamEvent::Delta(StreamDeltaEvent {
            delta: Delta::ToolCall(ToolCallDelta {
                input: "{\"a\":".into(),
                part_index: 0,
                id: Some("c".into()),
                name: Some("f".into()),
            }),
        }));
        acc.push(&StreamEvent::Delta(StreamDeltaEvent {
            delta: Delta::ToolCall(ToolCallDelta {
                input: " 1}".into(),
                part_index: 0,
                id: None,
                name: None,
            }),
        }));
        acc.push(&StreamEvent::Delta(StreamDeltaEvent {
            delta: Delta::Continuation(ContinuationDelta {
                provider: "x".into(),
                kind: "k".into(),
                data: JsonObject::new(),
                part_index: Some(2),
            }),
        }));
        acc.push(&end(Some(FinishReason::Stop), None, false));
        let response = acc.response().unwrap();
        assert_eq!(response.id.as_deref(), Some("id"));
        assert_eq!(response.model, "served");
        assert_eq!(response.finish_reason, FinishReason::ToolCall);
        assert_eq!(response.message.parts.len(), 3);
        assert!(matches!(&response.message.parts[0], Part::Text(t) if t.text == "Checking."));
        assert!(
            matches!(&response.message.parts[1], Part::ToolCall(c) if c.name == "f" && c.input["a"] == 1)
        );
        assert!(
            matches!(&response.message.parts[2], Part::Text(t) if t.text.is_empty() && t.continuation.len() == 1)
        );

        let mut acc = StreamAccumulator::new(&request());
        acc.push(&text(0, "hi"));
        acc.push(&StreamEvent::Delta(StreamDeltaEvent {
            delta: Delta::ToolCall(ToolCallDelta {
                input: "{}".into(),
                part_index: 1,
                id: None,
                name: None,
            }),
        }));
        let err = acc.response().unwrap_err();
        assert_eq!(err.class_name(), "StreamAssemblyError");
        assert_eq!(err.code().as_str(), "stream_assembly");
        let partial = err.partial().unwrap();
        assert_eq!(partial.message.parts, vec![Part::text("hi")]);
        assert_eq!(partial.finish_reason, FinishReason::Stop);
    }

    #[test]
    fn audio_chunks_become_wav_and_events_round_trip() {
        let mut acc = StreamAccumulator::new(&request());
        acc.push(&StreamEvent::Delta(StreamDeltaEvent {
            delta: Delta::Audio(AudioDelta {
                data: Some("AAEC".into()),
                part_index: 0,
                ..Default::default()
            }),
        }));
        let response = acc.response().unwrap();
        match &response.message.parts[0] {
            Part::Audio(a) => {
                assert_eq!(a.media_type, "audio/wav");
                let bytes = base64_decode(a.data.as_ref().unwrap()).unwrap();
                assert_eq!(&bytes[..4], b"RIFF");
                assert_eq!(bytes.len(), 47);
            }
            other => panic!("{other:?}"),
        }
        let events = response_to_events(&response).unwrap();
        let back = materialize_response(events.iter(), &request()).unwrap();
        assert_eq!(back.message.parts, response.message.parts);
    }
}

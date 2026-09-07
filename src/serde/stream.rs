//! Usage, logprobs, ErrorDetail, Delta, StreamEvent (serde.py `usage_*`,
//! `token_logprob_*`, `error_detail_*`, `delta_*`, `stream_event_*`).

use serde_json::Value;

use super::helpers::{float, opt_strings, Obj, Reader, VResult};
use super::{impl_serde_via_canonical, Canonical};
use crate::errors::ErrorCode;
use crate::types::{
    AudioDelta, CitationDelta, ContinuationDelta, Delta, ErrorDetail, FinishReason, ImageDelta,
    JsonObject, StreamDeltaEvent, StreamEndEvent, StreamErrorEvent, StreamEvent, StreamStartEvent,
    TextDelta, ThinkingDelta, TokenLogprob, ToolCallDelta, TopLogprob, Usage, ValidationError,
};

// ─── Usage ───────────────────────────────────────────────────────────

impl Canonical for Usage {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "Usage")?;
        let usage = Usage {
            input_tokens: r.opt_u64("input_tokens")?,
            output_tokens: r.opt_u64("output_tokens")?,
            total_tokens: r.opt_u64("total_tokens")?,
            cache_read_tokens: r.opt_u64("cache_read_tokens")?,
            cache_write_tokens: r.opt_u64("cache_write_tokens")?,
            reasoning_tokens: r.opt_u64("reasoning_tokens")?,
            input_audio_tokens: r.opt_u64("input_audio_tokens")?,
            output_audio_tokens: r.opt_u64("output_audio_tokens")?,
        }
        .normalized()?;
        usage.validate()?;
        Ok(usage)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::new();
        o.opt("input_tokens", self.input_tokens);
        o.opt("output_tokens", self.output_tokens);
        o.opt("total_tokens", self.total_tokens);
        o.opt("cache_read_tokens", self.cache_read_tokens);
        o.opt("cache_write_tokens", self.cache_write_tokens);
        o.opt("reasoning_tokens", self.reasoning_tokens);
        o.opt("input_audio_tokens", self.input_audio_tokens);
        o.opt("output_audio_tokens", self.output_audio_tokens);
        o.finish()
    }
}

/// A `usage` nest: absent or `{}` on the wire when nothing was reported.
pub(crate) fn usage_to_json_opt(usage: &Usage) -> Option<Value> {
    if usage.is_empty() {
        None
    } else {
        Some(usage.to_json())
    }
}

/// A telemetry nest read leniently (INV-042): a non-object reads as empty.
pub(crate) fn usage_from_parent(r: &Reader<'_>, key: &str) -> VResult<Usage> {
    match r.lenient_object(key) {
        Some(o) => Usage::from_json(&Value::Object(o.clone())),
        None => Ok(Usage::default()),
    }
}

// ─── Logprobs ────────────────────────────────────────────────────────

fn logprob_bytes(r: &Reader<'_>) -> VResult<Option<Vec<u64>>> {
    match r.get("bytes") {
        None => Ok(None),
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| match item {
                Value::Number(n) if n.is_u64() => Ok(n.as_u64().unwrap_or_default()),
                _ => Err(ValidationError::type_error(
                    "bytes must contain non-negative ints",
                )),
            })
            .collect::<VResult<Vec<u64>>>()
            .map(Some),
        Some(_) => Err(ValidationError::type_error("bytes must be a list of ints")),
    }
}

impl Canonical for TopLogprob {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "TopLogprob")?;
        let top = TopLogprob {
            token: r.req_str("token")?,
            logprob: r.req_f64("logprob")?,
            bytes: logprob_bytes(&r)?,
            token_id: r.opt_i64("token_id")?,
        };
        top.validate()?;
        Ok(top)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::new();
        // token and logprob are required: "" and 0.0 are data, not emptiness.
        o.set("token", self.token.as_str())
            .set("logprob", float(self.logprob));
        o.opt("bytes", opt_strings(self.bytes.as_deref()));
        o.opt("token_id", self.token_id);
        o.finish()
    }
}

impl Canonical for TokenLogprob {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "TokenLogprob")?;
        let token = TokenLogprob {
            token: r.req_str("token")?,
            logprob: r.req_f64("logprob")?,
            bytes: logprob_bytes(&r)?,
            token_id: r.opt_i64("token_id")?,
            top: r
                .array_or_empty("top")?
                .iter()
                .map(TopLogprob::from_json)
                .collect::<VResult<Vec<TopLogprob>>>()?,
        };
        token.validate()?;
        Ok(token)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::new();
        o.set("token", self.token.as_str())
            .set("logprob", float(self.logprob));
        o.opt("bytes", opt_strings(self.bytes.as_deref()));
        o.opt("token_id", self.token_id);
        if !self.top.is_empty() {
            o.set(
                "top",
                Value::Array(self.top.iter().map(Canonical::to_json).collect()),
            );
        }
        o.finish()
    }
}

pub(crate) fn logprobs_from_json(r: &Reader<'_>) -> VResult<Option<Vec<TokenLogprob>>> {
    r.opt_array("logprobs")?
        .map(|items| items.iter().map(TokenLogprob::from_json).collect())
        .transpose()
}

pub(crate) fn logprobs_to_json(logprobs: &[TokenLogprob]) -> Option<Value> {
    if logprobs.is_empty() {
        None
    } else {
        Some(Value::Array(
            logprobs.iter().map(Canonical::to_json).collect(),
        ))
    }
}

// ─── ErrorDetail ─────────────────────────────────────────────────────

impl Canonical for ErrorDetail {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "ErrorDetail")?;
        let detail = ErrorDetail {
            code: ErrorCode::parse(&r.req_str("code")?)?,
            message: r.str_or_empty("message")?,
            provider_code: r.opt_str("provider_code")?,
        };
        detail.validate()?;
        Ok(detail)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::new();
        o.set("code", self.code.as_str());
        o.omit_empty("message", self.message.as_str());
        o.omit_empty_opt("provider_code", self.provider_code.clone());
        o.finish()
    }
}

// ─── Delta ───────────────────────────────────────────────────────────

impl Canonical for Delta {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "Delta")?;
        let type_name = r.req_str("type")?;
        // INV-006/INV-045: part_index is an int >= 0, default 0.
        let part_index = r.opt_u64("part_index")?;
        let index = part_index.unwrap_or(0);
        let delta = match type_name.as_str() {
            "text" => Delta::Text(TextDelta {
                text: r.str_or_empty("text")?,
                part_index: index,
                logprobs: logprobs_from_json(&r)?.unwrap_or_default(),
            }),
            "thinking" => Delta::Thinking(ThinkingDelta {
                text: r.str_or_empty("text")?,
                part_index: index,
            }),
            "audio" => Delta::Audio(AudioDelta {
                data: r.opt_str("data")?,
                url: r.opt_str("url")?,
                file_id: r.opt_str("file_id")?,
                part_index: index,
                media_type: r.opt_str("media_type")?,
            }),
            "image" => Delta::Image(ImageDelta {
                data: r.opt_str("data")?,
                url: r.opt_str("url")?,
                file_id: r.opt_str("file_id")?,
                part_index: index,
                media_type: r.opt_str("media_type")?,
            }),
            "tool_call" => Delta::ToolCall(ToolCallDelta {
                input: r.str_or_empty("input")?,
                part_index: index,
                id: r.opt_str("id")?,
                name: r.opt_str("name")?,
            }),
            "citation" => Delta::Citation(CitationDelta {
                text: r.opt_str("text")?,
                url: r.opt_str("url")?,
                title: r.opt_str("title")?,
                part_index: index,
            }),
            "continuation" => Delta::Continuation(ContinuationDelta {
                provider: r.req_str("provider")?,
                kind: r.req_str("kind")?,
                data: r.object_or_empty("data")?,
                part_index,
            }),
            other => {
                return Err(ValidationError::value(format!(
                    "unsupported delta type: {other}"
                )))
            }
        };
        delta.validate()?;
        Ok(delta)
    }

    fn to_json(&self) -> Value {
        // Deltas drop only null fields: empty strings ARE emitted and
        // part_index is always emitted (ContinuationDelta: when not null).
        let mut o = Obj::typed(self.type_name());
        o.opt("part_index", self.part_index());
        match self {
            Delta::Text(d) => {
                o.set("text", d.text.as_str());
                o.opt("logprobs", logprobs_to_json(&d.logprobs));
            }
            Delta::Thinking(d) => {
                o.set("text", d.text.as_str());
            }
            Delta::Audio(d) => {
                o.opt("data", d.data.clone());
                o.opt("url", d.url.clone());
                o.opt("file_id", d.file_id.clone());
                o.opt("media_type", d.media_type.clone());
            }
            Delta::Image(d) => {
                o.opt("data", d.data.clone());
                o.opt("url", d.url.clone());
                o.opt("file_id", d.file_id.clone());
                o.opt("media_type", d.media_type.clone());
            }
            Delta::ToolCall(d) => {
                o.set("input", d.input.as_str());
                o.opt("id", d.id.clone());
                o.opt("name", d.name.clone());
            }
            Delta::Citation(d) => {
                o.opt("text", d.text.clone());
                o.opt("url", d.url.clone());
                o.opt("title", d.title.clone());
            }
            Delta::Continuation(d) => {
                o.set("provider", d.provider.as_str())
                    .set("kind", d.kind.as_str())
                    .set("data", Value::Object(d.data.clone()));
            }
        }
        o.finish()
    }
}

// ─── StreamEvent ─────────────────────────────────────────────────────

pub(crate) fn provider_data_from_json(r: &Reader<'_>) -> VResult<Option<JsonObject>> {
    r.opt_object("provider_data")
}

impl Canonical for StreamEvent {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "StreamEvent")?;
        let type_name = r.req_str("type")?;
        let event = match type_name.as_str() {
            "start" => StreamEvent::Start(StreamStartEvent {
                id: r.opt_str("id")?,
                model: r.opt_str("model")?,
            }),
            "delta" => StreamEvent::Delta(StreamDeltaEvent {
                delta: Delta::from_json(r.req("delta")?)?,
            }),
            "end" => StreamEvent::End(StreamEndEvent {
                finish_reason: r
                    .opt_str("finish_reason")?
                    .map(|s| FinishReason::parse(&s))
                    .transpose()?,
                usage: r
                    .lenient_object("usage")
                    .map(|o| Usage::from_json(&Value::Object(o.clone())))
                    .transpose()?,
                provider_data: provider_data_from_json(&r)?,
            }),
            "error" => StreamEvent::Error(StreamErrorEvent {
                error: ErrorDetail::from_json(r.req("error")?)?,
            }),
            other => {
                return Err(ValidationError::value(format!(
                    "unsupported stream event type: {other}"
                )))
            }
        };
        event.validate()?;
        Ok(event)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::typed(self.type_name());
        match self {
            StreamEvent::Start(e) => {
                o.omit_empty_opt("id", e.id.clone());
                o.omit_empty_opt("model", e.model.clone());
            }
            StreamEvent::Delta(e) => {
                o.set("delta", e.delta.to_json());
            }
            StreamEvent::End(e) => {
                o.opt("finish_reason", e.finish_reason.map(FinishReason::as_str));
                o.opt("usage", e.usage.as_ref().and_then(usage_to_json_opt));
                o.omit_empty_opt("provider_data", e.provider_data.clone().map(Value::Object));
            }
            StreamEvent::Error(e) => {
                o.set("error", e.error.to_json());
            }
        }
        o.finish()
    }
}

impl_serde_via_canonical!(
    Usage,
    TopLogprob,
    TokenLogprob,
    ErrorDetail,
    Delta,
    StreamEvent
);

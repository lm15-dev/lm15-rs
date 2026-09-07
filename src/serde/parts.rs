//! Parts and ContinuationState (serde.py `part_to_dict` / `part_from_dict`).

use std::path::PathBuf;

use serde_json::Value;

use super::helpers::{scalar_to_text, Obj, Reader, VResult};
use super::{impl_serde_via_canonical, Canonical};
use crate::types::{
    AudioPart, BinaryPart, CitationPart, ContinuationState, DocumentPart, ImageDetail, ImagePart,
    Part, RefusalPart, TextPart, ThinkingPart, ToolCallPart, ToolResultPart, ValidationError,
    VideoPart,
};

impl Canonical for ContinuationState {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "ContinuationState")?;
        let state = ContinuationState {
            provider: r.req_str("provider")?,
            kind: r.req_str("kind")?,
            data: r.object_or_empty("data")?,
        };
        state.validate()?;
        Ok(state)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::new();
        o.set("provider", self.provider.as_str())
            .set("kind", self.kind.as_str())
            .set("data", Value::Object(self.data.clone()));
        o.finish()
    }
}

pub(crate) fn continuation_from_json(r: &Reader<'_>) -> VResult<Vec<ContinuationState>> {
    match r.get("continuation") {
        None => Ok(Vec::new()),
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| match item {
                Value::Object(_) => ContinuationState::from_json(item),
                _ => Err(ValidationError::type_error(
                    "continuation entries must be objects",
                )),
            })
            .collect(),
        Some(_) => Err(ValidationError::type_error("continuation must be a list")),
    }
}

pub(crate) fn continuation_to_json(o: &mut Obj, states: &[ContinuationState]) {
    if !states.is_empty() {
        o.set(
            "continuation",
            Value::Array(states.iter().map(Canonical::to_json).collect()),
        );
    }
}

/// Parts inside a message or tool result: non-object entries become text
/// (INV-041, INV-047).
pub(crate) fn parts_from_list(items: &[Value]) -> VResult<Vec<Part>> {
    items
        .iter()
        .map(|item| match item {
            Value::Object(_) => Part::from_json(item),
            other => Ok(Part::text(scalar_to_text(other))),
        })
        .collect()
}

pub(crate) fn parts_to_json(parts: &[Part]) -> Value {
    Value::Array(parts.iter().map(Canonical::to_json).collect())
}

struct MediaFields {
    media_type: String,
    data: Option<String>,
    url: Option<String>,
    file_id: Option<String>,
    path: Option<PathBuf>,
}

fn media_fields(r: &Reader<'_>) -> VResult<MediaFields> {
    Ok(MediaFields {
        media_type: r.str_or_empty("media_type")?,
        data: r.opt_str("data")?,
        url: r.opt_str("url")?,
        file_id: r.opt_str("file_id")?,
        path: r.opt_str("path")?.map(PathBuf::from),
    })
}

fn media_to_json(
    o: &mut Obj,
    media_type: &str,
    data: Option<&String>,
    url: Option<&String>,
    file_id: Option<&String>,
    path: Option<&PathBuf>,
) {
    o.set("media_type", media_type);
    o.opt("data", data.cloned());
    o.opt("url", url.cloned());
    o.opt("file_id", file_id.cloned());
    o.opt("path", path.map(|p| p.to_string_lossy().into_owned()));
}

macro_rules! media_from_json {
    ($ty:ident, $r:expr, $continuation:expr $(, $extra:ident = $extra_value:expr)?) => {{
        let m = media_fields(&$r)?;
        $ty {
            media_type: m.media_type,
            data: m.data,
            url: m.url,
            file_id: m.file_id,
            path: m.path,
            $($extra: $extra_value,)?
            continuation: $continuation,
        }
    }};
}

impl Canonical for Part {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "Part")?;
        let type_name = r.req_str("type")?;
        let continuation = continuation_from_json(&r)?;
        let part = match type_name.as_str() {
            "text" => Part::Text(TextPart {
                text: r.str_or_empty("text")?,
                continuation,
            }),
            "thinking" => Part::Thinking(ThinkingPart {
                text: r.str_or_empty("text")?,
                continuation,
            }),
            "refusal" => Part::Refusal(RefusalPart {
                text: r.str_or_empty("text")?,
                continuation,
            }),
            "citation" => Part::Citation(CitationPart {
                url: r.opt_str("url")?,
                title: r.opt_str("title")?,
                text: r.opt_str("text")?,
                continuation,
            }),
            "image" => {
                let detail = r
                    .opt_str("detail")?
                    .map(|d| {
                        ImageDetail::parse(&d).map_err(|_| {
                            ValidationError::value(format!("unsupported ImagePart.detail: {d}"))
                        })
                    })
                    .transpose()?;
                Part::Image(media_from_json!(
                    ImagePart,
                    r,
                    continuation,
                    detail = detail
                ))
            }
            "audio" => Part::Audio(media_from_json!(AudioPart, r, continuation)),
            "video" => Part::Video(media_from_json!(VideoPart, r, continuation)),
            "document" => Part::Document(media_from_json!(DocumentPart, r, continuation)),
            "binary" => Part::Binary(media_from_json!(BinaryPart, r, continuation)),
            "tool_call" => Part::ToolCall(ToolCallPart {
                id: r.req_str("id")?,
                name: r.req_str("name")?,
                input: r.object_or_empty("input")?,
                continuation,
            }),
            "tool_result" => {
                // INV-041: a string is one TextPart (empty → no content →
                // rejected); a list may mix parts and scalars; any other
                // shape is empty content.
                let content = match r.get("content") {
                    Some(Value::String(s)) if s.is_empty() => Vec::new(),
                    Some(Value::String(s)) => vec![Part::text(s.clone())],
                    Some(Value::Array(items)) => parts_from_list(items)?,
                    _ => Vec::new(),
                };
                Part::ToolResult(ToolResultPart {
                    id: r.req_str("id")?,
                    content,
                    name: r.opt_str("name")?,
                    is_error: r.bool_or("is_error", false)?,
                    continuation,
                })
            }
            other => {
                return Err(ValidationError::value(format!(
                    "unsupported part type: {other}"
                )))
            }
        };
        part.validate()?;
        Ok(part)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::typed(self.type_name());
        match self {
            Part::Text(p) => {
                o.set("text", p.text.as_str());
            }
            Part::Thinking(p) => {
                o.set("text", p.text.as_str());
            }
            Part::Refusal(p) => {
                o.set("text", p.text.as_str());
            }
            Part::Citation(p) => {
                o.opt("text", p.text.clone());
                o.opt("url", p.url.clone());
                o.opt("title", p.title.clone());
            }
            Part::Image(p) => {
                media_to_json(
                    &mut o,
                    &p.media_type,
                    p.data.as_ref(),
                    p.url.as_ref(),
                    p.file_id.as_ref(),
                    p.path.as_ref(),
                );
                o.opt("detail", p.detail.map(ImageDetail::as_str));
            }
            Part::Audio(p) => media_to_json(
                &mut o,
                &p.media_type,
                p.data.as_ref(),
                p.url.as_ref(),
                p.file_id.as_ref(),
                p.path.as_ref(),
            ),
            Part::Video(p) => media_to_json(
                &mut o,
                &p.media_type,
                p.data.as_ref(),
                p.url.as_ref(),
                p.file_id.as_ref(),
                p.path.as_ref(),
            ),
            Part::Document(p) => media_to_json(
                &mut o,
                &p.media_type,
                p.data.as_ref(),
                p.url.as_ref(),
                p.file_id.as_ref(),
                p.path.as_ref(),
            ),
            Part::Binary(p) => media_to_json(
                &mut o,
                &p.media_type,
                p.data.as_ref(),
                p.url.as_ref(),
                p.file_id.as_ref(),
                p.path.as_ref(),
            ),
            Part::ToolCall(p) => {
                o.set("id", p.id.as_str())
                    .set("name", p.name.as_str())
                    .set("input", Value::Object(p.input.clone()));
            }
            Part::ToolResult(p) => {
                o.set("id", p.id.as_str());
                o.opt("name", p.name.clone());
                o.set("content", parts_to_json(&p.content));
                if p.is_error {
                    o.set("is_error", true);
                }
            }
        }
        continuation_to_json(&mut o, self.continuation());
        o.finish()
    }
}

impl_serde_via_canonical!(Part, ContinuationState);

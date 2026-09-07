//! Canonical messages and parts → Anthropic content blocks
//! (`lm15/providers/anthropic.py:409-465` is the reference shape; the
//! refusals are this port's reading of port.md rule 4 — no silent drops).
//!
//! Block shapes the wire has: `text`, `image`, `document` (each with a
//! `source`), `tool_use`, `tool_result`, `thinking`, `redacted_thinking`.
//! A canonical part with no block shape (audio, video, binary) is a
//! refusal, never an empty text block.

use std::path::Path;

use serde_json::{json, Map, Value};

use crate::compat::AnthropicThinkingReplay;
use crate::errors::Lm15Error;
use crate::types::{
    base64_encode, base64_payload, continuation_data, CitationPart, DocumentPart, ImagePart,
    Message, Part, Role, ThinkingPart, ToolResultPart,
};

use super::Refuse;

/// The `provider` a continuation state names for this dialect (MAP-7 rule
/// 8, D7): the dialect id, on every door of the wire.
pub const CONTINUATION_PROVIDER: &str = "anthropic";
/// `anthropic:thinking_signature` — `{"signature": str}`.
pub const KIND_THINKING_SIGNATURE: &str = "thinking_signature";
/// `anthropic:redacted_thinking` — `{"data": str}` (MAP-7 rule 11).
pub const KIND_REDACTED_THINKING: &str = "redacted_thinking";

/// The prefix a developer message carries on a wire without the role
/// (types.md § Message: "a prefixed user message").
pub const DEVELOPER_PREFIX: &str = "[developer]";

/// What the renderer needs besides the parts.
pub struct PartContext<'a> {
    pub refuse: &'a Refuse<'a>,
    pub thinking_replay: AnthropicThinkingReplay,
}

/// One canonical message as `{"role", "content": [blocks]}`.
pub fn message(msg: &Message, cx: &PartContext<'_>) -> Result<Value, Lm15Error> {
    let role = match msg.role {
        Role::Assistant => "assistant",
        // Tool results ride a user turn; a developer message is a prefixed
        // user message (the wire has no developer role).
        Role::User | Role::Tool | Role::Developer => "user",
    };
    let content = if msg.role == Role::Developer {
        developer_blocks(&msg.parts, cx)?
    } else {
        blocks(&msg.parts, cx)?
    };
    Ok(json!({"role": role, "content": content}))
}

/// Every part of a message, in order, as blocks. A thinking part that has
/// nothing this wire can carry (empty text, no Anthropic state) renders no
/// block: it would be an empty text block, which the API refuses.
fn blocks(parts: &[Part], cx: &PartContext<'_>) -> Result<Vec<Value>, Lm15Error> {
    let mut out = Vec::with_capacity(parts.len());
    for part in parts {
        if let Some(block) = block(part, cx)? {
            out.push(block);
        }
    }
    Ok(out)
}

/// A developer message: the text parts join into one prefixed block first
/// (`lm15/providers/anthropic.py:459-465`); media parts keep their blocks
/// after it instead of being dropped by a text-only rendering.
fn developer_blocks(parts: &[Part], cx: &PartContext<'_>) -> Result<Vec<Value>, Lm15Error> {
    let mut texts: Vec<&str> = Vec::new();
    let mut media = Vec::new();
    for part in parts {
        match part {
            Part::Text(t) => texts.push(&t.text),
            other => {
                if let Some(block) = block(other, cx)? {
                    media.push(block);
                }
            }
        }
    }
    let mut out = vec![text_block(&format!(
        "{DEVELOPER_PREFIX}\n{}",
        texts.join("\n")
    ))];
    out.extend(media);
    Ok(out)
}

pub fn text_block(text: &str) -> Value {
    json!({"type": "text", "text": text})
}

/// One part as a block, or `None` when the part has nothing to carry.
fn block(part: &Part, cx: &PartContext<'_>) -> Result<Option<Value>, Lm15Error> {
    Ok(Some(match part {
        Part::Text(t) => text_block(&t.text),
        Part::Image(image) => json!({"type": "image", "source": image_source(image, cx.refuse)?}),
        Part::Document(document) => {
            json!({"type": "document", "source": document_source(document, cx.refuse)?})
        }
        Part::ToolCall(call) => {
            json!({"type": "tool_use", "id": call.id, "name": call.name, "input": call.input})
        }
        Part::ToolResult(result) => tool_result_block(result, cx)?,
        Part::Thinking(thinking) => return thinking_block(thinking, cx),
        // The wire has no refusal block in a request; the refusal is what
        // the assistant said, so it replays as its text.
        Part::Refusal(refusal) => text_block(&refusal.text),
        Part::Citation(citation) => text_block(&citation_text(citation)),
        Part::Audio(_) | Part::Video(_) | Part::Binary(_) => {
            return Err(cx.refuse.feature(format!(
                "{} parts have no content block on the Messages API (text, image and document do); \
                 omit the part or send it as a document",
                part.type_name()
            )))
        }
    }))
}

/// `lm15/providers/anthropic.py:418-430`: `tool_use_id` plus the content —
/// one text part travels as a string, anything else as blocks (images and
/// documents survive that way); `is_error` only when true.
fn tool_result_block(result: &ToolResultPart, cx: &PartContext<'_>) -> Result<Value, Lm15Error> {
    let content = tool_result_content(&result.content, cx)?;
    let mut block = Map::new();
    block.insert("type".into(), "tool_result".into());
    block.insert("tool_use_id".into(), Value::String(result.id.clone()));
    match content.as_slice() {
        [single] if single.get("type") == Some(&Value::String("text".into())) => {
            block.insert("content".into(), single["text"].clone());
        }
        _ => {
            block.insert("content".into(), Value::Array(content));
        }
    }
    if result.is_error {
        block.insert("is_error".into(), Value::Bool(true));
    }
    Ok(Value::Object(block))
}

/// `lm15/providers/anthropic.py:450-457`: presentational parts only
/// (INV-013); text, image and document have blocks; a citation renders as
/// its text; audio, video and binary are refusals.
fn tool_result_content(parts: &[Part], cx: &PartContext<'_>) -> Result<Vec<Value>, Lm15Error> {
    parts
        .iter()
        .map(|part| match part {
            Part::Text(t) => Ok(text_block(&t.text)),
            Part::Image(image) => {
                Ok(json!({"type": "image", "source": image_source(image, cx.refuse)?}))
            }
            Part::Document(document) => {
                Ok(json!({"type": "document", "source": document_source(document, cx.refuse)?}))
            }
            Part::Citation(citation) => Ok(text_block(&citation_text(citation))),
            other => Err(cx.refuse.feature(format!(
                "{} parts have no tool_result content block on the Messages API",
                other.type_name()
            ))),
        })
        .collect()
}

/// `lm15/providers/anthropic.py:431-447` and MAP-7 rules 8 and 11: the
/// redacted blob goes back as `redacted_thinking`; a signed block goes
/// back signed; an unsigned block goes back as `thinking` only where the
/// compat says the server takes it (`thinking_replay="unsigned"`), else as
/// assistant text (decision G).
fn thinking_block(
    thinking: &ThinkingPart,
    cx: &PartContext<'_>,
) -> Result<Option<Value>, Lm15Error> {
    if let Some(redacted) = continuation_data(
        &thinking.continuation,
        CONTINUATION_PROVIDER,
        KIND_REDACTED_THINKING,
    ) {
        let mut block = Map::new();
        block.insert("type".into(), "redacted_thinking".into());
        for (key, value) in redacted {
            block.insert(key.clone(), value.clone());
        }
        return Ok(Some(Value::Object(block)));
    }
    let signature = continuation_data(
        &thinking.continuation,
        CONTINUATION_PROVIDER,
        KIND_THINKING_SIGNATURE,
    )
    .and_then(|data| data.get("signature"))
    .and_then(Value::as_str)
    .filter(|s| !s.is_empty());
    if let Some(signature) = signature {
        return Ok(Some(json!({
            "type": "thinking",
            "thinking": thinking.text,
            "signature": signature,
        })));
    }
    if thinking.text.is_empty() {
        // Hidden thinking from another wire (state this dialect cannot
        // read): nothing to replay here, and an empty text block is refused
        // by the API.
        return Ok(None);
    }
    if cx.thinking_replay == AnthropicThinkingReplay::Unsigned {
        return Ok(Some(json!({"type": "thinking", "thinking": thinking.text})));
    }
    Ok(Some(text_block(&thinking.text)))
}

/// `lm15/providers/common.py:62-65`: the text rendering of a citation.
pub fn citation_text(citation: &CitationPart) -> String {
    [&citation.title, &citation.url, &citation.text]
        .into_iter()
        .flatten()
        .filter(|s| !s.is_empty())
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(" — ")
}

fn image_source(image: &ImagePart, refuse: &Refuse<'_>) -> Result<Value, Lm15Error> {
    // `detail` is a resolution hint with no field on this wire; the image
    // is processed as-is. Stated in README-anthropic.md.
    media_source(
        "image",
        &image.media_type,
        image.data.as_deref(),
        image.url.as_deref(),
        image.file_id.as_deref(),
        image.path.as_deref(),
        refuse,
    )
}

fn document_source(document: &DocumentPart, refuse: &Refuse<'_>) -> Result<Value, Lm15Error> {
    media_source(
        "document",
        &document.media_type,
        document.data.as_deref(),
        document.url.as_deref(),
        document.file_id.as_deref(),
        document.path.as_deref(),
        refuse,
    )
}

/// `lm15/providers/common.py:227-237` `anthropic_source`: `url`, `file`,
/// or `base64`. Inline data is sent as its base64 payload (a data-URI
/// prefix and whitespace, which INV-012 tolerates on input, never reach
/// the wire); a local path is read now and encoded.
fn media_source(
    part_type: &str,
    media_type: &str,
    data: Option<&str>,
    url: Option<&str>,
    file_id: Option<&str>,
    path: Option<&Path>,
    refuse: &Refuse<'_>,
) -> Result<Value, Lm15Error> {
    if let Some(url) = url {
        return Ok(json!({"type": "url", "url": url}));
    }
    if let Some(file_id) = file_id {
        return Ok(json!({"type": "file", "file_id": file_id}));
    }
    if let Some(data) = data {
        return Ok(
            json!({"type": "base64", "media_type": media_type, "data": base64_payload(data)}),
        );
    }
    if let Some(path) = path {
        let bytes = std::fs::read(path).map_err(|err| {
            refuse.invalid_request(format!(
                "{part_type} part path {} cannot be read: {err}",
                path.display()
            ))
        })?;
        return Ok(
            json!({"type": "base64", "media_type": media_type, "data": base64_encode(&bytes)}),
        );
    }
    // INV-011 guarantees one source; a validated part never reaches here.
    Err(refuse.invalid_request(format!("{part_type} part has no source")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ContinuationState, JsonObject};

    fn cx<'a>(refuse: &'a Refuse<'a>, replay: AnthropicThinkingReplay) -> PartContext<'a> {
        PartContext {
            refuse,
            thinking_replay: replay,
        }
    }

    fn state(kind: &str, key: &str, value: &str) -> ContinuationState {
        let mut data = JsonObject::new();
        data.insert(key.into(), Value::String(value.into()));
        ContinuationState::new(CONTINUATION_PROVIDER, kind, data).unwrap()
    }

    #[test]
    fn thinking_replay_forms() {
        let refuse = Refuse {
            provider: "anthropic",
        };
        let signed = cx(&refuse, AnthropicThinkingReplay::Signed);
        let unsigned = cx(&refuse, AnthropicThinkingReplay::Unsigned);

        let mut part = ThinkingPart::new("why");
        assert_eq!(
            thinking_block(&part, &signed).unwrap().unwrap(),
            json!({"type": "text", "text": "why"})
        );
        assert_eq!(
            thinking_block(&part, &unsigned).unwrap().unwrap(),
            json!({"type": "thinking", "thinking": "why"})
        );
        part.continuation = vec![state(KIND_THINKING_SIGNATURE, "signature", "sig")];
        assert_eq!(
            thinking_block(&part, &signed).unwrap().unwrap(),
            json!({"type": "thinking", "thinking": "why", "signature": "sig"})
        );
        // An empty signature is no signature (Moonshot sends `""`).
        part.continuation = vec![state(KIND_THINKING_SIGNATURE, "signature", "")];
        assert_eq!(
            thinking_block(&part, &unsigned).unwrap().unwrap(),
            json!({"type": "thinking", "thinking": "why"})
        );
        // Hidden thinking: the blob goes back as redacted_thinking (D5).
        let hidden = ThinkingPart {
            text: String::new(),
            continuation: vec![state(KIND_REDACTED_THINKING, "data", "blob")],
        };
        assert_eq!(
            thinking_block(&hidden, &signed).unwrap().unwrap(),
            json!({"type": "redacted_thinking", "data": "blob"})
        );
        // Hidden thinking of another dialect: no block.
        let foreign = ThinkingPart {
            text: String::new(),
            continuation: vec![ContinuationState::new(
                "openai",
                "reasoning_item",
                JsonObject::new(),
            )
            .unwrap()],
        };
        assert_eq!(thinking_block(&foreign, &signed).unwrap(), None);
    }

    #[test]
    fn media_sources_and_refusals() {
        let refuse = Refuse {
            provider: "anthropic",
        };
        let image = ImagePart::from_data("image/png", "data:image/png;base64,aGk=").unwrap();
        assert_eq!(
            image_source(&image, &refuse).unwrap(),
            json!({"type": "base64", "media_type": "image/png", "data": "aGk="})
        );
        let image = ImagePart::from_file_id("file_1").unwrap();
        assert_eq!(
            image_source(&image, &refuse).unwrap(),
            json!({"type": "file", "file_id": "file_1"})
        );
        let missing = DocumentPart::from_path("/nonexistent/lm15-anthropic-test.pdf").unwrap();
        let err = document_source(&missing, &refuse).unwrap_err();
        assert_eq!(err.class_name(), "InvalidRequestError");

        let signed = cx(&refuse, AnthropicThinkingReplay::Signed);
        let audio = Part::Audio(crate::types::AudioPart::from_url("https://x/a.wav").unwrap());
        let err = block(&audio, &signed).unwrap_err();
        assert_eq!(err.class_name(), "UnsupportedFeatureError");
        assert_eq!(err.provider(), Some("anthropic"));
    }

    #[test]
    fn tool_results_and_developer_messages() {
        let refuse = Refuse {
            provider: "anthropic",
        };
        let signed = cx(&refuse, AnthropicThinkingReplay::Signed);
        let mut result = ToolResultPart::new("call_1", "4").unwrap();
        assert_eq!(
            tool_result_block(&result, &signed).unwrap(),
            json!({"type": "tool_result", "tool_use_id": "call_1", "content": "4"})
        );
        result.is_error = true;
        result.content = vec![
            Part::text("a"),
            Part::Image(ImagePart::from_url("https://x/i.png").unwrap()),
        ];
        assert_eq!(
            tool_result_block(&result, &signed).unwrap(),
            json!({
                "type": "tool_result",
                "tool_use_id": "call_1",
                "content": [
                    {"type": "text", "text": "a"},
                    {"type": "image", "source": {"type": "url", "url": "https://x/i.png"}},
                ],
                "is_error": true,
            })
        );

        let developer = Message::developer(vec![
            Part::text("Be brief."),
            Part::Image(ImagePart::from_url("https://x/i.png").unwrap()),
        ])
        .unwrap();
        assert_eq!(
            message(&developer, &signed).unwrap(),
            json!({"role": "user", "content": [
                {"type": "text", "text": "[developer]\nBe brief."},
                {"type": "image", "source": {"type": "url", "url": "https://x/i.png"}},
            ]})
        );
        let citation = CitationPart::new(Some("https://x"), Some("T"), None::<&str>).unwrap();
        assert_eq!(citation_text(&citation), "T — https://x");
    }
}

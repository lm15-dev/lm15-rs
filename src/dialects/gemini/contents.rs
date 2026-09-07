//! `contents` and `systemInstruction`: canonical messages and parts on
//! the Gemini wire (`lm15/providers/gemini.py:564-608`).
//!
//! Roles: `user` and `tool` → `user`; `assistant` → `model`; `developer`
//! → a `user` turn whose text is prefixed `[developer]\n` (spec/types.md
//! § Message: providers without a native developer role receive a
//! prefixed user message).
//!
//! Parts: text → `text`; media → `inlineData` (base64 + `mimeType`) or
//! `fileData` (`fileUri` from `url` or `file_id`); a local `path` is read
//! and inlined; tool call → `functionCall`; tool result →
//! `functionResponse` with `{"result": <text>}`; thinking → assistant
//! text (MAP-7.8 decision G), or a `thought` part when it carries a
//! `gemini:thought_signature` state; refusal → text; citation → its text
//! rendering. A `gemini:thought_signature` state on a text or tool-call
//! part replays as `thoughtSignature` (MAP-7.8; required on 3.x function
//! calls).
//!
//! No silent drops (playbooks/port.md rule 4): a media part where the
//! wire slot takes text only (`systemInstruction`, a developer turn, a
//! `functionResponse`) and `is_error` on a tool result are refusals.

use serde_json::{Map, Value};

use crate::errors::Lm15Error;
use crate::types::{
    continuation_data, CitationPart, Message, Part, Request, Role, SystemContent, ToolCallPart,
    ToolResultPart,
};
use crate::wire::BuildContext;

use super::{invalid, unsupported};

/// The `data.value` of a `gemini:thought_signature` state, when present.
fn thought_signature(part: &Part, cx: &BuildContext<'_>) -> Result<Option<String>, Lm15Error> {
    let Some(data) = continuation_data(part.continuation(), "gemini", "thought_signature") else {
        return Ok(None);
    };
    match data.get("value") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if s.is_empty() => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(other) => Err(invalid(
            cx,
            format!("gemini:thought_signature data.value must be a string (got {other})"),
        )),
    }
}

/// `parts_to_text` (`lm15/providers/common.py:53-66`) for one citation:
/// title, url, text joined by ` — `.
fn citation_text(part: &CitationPart) -> String {
    [&part.title, &part.url, &part.text]
        .into_iter()
        .filter_map(|field| field.as_deref().filter(|s| !s.is_empty()))
        .collect::<Vec<_>>()
        .join(" — ")
}

/// The lossy text rendering of prompt parts for a text-only slot
/// (`parts_to_text`): text and citations join with `\n`; any other part
/// is a refusal naming the slot.
fn text_only(parts: &[Part], slot: &str, cx: &BuildContext<'_>) -> Result<String, Lm15Error> {
    let mut out = Vec::new();
    for part in parts {
        match part {
            Part::Text(t) => out.push(t.text.clone()),
            Part::Citation(c) => {
                let text = citation_text(c);
                if !text.is_empty() {
                    out.push(text);
                }
            }
            other => {
                return Err(unsupported(
                    cx,
                    format!(
                        "a {} part cannot be sent in {slot} — that wire slot takes text only",
                        other.type_name()
                    ),
                ));
            }
        }
    }
    Ok(out.join("\n"))
}

/// `systemInstruction: {"parts": [{"text": ...}]}`.
pub fn system_instruction(
    system: &SystemContent,
    cx: &BuildContext<'_>,
) -> Result<Value, Lm15Error> {
    let text = match system {
        SystemContent::Text(text) => text.clone(),
        SystemContent::Parts(parts) => text_only(parts, "systemInstruction", cx)?,
    };
    let mut part = Map::new();
    part.insert("text".into(), Value::String(text));
    let mut out = Map::new();
    out.insert("parts".into(), Value::Array(vec![Value::Object(part)]));
    Ok(Value::Object(out))
}

fn media_part(
    media_type: &str,
    data: Option<&str>,
    url: Option<&str>,
    file_id: Option<&str>,
    path: Option<&std::path::Path>,
    cx: &BuildContext<'_>,
) -> Result<Value, Lm15Error> {
    let mut inner = Map::new();
    inner.insert("mimeType".into(), Value::String(media_type.to_string()));
    let key = if let Some(uri) = url.or(file_id) {
        inner.insert("fileUri".into(), Value::String(uri.to_string()));
        "fileData"
    } else if let Some(data) = data {
        inner.insert("data".into(), Value::String(data.to_string()));
        "inlineData"
    } else if let Some(path) = path {
        let bytes = std::fs::read(path).map_err(|err| {
            invalid(
                cx,
                format!("cannot read media part path {}: {err}", path.display()),
            )
        })?;
        inner.insert(
            "data".into(),
            Value::String(crate::types::base64_encode(&bytes)),
        );
        "inlineData"
    } else {
        // Unreachable through the constructors (INV-011).
        return Err(invalid(cx, "media part has no source".into()));
    };
    let mut out = Map::new();
    out.insert(key.into(), Value::Object(inner));
    Ok(Value::Object(out))
}

fn function_call(part: &ToolCallPart, signature: Option<String>) -> Value {
    let mut call = Map::new();
    if !part.id.is_empty() {
        call.insert("id".into(), Value::String(part.id.clone()));
    }
    call.insert("name".into(), Value::String(part.name.clone()));
    call.insert("args".into(), Value::Object(part.input.clone()));
    let mut out = Map::new();
    out.insert("functionCall".into(), Value::Object(call));
    if let Some(signature) = signature {
        out.insert("thoughtSignature".into(), Value::String(signature));
    }
    Value::Object(out)
}

/// The function name a result answers: the part's own `name`, else the
/// name of the tool call with the same id earlier in the transcript, else
/// the reference's placeholder `"tool"` (`lm15/providers/gemini.py:591`).
fn function_response_name(part: &ToolResultPart, request: &Request) -> String {
    if let Some(name) = &part.name {
        return name.clone();
    }
    request
        .messages
        .iter()
        .filter(|m| m.role == Role::Assistant)
        .flat_map(|m| m.parts.iter())
        .find_map(|p| match p {
            Part::ToolCall(call) if call.id == part.id => Some(call.name.clone()),
            _ => None,
        })
        .unwrap_or_else(|| "tool".to_string())
}

fn function_response(
    part: &ToolResultPart,
    request: &Request,
    cx: &BuildContext<'_>,
) -> Result<Value, Lm15Error> {
    if part.is_error {
        return Err(unsupported(
            cx,
            format!(
                "tool_result {:?} carries is_error=true — functionResponse has no error flag; put the failure in the result text (Anthropic carries it)",
                part.id
            ),
        ));
    }
    let text = text_only(&part.content, "a functionResponse", cx)?;
    let mut response = Map::new();
    response.insert("result".into(), Value::String(text));
    let mut fr = Map::new();
    if !part.id.is_empty() {
        fr.insert("id".into(), Value::String(part.id.clone()));
    }
    fr.insert(
        "name".into(),
        Value::String(function_response_name(part, request)),
    );
    fr.insert("response".into(), Value::Object(response));
    let mut out = Map::new();
    out.insert("functionResponse".into(), Value::Object(fr));
    Ok(Value::Object(out))
}

fn text_part(text: &str, signature: Option<String>, thought: bool) -> Value {
    let mut out = Map::new();
    out.insert("text".into(), Value::String(text.to_string()));
    if thought {
        out.insert("thought".into(), Value::Bool(true));
    }
    if let Some(signature) = signature {
        out.insert("thoughtSignature".into(), Value::String(signature));
    }
    Value::Object(out)
}

/// One canonical part on the wire (`_part`, `lm15/providers/gemini.py:564-602`).
fn part(part: &Part, request: &Request, cx: &BuildContext<'_>) -> Result<Value, Lm15Error> {
    Ok(match part {
        Part::Text(t) => text_part(&t.text, thought_signature(part, cx)?, false),
        Part::Image(p) => media_part(
            &p.media_type,
            p.data.as_deref(),
            p.url.as_deref(),
            p.file_id.as_deref(),
            p.path.as_deref(),
            cx,
        )?,
        Part::Audio(p) => media_part(
            &p.media_type,
            p.data.as_deref(),
            p.url.as_deref(),
            p.file_id.as_deref(),
            p.path.as_deref(),
            cx,
        )?,
        Part::Video(p) => media_part(
            &p.media_type,
            p.data.as_deref(),
            p.url.as_deref(),
            p.file_id.as_deref(),
            p.path.as_deref(),
            cx,
        )?,
        Part::Document(p) => media_part(
            &p.media_type,
            p.data.as_deref(),
            p.url.as_deref(),
            p.file_id.as_deref(),
            p.path.as_deref(),
            cx,
        )?,
        Part::Binary(p) => media_part(
            &p.media_type,
            p.data.as_deref(),
            p.url.as_deref(),
            p.file_id.as_deref(),
            p.path.as_deref(),
            cx,
        )?,
        Part::ToolCall(call) => function_call(call, thought_signature(part, cx)?),
        Part::ToolResult(result) => function_response(result, request, cx)?,
        Part::Thinking(t) => {
            // MAP-7.8: native replay when the signature is present; plain
            // assistant text without it (decision G).
            let signature = thought_signature(part, cx)?;
            let thought = signature.is_some();
            text_part(&t.text, signature, thought)
        }
        Part::Refusal(r) => text_part(&r.text, None, false),
        Part::Citation(c) => text_part(&citation_text(c), None, false),
    })
}

/// One message on the wire (`_message`, `lm15/providers/gemini.py:604-608`).
fn message(msg: &Message, request: &Request, cx: &BuildContext<'_>) -> Result<Value, Lm15Error> {
    let mut out = Map::new();
    if msg.role == Role::Developer {
        let text = text_only(&msg.parts, "a developer turn", cx)?;
        out.insert("role".into(), Value::String("user".into()));
        out.insert(
            "parts".into(),
            Value::Array(vec![text_part(
                &format!("[developer]\n{text}"),
                None,
                false,
            )]),
        );
        return Ok(Value::Object(out));
    }
    let role = match msg.role {
        Role::Assistant => "model",
        Role::User | Role::Tool | Role::Developer => "user",
    };
    let parts = msg
        .parts
        .iter()
        .map(|p| part(p, request, cx))
        .collect::<Result<Vec<Value>, Lm15Error>>()?;
    out.insert("role".into(), Value::String(role.into()));
    out.insert("parts".into(), Value::Array(parts));
    Ok(Value::Object(out))
}

/// `contents` for the messages the wire carries (`request` is the whole
/// transcript, for tool-name lookup).
pub fn contents(
    messages: &[Message],
    request: &Request,
    cx: &BuildContext<'_>,
) -> Result<Value, Lm15Error> {
    Ok(Value::Array(
        messages
            .iter()
            .map(|m| message(m, request, cx))
            .collect::<Result<Vec<Value>, Lm15Error>>()?,
    ))
}

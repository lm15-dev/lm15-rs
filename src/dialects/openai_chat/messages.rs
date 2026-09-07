//! The `messages` array (`lm15/providers/openai_chat.py:89-116`,
//! `:226-303`), in the reference's key order: system/developer
//! instructions first, then each canonical message as the wire spells
//! it — `tool` rows per tool result, assistant rows with `content`,
//! `reasoning_content` (per `thinking_replay`) and `tool_calls`, user
//! rows as a bare string or a content array.

use serde_json::{json, Map, Value};

use super::cache::{breakpoint_index, breakpoint_unsupported, stable_prefix};
use super::text::{data_uri, unsupported};
use crate::dialects::content;
use crate::compat::{
    IncludeOmit, OpenAIChatAssistantAfterToolResult, OpenAIChatAssistantReasoningContent,
    OpenAIChatThinkingReplay, ResolvedOpenAIChatCompat,
};
use crate::errors::Lm15Error;
use crate::types::{ImagePart, Message, Part, Request, Role, SystemContent, ToolResultPart};

const BREAKPOINT: &str = "prompt_cache_breakpoint";

fn breakpoint_mark() -> Value {
    json!({"mode": "explicit"})
}

fn object(pairs: Vec<(&str, Value)>) -> Value {
    let mut map = Map::with_capacity(pairs.len());
    for (key, value) in pairs {
        map.insert(key.to_string(), value);
    }
    Value::Object(map)
}

pub(super) fn build_messages(
    request: &Request,
    compat: &ResolvedOpenAIChatCompat,
    provider: &str,
) -> Result<Vec<Value>, Lm15Error> {
    let mut messages: Vec<Value> = Vec::with_capacity(request.messages.len() + 1);
    let role = compat.instruction_role.as_str();

    if let Some(system) = &request.system {
        let text = match system {
            SystemContent::Text(text) => text.clone(),
            SystemContent::Parts(parts) => content::parts_to_text(parts, provider, "the system message")?,
        };
        if stable_prefix(request, compat.cache_control) {
            // prefix="stable": the mark rides on the system message's text
            // content part (array form; a bare string cannot carry it).
            messages.push(object(vec![
                ("role", Value::String(role.into())),
                (
                    "content",
                    Value::Array(vec![object(vec![
                        ("type", Value::String("text".into())),
                        ("text", Value::String(text)),
                        (BREAKPOINT, breakpoint_mark()),
                    ])]),
                ),
            ]));
        } else {
            messages.push(object(vec![
                ("role", Value::String(role.into())),
                ("content", Value::String(text)),
            ]));
        }
    }

    let breakpoint = breakpoint_index(request, compat.cache_control);
    for (index, message) in request.messages.iter().enumerate() {
        let at_breakpoint = breakpoint == Some(index);
        if at_breakpoint && matches!(message.role, Role::Assistant | Role::Tool) {
            return Err(breakpoint_unsupported(provider, index, message.role));
        }
        match message.role {
            Role::Tool => {
                for part in &message.parts {
                    if let Part::ToolResult(result) = part {
                        messages.push(tool_row(result, compat, provider)?);
                    }
                }
                // `assistant_after_tool_result="insert"`: servers that
                // require an assistant turn between a tool row and the
                // next user turn get an empty one. Never at the end of the
                // transcript: the server produces that turn.
                let next = request.messages.get(index + 1);
                if compat.assistant_after_tool_result == OpenAIChatAssistantAfterToolResult::Insert
                    && next.is_some_and(|m| m.role != Role::Assistant && m.role != Role::Tool)
                {
                    messages.push(object(vec![
                        ("role", Value::String("assistant".into())),
                        ("content", Value::String(String::new())),
                    ]));
                }
            }
            Role::Assistant => messages.push(assistant_row(message, compat, provider)?),
            Role::User | Role::Developer => {
                let wire_role = if message.role == Role::Developer {
                    role
                } else {
                    "user"
                };
                let mut content = content_parts(message, at_breakpoint, provider)?;
                if at_breakpoint {
                    // The breakpoint rides on the last text content block of
                    // the prefix message (chat--create.md:
                    // ChatCompletionContentPartText carries it).
                    let last_is_text = content
                        .as_array_mut()
                        .and_then(|blocks| blocks.last_mut())
                        .and_then(Value::as_object_mut)
                        .filter(|block| block.get("type") == Some(&Value::String("text".into())))
                        .map(|block| {
                            block.insert(BREAKPOINT.into(), breakpoint_mark());
                        });
                    if last_is_text.is_none() {
                        return Err(breakpoint_unsupported(provider, index, message.role));
                    }
                }
                messages.push(object(vec![
                    ("role", Value::String(wire_role.into())),
                    ("content", content),
                ]));
            }
        }
    }
    Ok(messages)
}

/// One `tool` row (MAP-10): a string when the content is text-only; on a
/// preset that proved the array form live, text and image_url blocks; a
/// media part the preset does not admit raises first. `is_error` rides
/// as an `[error] ` prefix (rule 5: the wire has no flag).
fn tool_row(
    result: &ToolResultPart,
    compat: &ResolvedOpenAIChatCompat,
    provider: &str,
) -> Result<Value, Lm15Error> {
    content::check_tool_result_media(provider, result, compat.tool_result_media, "a Chat Completions tool row")?;
    let output = if content::text_only(&result.content) {
        Value::String(content::error_text(
            result,
            content::parts_to_text(&result.content, provider, "a Chat Completions tool row")?,
        ))
    } else {
        let mut blocks: Vec<Value> = Vec::with_capacity(result.content.len());
        for part in &result.content {
            blocks.push(match part {
                Part::Image(image) => image_block(image, provider)?,
                other => text_block(&content::parts_to_text(std::slice::from_ref(other), provider, "a Chat Completions tool row")?),
            });
        }
        if result.is_error {
            match blocks.iter_mut().find(|b| b.get("type") == Some(&Value::String("text".into()))) {
                Some(Value::Object(block)) => {
                    let text = block.get("text").and_then(Value::as_str).unwrap_or("").to_string();
                    block.insert("text".into(), Value::String(format!("[error] {text}")));
                }
                _ => blocks.insert(0, text_block("[error]")),
            }
        }
        Value::Array(blocks)
    };
    let mut pairs = vec![
        ("role", Value::String("tool".into())),
        ("tool_call_id", Value::String(result.id.clone())),
        ("content", output),
    ];
    if compat.tool_result_name == IncludeOmit::Include {
        if let Some(name) = result.name.as_ref().filter(|n| !n.is_empty()) {
            pairs.push(("name", Value::String(name.clone())));
        }
    }
    Ok(object(pairs))
}

/// One assistant row (`openai_chat.py:258-284`): text, refusal text and
/// (per `thinking_replay`) thinking text joined by newlines into
/// `content` (`null` when there is none), `reasoning_content` on the
/// native replay, `tool_calls` with compact JSON arguments.
fn assistant_row(
    message: &Message,
    compat: &ResolvedOpenAIChatCompat,
    provider: &str,
) -> Result<Value, Lm15Error> {
    let mut text_bits: Vec<&str> = Vec::new();
    let mut thinking_bits: Vec<&str> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    for part in &message.parts {
        match part {
            Part::Text(text) => text_bits.push(&text.text),
            Part::Refusal(refusal) if !refusal.text.is_empty() => text_bits.push(&refusal.text),
            Part::Refusal(_) => {}
            Part::Thinking(thinking) => {
                if thinking.text.is_empty() {
                    continue;
                }
                match compat.thinking_replay {
                    OpenAIChatThinkingReplay::AsText => text_bits.push(&thinking.text),
                    OpenAIChatThinkingReplay::Native => thinking_bits.push(&thinking.text),
                    OpenAIChatThinkingReplay::Omit => {}
                }
            }
            Part::ToolCall(call) => {
                let arguments = serde_json::to_string(&Value::Object(call.input.clone()))
                    .expect("a JSON object serializes");
                tool_calls.push(object(vec![
                    ("id", Value::String(call.id.clone())),
                    ("type", Value::String("function".into())),
                    (
                        "function",
                        object(vec![
                            ("name", Value::String(call.name.clone())),
                            ("arguments", Value::String(arguments)),
                        ]),
                    ),
                ]));
            }
            // Citations annotate the text they sit next to; the text is
            // replayed, the annotation has no assistant slot on this wire
            // (stated in README-openai-chat.md).
            Part::Citation(_) => {}
            Part::Image(_)
            | Part::Audio(_)
            | Part::Video(_)
            | Part::Document(_)
            | Part::Binary(_)
            | Part::ToolResult(_) => {
                return Err(unsupported(
                    provider,
                    format!(
                        "an assistant {} part cannot be replayed — the Chat Completions wire \
                         carries assistant text and tool calls only",
                        part.type_name()
                    ),
                ));
            }
        }
    }
    let content = if text_bits.is_empty() {
        Value::Null
    } else {
        Value::String(text_bits.join("\n"))
    };
    let mut pairs = vec![
        ("role", Value::String("assistant".into())),
        ("content", content),
    ];
    if compat.thinking_replay == OpenAIChatThinkingReplay::Native {
        let thinking = thinking_bits.join("\n");
        if !thinking.is_empty()
            || compat.assistant_reasoning_content
                == OpenAIChatAssistantReasoningContent::IncludeEmpty
        {
            pairs.push(("reasoning_content", Value::String(thinking)));
        }
    }
    if !tool_calls.is_empty() {
        pairs.push(("tool_calls", Value::Array(tool_calls)));
    }
    Ok(object(pairs))
}

/// User/developer content (`openai_chat.py:89-116`): a lone text part is
/// a bare string unless `force_array`; otherwise a content array of
/// `text` and `image_url` blocks. Parts the wire has no slot for refuse
/// (port.md rule 4; the reference dropped them, stated in
/// README-openai-chat.md). Prompt parts only reach here (INV-022).
fn content_parts(message: &Message, force_array: bool, provider: &str) -> Result<Value, Lm15Error> {
    if let [Part::Text(text)] = message.parts.as_slice() {
        if !force_array {
            return Ok(Value::String(text.text.clone()));
        }
    }
    let mut blocks: Vec<Value> = Vec::with_capacity(message.parts.len());
    for part in &message.parts {
        match part {
            Part::Text(text) => blocks.push(text_block(&text.text)),
            Part::Image(image) => blocks.push(image_block(image, provider)?),
            other => {
                return Err(unsupported(
                    provider,
                    format!(
                        "a {} part in a {} message has no slot on the Chat Completions wire \
                         (text and image_url only); pass provider syntax through extensions",
                        other.type_name(),
                        message.role
                    ),
                ));
            }
        }
    }
    Ok(Value::Array(blocks))
}

fn text_block(text: &str) -> Value {
    object(vec![
        ("type", Value::String("text".into())),
        ("text", Value::String(text.into())),
    ])
}

/// `{"type": "image_url", "image_url": {"url", "detail"?}}`: a URL
/// verbatim, inline data or a path (read now) as a data URI. `file_id`
/// has no slot on this wire (MAP-10: a raise, never empty text).
fn image_block(image: &ImagePart, provider: &str) -> Result<Value, Lm15Error> {
    let url = match (&image.url, &image.data, &image.path) {
        (Some(url), _, _) => url.clone(),
        (None, Some(data), _) => data_uri(&image.media_type, data),
        (None, None, Some(path)) => {
            let bytes = std::fs::read(path).map_err(|err| {
                unsupported(provider, format!("cannot read image part path {}: {err}", path.display()))
            })?;
            data_uri(&image.media_type, &crate::types::base64_encode(&bytes))
        }
        (None, None, None) => {
            return Err(unsupported(
                provider,
                "an image addressed by file_id cannot be sent on the Chat Completions wire (no file \
                 reference form); pass a URL or inline data",
            ))
        }
    };
    let mut inner = vec![("url", Value::String(url))];
    if let Some(detail) = image.detail {
        inner.push(("detail", Value::String(detail.as_str().into())));
    }
    Ok(object(vec![
        ("type", Value::String("image_url".into())),
        ("image_url", object(inner)),
    ]))
}

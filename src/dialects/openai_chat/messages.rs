//! The `messages` array (`lm15/providers/openai_chat.py:89-116`,
//! `:226-303`), in the reference's key order: system/developer
//! instructions first, then each canonical message as the wire spells
//! it — `tool` rows per tool result, assistant rows with `content`,
//! `reasoning_content` (per `thinking_replay`) and `tool_calls`, user
//! rows as a bare string or a content array.

use serde_json::{json, Map, Value};

use super::cache::{breakpoint_index, breakpoint_unsupported, stable_prefix};
use super::text::{data_uri, parts_to_text, unsupported};
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
            SystemContent::Parts(parts) => parts_to_text(parts),
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
                        messages.push(tool_row(result, compat));
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

/// One `tool` row (`openai_chat.py:242-256`). The output is the lossy
/// text of the result content; a result with no text at all names its
/// part types (the reference's placeholder, kept: a refusal here would
/// break the tool loop for a tool that returned media).
fn tool_row(result: &ToolResultPart, compat: &ResolvedOpenAIChatCompat) -> Value {
    let mut output = parts_to_text(&result.content);
    if output.is_empty() {
        let types: Vec<String> = result
            .content
            .iter()
            .map(|p| format!("{{\"type\": \"{}\"}}", p.type_name()))
            .collect();
        output = format!("[{}]", types.join(", "));
    }
    let mut pairs = vec![
        ("role", Value::String("tool".into())),
        ("tool_call_id", Value::String(result.id.clone())),
        ("content", Value::String(output)),
    ];
    if compat.tool_result_name == IncludeOmit::Include {
        if let Some(name) = result.name.as_ref().filter(|n| !n.is_empty()) {
            pairs.push(("name", Value::String(name.clone())));
        }
    }
    object(pairs)
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
/// verbatim or inline data as a data URI. `file_id` and `path` have no
/// slot (the reference raised an untyped error; here the pinned class).
fn image_block(image: &ImagePart, provider: &str) -> Result<Value, Lm15Error> {
    let url = match (&image.url, &image.data) {
        (Some(url), _) => url.clone(),
        (None, Some(data)) => data_uri(&image.media_type, data),
        (None, None) => {
            return Err(unsupported(
                provider,
                "an image addressed by file_id or path cannot be sent on the Chat Completions \
                 wire; pass a URL or inline data",
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

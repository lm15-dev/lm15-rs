//! MAP-12 (module 4b): a Chat Completions request body read INTO a
//! canonical [`Request`] — the decoder for `messages.rs` / `payload.rs`
//! under the same [`ResolvedOpenAIChatCompat`] they write with
//! (`lm15/providers/openai_chat.py` "Ingest" block).
//!
//! Every wire key has exactly one verdict
//! (`lm15-contract/tools/openai-chat-ingest-verdicts.json`, copied here as
//! data): `map`, `extensions`, `refuse`, `call-mode` (read and dropped:
//! `stream`, `stream_options`) or `default` (equal to the wire default,
//! reads as absent). A key with no verdict is refused — never dropped.
//! Refusals are `UnsupportedFeatureError`; malformed input (a wrong JSON
//! type, a missing required key, an unparsable tool-call arguments string)
//! is `InvalidRequestError`, the class `ProviderLM::build_request` gives a
//! Request that fails its own invariants (MAP-12 rule 6: the contract does
//! not pin that class).

use serde_json::{Map, Value};

use super::text::unsupported;
use crate::compat::{
    OpenAICacheControl, OpenAIChatBuiltinTools, OpenAIChatCompat, OpenAIChatThinkingFormat,
    OpenAIChatUserField, ResolvedOpenAIChatCompat,
};
use crate::errors::{ErrorMeta, Lm15Error};
use crate::types::{
    AudioPart, BuiltinTool, CacheConfig, CachePrefix, CacheRetention, Config, DocumentPart,
    FunctionTool, ImageDetail, ImagePart, JsonObject, Message, Part, Reasoning, ReasoningEffort,
    ReasoningSummary, RefusalPart, Request, Role, SystemContent, TextPart, ThinkingPart, Tool,
    ToolCallPart, ToolChoice, ToolChoiceMode, ToolResultPart, ValidationError,
};

// ─── verdict tables (data; the registry is normative) ────────────────

/// Top-level keys forwarded verbatim into `config.extensions`.
const EXTENSIONS_KEYS: &[&str] = &[
    "seed",
    "logit_bias",
    "presence_penalty",
    "frequency_penalty",
    "metadata",
    "verbosity",
    "moderation",
    "provider",
];

/// Top-level keys refused, with the reason a Request cannot carry them.
const REFUSED_KEYS: &[(&str, &str)] = &[
    ("n", "lm15 reads one choice per response; n>1 would silently lose choices — fan out in the caller"),
    ("functions", "the deprecated function-calling shape; declare tools with {type: function, function: {...}}"),
    ("function_call", "the deprecated function-calling shape; use tool_choice"),
    ("audio", "audio output parameters have no canonical slot on the chat surface"),
    ("modalities", "output modality selection has no canonical slot on the chat surface"),
    ("prediction", "predicted-output content has no canonical slot"),
    ("web_search_options", "a server-executed search the chat dialect cannot map to parts (MAP-1); the Responses dialect carries web_search as a BuiltinTool"),
    ("top_k", "the Chat Completions wire has no top_k (the builder raises on Config.top_k for the same reason); servers that take it do so through extensions"),
];

/// Call-mode keys: how the request is sent, not what is asked.
const CALL_MODE_KEYS: &[&str] = &["stream", "stream_options"];

/// Keys the config decoder consumes itself.
const CONFIG_KEYS: &[&str] = &[
    "model",
    "messages",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "max_completion_tokens",
    "max_tokens",
    "temperature",
    "top_p",
    "stop",
    "logprobs",
    "top_logprobs",
    "response_format",
    "service_tier",
    "store",
    "user",
    "safety_identifier",
    "user_id",
    "reasoning_effort",
    "reasoning",
    "thinking",
    "enable_thinking",
    "chat_template_kwargs",
    "reasoning_format",
    "prompt_cache_key",
    "prompt_cache_retention",
    "prompt_cache_options",
];

/// Groq's server-executed tool types → canonical builtin names
/// (the inverse of `payload.rs` `GROQ_BUILTIN_MAP`).
const GROQ_BUILTIN_INVERSE: &[(&str, &str)] = &[
    ("browser_search", "web_search"),
    ("code_interpreter", "code_execution"),
];

const AUDIO_MEDIA_TYPES: &[(&str, &str)] = &[("wav", "audio/wav"), ("mp3", "audio/mpeg")];

// ─── errors and small readers ────────────────────────────────────────

type R<T> = Result<T, Lm15Error>;

fn malformed(message: impl Into<String>) -> Lm15Error {
    Lm15Error::InvalidRequestError(ErrorMeta::new(message))
}

fn refuse(provider: &str, what: &str, why: &str) -> Lm15Error {
    unsupported(
        provider,
        format!("{what} cannot be carried by a canonical Request — {why}"),
    )
}

fn invalid(err: ValidationError) -> Lm15Error {
    malformed(err.message)
}

fn as_object<'a>(value: Option<&'a Value>, where_: &str) -> R<&'a JsonObject> {
    match value {
        Some(Value::Object(obj)) => Ok(obj),
        Some(other) => Err(malformed(format!(
            "{where_} must be a JSON object, got {}",
            type_name(other)
        ))),
        None => Err(malformed(format!(
            "{where_} must be a JSON object, got null"
        ))),
    }
}

fn as_str<'a>(value: Option<&'a Value>, where_: &str) -> R<&'a str> {
    match value {
        Some(Value::String(s)) => Ok(s),
        Some(other) => Err(malformed(format!(
            "{where_} must be a string, got {}",
            type_name(other)
        ))),
        None => Err(malformed(format!("{where_} must be a string, got null"))),
    }
}

fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// `null` reads as absent everywhere a caller might write it.
fn present<'a>(obj: &'a JsonObject, key: &str) -> Option<&'a Value> {
    match obj.get(key) {
        None | Some(Value::Null) => None,
        Some(v) => Some(v),
    }
}

/// An unlisted key inside a block or object is a refusal, never a drop.
fn only_keys(provider: &str, obj: &JsonObject, allowed: &[&str], where_: &str) -> R<()> {
    let mut extra: Vec<&str> = obj
        .keys()
        .map(String::as_str)
        .filter(|k| !allowed.contains(k))
        .collect();
    extra.sort_unstable();
    match extra.first() {
        Some(key) => Err(refuse(
            provider,
            &format!("{where_} key {key:?}"),
            "no canonical slot for it",
        )),
        None => Ok(()),
    }
}

/// `data:<media_type>;base64,<payload>` → (media_type, payload).
fn data_uri(value: &str, where_: &str) -> R<(String, String)> {
    let rest = value
        .strip_prefix("data:")
        .ok_or_else(|| malformed(format!("{where_} must be a base64 data URI")))?;
    let (head, payload) = rest.split_once(',').ok_or_else(|| {
        malformed(format!(
            "{where_} must be a base64 data URI (data:<media-type>;base64,<payload>)"
        ))
    })?;
    let media_type = head.strip_suffix(";base64").ok_or_else(|| {
        malformed(format!(
            "{where_} must be a base64 data URI (data:<media-type>;base64,<payload>)"
        ))
    })?;
    if payload.is_empty() {
        return Err(malformed(format!(
            "{where_} must be a base64 data URI (data:<media-type>;base64,<payload>)"
        )));
    }
    if media_type.is_empty() {
        return Err(malformed(format!("{where_} data URI has no media type")));
    }
    Ok((media_type.to_string(), payload.to_string()))
}

// ─── content blocks (inverse of messages.rs content_parts / image_block) ─

fn image_block(provider: &str, block: &JsonObject, where_: &str) -> R<Part> {
    only_keys(
        provider,
        block,
        &["type", "image_url", "prompt_cache_breakpoint"],
        where_,
    )?;
    let spec = as_object(present(block, "image_url"), &format!("{where_}.image_url"))?;
    only_keys(
        provider,
        spec,
        &["url", "detail"],
        &format!("{where_}.image_url"),
    )?;
    let url = as_str(present(spec, "url"), &format!("{where_}.image_url.url"))?;
    let detail = match present(spec, "detail") {
        None => None,
        Some(v) => Some(
            ImageDetail::parse(as_str(Some(v), &format!("{where_}.image_url.detail"))?)
                .map_err(invalid)?,
        ),
    };
    let mut part = if url.starts_with("data:") {
        let (media_type, payload) = data_uri(url, &format!("{where_}.image_url.url"))?;
        ImagePart::from_data(media_type, payload).map_err(invalid)?
    } else {
        // The wire carries no media type for a URL: guessed from the path,
        // else the ImagePart default (the reference: mimetypes.guess_type).
        let mut part = ImagePart::from_url(url).map_err(invalid)?;
        if let Some(guess) = guess_image_media_type(url) {
            part.media_type = guess.to_string();
        }
        part
    };
    part.detail = detail;
    Ok(Part::Image(part))
}

/// `mimetypes.guess_type(url)[0]` restricted to `image/*`, over the common
/// extensions the Python table knows (the reference's behaviour on a URL).
fn guess_image_media_type(url: &str) -> Option<&'static str> {
    let path = url.split(['?', '#']).next().unwrap_or(url);
    let ext = path.rsplit('.').next()?.to_ascii_lowercase();
    if path
        .rsplit('/')
        .next()
        .is_some_and(|last| !last.contains('.'))
    {
        return None;
    }
    Some(match ext.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" | "jpe" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "svg" => "image/svg+xml",
        "tif" | "tiff" => "image/tiff",
        "ico" => "image/vnd.microsoft.icon",
        "heic" => "image/heic",
        "heif" => "image/heif",
        "avif" => "image/avif",
        _ => return None,
    })
}

fn text_block(provider: &str, block: &JsonObject, where_: &str) -> R<Part> {
    only_keys(
        provider,
        block,
        &["type", "text", "prompt_cache_breakpoint"],
        where_,
    )?;
    Ok(Part::Text(TextPart {
        text: as_str(present(block, "text"), &format!("{where_}.text"))?.to_string(),
        continuation: Vec::new(),
    }))
}

fn has_breakpoint(block: &JsonObject, where_: &str) -> R<bool> {
    let Some(mark) = present(block, "prompt_cache_breakpoint") else {
        return Ok(false);
    };
    let mark = as_object(Some(mark), &format!("{where_}.prompt_cache_breakpoint"))?;
    let explicit = mark.len() == 1 && mark.get("mode").and_then(Value::as_str) == Some("explicit");
    if !explicit {
        return Err(malformed(format!(
            "{where_}.prompt_cache_breakpoint must be {{\"mode\": \"explicit\"}}"
        )));
    }
    if block.get("type").and_then(Value::as_str) != Some("text") {
        return Err(malformed(format!(
            "{where_}: a prompt_cache_breakpoint rides on a text block, not {:?}",
            block.get("type")
        )));
    }
    Ok(true)
}

/// A row's `content` → parts, plus whether its LAST block carries the
/// prompt-cache breakpoint. A string is one TextPart.
fn content_blocks(
    provider: &str,
    content: &Value,
    role: &str,
    where_: &str,
) -> R<(Vec<Part>, bool)> {
    let blocks = match content {
        Value::String(text) => return Ok((vec![Part::text(text.clone())], false)),
        Value::Array(blocks) => blocks,
        _ => {
            return Err(malformed(format!(
                "{where_}.content must be a string or an array of content parts"
            )))
        }
    };
    let mut parts = Vec::with_capacity(blocks.len());
    let mut breakpoint_at_end = false;
    let last = blocks.len().saturating_sub(1);
    for (index, block) in blocks.iter().enumerate() {
        let block_where = format!("{where_}.content[{index}]");
        let block = as_object(Some(block), &block_where)?;
        let kind = block.get("type").and_then(Value::as_str).unwrap_or("");
        let marked = has_breakpoint(block, &block_where)?;
        if marked && index != last {
            return Err(malformed(format!(
                "{block_where}: a prompt_cache_breakpoint marks the end of a message; it must be on the last block"
            )));
        }
        breakpoint_at_end |= marked;
        match (kind, role) {
            ("text", _) => parts.push(text_block(provider, block, &block_where)?),
            ("image_url", "user" | "tool") => parts.push(image_block(provider, block, &block_where)?),
            ("input_audio", "user") => {
                only_keys(provider, block, &["type", "input_audio", "prompt_cache_breakpoint"], &block_where)?;
                let spec = as_object(present(block, "input_audio"), &format!("{block_where}.input_audio"))?;
                only_keys(provider, spec, &["data", "format"], &format!("{block_where}.input_audio"))?;
                let fmt = as_str(present(spec, "format"), &format!("{block_where}.input_audio.format"))?;
                let media_type = AUDIO_MEDIA_TYPES
                    .iter()
                    .find(|(f, _)| *f == fmt)
                    .map(|(_, m)| *m)
                    .ok_or_else(|| {
                        malformed(format!(
                            "{block_where}.input_audio.format must be one of [\"mp3\", \"wav\"]"
                        ))
                    })?;
                let data = as_str(present(spec, "data"), &format!("{block_where}.input_audio.data"))?;
                parts.push(Part::Audio(AudioPart::from_data(media_type, data).map_err(invalid)?));
            }
            ("file", "user") => {
                only_keys(provider, block, &["type", "file", "prompt_cache_breakpoint"], &block_where)?;
                let spec = as_object(present(block, "file"), &format!("{block_where}.file"))?;
                only_keys(provider, spec, &["file_data", "file_id", "filename"], &format!("{block_where}.file"))?;
                if present(spec, "filename").is_some() {
                    return Err(refuse(provider, &format!("{block_where}.file.filename"), "DocumentPart has no filename slot"));
                }
                match (present(spec, "file_id"), present(spec, "file_data")) {
                    (Some(id), None) => {
                        let id = as_str(Some(id), &format!("{block_where}.file.file_id"))?;
                        parts.push(Part::Document(DocumentPart::from_file_id(id).map_err(invalid)?));
                    }
                    (None, Some(data)) => {
                        let data = as_str(Some(data), &format!("{block_where}.file.file_data"))?;
                        let (media_type, payload) = data_uri(data, &format!("{block_where}.file.file_data"))?;
                        parts.push(Part::Document(DocumentPart::from_data(media_type, payload).map_err(invalid)?));
                    }
                    _ => return Err(malformed(format!("{block_where}.file needs exactly one of file_data / file_id"))),
                }
            }
            ("refusal", "assistant") => {
                only_keys(provider, block, &["type", "refusal"], &block_where)?;
                let text = as_str(present(block, "refusal"), &format!("{block_where}.refusal"))?;
                parts.push(Part::Refusal(RefusalPart::new(text).map_err(invalid)?));
            }
            _ => {
                return Err(refuse(
                    provider,
                    &format!("{block_where} of type {kind:?} in a {role} message"),
                    "no canonical part for that block on this wire (a part is not a knob: there is no extensions door for content)",
                ))
            }
        }
    }
    Ok((parts, breakpoint_at_end))
}

fn tool_calls(provider: &str, calls: &Value, where_: &str) -> R<Vec<Part>> {
    let Value::Array(calls) = calls else {
        return Err(malformed(format!("{where_}.tool_calls must be an array")));
    };
    let mut out = Vec::with_capacity(calls.len());
    for (index, call) in calls.iter().enumerate() {
        let call_where = format!("{where_}.tool_calls[{index}]");
        let call = as_object(Some(call), &call_where)?;
        let kind = call
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("function");
        if kind != "function" {
            return Err(refuse(
                provider,
                &format!("{call_where} of type {kind:?}"),
                "only function tool calls have a canonical part",
            ));
        }
        only_keys(provider, call, &["id", "type", "function"], &call_where)?;
        let function = as_object(present(call, "function"), &format!("{call_where}.function"))?;
        only_keys(
            provider,
            function,
            &["name", "arguments"],
            &format!("{call_where}.function"),
        )?;
        // The builder writes json.dumps(input); the inverse is exact. A
        // caller-authored string that is not a JSON object is malformed
        // (the lenient provider-output parse is not used on caller input).
        let input: JsonObject = match present(function, "arguments") {
            None => Map::new(),
            Some(Value::String(s)) if s.is_empty() => Map::new(),
            Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
                Ok(Value::Object(obj)) => obj,
                Ok(_) => {
                    return Err(malformed(format!(
                        "{call_where}.function.arguments must encode a JSON object"
                    )))
                }
                Err(err) => {
                    return Err(malformed(format!(
                        "{call_where}.function.arguments is not JSON: {err}"
                    )))
                }
            },
            Some(Value::Object(obj)) => obj.clone(),
            Some(_) => {
                return Err(malformed(format!(
                    "{call_where}.function.arguments must encode a JSON object"
                )))
            }
        };
        let id = as_str(present(call, "id"), &format!("{call_where}.id"))?;
        let name = as_str(
            present(function, "name"),
            &format!("{call_where}.function.name"),
        )?;
        out.push(Part::ToolCall(
            ToolCallPart::new(id, name, input).map_err(invalid)?,
        ));
    }
    Ok(out)
}

// ─── rows (inverse of messages.rs build_messages) ────────────────────

struct Rows {
    system: Option<SystemContent>,
    messages: Vec<Message>,
    system_breakpoint: bool,
    breakpoint_index: Option<u64>,
}

fn rows(provider: &str, rows: &Value) -> R<Rows> {
    let Value::Array(rows) = rows else {
        return Err(malformed("messages must be an array"));
    };
    let mut out = Rows {
        system: None,
        messages: Vec::new(),
        system_breakpoint: false,
        breakpoint_index: None,
    };
    let mut pending: Vec<ToolResultPart> = Vec::new();

    fn flush(pending: &mut Vec<ToolResultPart>, messages: &mut Vec<Message>) -> R<()> {
        if !pending.is_empty() {
            let parts = std::mem::take(pending);
            messages.push(Message::tool_parts(parts).map_err(invalid)?);
        }
        Ok(())
    }

    for (index, row) in rows.iter().enumerate() {
        let where_ = format!("messages[{index}]");
        let row = as_object(Some(row), &where_)?;
        let role = row.get("role").and_then(Value::as_str).unwrap_or("");
        if present(row, "name").is_some() && role != "tool" {
            return Err(refuse(
                provider,
                &format!("{where_}.name"),
                "a per-message participant name has no canonical slot",
            ));
        }
        match role {
            "system" | "developer" => {
                flush(&mut pending, &mut out.messages)?;
                only_keys(provider, row, &["role", "content"], &where_)?;
                let content = row.get("content").unwrap_or(&Value::Null);
                let (parts, marked) = content_blocks(provider, content, "system", &where_)?;
                if index == 0 {
                    out.system_breakpoint |= marked;
                    out.system = Some(match parts.as_slice() {
                        [Part::Text(t)] => SystemContent::Text(t.text.clone()),
                        _ => SystemContent::Parts(parts),
                    });
                } else {
                    if marked {
                        out.breakpoint_index = Some(out.messages.len() as u64);
                    }
                    out.messages.push(Message::new(Role::Developer, parts).map_err(invalid)?);
                }
            }
            "user" => {
                flush(&mut pending, &mut out.messages)?;
                only_keys(provider, row, &["role", "content", "name"], &where_)?;
                let content = row.get("content").unwrap_or(&Value::Null);
                let (parts, marked) = content_blocks(provider, content, "user", &where_)?;
                if marked {
                    if out.breakpoint_index.is_some() || out.system_breakpoint {
                        return Err(malformed(format!("{where_}: a request carries at most one prompt_cache_breakpoint")));
                    }
                    out.breakpoint_index = Some(out.messages.len() as u64);
                }
                out.messages.push(Message::new(Role::User, parts).map_err(invalid)?);
            }
            "assistant" => {
                flush(&mut pending, &mut out.messages)?;
                only_keys(
                    provider, row,
                    &["role", "content", "tool_calls", "refusal", "reasoning_content", "name", "audio", "function_call"],
                    &where_,
                )?;
                if present(row, "audio").is_some() {
                    return Err(refuse(provider, &format!("{where_}.audio"), "an assistant audio reference has no canonical part"));
                }
                if present(row, "function_call").is_some() {
                    return Err(refuse(provider, &format!("{where_}.function_call"), "the deprecated function-calling shape; use tool_calls"));
                }
                let mut parts: Vec<Part> = Vec::new();
                if let Some(reasoning) = present(row, "reasoning_content") {
                    let text = as_str(Some(reasoning), &format!("{where_}.reasoning_content"))?;
                    parts.push(Part::Thinking(ThinkingPart { text: text.to_string(), continuation: Vec::new() }));
                }
                if let Some(content) = present(row, "content") {
                    let (text_parts, marked) = content_blocks(provider, content, "assistant", &where_)?;
                    if marked {
                        return Err(malformed(format!("{where_}: a prompt_cache_breakpoint cannot mark an assistant message (the builder refuses the same cell)")));
                    }
                    parts.extend(text_parts);
                }
                if let Some(refusal) = present(row, "refusal") {
                    let text = as_str(Some(refusal), &format!("{where_}.refusal"))?;
                    parts.push(Part::Refusal(RefusalPart::new(text).map_err(invalid)?));
                }
                if let Some(calls) = present(row, "tool_calls") {
                    parts.extend(tool_calls(provider, calls, &where_)?);
                }
                if parts.is_empty() {
                    parts.push(Part::text("")); // MAP-2, applied to history
                }
                out.messages.push(Message::new(Role::Assistant, parts).map_err(invalid)?);
            }
            "tool" => {
                only_keys(provider, row, &["role", "content", "tool_call_id", "name"], &where_)?;
                let content = row.get("content").unwrap_or(&Value::Null);
                let (parts, marked) = content_blocks(provider, content, "tool", &where_)?;
                if marked {
                    return Err(malformed(format!("{where_}: a prompt_cache_breakpoint cannot mark a tool message (the builder refuses the same cell)")));
                }
                let id = as_str(present(row, "tool_call_id"), &format!("{where_}.tool_call_id"))?;
                let name = match present(row, "name") {
                    None => None,
                    Some(v) => Some(as_str(Some(v), &format!("{where_}.name"))?.to_string()),
                };
                let mut part = ToolResultPart::new(id, parts).map_err(invalid)?;
                part.name = name;
                part.validate().map_err(invalid)?;
                pending.push(part);
            }
            "function" => {
                return Err(refuse(provider, &format!("{where_} with role 'function'"), "the deprecated function-calling shape; use a tool row with tool_call_id"));
            }
            other => {
                return Err(malformed(format!(
                    "{where_}.role must be one of system, developer, user, assistant, tool; got {other:?}"
                )))
            }
        }
    }
    flush(&mut pending, &mut out.messages)?;
    Ok(out)
}

// ─── tools, tool_choice, response_format (inverse of payload.rs) ─────

fn tools(provider: &str, raw: Option<&Value>, compat: &ResolvedOpenAIChatCompat) -> R<Vec<Tool>> {
    let Some(raw) = raw else {
        return Ok(Vec::new());
    };
    let Value::Array(entries) = raw else {
        return Err(malformed("tools must be an array"));
    };
    let mut out = Vec::with_capacity(entries.len());
    for (index, entry) in entries.iter().enumerate() {
        let where_ = format!("tools[{index}]");
        let entry = as_object(Some(entry), &where_)?;
        let kind = entry.get("type").and_then(Value::as_str).unwrap_or("");
        if kind == "function" {
            only_keys(provider, entry, &["type", "function"], &where_)?;
            let function = as_object(present(entry, "function"), &format!("{where_}.function"))?;
            only_keys(
                provider,
                function,
                &["name", "description", "parameters", "strict"],
                &format!("{where_}.function"),
            )?;
            if function.get("strict") == Some(&Value::Bool(true)) {
                return Err(refuse(
                    provider,
                    &format!("{where_}.function.strict = true"),
                    "no per-tool strict slot (compat.strict_tools is a preset policy)",
                ));
            }
            let name = as_str(
                present(function, "name"),
                &format!("{where_}.function.name"),
            )?;
            let description = match present(function, "description") {
                None => None,
                Some(v) => {
                    Some(as_str(Some(v), &format!("{where_}.function.description"))?.to_string())
                }
            };
            let parameters = match present(function, "parameters") {
                None => FunctionTool::default_parameters(),
                Some(v) => as_object(Some(v), &format!("{where_}.function.parameters"))?.clone(),
            };
            out.push(Tool::Function(
                FunctionTool::new(name, description, parameters).map_err(invalid)?,
            ));
        } else if let (Some((_, name)), OpenAIChatBuiltinTools::Groq) = (
            GROQ_BUILTIN_INVERSE.iter().find(|(wire, _)| *wire == kind),
            compat.builtin_tools,
        ) {
            let config: JsonObject = entry
                .iter()
                .filter(|(k, _)| k.as_str() != "type")
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            let config = if config.is_empty() {
                None
            } else {
                Some(config)
            };
            out.push(Tool::Builtin(
                BuiltinTool::new(*name, config).map_err(invalid)?,
            ));
        } else {
            return Err(refuse(
                provider,
                &format!("{where_} of type {kind:?}"),
                "only function tools (and, on the groq preset, its server-executed tools) have a canonical form",
            ));
        }
    }
    Ok(out)
}

fn tool_choice(
    provider: &str,
    raw: Option<&Value>,
    parallel: Option<&Value>,
) -> R<Option<ToolChoice>> {
    let mut mode: Option<ToolChoiceMode> = None;
    let mut allowed: Vec<String> = Vec::new();
    match raw {
        None => {}
        Some(Value::String(s)) if matches!(s.as_str(), "none" | "auto" | "required") => {
            mode = Some(ToolChoiceMode::parse(s).map_err(invalid)?);
        }
        Some(Value::Object(obj)) => match obj.get("type").and_then(Value::as_str) {
            Some("function") => {
                only_keys(provider, obj, &["type", "function"], "tool_choice")?;
                let function = as_object(present(obj, "function"), "tool_choice.function")?;
                only_keys(provider, function, &["name"], "tool_choice.function")?;
                mode = Some(ToolChoiceMode::Required);
                allowed = vec![
                    as_str(present(function, "name"), "tool_choice.function.name")?.to_string(),
                ];
            }
            Some("allowed_tools") => {
                only_keys(provider, obj, &["type", "allowed_tools"], "tool_choice")?;
                let spec = as_object(present(obj, "allowed_tools"), "tool_choice.allowed_tools")?;
                only_keys(
                    provider,
                    spec,
                    &["mode", "tools"],
                    "tool_choice.allowed_tools",
                )?;
                mode = Some(
                    ToolChoiceMode::parse(as_str(
                        present(spec, "mode"),
                        "tool_choice.allowed_tools.mode",
                    )?)
                    .map_err(invalid)?,
                );
                let entries = match present(spec, "tools") {
                    Some(Value::Array(entries)) if !entries.is_empty() => entries,
                    _ => {
                        return Err(malformed(
                            "tool_choice.allowed_tools.tools must be a non-empty array",
                        ))
                    }
                };
                for (index, entry) in entries.iter().enumerate() {
                    let where_ = format!("tool_choice.allowed_tools.tools[{index}]");
                    let entry = as_object(Some(entry), &where_)?;
                    if entry.get("type").and_then(Value::as_str) != Some("function") {
                        return Err(refuse(
                            provider,
                            &format!("{where_} of type {:?}", entry.get("type")),
                            "only function tools can be allowed on this wire",
                        ));
                    }
                    let function =
                        as_object(present(entry, "function"), &format!("{where_}.function"))?;
                    allowed.push(
                        as_str(
                            present(function, "name"),
                            &format!("{where_}.function.name"),
                        )?
                        .to_string(),
                    );
                }
            }
            Some("custom") => {
                return Err(refuse(
                    provider,
                    "tool_choice of type 'custom'",
                    "custom tools have no canonical form",
                ))
            }
            other => {
                return Err(malformed(format!(
                    "tool_choice.type must be function or allowed_tools; got {other:?}"
                )))
            }
        },
        Some(_) => {
            return Err(malformed(
                "tool_choice must be none, auto, required, or an object",
            ))
        }
    }
    let parallel = match parallel {
        None => None,
        Some(Value::Bool(b)) => Some(*b),
        Some(_) => return Err(malformed("parallel_tool_calls must be a boolean")),
    };
    if mode.is_none() && parallel.is_none() {
        return Ok(None);
    }
    let choice = ToolChoice {
        mode: mode.unwrap_or(ToolChoiceMode::Auto),
        allowed,
        parallel,
    };
    choice.validate().map_err(invalid)?;
    Ok(Some(choice))
}

fn response_format(provider: &str, raw: &Value) -> R<Option<JsonObject>> {
    let raw = as_object(Some(raw), "response_format")?;
    match raw.get("type").and_then(Value::as_str) {
        Some("text") => {
            only_keys(provider, raw, &["type"], "response_format")?;
            Ok(None)
        }
        Some("json_object") => {
            only_keys(provider, raw, &["type"], "response_format")?;
            let mut out = Map::new();
            out.insert("type".into(), Value::String("json_object".into()));
            Ok(Some(out))
        }
        Some("json_schema") => {
            only_keys(provider, raw, &["type", "json_schema"], "response_format")?;
            let inner = as_object(present(raw, "json_schema"), "response_format.json_schema")?;
            only_keys(
                provider,
                inner,
                &["name", "schema", "strict", "description"],
                "response_format.json_schema",
            )?;
            if present(inner, "description").is_some() {
                return Err(refuse(
                    provider,
                    "response_format.json_schema.description",
                    "the canonical response_format has no description slot (INV-050)",
                ));
            }
            let schema = as_object(
                present(inner, "schema"),
                "response_format.json_schema.schema",
            )?;
            let mut out = Map::new();
            out.insert("type".into(), Value::String("json_schema".into()));
            out.insert("schema".into(), Value::Object(schema.clone()));
            if let Some(name) = present(inner, "name") {
                let name = as_str(Some(name), "response_format.json_schema.name")?;
                if name != "response" {
                    out.insert("name".into(), Value::String(name.to_string()));
                }
            }
            match present(inner, "strict") {
                None => {}
                Some(Value::Bool(b)) => {
                    out.insert("strict".into(), Value::Bool(*b));
                }
                Some(_) => {
                    return Err(malformed(
                        "response_format.json_schema.strict must be a boolean",
                    ))
                }
            }
            Ok(Some(out))
        }
        other => Err(malformed(format!(
            "response_format.type must be text, json_object or json_schema; got {other:?}"
        ))),
    }
}

// ─── reasoning and cache (inverse of payload.rs / cache.rs) ──────────

fn reasoning(
    provider: &str,
    body: &JsonObject,
    compat: &ResolvedOpenAIChatCompat,
    extensions: &mut JsonObject,
) -> R<Option<Reasoning>> {
    const SPELLINGS: &[&str] = &[
        "reasoning_effort",
        "reasoning",
        "thinking",
        "enable_thinking",
        "chat_template_kwargs",
        "reasoning_format",
    ];
    let present_keys: Vec<&str> = SPELLINGS
        .iter()
        .copied()
        .filter(|k| body.contains_key(*k))
        .collect();
    if present_keys.is_empty() {
        return Ok(None);
    }
    let mut spelled_by: Vec<&str> = match compat.thinking_format {
        OpenAIChatThinkingFormat::ReasoningEffort => vec!["reasoning_effort"],
        OpenAIChatThinkingFormat::Openrouter => vec!["reasoning"],
        OpenAIChatThinkingFormat::Deepseek | OpenAIChatThinkingFormat::Kimi => {
            vec!["thinking", "reasoning_effort"]
        }
        OpenAIChatThinkingFormat::Qwen => vec!["enable_thinking"],
        OpenAIChatThinkingFormat::QwenChatTemplate => vec!["chat_template_kwargs"],
        OpenAIChatThinkingFormat::None => vec![],
    };
    if compat.builtin_tools == OpenAIChatBuiltinTools::Groq {
        spelled_by.push("reasoning_format");
    }
    if let Some(foreign) = present_keys.iter().find(|k| !spelled_by.contains(k)) {
        let mut sorted = spelled_by.clone();
        sorted.sort_unstable();
        let where_ = if sorted.is_empty() {
            "nowhere (no dial)".to_string()
        } else {
            format!("{sorted:?}")
        };
        return Err(refuse(
            provider,
            &format!("{foreign:?}"),
            &format!("this server's reasoning dial is spelled {where_}; another server's spelling would be sent and ignored"),
        ));
    }

    let mut effort: Option<ReasoningEffort> = None;
    let mut off = false;
    if let Some(word) = body.get("reasoning_effort") {
        let word = as_str(Some(word), "reasoning_effort")?;
        if word == "none" {
            off = true;
        } else {
            effort = Some(ReasoningEffort::parse(word).map_err(invalid)?);
        }
    }
    if let Some(spec) = body.get("thinking") {
        let spec = as_object(Some(spec), "thinking")?;
        only_keys(provider, spec, &["type"], "thinking")?;
        match spec.get("type").and_then(Value::as_str) {
            Some("disabled") => {
                if effort.is_some() {
                    return Err(malformed(
                        "thinking.type=disabled next to a reasoning_effort level is contradictory",
                    ));
                }
                off = true;
            }
            Some("enabled") => {
                if effort.is_none() && !off {
                    return Err(refuse(
                        provider,
                        "thinking.type=enabled without reasoning_effort",
                        "lm15's dial is a level (MAP-7); set config.reasoning with an effort word",
                    ));
                }
            }
            other => {
                return Err(malformed(format!(
                    "thinking.type must be enabled or disabled; got {other:?}"
                )))
            }
        }
    }
    if let Some(spec) = body.get("reasoning") {
        let spec = as_object(Some(spec), "reasoning")?;
        only_keys(provider, spec, &["effort", "enabled"], "reasoning")?;
        if spec.get("enabled") == Some(&Value::Bool(false)) {
            off = true;
        } else if let Some(word) = present(spec, "effort") {
            effort = Some(
                ReasoningEffort::parse(as_str(Some(word), "reasoning.effort")?).map_err(invalid)?,
            );
        } else {
            return Err(malformed("reasoning must carry effort or enabled: false"));
        }
    }
    if let Some(flag) = body.get("enable_thinking") {
        match flag {
            Value::Bool(false) => off = true,
            Value::Bool(true) => return Err(refuse(provider, "enable_thinking = true", "this wire has no effort level; lm15's dial is a level (MAP-7) — set config.reasoning yourself")),
            _ => return Err(malformed("enable_thinking must be a boolean")),
        }
    }
    if let Some(spec) = body.get("chat_template_kwargs") {
        let spec = as_object(Some(spec), "chat_template_kwargs")?;
        only_keys(
            provider,
            spec,
            &["enable_thinking", "preserve_thinking"],
            "chat_template_kwargs",
        )?;
        match spec.get("enable_thinking") {
            Some(Value::Bool(false)) => off = true,
            Some(Value::Bool(true)) => return Err(refuse(provider, "chat_template_kwargs.enable_thinking = true", "this wire has no effort level; lm15's dial is a level (MAP-7) — set config.reasoning yourself")),
            _ => return Err(malformed("chat_template_kwargs.enable_thinking must be a boolean")),
        }
    }
    let mut summary: Option<ReasoningSummary> = None;
    if let Some(value) = body.get("reasoning_format") {
        if value.as_str() != Some("parsed") {
            return Err(refuse(
                provider,
                &format!("reasoning_format = {value}"),
                "only 'parsed' maps (Reasoning.summary='auto', MAP-7 rule 7)",
            ));
        }
        if effort.is_none() {
            // The documented door: extensions={"reasoning_format": "parsed"}
            // with reasoning absent.
            extensions.insert("reasoning_format".into(), value.clone());
        } else {
            summary = Some(ReasoningSummary::Auto);
        }
    }
    if off {
        return Ok(Some(Reasoning::new(ReasoningEffort::Off)));
    }
    Ok(effort.map(|effort| Reasoning {
        effort,
        thinking_budget: None,
        summary,
    }))
}

fn cache(
    provider: &str,
    body: &JsonObject,
    compat: &ResolvedOpenAIChatCompat,
    system_breakpoint: bool,
    breakpoint_index: Option<u64>,
) -> R<Option<CacheConfig>> {
    const KEYS: &[&str] = &[
        "prompt_cache_key",
        "prompt_cache_retention",
        "prompt_cache_options",
    ];
    let mut keys: Vec<&str> = KEYS
        .iter()
        .copied()
        .filter(|k| body.contains_key(*k))
        .collect();
    keys.sort_unstable();
    let marked = system_breakpoint || breakpoint_index.is_some();
    if keys.is_empty() && !marked {
        return Ok(None);
    }
    if !matches!(
        compat.cache_control,
        OpenAICacheControl::OpenAI | OpenAICacheControl::OpenAIImplicit
    ) {
        let what = keys.first().copied().unwrap_or("prompt_cache_breakpoint");
        return Err(refuse(
            provider,
            &format!("{what:?}"),
            "this server has no OpenAI prompt-cache control (compat.cache_control)",
        ));
    }
    if marked && compat.cache_control != OpenAICacheControl::OpenAI {
        return Err(refuse(provider, "prompt_cache_breakpoint", "this server swallows an explicit breakpoint silently (compat.cache_control=openai_implicit)"));
    }
    let key = match present(body, "prompt_cache_key") {
        None => None,
        Some(v) => Some(as_str(Some(v), "prompt_cache_key")?.to_string()),
    };
    let mut retention = None;
    if let Some(value) = body.get("prompt_cache_retention") {
        if value.as_str() != Some("24h") {
            return Err(refuse(
                provider,
                &format!("prompt_cache_retention = {value}"),
                "only '24h' has a canonical value (CacheConfig.retention='long')",
            ));
        }
        retention = Some(CacheRetention::Long);
    }
    let mut explicit = false;
    if let Some(spec) = body.get("prompt_cache_options") {
        let spec = as_object(Some(spec), "prompt_cache_options")?;
        only_keys(provider, spec, &["mode", "ttl"], "prompt_cache_options")?;
        if present(spec, "ttl").is_some() {
            return Err(refuse(
                provider,
                "prompt_cache_options.ttl",
                "CacheConfig.retention names 24h only",
            ));
        }
        match spec.get("mode").and_then(Value::as_str) {
            Some("explicit") => explicit = true,
            Some("implicit") => {
                return Err(refuse(
                    provider,
                    "prompt_cache_options.mode = 'implicit'",
                    "the server default; a canonical CacheConfig names auto or off",
                ))
            }
            other => {
                return Err(malformed(format!(
                    "prompt_cache_options.mode must be explicit or implicit; got {other:?}"
                )))
            }
        }
    }
    let mut config = CacheConfig::default();
    if explicit && !marked {
        // Explicit mode with no mark is the cache-WRITE off switch (MAP-6 rule 2).
        if key.is_some() || retention.is_some() {
            return Err(malformed("prompt_cache_options.mode=explicit with no breakpoint is the off switch; it cannot carry a key or retention (INV-027)"));
        }
        config.mode = crate::types::CacheMode::Off;
        return Ok(Some(config));
    }
    config.key = key;
    config.retention = retention;
    if system_breakpoint {
        config.prefix = Some(CachePrefix::Stable);
    } else if let Some(index) = breakpoint_index {
        config.prefix_until_index = Some(index);
    }
    config.validate().map_err(invalid)?;
    Ok(Some(config))
}

fn config(
    provider: &str,
    body: &JsonObject,
    compat: &ResolvedOpenAIChatCompat,
    r: &Rows,
) -> R<Config> {
    let mut config = Config::default();
    let limits: Vec<&Value> = ["max_completion_tokens", "max_tokens"]
        .iter()
        .filter_map(|k| present(body, k))
        .collect();
    if let Some(first) = limits.first() {
        if limits.iter().any(|v| v != first) {
            return Err(malformed(format!(
                "max_tokens and max_completion_tokens disagree: {limits:?}"
            )));
        }
        config.max_tokens = Some(
            u64::try_from(crate::serde::int_from_value(first, "max_tokens").map_err(invalid)?)
                .map_err(|_| malformed("max_tokens must be >= 0"))?,
        );
    }
    if let Some(v) = present(body, "temperature") {
        config.temperature =
            Some(crate::serde::float_from_value(v, "temperature").map_err(invalid)?);
    }
    if let Some(v) = present(body, "top_p") {
        config.top_p = Some(crate::serde::float_from_value(v, "top_p").map_err(invalid)?);
    }
    if let Some(v) = present(body, "service_tier") {
        config.service_tier = Some(as_str(Some(v), "service_tier")?.to_string());
    }
    if let Some(v) = present(body, "store") {
        config.store = Some(
            v.as_bool()
                .ok_or_else(|| malformed("store must be a boolean"))?,
        );
    }
    match present(body, "stop") {
        None => {}
        Some(Value::String(s)) => config.stop = vec![s.clone()],
        Some(Value::Array(items)) => {
            config.stop = items
                .iter()
                .map(|v| {
                    v.as_str()
                        .map(str::to_string)
                        .ok_or_else(|| malformed("stop must contain strings"))
                })
                .collect::<R<_>>()?;
        }
        Some(_) => return Err(malformed("stop must be a string or an array of strings")),
    }
    match present(body, "logprobs") {
        Some(Value::Bool(true)) => {
            config.logprobs = Some(match present(body, "top_logprobs") {
                None => 0,
                Some(v) => {
                    u64::try_from(crate::serde::int_from_value(v, "top_logprobs").map_err(invalid)?)
                        .map_err(|_| malformed("top_logprobs must be >= 0"))?
                }
            });
        }
        Some(Value::Bool(false)) | None => {
            if present(body, "top_logprobs").is_some() {
                return Err(malformed("top_logprobs requires logprobs: true"));
            }
        }
        Some(_) => return Err(malformed("logprobs must be a boolean")),
    }
    if let Some(v) = present(body, "response_format") {
        config.response_format = response_format(provider, v)?;
    }
    config.tool_choice = tool_choice(
        provider,
        present(body, "tool_choice"),
        present(body, "parallel_tool_calls"),
    )?;

    let user_keys: Vec<&str> = ["user", "safety_identifier", "user_id"]
        .iter()
        .copied()
        .filter(|k| body.contains_key(*k))
        .collect();
    if user_keys.contains(&"user_id") && compat.user_field != OpenAIChatUserField::UserId {
        return Err(refuse(
            provider,
            "'user_id'",
            &format!(
                "this server spells the end-user field {:?}",
                compat.user_field.as_str()
            ),
        ));
    }
    if user_keys.len() > 1 {
        return Err(malformed(format!(
            "one end-user identifier only; got {user_keys:?}"
        )));
    }
    if let Some(key) = user_keys.first() {
        config.user_id = Some(as_str(present(body, key), key)?.to_string());
    }

    let mut extensions = JsonObject::new();
    config.reasoning = reasoning(provider, body, compat, &mut extensions)?;
    config.cache = cache(
        provider,
        body,
        compat,
        r.system_breakpoint,
        r.breakpoint_index,
    )?;
    for (key, value) in body {
        if EXTENSIONS_KEYS.contains(&key.as_str()) {
            extensions.insert(key.clone(), value.clone());
        }
    }
    config.extensions = if extensions.is_empty() {
        None
    } else {
        Some(extensions)
    };
    config.validate().map_err(invalid)?;
    Ok(config)
}

// ─── entry points ────────────────────────────────────────────────────

/// The decoder under an already-resolved compat and provider name (what
/// `ProviderLM::request_from_openai_chat` calls).
pub(crate) fn ingest(
    provider: &str,
    body: &Value,
    compat: &ResolvedOpenAIChatCompat,
) -> R<Request> {
    let Value::Object(body) = body else {
        return Err(malformed(format!(
            "a Chat Completions request body is a JSON object, got {}",
            type_name(body)
        )));
    };
    for key in body.keys() {
        let key = key.as_str();
        if let Some((_, why)) = REFUSED_KEYS.iter().find(|(k, _)| *k == key) {
            return Err(refuse(provider, &format!("{key:?}"), why));
        }
        if !CONFIG_KEYS.contains(&key)
            && !EXTENSIONS_KEYS.contains(&key)
            && !CALL_MODE_KEYS.contains(&key)
        {
            return Err(refuse(provider, &format!("{key:?}"), "no verdict for this key (lm15-contract/tools/openai-chat-ingest-verdicts.json); lm15 never drops a key silently"));
        }
    }
    let model = match body.get("model") {
        Some(Value::String(s)) if !s.is_empty() => s.clone(),
        _ => return Err(malformed("model must be a non-empty string")),
    };
    let Some(raw_rows) = body.get("messages") else {
        return Err(malformed("messages is required"));
    };
    let r = rows(provider, raw_rows)?;
    let tools = tools(provider, present(body, "tools"), compat)?;
    let config = config(provider, body, compat, &r)?;
    let request = Request {
        model,
        messages: r.messages,
        system: r.system,
        tools,
        config,
    };
    request.validate().map_err(invalid)?;
    Ok(request)
}

/// A Chat Completions request body → the canonical [`Request`] (MAP-12).
///
/// `body` is the JSON object a client would POST to `/chat/completions`.
/// `compat` names the server dialect whose spellings are read — a preset
/// (`OpenAIChatCompat::preset("groq")`) or `None` for OpenAI's own — the
/// same policy the chat dialect writes with, so what it emits for a
/// Request reads back as that Request wherever the wire can carry it.
/// Per-model overrides in the preset apply to `body.model`.
///
/// Every key has one verdict: it maps to a canonical field, passes
/// verbatim through `config.extensions`, or is refused with
/// `UnsupportedFeatureError` naming the key; `stream` / `stream_options`
/// are read and dropped. Malformed input is `InvalidRequestError`.
pub fn request_from_openai_chat(
    body: &Value,
    compat: Option<&OpenAIChatCompat>,
) -> Result<Request, Lm15Error> {
    let partial = compat.unwrap_or(&OpenAIChatCompat::EMPTY);
    let resolved = match body.get("model").and_then(Value::as_str) {
        Some(model) if !partial.model_overrides.is_empty() => partial.for_model(model).resolve(),
        _ => partial.resolve(),
    };
    ingest("openai-chat", body, &resolved)
}

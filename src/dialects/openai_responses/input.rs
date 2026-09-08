//! Canonical messages → Responses `input` items
//! (`lm15/providers/openai.py:682-776` `_build_input`;
//! `lm15/providers/common.py:53-76,162-220` for the part shapes).

use serde_json::{json, Map, Value};

use crate::compat::{IncludeOmit, OpenAIResponsesCommentaryPhase, ResolvedOpenAIResponsesCompat};
use crate::errors::Lm15Error;
use crate::types::{
    continuation_data, AudioPart, CitationPart, ImagePart, Message, Part, Role, ToolResultPart,
    VideoPart,
};

use super::{invalid_request, unsupported};
use crate::dialects::content;

/// The `input` array for `messages`. `breakpoint_index` is the message
/// that carries `prompt_cache_breakpoint` (MAP-6 `prefix_until_index`).
pub fn build_input(
    provider: &str,
    messages: &[Message],
    compat: &ResolvedOpenAIResponsesCompat,
    breakpoint_index: Option<usize>,
) -> Result<Vec<Value>, Lm15Error> {
    let mut items = Vec::new();
    for (index, message) in messages.iter().enumerate() {
        let marked = breakpoint_index == Some(index);
        if marked && matches!(message.role, Role::Assistant | Role::Tool) {
            return Err(breakpoint_unsupported(provider, index, message.role));
        }
        match message.role {
            Role::Tool => {
                for part in &message.parts {
                    if let Part::ToolResult(result) = part {
                        items.push(tool_result_item(provider, result, compat)?);
                    }
                }
            }
            Role::Assistant => {
                let mut content = Vec::new();
                for part in &message.parts {
                    match part {
                        Part::Text(text) => content.push(output_text(&text.text)),
                        Part::Refusal(refusal) => {
                            content.push(json!({"type": "refusal", "refusal": refusal.text}))
                        }
                        Part::Thinking(thinking) => {
                            // MAP-7.8: native replay when the state is
                            // present, as its own item before the message
                            // it preceded; assistant text otherwise
                            // (decision G, changes/2026-09-01-*).
                            match continuation_data(
                                &thinking.continuation,
                                "openai",
                                "reasoning_item",
                            ) {
                                Some(state) => items.push(reasoning_item(state, &thinking.text)),
                                None if !thinking.text.is_empty() => {
                                    content.push(output_text(&thinking.text))
                                }
                                None => {}
                            }
                        }
                        // Annotations of the text already replayed; the
                        // wire's `output_text.annotations` is output-side.
                        Part::Citation(_) => {}
                        // Emitted as their own items after the message.
                        Part::ToolCall(_) => {}
                        // INV-023: unreachable.
                        Part::ToolResult(_) => {}
                        Part::Image(_)
                        | Part::Audio(_)
                        | Part::Video(_)
                        | Part::Document(_)
                        | Part::Binary(_) => {
                            return Err(unsupported(
                                provider,
                                format!(
                                    "assistant {} parts cannot be replayed — the Responses wire \
                                     takes output_text and refusal in an assistant message; \
                                     send generated media back as a user part",
                                    part.type_name()
                                ),
                            ))
                        }
                    }
                }
                push_message(
                    &mut items, message, content, compat, marked, provider, index,
                )?;
            }
            Role::User | Role::Developer => {
                let mut content = Vec::with_capacity(message.parts.len());
                for part in &message.parts {
                    if !matches!(part, Part::ToolCall(_) | Part::ToolResult(_)) {
                        content.push(part_to_input(provider, part)?);
                    }
                }
                push_message(
                    &mut items, message, content, compat, marked, provider, index,
                )?;
            }
        }
        for part in &message.parts {
            if let Part::ToolCall(call) = part {
                items.push(json!({
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": compact_json(&call.input),
                }));
            }
        }
    }
    Ok(items)
}

/// The message item for `content`, with the breakpoint mark and the
/// commentary tag where they apply; nothing when `content` is empty.
fn push_message(
    items: &mut Vec<Value>,
    message: &Message,
    mut content: Vec<Value>,
    compat: &ResolvedOpenAIResponsesCompat,
    marked: bool,
    provider: &str,
    index: usize,
) -> Result<(), Lm15Error> {
    if marked {
        // The wire carries `prompt_cache_breakpoint` on input_text blocks
        // only (gpt-5.6+, live 2026-09-01; pre-5.6 answers HTTP 400 — the
        // loud failure is the contract).
        let last_is_text = content
            .last()
            .and_then(|block| block.get("type"))
            .and_then(Value::as_str)
            == Some("input_text");
        if !last_is_text {
            return Err(breakpoint_unsupported(provider, index, message.role));
        }
        if let Some(Value::Object(block)) = content.last_mut() {
            block.insert(
                "prompt_cache_breakpoint".into(),
                json!({"mode": "explicit"}),
            );
        }
    }
    if content.is_empty() {
        return Ok(());
    }
    let role = match message.role {
        Role::Developer => compat.developer_role.as_str(),
        other => other.as_str(),
    };
    let mut item = Map::new();
    item.insert("role".into(), Value::String(role.into()));
    item.insert("content".into(), Value::Array(content));
    if compat.commentary_phase == OpenAIResponsesCommentaryPhase::Tag
        && message.role == Role::Assistant
        && message
            .parts
            .iter()
            .any(|part| matches!(part, Part::ToolCall(_)))
    {
        // Assistant text before a function_call in the same turn is
        // "commentary" on this server (Meta, protocols--responses.md §
        // Message phase; changes/2026-09-03-meta-live.md §2).
        item.insert("phase".into(), Value::String("commentary".into()));
    }
    items.push(Value::Object(item));
    Ok(())
}

/// `{"type": "reasoning", id?, encrypted_content?, "summary": [...]}`.
/// `summary` is required on a replayed item, even empty (HTTP 400
/// "Missing required parameter: input[1].summary", live 2026-09-02).
fn reasoning_item(state: &Map<String, Value>, text: &str) -> Value {
    let mut item = Map::new();
    item.insert("type".into(), Value::String("reasoning".into()));
    for key in ["id", "encrypted_content"] {
        if let Some(value) = state.get(key) {
            item.insert(key.into(), value.clone());
        }
    }
    let summary = if text.is_empty() {
        Vec::new()
    } else {
        vec![json!({"type": "summary_text", "text": text})]
    };
    item.insert("summary".into(), Value::Array(summary));
    Value::Object(item)
}

/// `function_call_output` (MAP-10; `lm15/providers/common.py`
/// `tool_result_output_openai`): a string when the content is text-only,
/// the documented array of input_text/input_image/input_file blocks
/// otherwise; a media part the preset does not admit raises first.
/// `is_error` rides as an `[error] ` prefix on the text (rule 5).
fn tool_result_item(
    provider: &str,
    result: &ToolResultPart,
    compat: &ResolvedOpenAIResponsesCompat,
) -> Result<Value, Lm15Error> {
    content::check_tool_result_media(provider, result, compat.tool_result_media, "function_call_output")?;
    let output = if content::text_only(&result.content) {
        Value::String(content::error_text(
            result,
            content::parts_to_text(&result.content, provider, "function_call_output")?,
        ))
    } else {
        let mut blocks: Vec<Value> = Vec::with_capacity(result.content.len());
        for part in &result.content {
            blocks.push(part_to_input(provider, part)?);
        }
        if result.is_error {
            match blocks.iter_mut().find(|b| b.get("type") == Some(&json!("input_text"))) {
                Some(Value::Object(block)) => {
                    let text = block.get("text").and_then(Value::as_str).unwrap_or("").to_string();
                    block.insert("text".into(), Value::String(format!("[error] {text}")));
                }
                _ => blocks.insert(0, json!({"type": "input_text", "text": "[error]"})),
            }
        }
        Value::Array(blocks)
    };
    let mut item = Map::new();
    item.insert("type".into(), Value::String("function_call_output".into()));
    item.insert("call_id".into(), Value::String(result.id.clone()));
    item.insert("output".into(), output);
    if compat.tool_result_name == IncludeOmit::Include {
        if let Some(name) = &result.name {
            item.insert("name".into(), Value::String(name.clone()));
        }
    }
    Ok(Value::Object(item))
}

fn output_text(text: &str) -> Value {
    json!({"type": "output_text", "text": text})
}

/// `json.dumps(value, separators=(",", ":"))`: compact; non-ASCII is sent
/// raw where Python would `\u`-escape it (the same JSON value).
fn compact_json(value: &Map<String, Value>) -> String {
    serde_json::to_string(value).expect("a JSON object serializes")
}

/// Text for a text-only field (MAP-10: a media part raises; see
/// `dialects::content::parts_to_text`).
pub fn parts_to_text(parts: &[Part], provider: &str, where_: &str) -> Result<String, Lm15Error> {
    content::parts_to_text(parts, provider, where_)
}

fn citation_text(citation: &CitationPart) -> Option<String> {
    let bits: Vec<&str> = [&citation.title, &citation.url, &citation.text]
        .into_iter()
        .filter_map(|field| field.as_deref())
        .filter(|s| !s.is_empty())
        .collect();
    if bits.is_empty() {
        None
    } else {
        Some(bits.join(" — "))
    }
}

/// `lm15/providers/common.py:162-220` `part_to_openai_input`, for the
/// prompt parts a user/developer message may carry (INV-024). A
/// path-addressed part is read and inlined (the Anthropic dialect's
/// precedent, `common.py:234`); the reference's Responses path sends an
/// empty `input_text` instead — a silent drop, refused here.
pub(crate) fn part_to_input(provider: &str, part: &Part) -> Result<Value, Lm15Error> {
    Ok(match part {
        Part::Text(text) => json!({"type": "input_text", "text": text.text}),
        Part::Image(image) => image_input(provider, image)?,
        Part::Audio(audio) => audio_input(provider, audio)?,
        Part::Document(document) => file_input(
            provider,
            &document.media_type,
            document.data.as_deref(),
            document.url.as_deref(),
            document.file_id.as_deref(),
            document.path.as_deref(),
        )?,
        Part::Binary(binary) => file_input(
            provider,
            &binary.media_type,
            binary.data.as_deref(),
            binary.url.as_deref(),
            binary.file_id.as_deref(),
            binary.path.as_deref(),
        )?,
        Part::Video(video) => video_input(provider, video)?,
        Part::Citation(citation) => {
            json!({"type": "input_text", "text": citation_text(citation).unwrap_or_default()})
        }
        Part::Thinking(thinking) => json!({"type": "input_text", "text": thinking.text}),
        // INV-013/INV-024 keep these out of prompt and tool-result content.
        Part::Refusal(_) | Part::ToolResult(_) | Part::ToolCall(_) => {
            return Err(unsupported(
                provider,
                format!("a {} part has no input block on the Responses wire (MAP-10)", part.type_name()),
            ))
        }
    })
}

fn image_input(provider: &str, image: &ImagePart) -> Result<Value, Lm15Error> {
    let mut payload = Map::new();
    payload.insert("type".into(), Value::String("input_image".into()));
    if let Some(file_id) = &image.file_id {
        payload.insert("file_id".into(), Value::String(file_id.clone()));
        return Ok(Value::Object(payload));
    }
    let url = match (&image.url, &image.data, &image.path) {
        (Some(url), _, _) => url.clone(),
        (None, Some(data), _) => data_uri(&image.media_type, data),
        (None, None, Some(path)) => data_uri(&image.media_type, &read_base64(provider, path)?),
        (None, None, None) => unreachable!("INV-011: one media source"),
    };
    payload.insert("image_url".into(), Value::String(url));
    if let Some(detail) = image.detail {
        payload.insert("detail".into(), Value::String(detail.as_str().into()));
    }
    Ok(Value::Object(payload))
}

fn audio_input(provider: &str, audio: &AudioPart) -> Result<Value, Lm15Error> {
    let inline = |data: &str| {
        // `common.py:181-185`: the subtype is the format; mpeg is mp3.
        let mut media = audio
            .media_type
            .split_once('/')
            .map(|(_, sub)| sub)
            .unwrap_or(&audio.media_type)
            .to_string();
        if media == "mpeg" {
            media = "mp3".into();
        }
        json!({"type": "input_audio", "audio": data, "format": media})
    };
    Ok(
        match (&audio.data, &audio.url, &audio.file_id, &audio.path) {
            (Some(data), _, _, _) => inline(data),
            (None, Some(url), _, _) => json!({"type": "input_audio", "audio_url": url}),
            (None, None, Some(file_id), _) => json!({"type": "input_audio", "file_id": file_id}),
            (None, None, None, Some(path)) => inline(&read_base64(provider, path)?),
            (None, None, None, None) => unreachable!("INV-011: one media source"),
        },
    )
}

fn file_input(
    provider: &str,
    media_type: &str,
    data: Option<&str>,
    url: Option<&str>,
    file_id: Option<&str>,
    path: Option<&std::path::Path>,
) -> Result<Value, Lm15Error> {
    let inline = |data: &str| {
        // A filename is required next to inline file_data (live 2026-06-11:
        // 400 missing_required_parameter); derived from the subtype.
        let ext = media_type
            .split_once('/')
            .map(|(_, sub)| sub)
            .unwrap_or(media_type)
            .split('+')
            .next()
            .filter(|s| !s.is_empty())
            .unwrap_or("bin");
        json!({
            "type": "input_file",
            "filename": format!("file.{ext}"),
            "file_data": data_uri(media_type, data),
        })
    };
    Ok(match (url, data, file_id, path) {
        (Some(url), _, _, _) => json!({"type": "input_file", "file_url": url}),
        (None, Some(data), _, _) => inline(data),
        (None, None, Some(file_id), _) => json!({"type": "input_file", "file_id": file_id}),
        (None, None, None, Some(path)) => inline(&read_base64(provider, path)?),
        (None, None, None, None) => unreachable!("INV-011: one media source"),
    })
}

fn video_input(provider: &str, video: &VideoPart) -> Result<Value, Lm15Error> {
    Ok(
        match (&video.url, &video.data, &video.file_id, &video.path) {
            (Some(url), _, _, _) => json!({"type": "input_video", "video_url": url}),
            (None, Some(data), _, _) => {
                json!({"type": "input_video", "video_data": data_uri(&video.media_type, data)})
            }
            (None, None, Some(file_id), _) => json!({"type": "input_video", "file_id": file_id}),
            (None, None, None, Some(path)) => json!({
                "type": "input_video",
                "video_data": data_uri(&video.media_type, &read_base64(provider, path)?),
            }),
            (None, None, None, None) => unreachable!("INV-011: one media source"),
        },
    )
}

/// `common.py:73-76` `media_data_uri`.
fn data_uri(media_type: &str, data: &str) -> String {
    format!("data:{media_type};base64,{data}")
}

fn read_base64(provider: &str, path: &std::path::Path) -> Result<String, Lm15Error> {
    let bytes = std::fs::read(path).map_err(|err| {
        invalid_request(
            provider,
            format!("cannot read media part path {}: {err}", path.display()),
        )
    })?;
    Ok(crate::types::base64_encode(&bytes))
}

fn breakpoint_unsupported(provider: &str, index: usize, role: Role) -> Lm15Error {
    unsupported(
        provider,
        format!(
            "cache.prefix_until_index={index} points at a {} message whose last block is not \
             text — the wire carries prompt_cache_breakpoint on text input blocks only. Point \
             the prefix at a user/developer message that ends with text, or omit \
             prefix_until_index (implicit caching still applies).",
            role.as_str()
        ),
    )
}

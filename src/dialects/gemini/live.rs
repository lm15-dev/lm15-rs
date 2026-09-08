//! The Gemini Live (BidiGenerateContent) websocket codec
//! (`gemini.py:1249-1418`): a `setup` frame at connect (the key rides in
//! the URL query), `clientContent` / `realtimeInput` / `toolResponse`
//! client frames, `serverContent` / `toolCall` server frames. Audio-native
//! models (`live-preview`, `native-audio`) answer AUDIO with a transcript.

use serde_json::{json, Map, Value};

use crate::errors::Lm15Error;
use crate::types::{
    ErrorDetail, LiveClientEvent, LiveConfig, LiveServerAudioEvent, LiveServerErrorEvent,
    LiveServerEvent, LiveServerInterruptedEvent, LiveServerTextEvent, LiveServerToolCallEvent,
    LiveServerTurnEndEvent, LiveServerUsageEvent, SystemContent, Tool, Usage,
};
use crate::wire::BuildContext;

use super::contents::live_part;
use super::model_path;
use super::response::{stream_error_class, usage_from_metadata};

pub fn is_audio_native(model: &str) -> bool {
    let lowered = model.to_ascii_lowercase();
    lowered.contains("live-preview") || lowered.contains("native-audio")
}

/// The socket URL without the key: the adapter appends `key=` from the
/// credential (the Gemini door's `query-key` scheme).
pub fn url(cx: &BuildContext<'_>) -> String {
    let base = cx.base_url.trim_end_matches('/');
    let (scheme, rest) = base.split_once("://").unwrap_or(("https", base));
    let ws_scheme = if scheme == "https" { "wss" } else { "ws" };
    let netloc = rest.split('/').next().unwrap_or(rest);
    format!("{ws_scheme}://{netloc}/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent")
}

fn system_text(cx: &BuildContext<'_>, system: &SystemContent) -> Result<String, Lm15Error> {
    match system {
        SystemContent::Text(text) => Ok(text.clone()),
        SystemContent::Parts(parts) => {
            crate::dialects::content::parts_to_text(parts, cx.provider, "systemInstruction")
        }
    }
}

pub fn setup_frame(cx: &BuildContext<'_>, config: &LiveConfig) -> Result<Value, Lm15Error> {
    let mut setup = Map::new();
    setup.insert(
        "model".into(),
        Value::String(model_path(crate::wire::wire_model(
            cx.provider,
            &config.model,
        ))),
    );
    if let Some(system) = &config.system {
        setup.insert(
            "systemInstruction".into(),
            json!({"parts": [{"text": system_text(cx, system)?}]}),
        );
    }
    let functions: Vec<Value> = config
        .tools
        .iter()
        .filter_map(|t| match t {
            Tool::Function(f) => Some(
                json!({"name": f.name, "description": f.description, "parameters": f.parameters}),
            ),
            _ => None,
        })
        .collect();
    if !functions.is_empty() {
        setup.insert("tools".into(), json!([{"functionDeclarations": functions}]));
    }
    let mut generation = Map::new();
    if config.output_format.is_some() || is_audio_native(&config.model) {
        generation.insert("responseModalities".into(), json!(["AUDIO"]));
    }
    if let Some(voice) = &config.voice {
        generation.entry("speechConfig").or_insert_with(
            || json!({"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}),
        );
    }
    if !generation.is_empty() {
        setup.insert("generationConfig".into(), Value::Object(generation));
    }
    if let Some(extensions) = &config.extensions {
        setup.extend(extensions.clone());
    }
    if is_audio_native(&config.model) {
        setup
            .entry("outputAudioTranscription")
            .or_insert_with(|| json!({}));
    }
    Ok(json!({"setup": setup}))
}

pub fn encode(
    cx: &BuildContext<'_>,
    config: &LiveConfig,
    event: &LiveClientEvent,
) -> Result<Vec<Value>, Lm15Error> {
    Ok(match event {
        LiveClientEvent::Text(text) if is_audio_native(&config.model) => {
            vec![json!({"realtimeInput": {"text": text.text}})]
        }
        LiveClientEvent::Text(text) => vec![
            json!({"clientContent": {"turns": [{"role": "user", "parts": [{"text": text.text}]}], "turnComplete": true}}),
        ],
        LiveClientEvent::Turn(turn) => {
            let parts = turn
                .parts
                .iter()
                .map(|p| live_part(p, cx))
                .collect::<Result<Vec<_>, _>>()?;
            vec![
                json!({"clientContent": {"turns": [{"role": "user", "parts": parts}], "turnComplete": turn.turn_complete}}),
            ]
        }
        LiveClientEvent::Audio(audio) => vec![
            json!({"realtimeInput": {"audio": {"mimeType": audio.media_type, "data": audio.data}}}),
        ],
        LiveClientEvent::Image(image) => vec![
            json!({"realtimeInput": {"video": {"mimeType": image.media_type, "data": image.data}}}),
        ],
        LiveClientEvent::Interrupt(_) => vec![json!({"clientContent": {"turnComplete": true}})],
        LiveClientEvent::EndAudio(_) => vec![json!({"realtimeInput": {"audioStreamEnd": true}})],
        LiveClientEvent::ToolResult(result) => {
            let text = crate::dialects::content::parts_to_text(
                &result.content,
                cx.provider,
                "a functionResponse",
            )?;
            vec![
                json!({"toolResponse": {"functionResponses": [{"id": result.id, "response": {"output": [{"text": text}]}}]}}),
            ]
        }
    })
}

fn str_of(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

fn live_usage(
    cx: &BuildContext<'_>,
    payload: &Map<String, Value>,
    server: &Map<String, Value>,
) -> Usage {
    let metadata = payload
        .get("usageMetadata")
        .filter(|v| v.is_object())
        .or_else(|| server.get("usageMetadata"));
    usage_from_metadata(
        cx.provider,
        metadata,
        &["responseTokenCount", "candidatesTokenCount"],
    )
    .unwrap_or_default()
}

fn tool_call(fc: &Map<String, Value>) -> LiveServerToolCallEvent {
    let id = str_of(fc.get("id"));
    let name = str_of(fc.get("name"));
    LiveServerToolCallEvent {
        id: if id.is_empty() { "fc_0".into() } else { id },
        name: if name.is_empty() { "tool".into() } else { name },
        input: fc
            .get("args")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default(),
    }
}

/// Setup-phase frames: `true` on `setupComplete`; an error frame is the
/// typed error; anything else means keep waiting.
pub fn setup_complete(cx: &BuildContext<'_>, frame: &[u8]) -> Result<bool, Lm15Error> {
    let Ok(Value::Object(payload)) = serde_json::from_slice::<Value>(frame) else {
        return Ok(false);
    };
    if payload.contains_key("setupComplete") {
        return Ok(true);
    }
    if let Some(err) = payload.get("error") {
        let (message, code) = match err {
            Value::Object(e) => (str_of(e.get("message")), {
                let s = str_of(e.get("status"));
                if s.is_empty() {
                    "live_setup".to_string()
                } else {
                    s
                }
            }),
            other => (other.to_string(), "live_setup".into()),
        };
        let mut meta =
            crate::errors::ErrorMeta::new(format!("{}: Live setup failed: {message}", cx.provider));
        meta.provider = Some(cx.provider.to_string());
        meta.provider_code = Some(code);
        return Err(Lm15Error::InvalidRequestError(meta));
    }
    Ok(false)
}

pub fn decode(cx: &BuildContext<'_>, frame: &[u8]) -> Vec<LiveServerEvent> {
    let Ok(Value::Object(payload)) = serde_json::from_slice::<Value>(frame) else {
        return Vec::new();
    };
    if let Some(err) = payload.get("error") {
        let (provider_code, message) = match err {
            Value::Object(e) => {
                let mut code = str_of(e.get("status"));
                if code.is_empty() {
                    code = str_of(e.get("code"));
                }
                if code.is_empty() {
                    code = "provider".into();
                }
                (code, str_of(e.get("message")))
            }
            _ => ("provider".to_string(), String::new()),
        };
        let class = stream_error_class(&provider_code, &message);
        return vec![LiveServerEvent::Error(LiveServerErrorEvent {
            error: ErrorDetail {
                code: class.code(),
                message: if message.is_empty() {
                    provider_code.clone()
                } else {
                    message
                },
                provider_code: Some(provider_code),
            },
        })];
    }
    let mut events = Vec::new();
    if let Some(calls) = payload
        .get("toolCall")
        .and_then(Value::as_object)
        .and_then(|t| t.get("functionCalls"))
        .and_then(Value::as_array)
    {
        for fc in calls.iter().filter_map(Value::as_object) {
            events.push(LiveServerEvent::ToolCall(tool_call(fc)));
        }
    }
    let Some(server) = payload.get("serverContent").and_then(Value::as_object) else {
        return events;
    };
    if let Some(parts) = server
        .get("modelTurn")
        .and_then(Value::as_object)
        .and_then(|m| m.get("parts"))
        .and_then(Value::as_array)
    {
        for part in parts.iter().filter_map(Value::as_object) {
            if let Some(text) = part.get("text") {
                events.push(LiveServerEvent::Text(LiveServerTextEvent {
                    text: str_of(Some(text)),
                }));
            } else if let Some(inline) = part.get("inlineData").and_then(Value::as_object) {
                let mime = str_of(inline.get("mimeType"));
                if mime.starts_with("audio/") {
                    events.push(LiveServerEvent::Audio(LiveServerAudioEvent {
                        data: str_of(inline.get("data")),
                        media_type: if mime.is_empty() { None } else { Some(mime) },
                    }));
                }
            } else if let Some(fc) = part.get("functionCall").and_then(Value::as_object) {
                events.push(LiveServerEvent::ToolCall(tool_call(fc)));
            }
        }
    }
    if let Some(text) = server
        .get("outputTranscription")
        .and_then(Value::as_object)
        .and_then(|t| t.get("text"))
        .and_then(Value::as_str)
        .filter(|t| !t.is_empty())
    {
        events.push(LiveServerEvent::Text(LiveServerTextEvent {
            text: text.to_string(),
        }));
    }
    let turn_complete = server
        .get("turnComplete")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let has_usage = payload.get("usageMetadata").is_some_and(Value::is_object)
        || server.get("usageMetadata").is_some_and(Value::is_object);
    if has_usage && !turn_complete {
        events.push(LiveServerEvent::Usage(LiveServerUsageEvent {
            usage: live_usage(cx, &payload, server),
        }));
    }
    if server
        .get("interrupted")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        events.push(LiveServerEvent::Interrupted(LiveServerInterruptedEvent));
    }
    if turn_complete {
        events.push(LiveServerEvent::TurnEnd(LiveServerTurnEndEvent {
            usage: live_usage(cx, &payload, server),
        }));
    }
    events
}

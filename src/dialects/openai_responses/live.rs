//! The GA Realtime websocket codec (`openai.py:1418-1622`, captured live
//! 2026-09-01): `session.update` at connect, client events as
//! `conversation.item.create` + `response.create` bundles, server frames
//! decoded to the canonical live events. The beta header HARD-CLOSES
//! the socket (4000 `beta_api_shape_disabled`), so the connect headers
//! are the access policy's, as for HTTP.

use serde_json::{json, Map, Value};

use crate::errors::Lm15Error;
use crate::types::{
    AudioEncoding, AudioFormat, ErrorDetail, LiveClientEvent, LiveConfig, LiveServerAudioEvent,
    LiveServerErrorEvent, LiveServerEvent, LiveServerInterruptedEvent, LiveServerTextEvent,
    LiveServerToolCallDeltaEvent, LiveServerToolCallEvent, LiveServerTurnEndEvent,
    LiveServerUsageEvent, SystemContent, Tool, Usage,
};
use crate::wire::{apply_static_headers, BuildContext};

use super::input::{part_to_input, parts_to_text};
use super::response::stream_error_class;

pub fn url(cx: &BuildContext<'_>, config: &LiveConfig) -> (String, Vec<(String, String)>) {
    let base = cx.base_url.trim_end_matches('/');
    let (scheme, rest) = base.split_once("://").unwrap_or(("https", base));
    let ws_scheme = if scheme == "https" { "wss" } else { "ws" };
    let (netloc, path) = rest.split_once('/').unwrap_or((rest, ""));
    let path = if path.is_empty() {
        "/realtime".to_string()
    } else {
        format!("/{}/realtime", path.trim_end_matches('/'))
    };
    let url = crate::wire::full_url(
        &format!("{ws_scheme}://{netloc}{path}"),
        &[("model".to_string(), config.model.clone())],
    );
    let mut headers = Vec::new();
    apply_static_headers(&mut headers, cx.policy);
    (url, headers)
}

fn audio_format(fmt: &AudioFormat) -> Value {
    match fmt.encoding {
        AudioEncoding::Pcm16 => json!({"type": "audio/pcm", "rate": fmt.sample_rate}),
        other => json!({"type": format!("audio/{}", other.as_str())}),
    }
}

fn system_text(cx: &BuildContext<'_>, system: &SystemContent) -> Result<String, Lm15Error> {
    match system {
        SystemContent::Text(text) => Ok(text.clone()),
        SystemContent::Parts(parts) => parts_to_text(parts, cx.provider, "session instructions"),
    }
}

pub fn session_update(cx: &BuildContext<'_>, config: &LiveConfig) -> Result<Value, Lm15Error> {
    let mut session = Map::new();
    session.insert("type".into(), json!("realtime"));
    if let Some(system) = &config.system {
        session.insert(
            "instructions".into(),
            Value::String(system_text(cx, system)?),
        );
    }
    let mut audio = Map::new();
    if config.output_format.is_some() || config.voice.is_some() {
        session.insert("output_modalities".into(), json!(["audio"]));
        let mut output = Map::new();
        if let Some(fmt) = &config.output_format {
            output.insert("format".into(), audio_format(fmt));
        }
        if let Some(voice) = &config.voice {
            output.insert("voice".into(), Value::String(voice.clone()));
        }
        audio.insert("output".into(), Value::Object(output));
    } else {
        session.insert("output_modalities".into(), json!(["text"]));
    }
    if let Some(fmt) = &config.input_format {
        // turn_detection null = server VAD OFF: a turn happens exactly
        // when the caller sends end_audio() (commit + response.create).
        audio.insert(
            "input".into(),
            json!({"format": audio_format(fmt), "turn_detection": null}),
        );
    }
    if !audio.is_empty() {
        session.insert("audio".into(), Value::Object(audio));
    }
    if !config.tools.is_empty() {
        let tools: Vec<Value> = config
            .tools
            .iter()
            .filter_map(|t| match t {
                Tool::Function(f) => Some(json!({"type": "function", "name": f.name, "description": f.description, "parameters": f.parameters})),
                _ => None,
            })
            .collect();
        session.insert("tools".into(), Value::Array(tools));
    }
    if let Some(extensions) = &config.extensions {
        session.extend(extensions.clone());
    }
    Ok(json!({"type": "session.update", "session": session}))
}

fn user_item(content: Vec<Value>) -> Value {
    json!({"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": content}})
}

pub fn encode(cx: &BuildContext<'_>, event: &LiveClientEvent) -> Result<Vec<Value>, Lm15Error> {
    Ok(match event {
        LiveClientEvent::Audio(audio) => {
            vec![json!({"type": "input_audio_buffer.append", "audio": audio.data})]
        }
        LiveClientEvent::EndAudio(_) => vec![
            json!({"type": "input_audio_buffer.commit"}),
            json!({"type": "response.create"}),
        ],
        LiveClientEvent::Interrupt(_) => vec![json!({"type": "response.cancel"})],
        LiveClientEvent::Text(text) => vec![
            user_item(vec![json!({"type": "input_text", "text": text.text})]),
            json!({"type": "response.create"}),
        ],
        LiveClientEvent::Turn(turn) => {
            let content = turn
                .parts
                .iter()
                .map(|p| part_to_input(cx.provider, p))
                .collect::<Result<Vec<_>, _>>()?;
            let mut frames = vec![user_item(content)];
            if turn.turn_complete {
                frames.push(json!({"type": "response.create"}));
            }
            frames
        }
        LiveClientEvent::Image(image) => vec![
            user_item(vec![
                json!({"type": "input_image", "image_url": format!("data:{};base64,{}", image.media_type, image.data)}),
            ]),
            json!({"type": "response.create"}),
        ],
        LiveClientEvent::ToolResult(result) => {
            let output = parts_to_text(
                &result.content,
                cx.provider,
                "a Realtime function_call_output",
            )?;
            vec![
                json!({"type": "conversation.item.create", "item": {"type": "function_call_output", "call_id": result.id, "output": output}}),
                json!({"type": "response.create"}),
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

/// Usage from a Realtime `response.done` payload, or `None` when absent.
fn usage_from_response(response: &Map<String, Value>) -> Option<Usage> {
    let usage = response.get("usage")?.as_object()?;
    let details = |a: &str, b: &str| -> Map<String, Value> {
        usage
            .get(a)
            .or_else(|| usage.get(b))
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default()
    };
    let input = details("input_token_details", "input_tokens_details");
    let output = details("output_token_details", "output_tokens_details");
    let count = |map: &Map<String, Value>, key: &str| map.get(key).and_then(Value::as_u64);
    Some(Usage {
        input_tokens: count(usage, "input_tokens"),
        output_tokens: count(usage, "output_tokens"),
        total_tokens: count(usage, "total_tokens"),
        reasoning_tokens: count(&output, "reasoning_tokens"),
        cache_read_tokens: count(&input, "cached_tokens"),
        cache_write_tokens: count(&input, "cache_write_tokens"),
        input_audio_tokens: count(&input, "audio_tokens"),
        output_audio_tokens: count(&output, "audio_tokens"),
    })
}

fn error_detail(provider_code: &str, message: &str) -> ErrorDetail {
    ErrorDetail {
        code: stream_error_class(provider_code).code(),
        message: if message.is_empty() {
            if provider_code.is_empty() {
                "provider error".into()
            } else {
                provider_code.into()
            }
        } else {
            message.into()
        },
        provider_code: Some(if provider_code.is_empty() {
            "provider".into()
        } else {
            provider_code.into()
        }),
    }
}

pub fn decode(frame: &[u8]) -> Vec<LiveServerEvent> {
    let Ok(Value::Object(payload)) = serde_json::from_slice::<Value>(frame) else {
        return Vec::new();
    };
    let et = str_of(payload.get("type"));
    let mut events = Vec::new();
    match et.as_str() {
        "response.output_text.delta"
        | "response.text.delta"
        | "response.output_audio_transcript.delta"
        | "response.audio_transcript.delta" => {
            let delta = {
                let d = str_of(payload.get("delta"));
                if d.is_empty() {
                    str_of(payload.get("text"))
                } else {
                    d
                }
            };
            if !delta.is_empty() {
                events.push(LiveServerEvent::Text(LiveServerTextEvent { text: delta }));
            }
        }
        "response.output_audio.delta" => {
            let delta = str_of(payload.get("delta"));
            if !delta.is_empty() {
                events.push(LiveServerEvent::Audio(LiveServerAudioEvent {
                    data: delta,
                    media_type: None,
                }));
            }
        }
        "response.function_call_arguments.delta" => {
            let delta = str_of(payload.get("delta"));
            if !delta.is_empty() {
                let id = {
                    let c = str_of(payload.get("call_id"));
                    if c.is_empty() {
                        str_of(payload.get("id"))
                    } else {
                        c
                    }
                };
                let name = str_of(payload.get("name"));
                events.push(LiveServerEvent::ToolCallDelta(
                    LiveServerToolCallDeltaEvent {
                        input_delta: delta,
                        id: if id.is_empty() { None } else { Some(id) },
                        name: if name.is_empty() { None } else { Some(name) },
                    },
                ));
            }
        }
        "response.output_item.done" => {
            // The ONLY tool-call emission point (`function_call_arguments.done`
            // also carries the full call; mapping both double-fires it).
            if let Some(item) = payload.get("item").and_then(Value::as_object) {
                if item.get("type").and_then(Value::as_str) == Some("function_call") {
                    let call_id = {
                        let c = str_of(item.get("call_id"));
                        if c.is_empty() {
                            str_of(item.get("id"))
                        } else {
                            c
                        }
                    };
                    if !call_id.is_empty() {
                        let name = str_of(item.get("name"));
                        events.push(LiveServerEvent::ToolCall(LiveServerToolCallEvent {
                            id: call_id,
                            name: if name.is_empty() { "tool".into() } else { name },
                            input: crate::dialects::wire_json::parse_json_object(
                                item.get("arguments"),
                            ),
                        }));
                    }
                }
            }
        }
        "response.done" | "response.completed" => {
            let response = payload
                .get("response")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            let output = response
                .get("output")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let usage = usage_from_response(&response);
            if response.get("status").and_then(Value::as_str) == Some("cancelled") {
                // Barge-in via response.done status=cancelled; the cancelled
                // response still consumed tokens: a usage event first.
                if let Some(usage) = usage {
                    events.push(LiveServerEvent::Usage(LiveServerUsageEvent { usage }));
                }
                events.push(LiveServerEvent::Interrupted(LiveServerInterruptedEvent));
            } else if output
                .iter()
                .any(|i| i.get("type").and_then(Value::as_str) == Some("function_call"))
            {
                // A response requesting tool calls does not end the turn.
                if let Some(usage) = usage {
                    events.push(LiveServerEvent::Usage(LiveServerUsageEvent { usage }));
                }
            } else {
                events.push(LiveServerEvent::TurnEnd(LiveServerTurnEndEvent {
                    usage: usage.unwrap_or_default(),
                }));
            }
        }
        "response.cancelled" | "response.canceled" => {
            events.push(LiveServerEvent::Interrupted(LiveServerInterruptedEvent));
        }
        "error" | "response.error" => {
            let (provider_code, message) = match payload.get("error") {
                Some(Value::Object(err)) => {
                    let mut code = str_of(err.get("code"));
                    if code.is_empty() {
                        code = str_of(err.get("type"));
                    }
                    if code.is_empty() {
                        code = str_of(payload.get("code"));
                    }
                    if code.is_empty() {
                        code = "provider".into();
                    }
                    let mut message = str_of(err.get("message"));
                    if message.is_empty() {
                        message = str_of(payload.get("message"));
                    }
                    (code, message)
                }
                _ => {
                    let mut code = str_of(payload.get("code"));
                    if code.is_empty() {
                        code = str_of(payload.get("error_type"));
                    }
                    if code.is_empty() {
                        code = "provider".into();
                    }
                    (code, str_of(payload.get("message")))
                }
            };
            if provider_code == "response_cancel_not_active" {
                // Benign barge-in race: the response finished before the
                // cancel arrived, or interrupt() was pressed twice.
                return events;
            }
            events.push(LiveServerEvent::Error(LiveServerErrorEvent {
                error: error_detail(&provider_code, &message),
            }));
        }
        _ => {}
    }
    events
}

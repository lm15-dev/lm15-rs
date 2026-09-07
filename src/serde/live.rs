//! AudioFormat, LiveConfig, live client/server events (serde.py
//! `audio_format_*`, `live_*`).

use serde_json::Value;

use super::config::extensions_to_json;
use super::helpers::{Obj, Reader, VResult};
use super::message::{system_from_json, system_to_json, tools_from_json, tools_to_json};
use super::parts::{parts_from_list, parts_to_json};
use super::stream::{usage_from_parent, usage_to_json_opt};
use super::{impl_serde_via_canonical, Canonical};
use crate::types::{
    AudioEncoding, AudioFormat, ErrorDetail, LiveClientAudioEvent, LiveClientEndAudioEvent,
    LiveClientEvent, LiveClientImageEvent, LiveClientInterruptEvent, LiveClientTextEvent,
    LiveClientToolResultEvent, LiveClientTurnEvent, LiveConfig, LiveServerAudioEvent,
    LiveServerErrorEvent, LiveServerEvent, LiveServerInterruptedEvent, LiveServerTextEvent,
    LiveServerToolCallDeltaEvent, LiveServerToolCallEvent, LiveServerTurnEndEvent,
    LiveServerUsageEvent, ValidationError,
};

impl Canonical for AudioFormat {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "AudioFormat")?;
        let format = AudioFormat {
            encoding: AudioEncoding::parse(&r.req_str("encoding")?)?,
            sample_rate: r.req_u64("sample_rate")?,
            channels: r.opt_u64("channels")?.unwrap_or(1),
        };
        format.validate()?;
        Ok(format)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::new();
        o.set("encoding", self.encoding.as_str())
            .set("sample_rate", self.sample_rate)
            .set("channels", self.channels);
        o.finish()
    }
}

impl Canonical for LiveConfig {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "LiveConfig")?;
        let audio_format = |key: &str| -> VResult<Option<AudioFormat>> {
            r.lenient_object(key)
                .map(|o| AudioFormat::from_json(&Value::Object(o.clone())))
                .transpose()
        };
        let config = LiveConfig {
            model: r.req_str("model")?,
            system: system_from_json(&r)?,
            tools: tools_from_json(&r)?,
            voice: r.opt_str("voice")?,
            input_format: audio_format("input_format")?,
            output_format: audio_format("output_format")?,
            extensions: r.opt_object("extensions")?,
        }
        .normalized();
        config.validate()?;
        Ok(config)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::new();
        o.set("model", self.model.as_str());
        o.omit_empty_opt("system", self.system.as_ref().map(system_to_json));
        o.omit_empty("tools", tools_to_json(&self.tools));
        o.omit_empty_opt("voice", self.voice.clone());
        o.opt(
            "input_format",
            self.input_format.as_ref().map(Canonical::to_json),
        );
        o.opt(
            "output_format",
            self.output_format.as_ref().map(Canonical::to_json),
        );
        extensions_to_json(&mut o, self.extensions.as_ref());
        o.finish()
    }
}

impl Canonical for LiveClientEvent {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "LiveClientEvent")?;
        let type_name = r.req_str("type")?;
        let event = match type_name.as_str() {
            "turn" => LiveClientEvent::Turn(LiveClientTurnEvent {
                parts: parts_from_list(r.array_or_empty("parts")?)?,
                turn_complete: r.bool_or("turn_complete", true)?,
            }),
            "audio" => LiveClientEvent::Audio(LiveClientAudioEvent {
                data: r.req_str("data")?,
                media_type: r.str_or("media_type", LiveClientAudioEvent::DEFAULT_MEDIA_TYPE)?,
            }),
            "image" => LiveClientEvent::Image(LiveClientImageEvent {
                data: r.req_str("data")?,
                media_type: r.str_or("media_type", LiveClientImageEvent::DEFAULT_MEDIA_TYPE)?,
            }),
            "text" => LiveClientEvent::Text(LiveClientTextEvent {
                text: r.str_or_empty("text")?,
            }),
            "tool_result" => LiveClientEvent::ToolResult(LiveClientToolResultEvent {
                id: r.req_str("id")?,
                content: parts_from_list(r.array_or_empty("content")?)?,
            }),
            "interrupt" => LiveClientEvent::Interrupt(LiveClientInterruptEvent),
            "end_audio" => LiveClientEvent::EndAudio(LiveClientEndAudioEvent),
            other => {
                return Err(ValidationError::value(format!(
                    "unsupported live client event type: {other}"
                )))
            }
        };
        event.validate()?;
        Ok(event)
    }

    fn to_json(&self) -> Value {
        // Live client events emit ALL their fields verbatim (no cleaning).
        let mut o = Obj::typed(self.type_name());
        match self {
            LiveClientEvent::Turn(e) => {
                o.set("parts", parts_to_json(&e.parts))
                    .set("turn_complete", e.turn_complete);
            }
            LiveClientEvent::Audio(e) => {
                o.set("data", e.data.as_str())
                    .set("media_type", e.media_type.as_str());
            }
            LiveClientEvent::Image(e) => {
                o.set("data", e.data.as_str())
                    .set("media_type", e.media_type.as_str());
            }
            LiveClientEvent::Text(e) => {
                o.set("text", e.text.as_str());
            }
            LiveClientEvent::ToolResult(e) => {
                o.set("id", e.id.as_str())
                    .set("content", parts_to_json(&e.content));
            }
            LiveClientEvent::Interrupt(_) | LiveClientEvent::EndAudio(_) => {}
        }
        o.finish()
    }
}

impl Canonical for LiveServerEvent {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "LiveServerEvent")?;
        let type_name = r.req_str("type")?;
        let event = match type_name.as_str() {
            "audio" => LiveServerEvent::Audio(LiveServerAudioEvent {
                data: r.req_str("data")?,
                media_type: r.opt_str("media_type")?,
            }),
            "text" => LiveServerEvent::Text(LiveServerTextEvent {
                text: r.str_or_empty("text")?,
            }),
            "tool_call" => LiveServerEvent::ToolCall(LiveServerToolCallEvent {
                id: r.req_str("id")?,
                name: r.req_str("name")?,
                input: r.object_or_empty("input")?,
            }),
            "tool_call_delta" => LiveServerEvent::ToolCallDelta(LiveServerToolCallDeltaEvent {
                input_delta: r.str_or_empty("input_delta")?,
                id: r.opt_str("id")?,
                name: r.opt_str("name")?,
            }),
            "interrupted" => LiveServerEvent::Interrupted(LiveServerInterruptedEvent),
            "turn_end" => LiveServerEvent::TurnEnd(LiveServerTurnEndEvent {
                usage: usage_from_parent(&r, "usage")?,
            }),
            "usage" => LiveServerEvent::Usage(LiveServerUsageEvent {
                usage: usage_from_parent(&r, "usage")?,
            }),
            "error" => LiveServerEvent::Error(LiveServerErrorEvent {
                error: ErrorDetail::from_json(r.req("error")?)?,
            }),
            other => {
                return Err(ValidationError::value(format!(
                    "unsupported live server event type: {other}"
                )))
            }
        };
        event.validate()?;
        Ok(event)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::typed(self.type_name());
        match self {
            LiveServerEvent::Audio(e) => {
                o.set("data", e.data.as_str());
                o.omit_empty_opt("media_type", e.media_type.clone());
            }
            LiveServerEvent::Text(e) => {
                o.set("text", e.text.as_str());
            }
            LiveServerEvent::ToolCall(e) => {
                o.set("id", e.id.as_str())
                    .set("name", e.name.as_str())
                    .set("input", Value::Object(e.input.clone()));
            }
            LiveServerEvent::ToolCallDelta(e) => {
                o.omit_empty_opt("id", e.id.clone());
                o.omit_empty_opt("name", e.name.clone());
                o.omit_empty("input_delta", e.input_delta.as_str());
            }
            LiveServerEvent::Interrupted(_) => {}
            LiveServerEvent::TurnEnd(e) => {
                o.opt("usage", usage_to_json_opt(&e.usage));
            }
            LiveServerEvent::Usage(e) => {
                o.opt("usage", usage_to_json_opt(&e.usage));
            }
            LiveServerEvent::Error(e) => {
                o.set("error", e.error.to_json());
            }
        }
        o.finish()
    }
}

impl_serde_via_canonical!(AudioFormat, LiveConfig, LiveClientEvent, LiveServerEvent);

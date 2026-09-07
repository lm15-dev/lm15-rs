//! Audio / Live (types.md § Audio / Live (realtime)).

use super::json::{
    non_empty, opt_non_empty, positive, validate_base64, JsonObject, VResult, ValidationError,
};
use super::message::SystemContent;
use super::parts::Part;
use super::stream::ErrorDetail;
use super::tools::{validate_tools, Tool};
use super::usage::Usage;
use super::vocab::AudioEncoding;

/// PCM/opus/... framing for a live session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioFormat {
    pub encoding: AudioEncoding,
    pub sample_rate: u64,
    pub channels: u64,
}

impl AudioFormat {
    pub fn new(encoding: AudioEncoding, sample_rate: u64) -> VResult<Self> {
        let format = AudioFormat {
            encoding,
            sample_rate,
            channels: 1,
        };
        format.validate()?;
        Ok(format)
    }

    pub fn validate(&self) -> VResult<()> {
        positive(Some(self.sample_rate), "sample_rate")?;
        positive(Some(self.channels), "channels")
    }
}

/// A live session configuration.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LiveConfig {
    pub model: String,
    pub system: Option<SystemContent>,
    pub tools: Vec<Tool>,
    pub voice: Option<String>,
    pub input_format: Option<AudioFormat>,
    pub output_format: Option<AudioFormat>,
    pub extensions: Option<JsonObject>,
}

impl LiveConfig {
    pub fn normalized(mut self) -> Self {
        self.extensions = self.extensions.filter(|e| !e.is_empty());
        self
    }

    pub fn validate(&self) -> VResult<()> {
        if self.model.is_empty() {
            return Err(ValidationError::value("model is required"));
        }
        validate_tools("LiveConfig", &self.tools)?;
        if let Some(system) = &self.system {
            system.validate()?;
        }
        if let Some(f) = &self.input_format {
            f.validate()?;
        }
        if let Some(f) = &self.output_format {
            f.validate()?;
        }
        opt_non_empty(self.voice.as_ref(), "LiveConfig.voice")
    }
}

// ─── Client events ───────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveClientTurnEvent {
    pub parts: Vec<Part>,
    pub turn_complete: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveClientAudioEvent {
    pub data: String,
    pub media_type: String,
}

impl LiveClientAudioEvent {
    pub const DEFAULT_MEDIA_TYPE: &'static str = "audio/pcm;rate=16000";
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveClientImageEvent {
    pub data: String,
    pub media_type: String,
}

impl LiveClientImageEvent {
    pub const DEFAULT_MEDIA_TYPE: &'static str = "image/jpeg";
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LiveClientTextEvent {
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveClientToolResultEvent {
    pub id: String,
    pub content: Vec<Part>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LiveClientInterruptEvent;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LiveClientEndAudioEvent;

/// vocabularies.md § LiveClientEventType.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LiveClientEvent {
    Turn(LiveClientTurnEvent),
    Audio(LiveClientAudioEvent),
    Image(LiveClientImageEvent),
    Text(LiveClientTextEvent),
    ToolResult(LiveClientToolResultEvent),
    Interrupt(LiveClientInterruptEvent),
    EndAudio(LiveClientEndAudioEvent),
}

fn validate_live_media(
    owner: &str,
    data: &str,
    media_type: Option<&str>,
    prefix: &str,
) -> VResult<()> {
    non_empty(data, &format!("{owner}.data"))?;
    validate_base64(owner, data)?;
    if let Some(media_type) = media_type {
        non_empty(media_type, &format!("{owner}.media_type"))?;
        if !media_type.starts_with(prefix) {
            return Err(ValidationError::value(format!(
                "{owner}.media_type must start with '{prefix}'"
            )));
        }
    }
    Ok(())
}

impl LiveClientEvent {
    pub const TYPES: &'static [&'static str] = &[
        "turn",
        "audio",
        "image",
        "text",
        "tool_result",
        "interrupt",
        "end_audio",
    ];

    pub fn type_name(&self) -> &'static str {
        match self {
            LiveClientEvent::Turn(_) => "turn",
            LiveClientEvent::Audio(_) => "audio",
            LiveClientEvent::Image(_) => "image",
            LiveClientEvent::Text(_) => "text",
            LiveClientEvent::ToolResult(_) => "tool_result",
            LiveClientEvent::Interrupt(_) => "interrupt",
            LiveClientEvent::EndAudio(_) => "end_audio",
        }
    }

    pub fn validate(&self) -> VResult<()> {
        match self {
            LiveClientEvent::Turn(e) => {
                if e.parts.is_empty() {
                    return Err(ValidationError::value(
                        "LiveClientTurnEvent requires at least one part",
                    ));
                }
                for part in &e.parts {
                    part.validate()?;
                    if part.is_prompt_forbidden() {
                        return Err(ValidationError::type_error(
                            "LiveClientTurnEvent.parts cannot contain model/tool protocol parts",
                        ));
                    }
                }
                Ok(())
            }
            LiveClientEvent::Audio(e) => validate_live_media(
                "LiveClientAudioEvent",
                &e.data,
                Some(&e.media_type),
                "audio/",
            ),
            LiveClientEvent::Image(e) => validate_live_media(
                "LiveClientImageEvent",
                &e.data,
                Some(&e.media_type),
                "image/",
            ),
            LiveClientEvent::Text(_) => Ok(()),
            LiveClientEvent::ToolResult(e) => {
                non_empty(&e.id, "LiveClientToolResultEvent.id")?;
                if e.content.is_empty() {
                    return Err(ValidationError::value(
                        "LiveClientToolResultEvent requires content",
                    ));
                }
                for part in &e.content {
                    part.validate()?;
                    if part.is_tool_result_forbidden() {
                        return Err(ValidationError::type_error(
                            "LiveClientToolResultEvent.content cannot contain model or protocol parts",
                        ));
                    }
                }
                Ok(())
            }
            LiveClientEvent::Interrupt(_) | LiveClientEvent::EndAudio(_) => Ok(()),
        }
    }
}

// ─── Server events ───────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveServerAudioEvent {
    pub data: String,
    pub media_type: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LiveServerTextEvent {
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveServerToolCallEvent {
    pub id: String,
    pub name: String,
    pub input: JsonObject,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LiveServerToolCallDeltaEvent {
    pub input_delta: String,
    pub id: Option<String>,
    pub name: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LiveServerInterruptedEvent;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveServerTurnEndEvent {
    pub usage: Usage,
}

/// Billed usage for a response that did not end the turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveServerUsageEvent {
    pub usage: Usage,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveServerErrorEvent {
    pub error: ErrorDetail,
}

/// vocabularies.md § LiveServerEventType.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LiveServerEvent {
    Audio(LiveServerAudioEvent),
    Text(LiveServerTextEvent),
    ToolCall(LiveServerToolCallEvent),
    ToolCallDelta(LiveServerToolCallDeltaEvent),
    Interrupted(LiveServerInterruptedEvent),
    TurnEnd(LiveServerTurnEndEvent),
    Usage(LiveServerUsageEvent),
    Error(LiveServerErrorEvent),
}

impl LiveServerEvent {
    pub const TYPES: &'static [&'static str] = &[
        "audio",
        "text",
        "tool_call",
        "tool_call_delta",
        "interrupted",
        "turn_end",
        "usage",
        "error",
    ];

    pub fn type_name(&self) -> &'static str {
        match self {
            LiveServerEvent::Audio(_) => "audio",
            LiveServerEvent::Text(_) => "text",
            LiveServerEvent::ToolCall(_) => "tool_call",
            LiveServerEvent::ToolCallDelta(_) => "tool_call_delta",
            LiveServerEvent::Interrupted(_) => "interrupted",
            LiveServerEvent::TurnEnd(_) => "turn_end",
            LiveServerEvent::Usage(_) => "usage",
            LiveServerEvent::Error(_) => "error",
        }
    }

    pub fn validate(&self) -> VResult<()> {
        match self {
            LiveServerEvent::Audio(e) => validate_live_media(
                "LiveServerAudioEvent",
                &e.data,
                e.media_type.as_deref(),
                "audio/",
            ),
            LiveServerEvent::Text(_) | LiveServerEvent::Interrupted(_) => Ok(()),
            LiveServerEvent::ToolCall(e) => {
                non_empty(&e.id, "LiveServerToolCallEvent.id")?;
                non_empty(&e.name, "LiveServerToolCallEvent.name")
            }
            LiveServerEvent::ToolCallDelta(e) => {
                opt_non_empty(e.id.as_ref(), "LiveServerToolCallDeltaEvent.id")?;
                opt_non_empty(e.name.as_ref(), "LiveServerToolCallDeltaEvent.name")
            }
            LiveServerEvent::TurnEnd(e) => e.usage.validate(),
            LiveServerEvent::Usage(e) => e.usage.validate(),
            LiveServerEvent::Error(e) => e.error.validate(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_format_positive_ints() {
        assert!(AudioFormat::new(AudioEncoding::Pcm16, 0).is_err());
        let f = AudioFormat::new(AudioEncoding::Pcm16, 24000).unwrap();
        assert_eq!(f.channels, 1);
    }

    #[test]
    fn live_client_turn_rejects_protocol_parts_and_empty() {
        let empty = LiveClientEvent::Turn(LiveClientTurnEvent {
            parts: vec![],
            turn_complete: true,
        });
        assert!(empty.validate().is_err());
        let thinking = LiveClientEvent::Turn(LiveClientTurnEvent {
            parts: vec![Part::thinking("x")],
            turn_complete: true,
        });
        assert!(thinking.validate().is_err());
    }

    #[test]
    fn live_media_prefix_and_base64() {
        let bad_prefix = LiveClientEvent::Audio(LiveClientAudioEvent {
            data: "aGk=".into(),
            media_type: "image/png".into(),
        });
        assert!(bad_prefix.validate().is_err());
        let bad_data = LiveServerEvent::Audio(LiveServerAudioEvent {
            data: "###".into(),
            media_type: None,
        });
        assert!(bad_data.validate().is_err());
    }
}

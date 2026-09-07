//! Deltas (types.md § Deltas; INV-006, INV-018, INV-019). `Delta` is a
//! closed sum discriminated on the `type` key.

use super::continuation::ContinuationState;
use super::json::{non_empty, opt_non_empty, JsonObject, VResult, ValidationError};
use super::usage::TokenLogprob;

/// A text fragment. `logprobs` are the tokens of exactly this fragment.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TextDelta {
    pub text: String,
    pub part_index: u64,
    pub logprobs: Vec<TokenLogprob>,
}

/// A reasoning fragment.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ThinkingDelta {
    pub text: String,
    pub part_index: u64,
}

/// An audio chunk: at most one address, possibly none (INV-018); `data`
/// may be unaligned partial base64.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AudioDelta {
    pub data: Option<String>,
    pub url: Option<String>,
    pub file_id: Option<String>,
    pub part_index: u64,
    pub media_type: Option<String>,
}

/// An image chunk (same shape as [`AudioDelta`]).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ImageDelta {
    pub data: Option<String>,
    pub url: Option<String>,
    pub file_id: Option<String>,
    pub part_index: u64,
    pub media_type: Option<String>,
}

/// A tool-call input fragment (raw JSON text), optionally carrying identity.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ToolCallDelta {
    pub input: String,
    pub part_index: u64,
    pub id: Option<String>,
    pub name: Option<String>,
}

/// A citation fragment; at least one of text/url/title (INV-019).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CitationDelta {
    pub text: Option<String>,
    pub url: Option<String>,
    pub title: Option<String>,
    pub part_index: u64,
}

/// Opaque continuation state arriving mid-stream. `part_index: None`
/// attaches to the message, `Some(i)` to the completed part `i`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ContinuationDelta {
    pub provider: String,
    pub kind: String,
    pub data: JsonObject,
    pub part_index: Option<u64>,
}

impl ContinuationDelta {
    pub fn to_state(&self) -> ContinuationState {
        ContinuationState {
            provider: self.provider.clone(),
            kind: self.kind.clone(),
            data: self.data.clone(),
        }
    }
}

/// A typed fragment of a Part being assembled (vocabularies.md § DeltaType).
#[derive(Debug, Clone, PartialEq)]
pub enum Delta {
    Text(TextDelta),
    Thinking(ThinkingDelta),
    Audio(AudioDelta),
    Image(ImageDelta),
    ToolCall(ToolCallDelta),
    Citation(CitationDelta),
    Continuation(ContinuationDelta),
}

fn validate_addresses(
    delta_type: &str,
    data: Option<&String>,
    url: Option<&String>,
    file_id: Option<&String>,
) -> VResult<()> {
    let count = [data, url, file_id].iter().filter(|v| v.is_some()).count();
    if count > 1 {
        return Err(ValidationError::value(format!(
            "{delta_type} can include at most one of data, url, or file_id"
        )));
    }
    Ok(())
}

impl Delta {
    /// Every `type` discriminator, in spec order (vocabularies.md § DeltaType).
    pub const TYPES: &'static [&'static str] = &[
        "text",
        "thinking",
        "audio",
        "image",
        "tool_call",
        "citation",
        "continuation",
    ];

    pub fn type_name(&self) -> &'static str {
        match self {
            Delta::Text(_) => "text",
            Delta::Thinking(_) => "thinking",
            Delta::Audio(_) => "audio",
            Delta::Image(_) => "image",
            Delta::ToolCall(_) => "tool_call",
            Delta::Citation(_) => "citation",
            Delta::Continuation(_) => "continuation",
        }
    }

    /// The addressed part, `None` for a message-level continuation.
    pub fn part_index(&self) -> Option<u64> {
        match self {
            Delta::Text(d) => Some(d.part_index),
            Delta::Thinking(d) => Some(d.part_index),
            Delta::Audio(d) => Some(d.part_index),
            Delta::Image(d) => Some(d.part_index),
            Delta::ToolCall(d) => Some(d.part_index),
            Delta::Citation(d) => Some(d.part_index),
            Delta::Continuation(d) => d.part_index,
        }
    }

    pub fn validate(&self) -> VResult<()> {
        match self {
            Delta::Text(d) => d.logprobs.iter().try_for_each(TokenLogprob::validate),
            Delta::Thinking(_) => Ok(()),
            Delta::Audio(d) => {
                validate_addresses(
                    "AudioDelta",
                    d.data.as_ref(),
                    d.url.as_ref(),
                    d.file_id.as_ref(),
                )?;
                opt_non_empty(d.media_type.as_ref(), "AudioDelta.media_type")
            }
            Delta::Image(d) => {
                validate_addresses(
                    "ImageDelta",
                    d.data.as_ref(),
                    d.url.as_ref(),
                    d.file_id.as_ref(),
                )?;
                opt_non_empty(d.media_type.as_ref(), "ImageDelta.media_type")
            }
            Delta::ToolCall(d) => {
                opt_non_empty(d.id.as_ref(), "ToolCallDelta.id")?;
                opt_non_empty(d.name.as_ref(), "ToolCallDelta.name")
            }
            Delta::Citation(d) => {
                if d.text.is_none() && d.url.is_none() && d.title.is_none() {
                    return Err(ValidationError::value(
                        "CitationDelta requires at least one of text, url, or title",
                    ));
                }
                Ok(())
            }
            Delta::Continuation(d) => {
                non_empty(&d.provider, "ContinuationDelta.provider")?;
                non_empty(&d.kind, "ContinuationDelta.kind")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inv_018_media_deltas_carry_at_most_one_address() {
        let two = Delta::Audio(AudioDelta {
            data: Some("aG".into()),
            url: Some("https://a".into()),
            ..Default::default()
        });
        assert!(two.validate().is_err());
        let none = Delta::Audio(AudioDelta::default());
        assert!(none.validate().is_ok());
        let unaligned = Delta::Image(ImageDelta {
            data: Some("aG".into()),
            ..Default::default()
        });
        assert!(unaligned.validate().is_ok());
    }

    #[test]
    fn inv_019_citation_delta_needs_one_field() {
        assert!(Delta::Citation(CitationDelta::default())
            .validate()
            .is_err());
        assert!(Delta::Citation(CitationDelta {
            url: Some("https://a".into()),
            ..Default::default()
        })
        .validate()
        .is_ok());
    }

    #[test]
    fn continuation_delta_to_state() {
        let d = ContinuationDelta {
            provider: "anthropic".into(),
            kind: "thinking_signature".into(),
            data: JsonObject::new(),
            part_index: Some(1),
        };
        assert_eq!(d.to_state().kind, "thinking_signature");
        assert!(Delta::Continuation(d).validate().is_ok());
        assert!(Delta::Continuation(ContinuationDelta::default())
            .validate()
            .is_err());
    }
}

//! Request and Response (types.md § Request / Response; INV-030, INV-031,
//! INV-036).

use super::config::Config;
use super::json::{non_empty, opt_non_empty, JsonObject, VResult, ValidationError};
use super::message::{Message, SystemContent};
use super::parts::{CitationPart, DataPart, Part, ToolCallPart};
use super::tools::{validate_tools, Tool};
use super::usage::{TokenLogprob, Usage};
use super::vocab::{FinishReason, Role};
use super::{Adaptation, JudgmentMethod};

/// A complete request to a model.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Request {
    pub model: String,
    pub messages: Vec<Message>,
    pub system: Option<SystemContent>,
    pub tools: Vec<Tool>,
    pub config: Config,
}

impl Request {
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> VResult<Self> {
        let request = Request {
            model: model.into(),
            messages,
            ..Default::default()
        };
        request.validate()?;
        Ok(request)
    }

    pub fn validate(&self) -> VResult<()> {
        if self.model.is_empty() {
            return Err(ValidationError::value("model is required"));
        }
        if self.messages.is_empty() {
            return Err(ValidationError::value("at least one message is required"));
        }
        for message in &self.messages {
            message.validate()?;
        }
        if let Some(system) = &self.system {
            system.validate()?;
        }
        validate_tools("Request", &self.tools)?;
        self.config.validate()?;
        if let Some(choice) = &self.config.tool_choice {
            let mut missing: Vec<&str> = choice
                .allowed
                .iter()
                .filter(|name| !self.tools.iter().any(|t| t.name() == name.as_str()))
                .map(String::as_str)
                .collect();
            if !missing.is_empty() {
                missing.sort_unstable();
                return Err(ValidationError::value(format!(
                    "ToolChoice.allowed contains tools not present in Request.tools: {missing:?}"
                )));
            }
        }
        Ok(())
    }
}

/// The artifact a model returns.
#[derive(Debug, Clone, PartialEq)]
pub struct Response {
    pub id: Option<String>,
    pub model: String,
    pub message: Message,
    pub finish_reason: FinishReason,
    pub usage: Usage,
    pub logprobs: Option<Vec<TokenLogprob>>,
    pub logprobs_complete: bool,
    pub adaptations: Vec<Adaptation>,
    pub provider_data: Option<JsonObject>,
}

impl Response {
    pub fn validate(&self) -> VResult<()> {
        opt_non_empty(self.id.as_ref(), "Response.id")?;
        non_empty(&self.model, "Response.model")?;
        self.message.validate()?;
        if self.message.role != Role::Assistant {
            return Err(ValidationError::value(
                "Response.message must have role 'assistant'",
            ));
        }
        self.usage.validate()?;
        self.adaptations.iter().try_for_each(Adaptation::validate)?;
        if let Some(logprobs) = &self.logprobs {
            logprobs.iter().try_for_each(TokenLogprob::validate)?;
        }
        Ok(())
    }

    /// Concatenated assistant text; citation and thinking parts are metadata.
    pub fn text(&self) -> Option<String> {
        if let Some(text) = self.message.text() {
            return Some(text);
        }
        let all_textual = self
            .message
            .parts
            .iter()
            .all(|p| matches!(p, Part::Text(_) | Part::Citation(_) | Part::Thinking(_)));
        if !all_textual {
            return None;
        }
        let texts: Vec<&str> = self
            .message
            .parts
            .iter()
            .filter_map(|p| match p {
                Part::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect();
        if texts.is_empty() {
            None
        } else {
            Some(texts.join("\n"))
        }
    }

    pub fn tool_calls(&self) -> Vec<&ToolCallPart> {
        self.message
            .parts
            .iter()
            .filter_map(|p| match p {
                Part::ToolCall(c) => Some(c),
                _ => None,
            })
            .collect()
    }

    pub fn citations(&self) -> Vec<&CitationPart> {
        self.message
            .parts
            .iter()
            .filter_map(|p| match p {
                Part::Citation(c) => Some(c),
                _ => None,
            })
            .collect()
    }

    /// The first structured answer part, with its measurement metadata.
    pub fn data_part(&self) -> Option<&DataPart> {
        self.message.parts.iter().find_map(|part| match part {
            Part::Data(data) => Some(data),
            _ => None,
        })
    }

    /// A structured value, falling back to JSON text for ordinary schemas.
    pub fn data(&self) -> Option<serde_json::Value> {
        self.json()
    }

    pub fn probabilities(&self) -> Option<&JsonObject> {
        self.data_part()?.probabilities.as_ref()
    }

    pub fn method(&self) -> Option<JudgmentMethod> {
        self.data_part()?.method
    }

    pub fn expected_level(&self, field: &str) -> Option<f64> {
        self.data_part()?.expected_level(field)
    }

    pub fn expected(&self, field: &str) -> Option<f64> {
        self.expected_level(field)
    }

    /// Structured value (including null), otherwise parsed response text.
    pub fn json(&self) -> Option<serde_json::Value> {
        if let Some(data) = self.data_part() {
            return Some(data.value.clone());
        }
        serde_json::from_str(self.text()?.trim()).ok()
    }

    pub fn parse_json(&self, default: serde_json::Value) -> serde_json::Value {
        self.json().unwrap_or(default)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{FunctionTool, ToolChoice, ToolChoiceMode};

    fn tool(name: &str) -> Tool {
        Tool::Function(FunctionTool {
            name: name.into(),
            ..Default::default()
        })
    }

    #[test]
    fn inv_031_allowed_subset_of_tools() {
        let mut req = Request::new("m", vec![Message::user("hi").unwrap()]).unwrap();
        req.tools = vec![tool("lookup")];
        req.config.tool_choice = Some(ToolChoice {
            mode: ToolChoiceMode::Required,
            allowed: vec!["other".into()],
            parallel: None,
        });
        assert!(req.validate().is_err());
        req.config.tool_choice.as_mut().unwrap().allowed = vec!["lookup".into()];
        assert!(req.validate().is_ok());
        req.tools.push(tool("lookup"));
        assert!(req.validate().is_err(), "INV-030 duplicate names");
    }

    #[test]
    fn request_requires_model_and_messages() {
        assert!(Request::new("", vec![Message::user("hi").unwrap()]).is_err());
        assert!(Request::new("m", vec![]).is_err());
    }

    #[test]
    fn inv_036_response_message_is_assistant() {
        let response = Response {
            id: None,
            model: "m".into(),
            message: Message::user("hi").unwrap(),
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
            logprobs: None,
            logprobs_complete: true,
            adaptations: Vec::new(),
            provider_data: None,
        };
        assert!(response.validate().is_err());
        let ok = Response {
            message: Message::assistant(vec![Part::text("a"), Part::thinking("t")]).unwrap(),
            ..response
        };
        assert!(ok.validate().is_ok());
        assert_eq!(ok.text(), Some("a".to_string()));
    }
}

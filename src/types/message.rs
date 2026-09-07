//! Message (types.md § Messages; INV-020..025).

use super::continuation::{validate_continuation, ContinuationState};
use super::json::{VResult, ValidationError};
use super::parts::{ContentInput, Part, ToolResultPart};
use super::vocab::Role;

/// A contribution to a conversation, attributed to a speaker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub parts: Vec<Part>,
    pub continuation: Vec<ContinuationState>,
}

/// `Request.system` / `LiveConfig.system`: a non-empty string or prompt parts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SystemContent {
    Text(String),
    Parts(Vec<Part>),
}

impl SystemContent {
    /// INV-021/INV-024: strings normalize, protocol parts are rejected.
    pub fn validate(&self) -> VResult<()> {
        match self {
            SystemContent::Text(text) => {
                if text.is_empty() {
                    return Err(ValidationError::value("system cannot be empty"));
                }
                Ok(())
            }
            SystemContent::Parts(parts) => {
                if parts.is_empty() {
                    return Err(ValidationError::value("content sequence cannot be empty"));
                }
                for part in parts {
                    part.validate()?;
                    if part.is_prompt_forbidden() {
                        return Err(ValidationError::type_error(
                            "system parts cannot contain model/tool protocol parts",
                        ));
                    }
                }
                Ok(())
            }
        }
    }
}

impl From<&str> for SystemContent {
    fn from(value: &str) -> Self {
        SystemContent::Text(value.to_string())
    }
}

impl From<String> for SystemContent {
    fn from(value: String) -> Self {
        SystemContent::Text(value)
    }
}

impl From<Vec<Part>> for SystemContent {
    fn from(value: Vec<Part>) -> Self {
        SystemContent::Parts(value)
    }
}

impl Message {
    pub fn new(role: Role, parts: Vec<Part>) -> VResult<Self> {
        let message = Message {
            role,
            parts,
            continuation: Vec::new(),
        };
        message.validate()?;
        Ok(message)
    }

    /// A user message: prompt parts only (INV-021, INV-024).
    pub fn user(content: impl Into<ContentInput>) -> VResult<Self> {
        Message::new(Role::User, Part::normalize_content(content.into())?)
    }

    /// A developer message: prompt parts only.
    pub fn developer(content: impl Into<ContentInput>) -> VResult<Self> {
        Message::new(Role::Developer, Part::normalize_content(content.into())?)
    }

    /// An assistant message: no ToolResultPart (INV-023).
    pub fn assistant(content: impl Into<ContentInput>) -> VResult<Self> {
        Message::new(Role::Assistant, Part::normalize_content(content.into())?)
    }

    /// One tool result: `Message::tool(&call.id, result)` (INV-025 single form).
    pub fn tool(call_id: &str, output: impl Into<ContentInput>) -> VResult<Self> {
        Message::new(Role::Tool, vec![Part::tool_result(call_id, output)?])
    }

    /// Several tool results at once, in order (INV-025 map form).
    pub fn tool_results<I, C>(results: I) -> VResult<Self>
    where
        I: IntoIterator<Item = (String, C)>,
        C: Into<ContentInput>,
    {
        let parts = results
            .into_iter()
            .map(|(id, output)| Part::tool_result(id, output))
            .collect::<VResult<Vec<Part>>>()?;
        Message::new(Role::Tool, parts)
    }

    /// Explicit ToolResultParts.
    pub fn tool_parts(parts: Vec<ToolResultPart>) -> VResult<Self> {
        Message::new(
            Role::Tool,
            parts.into_iter().map(Part::ToolResult).collect(),
        )
    }

    pub fn validate(&self) -> VResult<()> {
        if self.parts.is_empty() {
            return Err(ValidationError::value("Message requires at least one part"));
        }
        for part in &self.parts {
            part.validate()?;
        }
        validate_continuation(&self.continuation)?;
        validate_message_parts(self.role, &self.parts)
    }

    /// Text only when every part is a TextPart, joined with `\n`.
    pub fn text(&self) -> Option<String> {
        let mut texts = Vec::with_capacity(self.parts.len());
        for part in &self.parts {
            match part {
                Part::Text(p) => texts.push(p.text.as_str()),
                _ => return None,
            }
        }
        Some(texts.join("\n"))
    }
}

fn validate_message_parts(role: Role, parts: &[Part]) -> VResult<()> {
    match role {
        Role::Tool => {
            if parts.iter().any(|p| !matches!(p, Part::ToolResult(_))) {
                return Err(ValidationError::type_error(
                    "tool messages may only contain ToolResultPart objects",
                ));
            }
        }
        Role::Assistant => {
            if parts.iter().any(|p| matches!(p, Part::ToolResult(_))) {
                return Err(ValidationError::type_error(
                    "assistant messages cannot contain ToolResultPart objects",
                ));
            }
        }
        Role::User | Role::Developer => {
            if parts.iter().any(Part::is_prompt_forbidden) {
                return Err(ValidationError::type_error(format!(
                    "{role} messages cannot contain model/tool protocol parts"
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::JsonObject;

    #[test]
    fn inv_020_021_factories_normalize_content() {
        let m = Message::user("hi").unwrap();
        assert_eq!(m.parts, vec![Part::text("hi")]);
        assert_eq!(m.text(), Some("hi".to_string()));
        assert!(Message::user(Vec::<Part>::new()).is_err());
        assert!(Message::new(Role::User, vec![]).is_err());
    }

    #[test]
    fn inv_022_tool_messages_only_tool_results() {
        assert!(Message::new(Role::Tool, vec![Part::text("x")]).is_err());
        assert!(Message::tool("call_1", "ok").is_ok());
    }

    #[test]
    fn inv_023_assistant_never_tool_result() {
        let result = Part::tool_result("c", "ok").unwrap();
        assert!(Message::assistant(result).is_err());
        let call = Part::tool_call("c", "f", JsonObject::new()).unwrap();
        assert!(Message::assistant(call).is_ok());
    }

    #[test]
    fn inv_024_prompts_reject_protocol_parts() {
        let call = Part::tool_call("c", "f", JsonObject::new()).unwrap();
        assert!(Message::user(call.clone()).is_err());
        assert!(Message::developer(call).is_err());
        let cite = Part::Citation(
            super::super::parts::CitationPart::new(Some("https://a"), None::<&str>, None::<&str>)
                .unwrap(),
        );
        assert!(Message::user(cite).is_err());
        assert!(SystemContent::Text(String::new()).validate().is_err());
        assert!(SystemContent::Parts(vec![Part::thinking("x")])
            .validate()
            .is_err());
    }

    #[test]
    fn inv_025_tool_results_map_form_keeps_order() {
        let m =
            Message::tool_results(vec![("a".to_string(), "1"), ("b".to_string(), "2")]).unwrap();
        let ids: Vec<&str> = m
            .parts
            .iter()
            .map(|p| match p {
                Part::ToolResult(r) => r.id.as_str(),
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(ids, vec!["a", "b"]);
    }
}

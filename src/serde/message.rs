//! Message, tools, ToolChoice (serde.py `message_*`, `tool_*`).

use serde_json::Value;

use super::helpers::{strings, Obj, Reader, VResult};
use super::parts::{continuation_from_json, continuation_to_json, parts_from_list, parts_to_json};
use super::{impl_serde_via_canonical, Canonical};
use crate::types::{
    BuiltinTool, FunctionTool, Message, Part, Role, SystemContent, Tool, ToolChoice,
    ToolChoiceMode, ValidationError,
};

impl Canonical for Message {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "Message")?;
        let role_text = r.req_str("role")?;
        // INV-047: non-object part entries become text; an empty or
        // missing `parts` is rejected with a role-naming error.
        let parts = parts_from_list(r.array_or_empty("parts")?)?;
        if parts.is_empty() {
            return Err(ValidationError::value(format!(
                "message for role '{role_text}' has no parts"
            )));
        }
        let message = Message {
            role: Role::parse(&role_text)?,
            parts,
            continuation: continuation_from_json(&r)?,
        };
        message.validate()?;
        Ok(message)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::new();
        o.set("role", self.role.as_str())
            .set("parts", parts_to_json(&self.parts));
        continuation_to_json(&mut o, &self.continuation);
        o.finish()
    }
}

/// `system`: a string, a list of parts, or absent.
pub(crate) fn system_from_json(r: &Reader<'_>) -> VResult<Option<SystemContent>> {
    match r.get("system") {
        None => Ok(None),
        Some(Value::String(s)) => Ok(Some(SystemContent::Text(s.clone()))),
        Some(Value::Array(items)) => Ok(Some(SystemContent::Parts(
            items
                .iter()
                .map(Part::from_json)
                .collect::<VResult<Vec<Part>>>()?,
        ))),
        Some(_) => Err(ValidationError::type_error(
            "system must be a string, Part, or sequence of Parts",
        )),
    }
}

pub(crate) fn system_to_json(system: &SystemContent) -> Value {
    match system {
        SystemContent::Text(s) => Value::String(s.clone()),
        SystemContent::Parts(parts) => parts_to_json(parts),
    }
}

pub(crate) fn tools_from_json(r: &Reader<'_>) -> VResult<Vec<Tool>> {
    r.array_or_empty("tools")?
        .iter()
        .map(Tool::from_json)
        .collect()
}

pub(crate) fn tools_to_json(tools: &[Tool]) -> Value {
    Value::Array(tools.iter().map(Canonical::to_json).collect())
}

impl Canonical for Tool {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "Tool")?;
        // INV-034: "builtin" → BuiltinTool; any other or missing type →
        // FunctionTool.
        let tool = if r.get("type").and_then(Value::as_str) == Some("builtin") {
            Tool::Builtin(BuiltinTool {
                name: r.req_str("name")?,
                config: r.opt_object("config")?,
            })
        } else {
            let parameters = match r.get("parameters") {
                None => FunctionTool::default_parameters(),
                Some(Value::Object(o)) => o.clone(),
                Some(_) => {
                    return Err(ValidationError::type_error(
                        "parameters must be a JSON object",
                    ))
                }
            };
            Tool::Function(FunctionTool {
                name: r.req_str("name")?,
                description: r.opt_str("description")?,
                parameters,
            })
        };
        tool.validate()?;
        Ok(tool)
    }

    fn to_json(&self) -> Value {
        match self {
            Tool::Function(t) => {
                let mut o = Obj::typed("function");
                o.set("name", t.name.as_str());
                o.omit_empty_opt("description", t.description.clone());
                // INV-033: required-with-shape; `{}` round-trips verbatim.
                o.set("parameters", Value::Object(t.parameters.clone()));
                o.finish()
            }
            Tool::Builtin(t) => {
                let mut o = Obj::typed("builtin");
                o.set("name", t.name.as_str());
                o.omit_empty_opt("config", t.config.clone().map(Value::Object));
                o.finish()
            }
        }
    }
}

impl Canonical for ToolChoice {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "ToolChoice")?;
        let choice = ToolChoice {
            mode: ToolChoiceMode::parse(&r.str_or("mode", "auto")?)?,
            allowed: r.str_list("allowed")?,
            parallel: r.opt_bool("parallel")?,
        };
        choice.validate()?;
        Ok(choice)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::new();
        o.set("mode", self.mode.as_str());
        o.omit_empty("allowed", strings(&self.allowed));
        o.opt("parallel", self.parallel);
        o.finish()
    }
}

impl_serde_via_canonical!(Message, Tool, ToolChoice);

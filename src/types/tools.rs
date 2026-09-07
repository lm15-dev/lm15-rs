//! Tools (types.md § Tools; INV-030, INV-033, INV-034).

use serde_json::{json, Value};

use super::json::{non_empty, JsonObject, VResult, ValidationError};

/// A function the model may call. `parameters` is an opaque JSON Schema,
/// required-with-shape: always emitted, `{}` round-trips verbatim (INV-033).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionTool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: JsonObject,
}

/// A provider-native tool (`web_search`, `code_execution`, ...).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuiltinTool {
    pub name: String,
    pub config: Option<JsonObject>,
}

/// vocabularies.md: `"type": "builtin"` is a BuiltinTool, anything else a
/// FunctionTool (INV-034).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Tool {
    Function(FunctionTool),
    Builtin(BuiltinTool),
}

impl FunctionTool {
    /// The default schema: `{"type": "object", "properties": {}}`.
    pub fn default_parameters() -> JsonObject {
        match json!({"type": "object", "properties": {}}) {
            Value::Object(map) => map,
            _ => unreachable!("literal is an object"),
        }
    }

    pub fn new(
        name: impl Into<String>,
        description: Option<String>,
        parameters: JsonObject,
    ) -> VResult<Self> {
        let tool = FunctionTool {
            name: name.into(),
            description,
            parameters,
        };
        tool.validate()?;
        Ok(tool)
    }

    pub fn validate(&self) -> VResult<()> {
        non_empty(&self.name, "FunctionTool.name")
    }
}

impl Default for FunctionTool {
    fn default() -> Self {
        FunctionTool {
            name: String::new(),
            description: None,
            parameters: FunctionTool::default_parameters(),
        }
    }
}

impl BuiltinTool {
    pub fn new(name: impl Into<String>, config: Option<JsonObject>) -> VResult<Self> {
        let tool = BuiltinTool {
            name: name.into(),
            config,
        };
        tool.validate()?;
        Ok(tool)
    }

    pub fn validate(&self) -> VResult<()> {
        non_empty(&self.name, "BuiltinTool.name")
    }
}

impl Tool {
    pub fn name(&self) -> &str {
        match self {
            Tool::Function(t) => &t.name,
            Tool::Builtin(t) => &t.name,
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Tool::Function(_) => "function",
            Tool::Builtin(_) => "builtin",
        }
    }

    pub fn validate(&self) -> VResult<()> {
        match self {
            Tool::Function(t) => t.validate(),
            Tool::Builtin(t) => t.validate(),
        }
    }
}

/// INV-030: tool names are unique within a request.
pub(crate) fn validate_tools(owner: &str, tools: &[Tool]) -> VResult<()> {
    let mut names: Vec<&str> = Vec::with_capacity(tools.len());
    for tool in tools {
        tool.validate()?;
        if names.contains(&tool.name()) {
            return Err(ValidationError::value(format!(
                "{owner}.tools cannot contain duplicate tool names"
            )));
        }
        names.push(tool.name());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inv_030_duplicate_tool_names_rejected() {
        let a = Tool::Function(
            FunctionTool::new("lookup", None, FunctionTool::default_parameters()).unwrap(),
        );
        let b = Tool::Builtin(BuiltinTool::new("lookup", None).unwrap());
        assert!(validate_tools("Request", &[a.clone(), b]).is_err());
        assert!(validate_tools("Request", &[a]).is_ok());
        assert!(FunctionTool::new("", None, JsonObject::new()).is_err());
    }
}

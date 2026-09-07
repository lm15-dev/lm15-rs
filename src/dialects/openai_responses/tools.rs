//! Tools and `tool_choice` (`lm15/providers/openai.py:119-139,289-294,
//! 778-808,846-857`; MAP-8; spec/types.md § ToolChoice, kind-aware name
//! resolution).

use serde_json::{json, Map, Value};

use crate::compat::{IncludeOmit, OpenAIResponsesBuiltinTools, ResolvedOpenAIResponsesCompat};
use crate::types::{BuiltinTool, Request, Tool, ToolChoiceMode};

/// Canonical builtin name → OpenAI Responses tool type
/// (`lm15/providers/openai.py:120-126` `_OPENAI_BUILTIN_MAP`).
const OPENAI_BUILTIN_MAP: &[(&str, &str)] = &[
    ("web_search", "web_search_preview"),
    ("code_execution", "code_interpreter"),
    ("file_search", "file_search"),
    ("computer_use", "computer_use_preview"),
];

/// `lm15/providers/openai.py:139-140` `_builtin_type`: the compat table
/// for the name, else the name verbatim (the server refuses an unknown
/// type loudly — Meta and Moonshot HTTP 400, live 2026-09-03).
pub fn builtin_type<'a>(tool: &'a BuiltinTool, compat: &ResolvedOpenAIResponsesCompat) -> &'a str {
    let table: &[(&str, &str)] = match compat.builtin_tools {
        OpenAIResponsesBuiltinTools::OpenAI => OPENAI_BUILTIN_MAP,
        OpenAIResponsesBuiltinTools::Verbatim => &[],
    };
    table
        .iter()
        .find(|(name, _)| *name == tool.name)
        .map(|(_, wire)| *wire)
        .unwrap_or(&tool.name)
}

/// The `tools` array (`openai.py:846-857`).
pub fn tools_payload(tools: &[Tool], compat: &ResolvedOpenAIResponsesCompat) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| match tool {
            Tool::Function(function) => {
                let mut payload = Map::new();
                payload.insert("type".into(), Value::String("function".into()));
                payload.insert("name".into(), Value::String(function.name.clone()));
                // The reference sends `description: null` when absent; the
                // schema takes it and the bytes match.
                payload.insert(
                    "description".into(),
                    function
                        .description
                        .clone()
                        .map(Value::String)
                        .unwrap_or(Value::Null),
                );
                payload.insert(
                    "parameters".into(),
                    Value::Object(function.parameters.clone()),
                );
                if compat.strict_tools == IncludeOmit::Include {
                    payload.insert("strict".into(), Value::Bool(false));
                }
                Value::Object(payload)
            }
            Tool::Builtin(builtin) => builtin_payload(builtin, compat),
        })
        .collect()
}

/// `openai.py:289-294` `_builtin_to_openai`: the type, then the config
/// keys verbatim.
fn builtin_payload(tool: &BuiltinTool, compat: &ResolvedOpenAIResponsesCompat) -> Value {
    let mut payload = Map::new();
    payload.insert(
        "type".into(),
        Value::String(builtin_type(tool, compat).into()),
    );
    if let Some(config) = &tool.config {
        for (key, value) in config {
            payload.insert(key.clone(), value.clone());
        }
    }
    Value::Object(payload)
}

/// `openai.py:778-808` `_tool_choice_payload`: `None` when the request
/// has no tool choice.
pub fn tool_choice_payload(
    request: &Request,
    compat: &ResolvedOpenAIResponsesCompat,
) -> Option<Value> {
    let choice = request.config.tool_choice.as_ref()?;
    if choice.mode == ToolChoiceMode::None {
        return Some(json!("none"));
    }
    if !choice.allowed.is_empty() {
        // INV-031 guarantees every allowed name is declared. A single
        // name under `required` is the forced form — function or hosted
        // tool (live 2026-09-01); anything else is `allowed_tools`, which
        // keeps `auto` honest.
        let entries: Vec<&Tool> = choice
            .allowed
            .iter()
            .filter_map(|name| request.tools.iter().find(|tool| tool.name() == name))
            .collect();
        let forced = |tool: &Tool| match tool {
            Tool::Builtin(builtin) => json!({"type": builtin_type(builtin, compat)}),
            Tool::Function(function) => json!({"type": "function", "name": function.name}),
        };
        if entries.len() == 1 && choice.mode == ToolChoiceMode::Required {
            return Some(forced(entries[0]));
        }
        let tools: Vec<Value> = entries.into_iter().map(forced).collect();
        return Some(json!({
            "type": "allowed_tools",
            "mode": choice.mode.as_str(),
            "tools": tools,
        }));
    }
    Some(json!(choice.mode.as_str()))
}

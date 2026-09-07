//! OpenAI Responses compat (`lm15/compat.py:48-297`, `:1047-1118`). Data
//! only: the Responses dialect (module 4 W2) consults the resolved value.

use super::{IncludeOmit, JsonObject, Knob, OpenAICacheControl};

/// `lm15/compat.py:50` `OpenAIResponsesDeveloperRole`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpenAIResponsesDeveloperRole {
    Developer,
    System,
}

impl OpenAIResponsesDeveloperRole {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "developer" => Some(OpenAIResponsesDeveloperRole::Developer),
            "system" => Some(OpenAIResponsesDeveloperRole::System),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            OpenAIResponsesDeveloperRole::Developer => "developer",
            OpenAIResponsesDeveloperRole::System => "system",
        }
    }
}

/// `lm15/compat.py:51-56` `OpenAIResponsesMaxOutputTokensField`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpenAIResponsesMaxOutputTokensField {
    MaxOutputTokens,
    MaxCompletionTokens,
    MaxTokens,
}

impl OpenAIResponsesMaxOutputTokensField {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "max_output_tokens" => Some(OpenAIResponsesMaxOutputTokensField::MaxOutputTokens),
            "max_completion_tokens" => {
                Some(OpenAIResponsesMaxOutputTokensField::MaxCompletionTokens)
            }
            "max_tokens" => Some(OpenAIResponsesMaxOutputTokensField::MaxTokens),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            OpenAIResponsesMaxOutputTokensField::MaxOutputTokens => "max_output_tokens",
            OpenAIResponsesMaxOutputTokensField::MaxCompletionTokens => "max_completion_tokens",
            OpenAIResponsesMaxOutputTokensField::MaxTokens => "max_tokens",
        }
    }
}

/// `lm15/compat.py:57-67` `OpenAIResponsesReasoningFormat`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpenAIResponsesReasoningFormat {
    None,
    ResponsesReasoning,
    ReasoningEffort,
    Openrouter,
    Deepseek,
    Qwen,
    QwenChatTemplate,
    Zai,
}

impl OpenAIResponsesReasoningFormat {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "none" => Some(OpenAIResponsesReasoningFormat::None),
            "responses_reasoning" => Some(OpenAIResponsesReasoningFormat::ResponsesReasoning),
            "reasoning_effort" => Some(OpenAIResponsesReasoningFormat::ReasoningEffort),
            "openrouter" => Some(OpenAIResponsesReasoningFormat::Openrouter),
            "deepseek" => Some(OpenAIResponsesReasoningFormat::Deepseek),
            "qwen" => Some(OpenAIResponsesReasoningFormat::Qwen),
            "qwen_chat_template" => Some(OpenAIResponsesReasoningFormat::QwenChatTemplate),
            "zai" => Some(OpenAIResponsesReasoningFormat::Zai),
            _ => None,
        }
    }
}

/// `lm15/compat.py:86` `OpenAIResponsesCommentaryPhase` (marked for
/// demotion, api-family.md § Providers, direct).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpenAIResponsesCommentaryPhase {
    Omit,
    Tag,
}

impl OpenAIResponsesCommentaryPhase {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "omit" => Some(OpenAIResponsesCommentaryPhase::Omit),
            "tag" => Some(OpenAIResponsesCommentaryPhase::Tag),
            _ => None,
        }
    }
}

/// `lm15/compat.py:91` `OpenAIResponsesEditImageField` (marked for demotion).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpenAIResponsesEditImageField {
    Array,
    Indexed,
}

impl OpenAIResponsesEditImageField {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "array" => Some(OpenAIResponsesEditImageField::Array),
            "indexed" => Some(OpenAIResponsesEditImageField::Indexed),
            _ => None,
        }
    }
}

/// `lm15/compat.py:101` `OpenAIResponsesBuiltinTools`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpenAIResponsesBuiltinTools {
    /// OpenAI's Responses vocabulary (`web_search_preview`, ...).
    OpenAI,
    /// The canonical name IS the wire type (Meta, Moonshot).
    Verbatim,
}

impl OpenAIResponsesBuiltinTools {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "openai" => Some(OpenAIResponsesBuiltinTools::OpenAI),
            "verbatim" => Some(OpenAIResponsesBuiltinTools::Verbatim),
            _ => None,
        }
    }
}

/// The shared two-value knob's JSON spelling (`lm15/compat.py:68-69`).
fn parse_include_omit(value: &str) -> Option<IncludeOmit> {
    match value {
        "include" => Some(IncludeOmit::Include),
        "omit" => Some(IncludeOmit::Omit),
        _ => None,
    }
}

/// `lm15/compat.py:78` `OpenAICacheControl`.
fn parse_cache_control(value: &str) -> Option<OpenAICacheControl> {
    match value {
        "none" => Some(OpenAICacheControl::None),
        "openai" => Some(OpenAICacheControl::OpenAI),
        "openai_implicit" => Some(OpenAICacheControl::OpenAIImplicit),
        "anthropic" => Some(OpenAICacheControl::Anthropic),
        _ => None,
    }
}

/// Partial OpenAI Responses compat (`lm15/compat.py:104-153`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct OpenAIResponsesCompat {
    pub developer_role: Option<Knob<OpenAIResponsesDeveloperRole>>,
    pub max_output_tokens_field: Option<Knob<OpenAIResponsesMaxOutputTokensField>>,
    pub reasoning_format: Option<Knob<OpenAIResponsesReasoningFormat>>,
    pub tool_result_name: Option<Knob<IncludeOmit>>,
    pub strict_tools: Option<Knob<IncludeOmit>>,
    pub cache_control: Option<Knob<OpenAICacheControl>>,
    pub commentary_phase: Option<Knob<OpenAIResponsesCommentaryPhase>>,
    pub edit_image_field: Option<Knob<OpenAIResponsesEditImageField>>,
    pub builtin_tools: Option<Knob<OpenAIResponsesBuiltinTools>>,
    pub routing: Option<JsonObject>,
    pub extensions: Option<JsonObject>,
}

impl OpenAIResponsesCompat {
    pub const EMPTY: OpenAIResponsesCompat = OpenAIResponsesCompat {
        developer_role: None,
        max_output_tokens_field: None,
        reasoning_format: None,
        tool_result_name: None,
        strict_tools: None,
        cache_control: None,
        commentary_phase: None,
        edit_image_field: None,
        builtin_tools: None,
        routing: None,
        extensions: None,
    };

    /// The named preset (`lm15/compat.py:142-153`; aliases accepted).
    pub fn preset(name: &str) -> Option<&'static OpenAIResponsesCompat> {
        super::find(OPENAI_RESPONSES_PRESETS, name)
    }

    /// `lm15/compat.py:1047-1065` `merge_openai_responses_compat`: `None`
    /// fields inherit; set fields (including `Auto`) override; the two
    /// `extensions` objects merge key-wise, `over` winning.
    pub fn merge(&self, over: &OpenAIResponsesCompat) -> OpenAIResponsesCompat {
        OpenAIResponsesCompat {
            developer_role: over.developer_role.or(self.developer_role),
            max_output_tokens_field: over
                .max_output_tokens_field
                .or(self.max_output_tokens_field),
            reasoning_format: over.reasoning_format.or(self.reasoning_format),
            tool_result_name: over.tool_result_name.or(self.tool_result_name),
            strict_tools: over.strict_tools.or(self.strict_tools),
            cache_control: over.cache_control.or(self.cache_control),
            commentary_phase: over.commentary_phase.or(self.commentary_phase),
            edit_image_field: over.edit_image_field.or(self.edit_image_field),
            builtin_tools: over.builtin_tools.or(self.builtin_tools),
            routing: over.routing.clone().or_else(|| self.routing.clone()),
            extensions: match (&self.extensions, &over.extensions) {
                (None, b) => b.clone(),
                (a, None) => a.clone(),
                (Some(a), Some(b)) => {
                    let mut merged = a.clone();
                    merged.extend(b.iter().map(|(k, v)| (k.clone(), v.clone())));
                    Some(merged)
                }
            },
        }
    }

    /// The request-level override (`lm15/profiles.py:146-181`
    /// `openai_responses_compat_from_extensions`): `extensions.
    /// openai_responses_compat`, else `openai_compat`, else
    /// `compat.openai_responses` / `compat.openai`. A non-object is no
    /// override; a knob with an unknown value is an error (the reference's
    /// dataclass takes it and the wire sees garbage — the port refuses).
    pub fn from_extensions(
        extensions: Option<&JsonObject>,
    ) -> Result<Option<OpenAIResponsesCompat>, String> {
        let Some(extensions) = extensions else {
            return Ok(None);
        };
        let raw = extensions
            .get("openai_responses_compat")
            .or_else(|| extensions.get("openai_compat"))
            .or_else(|| {
                extensions.get("compat").and_then(|compat| {
                    let compat = compat.as_object()?;
                    compat
                        .get("openai_responses")
                        .or_else(|| compat.get("openai"))
                })
            });
        match raw.and_then(serde_json::Value::as_object) {
            Some(object) => OpenAIResponsesCompat::from_json(object).map(Some),
            None => Ok(None),
        }
    }

    /// A partial compat from its JSON spelling (the field names and values
    /// of `lm15/compat.py:50-101`); unknown keys are ignored, as in the
    /// reference's `allowed` filter (`lm15/profiles.py:166-180`).
    pub fn from_json(object: &JsonObject) -> Result<OpenAIResponsesCompat, String> {
        fn knob<T: Copy>(
            object: &JsonObject,
            key: &str,
            parse: fn(&str) -> Option<T>,
        ) -> Result<Option<Knob<T>>, String> {
            match object.get(key) {
                None | Some(serde_json::Value::Null) => Ok(None),
                Some(serde_json::Value::String(s)) if s == "auto" => Ok(Some(Knob::Auto)),
                Some(serde_json::Value::String(s)) => parse(s)
                    .map(|v| Some(Knob::Set(v)))
                    .ok_or_else(|| format!("OpenAIResponsesCompat.{key}: unknown value {s:?}")),
                Some(other) => Err(format!(
                    "OpenAIResponsesCompat.{key}: expected a string, got {other}"
                )),
            }
        }
        fn object_field(object: &JsonObject, key: &str) -> Result<Option<JsonObject>, String> {
            match object.get(key) {
                None | Some(serde_json::Value::Null) => Ok(None),
                Some(serde_json::Value::Object(o)) => Ok(Some(o.clone())),
                Some(other) => Err(format!(
                    "OpenAIResponsesCompat.{key}: expected an object, got {other}"
                )),
            }
        }
        Ok(OpenAIResponsesCompat {
            developer_role: knob(
                object,
                "developer_role",
                OpenAIResponsesDeveloperRole::parse,
            )?,
            max_output_tokens_field: knob(
                object,
                "max_output_tokens_field",
                OpenAIResponsesMaxOutputTokensField::parse,
            )?,
            reasoning_format: knob(
                object,
                "reasoning_format",
                OpenAIResponsesReasoningFormat::parse,
            )?,
            tool_result_name: knob(object, "tool_result_name", parse_include_omit)?,
            strict_tools: knob(object, "strict_tools", parse_include_omit)?,
            cache_control: knob(object, "cache_control", parse_cache_control)?,
            commentary_phase: knob(
                object,
                "commentary_phase",
                OpenAIResponsesCommentaryPhase::parse,
            )?,
            edit_image_field: knob(
                object,
                "edit_image_field",
                OpenAIResponsesEditImageField::parse,
            )?,
            builtin_tools: knob(object, "builtin_tools", OpenAIResponsesBuiltinTools::parse)?,
            routing: object_field(object, "routing")?,
            extensions: object_field(object, "extensions")?,
        })
    }

    /// `lm15/compat.py:1068-1118` `resolve_openai_responses_compat`.
    pub fn resolve(&self) -> ResolvedOpenAIResponsesCompat {
        ResolvedOpenAIResponsesCompat {
            developer_role: Knob::resolve(
                self.developer_role,
                OpenAIResponsesDeveloperRole::Developer,
            ),
            max_output_tokens_field: Knob::resolve(
                self.max_output_tokens_field,
                OpenAIResponsesMaxOutputTokensField::MaxOutputTokens,
            ),
            reasoning_format: Knob::resolve(
                self.reasoning_format,
                OpenAIResponsesReasoningFormat::ResponsesReasoning,
            ),
            tool_result_name: Knob::resolve(self.tool_result_name, IncludeOmit::Omit),
            strict_tools: Knob::resolve(self.strict_tools, IncludeOmit::Omit),
            cache_control: Knob::resolve(self.cache_control, OpenAICacheControl::OpenAI),
            commentary_phase: Knob::resolve(
                self.commentary_phase,
                OpenAIResponsesCommentaryPhase::Omit,
            ),
            edit_image_field: Knob::resolve(
                self.edit_image_field,
                OpenAIResponsesEditImageField::Array,
            ),
            builtin_tools: Knob::resolve(self.builtin_tools, OpenAIResponsesBuiltinTools::OpenAI),
            routing: self.routing.clone(),
            extensions: self.extensions.clone(),
        }
    }
}

/// Fully resolved OpenAI Responses compat (`lm15/compat.py:274-297`).
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedOpenAIResponsesCompat {
    pub developer_role: OpenAIResponsesDeveloperRole,
    pub max_output_tokens_field: OpenAIResponsesMaxOutputTokensField,
    pub reasoning_format: OpenAIResponsesReasoningFormat,
    pub tool_result_name: IncludeOmit,
    pub strict_tools: IncludeOmit,
    pub cache_control: OpenAICacheControl,
    pub commentary_phase: OpenAIResponsesCommentaryPhase,
    pub edit_image_field: OpenAIResponsesEditImageField,
    pub builtin_tools: OpenAIResponsesBuiltinTools,
    pub routing: Option<JsonObject>,
    pub extensions: Option<JsonObject>,
}

impl Default for ResolvedOpenAIResponsesCompat {
    fn default() -> Self {
        OpenAIResponsesCompat::EMPTY.resolve()
    }
}

use Knob::Set;
use OpenAIResponsesDeveloperRole::{Developer, System};
use OpenAIResponsesMaxOutputTokensField::{MaxOutputTokens, MaxTokens};
use OpenAIResponsesReasoningFormat as Fmt;

const fn preset(
    developer_role: OpenAIResponsesDeveloperRole,
    max_output_tokens_field: OpenAIResponsesMaxOutputTokensField,
    reasoning_format: OpenAIResponsesReasoningFormat,
    cache_control: OpenAICacheControl,
) -> OpenAIResponsesCompat {
    OpenAIResponsesCompat {
        developer_role: Some(Set(developer_role)),
        max_output_tokens_field: Some(Set(max_output_tokens_field)),
        reasoning_format: Some(Set(reasoning_format)),
        tool_result_name: Some(Set(IncludeOmit::Omit)),
        strict_tools: Some(Set(IncludeOmit::Omit)),
        cache_control: Some(Set(cache_control)),
        ..OpenAIResponsesCompat::EMPTY
    }
}

/// The single-server knobs over a common preset (a const item cannot use
/// struct-update syntax over a value with a destructor).
const fn with(
    mut compat: OpenAIResponsesCompat,
    commentary_phase: Option<OpenAIResponsesCommentaryPhase>,
    edit_image_field: Option<OpenAIResponsesEditImageField>,
    builtin_tools: Option<OpenAIResponsesBuiltinTools>,
) -> OpenAIResponsesCompat {
    if let Some(phase) = commentary_phase {
        compat.commentary_phase = Some(Set(phase));
    }
    if let Some(field) = edit_image_field {
        compat.edit_image_field = Some(Set(field));
    }
    if let Some(tools) = builtin_tools {
        compat.builtin_tools = Some(Set(tools));
    }
    compat
}

/// `lm15/compat.py:159-260` `OPENAI_RESPONSES_PRESETS`.
pub const OPENAI_RESPONSES_PRESETS: &[(&str, OpenAIResponsesCompat)] = &[
    (
        "openai",
        preset(
            Developer,
            MaxOutputTokens,
            Fmt::ResponsesReasoning,
            OpenAICacheControl::OpenAI,
        ),
    ),
    (
        "openrouter",
        preset(
            Developer,
            MaxTokens,
            Fmt::Openrouter,
            OpenAICacheControl::OpenAI,
        ),
    ),
    (
        "ollama",
        preset(System, MaxTokens, Fmt::None, OpenAICacheControl::None),
    ),
    (
        "vllm",
        preset(
            System,
            MaxTokens,
            Fmt::ReasoningEffort,
            OpenAICacheControl::None,
        ),
    ),
    (
        "sglang",
        preset(
            System,
            MaxTokens,
            Fmt::ReasoningEffort,
            OpenAICacheControl::None,
        ),
    ),
    (
        "qwen",
        preset(System, MaxTokens, Fmt::Qwen, OpenAICacheControl::None),
    ),
    (
        "deepseek",
        preset(System, MaxTokens, Fmt::Deepseek, OpenAICacheControl::None),
    ),
    (
        "zai",
        preset(System, MaxTokens, Fmt::Zai, OpenAICacheControl::None),
    ),
    // Meta Model API (`lm15/compat.py:232-242`).
    (
        "meta",
        with(
            preset(
                Developer,
                MaxOutputTokens,
                Fmt::ResponsesReasoning,
                OpenAICacheControl::OpenAIImplicit,
            ),
            Some(OpenAIResponsesCommentaryPhase::Tag),
            Some(OpenAIResponsesEditImageField::Indexed),
            Some(OpenAIResponsesBuiltinTools::Verbatim),
        ),
    ),
    // Moonshot AI over the Responses wire (`lm15/compat.py:251-259`).
    (
        "moonshotai",
        with(
            preset(
                Developer,
                MaxOutputTokens,
                Fmt::ResponsesReasoning,
                OpenAICacheControl::OpenAIImplicit,
            ),
            None,
            None,
            Some(OpenAIResponsesBuiltinTools::Verbatim),
        ),
    ),
];

/// `lm15/compat.py:264-271` `OPENAI_RESPONSES_PRESET_BASE_URLS`.
pub const OPENAI_RESPONSES_PRESET_BASE_URLS: &[(&str, &str)] = &[
    ("openai", "https://api.openai.com/v1"),
    ("openrouter", "https://openrouter.ai/api/v1"),
    ("meta", "https://api.meta.ai/v1"),
    ("moonshotai", "https://api.moonshot.ai/v1"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_empty_partial_and_the_openai_preset_resolve_alike() {
        let empty = OpenAIResponsesCompat::EMPTY.resolve();
        let openai = OpenAIResponsesCompat::preset("openai").unwrap().resolve();
        assert_eq!(empty, openai);
        assert_eq!(empty.builtin_tools, OpenAIResponsesBuiltinTools::OpenAI);
        assert_eq!(empty.edit_image_field, OpenAIResponsesEditImageField::Array);
    }

    #[test]
    fn meta_sets_the_single_provider_knobs() {
        let meta = OpenAIResponsesCompat::preset("meta").unwrap().resolve();
        assert_eq!(meta.commentary_phase, OpenAIResponsesCommentaryPhase::Tag);
        assert_eq!(meta.cache_control, OpenAICacheControl::OpenAIImplicit);
        assert_eq!(meta.builtin_tools, OpenAIResponsesBuiltinTools::Verbatim);
        assert_eq!(
            OpenAIResponsesCompat::preset("responses").unwrap(),
            OpenAIResponsesCompat::preset("openai").unwrap()
        );
        assert_eq!(OPENAI_RESPONSES_PRESETS.len(), 10);
    }
}

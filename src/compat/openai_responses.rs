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

/// `lm15/compat.py:86` `OpenAIResponsesCommentaryPhase` (marked for
/// demotion, api-family.md § Providers, direct).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpenAIResponsesCommentaryPhase {
    Omit,
    Tag,
}

/// `lm15/compat.py:91` `OpenAIResponsesEditImageField` (marked for demotion).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpenAIResponsesEditImageField {
    Array,
    Indexed,
}

/// `lm15/compat.py:101` `OpenAIResponsesBuiltinTools`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpenAIResponsesBuiltinTools {
    /// OpenAI's Responses vocabulary (`web_search_preview`, ...).
    OpenAI,
    /// The canonical name IS the wire type (Meta, Moonshot).
    Verbatim,
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

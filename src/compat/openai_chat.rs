//! OpenAI Chat Completions compat (`lm15/compat.py:300-828`). Data only:
//! the chat dialect (module 4 W3) consults the resolved value, after
//! `for_model` applied the door's per-model overrides.

use super::{IncludeOmit, JsonObject, Knob, OpenAICacheControl, ReasoningEfforts, SendReject};
use crate::types::ReasoningEffort;

/// `lm15/compat.py:309` `OpenAIChatInstructionRole`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpenAIChatInstructionRole {
    Developer,
    System,
}

impl OpenAIChatInstructionRole {
    pub fn as_str(self) -> &'static str {
        match self {
            OpenAIChatInstructionRole::Developer => "developer",
            OpenAIChatInstructionRole::System => "system",
        }
    }
}

/// `lm15/compat.py:310` `OpenAIChatMaxTokensField`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpenAIChatMaxTokensField {
    MaxCompletionTokens,
    MaxTokens,
}

impl OpenAIChatMaxTokensField {
    pub fn as_str(self) -> &'static str {
        match self {
            OpenAIChatMaxTokensField::MaxCompletionTokens => "max_completion_tokens",
            OpenAIChatMaxTokensField::MaxTokens => "max_tokens",
        }
    }
}

/// `lm15/compat.py:311` `OpenAIChatStreamUsage`.
pub type OpenAIChatStreamUsage = IncludeOmit;

/// `lm15/compat.py:312` `OpenAIChatAssistantAfterToolResult`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpenAIChatAssistantAfterToolResult {
    Insert,
    Omit,
}

/// `lm15/compat.py:313` `OpenAIChatThinkingReplay`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpenAIChatThinkingReplay {
    Native,
    AsText,
    Omit,
}

/// `lm15/compat.py:314` `OpenAIChatAssistantReasoningContent`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpenAIChatAssistantReasoningContent {
    IncludeEmpty,
    Omit,
}

/// `lm15/compat.py:325-334` `OpenAIChatThinkingFormat`. `deepseek` names a
/// wire SHAPE (`thinking: {type}` + `reasoning_effort`), not a company.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpenAIChatThinkingFormat {
    None,
    ReasoningEffort,
    Openrouter,
    Deepseek,
    Kimi,
    Qwen,
    QwenChatTemplate,
}

/// `lm15/compat.py:342` `OpenAIChatBuiltinTools`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpenAIChatBuiltinTools {
    /// Raise: the base wire carries function/custom tools only.
    Reject,
    /// Groq's server-executed tool types.
    Groq,
}

/// `lm15/compat.py:350` `OpenAIChatUserField`: which request field carries
/// `Config.user_id`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpenAIChatUserField {
    User,
    UserId,
    SafetyIdentifier,
}

impl OpenAIChatUserField {
    pub fn as_str(self) -> &'static str {
        match self {
            OpenAIChatUserField::User => "user",
            OpenAIChatUserField::UserId => "user_id",
            OpenAIChatUserField::SafetyIdentifier => "safety_identifier",
        }
    }
}

/// `lm15/compat.py:356` `OpenAIChatForcedToolChoice`.
pub type OpenAIChatForcedToolChoice = SendReject;
/// `lm15/compat.py:361` `OpenAIChatJsonSchema`.
pub type OpenAIChatJsonSchema = SendReject;

/// Partial OpenAI Chat Completions compat (`lm15/compat.py:372-469`).
///
/// `model_overrides`: per-model-family overrides on a door that forwards
/// knobs to many vendors' models; `(model-id prefix, knobs)`, the first
/// matching prefix wins, resolved per request by [`OpenAIChatCompat::for_model`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct OpenAIChatCompat {
    pub instruction_role: Option<Knob<OpenAIChatInstructionRole>>,
    pub max_tokens_field: Option<Knob<OpenAIChatMaxTokensField>>,
    pub stream_usage: Option<Knob<OpenAIChatStreamUsage>>,
    pub tool_result_name: Option<Knob<IncludeOmit>>,
    pub assistant_after_tool_result: Option<Knob<OpenAIChatAssistantAfterToolResult>>,
    pub thinking_format: Option<Knob<OpenAIChatThinkingFormat>>,
    pub thinking_replay: Option<Knob<OpenAIChatThinkingReplay>>,
    pub assistant_reasoning_content: Option<Knob<OpenAIChatAssistantReasoningContent>>,
    pub strict_tools: Option<Knob<IncludeOmit>>,
    pub builtin_tools: Option<Knob<OpenAIChatBuiltinTools>>,
    pub cache_control: Option<Knob<OpenAICacheControl>>,
    pub user_field: Option<Knob<OpenAIChatUserField>>,
    pub forced_tool_choice: Option<Knob<OpenAIChatForcedToolChoice>>,
    pub json_schema: Option<Knob<OpenAIChatJsonSchema>>,
    pub reasoning_efforts: Option<ReasoningEfforts>,
    pub routing: Option<JsonObject>,
    pub extensions: Option<JsonObject>,
    pub model_overrides: &'static [(&'static str, ChatModelOverride)],
}

/// The knobs a per-model override may set (`lm15/compat.py:303-307`
/// `_CHAT_OVERRIDABLE`). `None` keeps the door's value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ChatModelOverride {
    pub instruction_role: Option<Knob<OpenAIChatInstructionRole>>,
    pub max_tokens_field: Option<Knob<OpenAIChatMaxTokensField>>,
    pub stream_usage: Option<Knob<OpenAIChatStreamUsage>>,
    pub thinking_format: Option<Knob<OpenAIChatThinkingFormat>>,
    pub thinking_replay: Option<Knob<OpenAIChatThinkingReplay>>,
    pub assistant_reasoning_content: Option<Knob<OpenAIChatAssistantReasoningContent>>,
    pub strict_tools: Option<Knob<IncludeOmit>>,
    pub cache_control: Option<Knob<OpenAICacheControl>>,
    pub user_field: Option<Knob<OpenAIChatUserField>>,
    pub forced_tool_choice: Option<Knob<OpenAIChatForcedToolChoice>>,
    pub json_schema: Option<Knob<OpenAIChatJsonSchema>>,
    pub reasoning_efforts: Option<ReasoningEfforts>,
}

impl ChatModelOverride {
    pub const NONE: ChatModelOverride = ChatModelOverride {
        instruction_role: None,
        max_tokens_field: None,
        stream_usage: None,
        thinking_format: None,
        thinking_replay: None,
        assistant_reasoning_content: None,
        strict_tools: None,
        cache_control: None,
        user_field: None,
        forced_tool_choice: None,
        json_schema: None,
        reasoning_efforts: None,
    };
}

impl OpenAIChatCompat {
    pub const EMPTY: OpenAIChatCompat = OpenAIChatCompat {
        instruction_role: None,
        max_tokens_field: None,
        stream_usage: None,
        tool_result_name: None,
        assistant_after_tool_result: None,
        thinking_format: None,
        thinking_replay: None,
        assistant_reasoning_content: None,
        strict_tools: None,
        builtin_tools: None,
        cache_control: None,
        user_field: None,
        forced_tool_choice: None,
        json_schema: None,
        reasoning_efforts: None,
        routing: None,
        extensions: None,
        model_overrides: &[],
    };

    /// The named preset (`lm15/compat.py:458-469`; aliases accepted).
    pub fn preset(name: &str) -> Option<&'static OpenAIChatCompat> {
        super::find(OPENAI_CHAT_PRESETS, name)
    }

    /// This compat with the first matching `model_overrides` entry applied
    /// (`lm15/compat.py:451-456`); the overrides are consumed.
    pub fn for_model(&self, model: &str) -> OpenAIChatCompat {
        let mut out = self.clone();
        out.model_overrides = &[];
        if let Some((_, knobs)) = self
            .model_overrides
            .iter()
            .find(|(prefix, _)| model.starts_with(prefix))
        {
            macro_rules! apply {
                ($($field:ident),+) => {
                    $(if knobs.$field.is_some() { out.$field = knobs.$field; })+
                };
            }
            apply!(
                instruction_role,
                max_tokens_field,
                stream_usage,
                thinking_format,
                thinking_replay,
                assistant_reasoning_content,
                strict_tools,
                cache_control,
                user_field,
                forced_tool_choice,
                json_schema,
                reasoning_efforts
            );
        }
        out
    }

    /// `lm15/compat.py:778-828` `resolve_openai_chat_compat` over
    /// `_CHAT_AUTO_DEFAULTS`.
    pub fn resolve(&self) -> ResolvedOpenAIChatCompat {
        ResolvedOpenAIChatCompat {
            instruction_role: Knob::resolve(
                self.instruction_role,
                OpenAIChatInstructionRole::System,
            ),
            max_tokens_field: Knob::resolve(
                self.max_tokens_field,
                OpenAIChatMaxTokensField::MaxCompletionTokens,
            ),
            stream_usage: Knob::resolve(self.stream_usage, IncludeOmit::Include),
            tool_result_name: Knob::resolve(self.tool_result_name, IncludeOmit::Omit),
            assistant_after_tool_result: Knob::resolve(
                self.assistant_after_tool_result,
                OpenAIChatAssistantAfterToolResult::Omit,
            ),
            thinking_format: Knob::resolve(
                self.thinking_format,
                OpenAIChatThinkingFormat::ReasoningEffort,
            ),
            // Decision G (2026-09-01): unsigned thinking is replayed as text, never dropped.
            thinking_replay: Knob::resolve(self.thinking_replay, OpenAIChatThinkingReplay::AsText),
            assistant_reasoning_content: Knob::resolve(
                self.assistant_reasoning_content,
                OpenAIChatAssistantReasoningContent::Omit,
            ),
            strict_tools: Knob::resolve(self.strict_tools, IncludeOmit::Omit),
            builtin_tools: Knob::resolve(self.builtin_tools, OpenAIChatBuiltinTools::Reject),
            cache_control: Knob::resolve(self.cache_control, OpenAICacheControl::OpenAI),
            user_field: Knob::resolve(self.user_field, OpenAIChatUserField::User),
            forced_tool_choice: Knob::resolve(self.forced_tool_choice, SendReject::Send),
            json_schema: Knob::resolve(self.json_schema, SendReject::Send),
            reasoning_efforts: self.reasoning_efforts,
            routing: self.routing.clone(),
            extensions: self.extensions.clone(),
        }
    }
}

/// Fully resolved OpenAI Chat compat (`lm15/compat.py:747-775`).
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedOpenAIChatCompat {
    pub instruction_role: OpenAIChatInstructionRole,
    pub max_tokens_field: OpenAIChatMaxTokensField,
    pub stream_usage: OpenAIChatStreamUsage,
    pub tool_result_name: IncludeOmit,
    pub assistant_after_tool_result: OpenAIChatAssistantAfterToolResult,
    pub thinking_format: OpenAIChatThinkingFormat,
    pub thinking_replay: OpenAIChatThinkingReplay,
    pub assistant_reasoning_content: OpenAIChatAssistantReasoningContent,
    pub strict_tools: IncludeOmit,
    pub builtin_tools: OpenAIChatBuiltinTools,
    pub cache_control: OpenAICacheControl,
    pub user_field: OpenAIChatUserField,
    pub forced_tool_choice: OpenAIChatForcedToolChoice,
    pub json_schema: OpenAIChatJsonSchema,
    pub reasoning_efforts: Option<ReasoningEfforts>,
    pub routing: Option<JsonObject>,
    pub extensions: Option<JsonObject>,
}

impl Default for ResolvedOpenAIChatCompat {
    fn default() -> Self {
        OpenAIChatCompat::EMPTY.resolve()
    }
}

use Knob::Set;
use OpenAIChatMaxTokensField::{MaxCompletionTokens, MaxTokens};
use OpenAIChatThinkingFormat as Think;

/// The seven fields every chat preset sets (`lm15/compat.py:499-507`).
const fn preset(
    max_tokens_field: OpenAIChatMaxTokensField,
    thinking_format: OpenAIChatThinkingFormat,
    cache_control: OpenAICacheControl,
) -> OpenAIChatCompat {
    OpenAIChatCompat {
        instruction_role: Some(Set(OpenAIChatInstructionRole::System)),
        max_tokens_field: Some(Set(max_tokens_field)),
        stream_usage: Some(Set(IncludeOmit::Include)),
        thinking_format: Some(Set(thinking_format)),
        tool_result_name: Some(Set(IncludeOmit::Omit)),
        strict_tools: Some(Set(IncludeOmit::Omit)),
        cache_control: Some(Set(cache_control)),
        ..OpenAIChatCompat::EMPTY
    }
}

/// A preset's extra knobs over the seven common ones. A const item cannot
/// use struct-update syntax over a value with a destructor (`extensions`),
/// so the overrides are applied field by field.
const fn with(
    mut compat: OpenAIChatCompat,
    knobs: ChatModelOverride,
    builtin_tools: Option<OpenAIChatBuiltinTools>,
    model_overrides: &'static [(&'static str, ChatModelOverride)],
) -> OpenAIChatCompat {
    if knobs.instruction_role.is_some() {
        compat.instruction_role = knobs.instruction_role;
    }
    if knobs.thinking_replay.is_some() {
        compat.thinking_replay = knobs.thinking_replay;
    }
    if knobs.assistant_reasoning_content.is_some() {
        compat.assistant_reasoning_content = knobs.assistant_reasoning_content;
    }
    if knobs.user_field.is_some() {
        compat.user_field = knobs.user_field;
    }
    if knobs.forced_tool_choice.is_some() {
        compat.forced_tool_choice = knobs.forced_tool_choice;
    }
    if knobs.json_schema.is_some() {
        compat.json_schema = knobs.json_schema;
    }
    if knobs.reasoning_efforts.is_some() {
        compat.reasoning_efforts = knobs.reasoning_efforts;
    }
    if let Some(tools) = builtin_tools {
        compat.builtin_tools = Some(Set(tools));
    }
    compat.model_overrides = model_overrides;
    compat
}

/// `lm15/compat.py:630`, `:657`: gpt-oss ignores both knobs (HTTP 200).
const GPT_OSS_OVERRIDE: ChatModelOverride = ChatModelOverride {
    forced_tool_choice: Some(Set(SendReject::Reject)),
    json_schema: Some(Set(SendReject::Reject)),
    ..ChatModelOverride::NONE
};

/// `lm15/compat.py:498-718` `OPENAI_CHAT_PRESETS`.
pub const OPENAI_CHAT_PRESETS: &[(&str, OpenAIChatCompat)] = &[
    (
        "openai",
        preset(
            MaxCompletionTokens,
            Think::ReasoningEffort,
            OpenAICacheControl::OpenAI,
        ),
    ),
    // ollama / LM Studio: max_tokens, no reasoning dial (`:509-517`).
    (
        "ollama",
        preset(MaxTokens, Think::None, OpenAICacheControl::None),
    ),
    // Groq: server-executed builtin tools (`:520-529`).
    (
        "groq",
        with(
            preset(MaxTokens, Think::ReasoningEffort, OpenAICacheControl::None),
            ChatModelOverride::NONE,
            Some(OpenAIChatBuiltinTools::Groq),
            &[],
        ),
    ),
    // OpenRouter: unified reasoning object; OpenAI-shaped cache_control (`:531-539`).
    (
        "openrouter",
        preset(MaxTokens, Think::Openrouter, OpenAICacheControl::OpenAI),
    ),
    // xAI, pinned live 2026-09-01 against grok-4.6 (`:544-552`).
    (
        "xai",
        preset(MaxTokens, Think::Deepseek, OpenAICacheControl::None),
    ),
    (
        "vllm",
        preset(MaxTokens, Think::ReasoningEffort, OpenAICacheControl::None),
    ),
    (
        "sglang",
        preset(MaxTokens, Think::ReasoningEffort, OpenAICacheControl::None),
    ),
    // DeepSeek (`:580-591`).
    (
        "deepseek",
        with(
            preset(MaxTokens, Think::Deepseek, OpenAICacheControl::None),
            ChatModelOverride {
                thinking_replay: Some(Set(OpenAIChatThinkingReplay::Native)),
                assistant_reasoning_content: Some(Set(
                    OpenAIChatAssistantReasoningContent::IncludeEmpty,
                )),
                user_field: Some(Set(OpenAIChatUserField::UserId)),
                ..ChatModelOverride::NONE
            },
            None,
            &[],
        ),
    ),
    (
        "qwen",
        preset(MaxTokens, Think::Qwen, OpenAICacheControl::None),
    ),
    // Amazon Bedrock's Chat Completions door on bedrock-runtime (`:616-633`).
    (
        "bedrock",
        with(
            preset(
                MaxCompletionTokens,
                Think::ReasoningEffort,
                OpenAICacheControl::None,
            ),
            ChatModelOverride {
                user_field: Some(Set(OpenAIChatUserField::User)),
                forced_tool_choice: Some(Set(SendReject::Send)),
                json_schema: Some(Set(SendReject::Send)),
                ..ChatModelOverride::NONE
            },
            None,
            &[
                ("openai.gpt-oss", GPT_OSS_OVERRIDE),
                (
                    "google.gemma",
                    ChatModelOverride {
                        forced_tool_choice: Some(Set(SendReject::Reject)),
                        ..ChatModelOverride::NONE
                    },
                ),
            ],
        ),
    ),
    // Bedrock-mantle Chat Completions (`:645-659`).
    (
        "bedrock_mantle",
        with(
            preset(
                MaxCompletionTokens,
                Think::ReasoningEffort,
                OpenAICacheControl::None,
            ),
            ChatModelOverride {
                user_field: Some(Set(OpenAIChatUserField::User)),
                forced_tool_choice: Some(Set(SendReject::Send)),
                json_schema: Some(Set(SendReject::Send)),
                ..ChatModelOverride::NONE
            },
            None,
            &[("openai.gpt-oss", GPT_OSS_OVERRIDE)],
        ),
    ),
    // Z.AI (`:660-672`).
    (
        "zai",
        with(
            preset(MaxTokens, Think::Deepseek, OpenAICacheControl::None),
            ChatModelOverride {
                thinking_replay: Some(Set(OpenAIChatThinkingReplay::Native)),
                user_field: Some(Set(OpenAIChatUserField::UserId)),
                forced_tool_choice: Some(Set(SendReject::Reject)),
                json_schema: Some(Set(SendReject::Reject)),
                ..ChatModelOverride::NONE
            },
            None,
            &[],
        ),
    ),
    // Meta Model API (`:684-693`).
    (
        "meta",
        with(
            preset(
                MaxCompletionTokens,
                Think::ReasoningEffort,
                OpenAICacheControl::OpenAIImplicit,
            ),
            ChatModelOverride {
                instruction_role: Some(Set(OpenAIChatInstructionRole::Developer)),
                user_field: Some(Set(OpenAIChatUserField::SafetyIdentifier)),
                ..ChatModelOverride::NONE
            },
            None,
            &[],
        ),
    ),
    // Moonshot AI / Kimi API Platform (`:706-717`).
    (
        "moonshotai",
        with(
            preset(
                MaxCompletionTokens,
                Think::Kimi,
                OpenAICacheControl::OpenAIImplicit,
            ),
            ChatModelOverride {
                thinking_replay: Some(Set(OpenAIChatThinkingReplay::Native)),
                user_field: Some(Set(OpenAIChatUserField::SafetyIdentifier)),
                reasoning_efforts: Some(&[
                    ReasoningEffort::Low,
                    ReasoningEffort::High,
                    ReasoningEffort::Max,
                ]),
                ..ChatModelOverride::NONE
            },
            None,
            &[],
        ),
    ),
];

/// `lm15/compat.py:725-744` `OPENAI_CHAT_PRESET_BASE_URLS`.
pub const OPENAI_CHAT_PRESET_BASE_URLS: &[(&str, &str)] = &[
    ("openai", "https://api.openai.com/v1"),
    ("ollama", "http://localhost:11434/v1"),
    ("groq", "https://api.groq.com/openai/v1"),
    ("openrouter", "https://openrouter.ai/api/v1"),
    ("xai", "https://api.x.ai/v1"),
    ("vllm", "http://localhost:8000/v1"),
    ("sglang", "http://localhost:30000/v1"),
    ("deepseek", "https://api.deepseek.com"),
    ("zai", "https://api.z.ai/api/paas/v4"),
    ("meta", "https://api.meta.ai/v1"),
    ("moonshotai", "https://api.moonshot.ai/v1"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_defaults_match_the_reference_table() {
        let resolved = OpenAIChatCompat::EMPTY.resolve();
        assert_eq!(resolved.instruction_role, OpenAIChatInstructionRole::System);
        assert_eq!(resolved.max_tokens_field, MaxCompletionTokens);
        assert_eq!(resolved.thinking_replay, OpenAIChatThinkingReplay::AsText);
        assert_eq!(resolved.builtin_tools, OpenAIChatBuiltinTools::Reject);
        assert_eq!(resolved.user_field, OpenAIChatUserField::User);
        assert_eq!(resolved.json_schema, SendReject::Send);
    }

    #[test]
    fn model_overrides_apply_by_prefix_and_are_consumed() {
        let bedrock = OpenAIChatCompat::preset("bedrock").unwrap();
        let oss = bedrock.for_model("openai.gpt-oss-20b-1:0");
        assert!(oss.model_overrides.is_empty());
        assert_eq!(oss.resolve().forced_tool_choice, SendReject::Reject);
        assert_eq!(oss.resolve().json_schema, SendReject::Reject);
        let gemma = bedrock.for_model("google.gemma-3-27b").resolve();
        assert_eq!(gemma.forced_tool_choice, SendReject::Reject);
        assert_eq!(gemma.json_schema, SendReject::Send);
        let other = bedrock.for_model("deepseek.v3").resolve();
        assert_eq!(other.forced_tool_choice, SendReject::Send);
        let mantle = OpenAIChatCompat::preset("bedrock-mantle").unwrap();
        assert_eq!(
            mantle
                .for_model("google.gemma")
                .resolve()
                .forced_tool_choice,
            SendReject::Send
        );
    }

    #[test]
    fn presets_carry_the_documented_user_fields() {
        assert_eq!(
            OpenAIChatCompat::preset("deepseek")
                .unwrap()
                .resolve()
                .user_field,
            OpenAIChatUserField::UserId
        );
        assert_eq!(
            OpenAIChatCompat::preset("meta")
                .unwrap()
                .resolve()
                .user_field,
            OpenAIChatUserField::SafetyIdentifier
        );
        assert_eq!(
            OpenAIChatCompat::preset("groq")
                .unwrap()
                .resolve()
                .builtin_tools,
            OpenAIChatBuiltinTools::Groq
        );
        assert_eq!(OPENAI_CHAT_PRESETS.len(), 14);
        assert_eq!(OPENAI_CHAT_PRESET_BASE_URLS.len(), 11);
    }
}

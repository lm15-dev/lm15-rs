//! OpenAI Chat Completions compat (`lm15/compat.py:300-828`). Data only:
//! the chat dialect (module 4 W3) consults the resolved value, after
//! `for_model` applied the door's per-model overrides.

use super::{
    IncludeOmit, JsonObject, Knob, OpenAICacheControl, ReasoningEfforts, SendReject,
    ToolResultMedia,
};

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

/// Named-token scoring is proven only on vLLM >= 0.29 (MAP-14).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum OpenAIChatTokenScoring {
    #[default]
    None,
    LogprobTokenIds,
}
impl OpenAIChatTokenScoring {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::LogprobTokenIds => "logprob_token_ids",
        }
    }
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

/// `lm15/compat.py` `OpenAIChatReasoningOff`: what an explicit
/// reasoning-off becomes. `Send` puts the dial's off word on the wire
/// (MAP-5). `Lowest` is for a model that cannot stop reasoning on a server
/// that accepts the off word and reasons anyway: the lowest level
/// (`reasoning_efforts[0]`, else `low`) is sent and the substitution
/// recorded (MAP-13 §4.2). Ratified 2026-09-26
/// (`changes/2026-09-26-inference-hosts-live.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpenAIChatReasoningOff {
    Send,
    Lowest,
}

impl OpenAIChatReasoningOff {
    pub fn as_str(self) -> &'static str {
        match self {
            OpenAIChatReasoningOff::Send => "send",
            OpenAIChatReasoningOff::Lowest => "lowest",
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
    /// MAP-10: media inside a tool row (`lm15/compat.py` `tool_result_media`).
    pub tool_result_media: Option<Knob<ToolResultMedia>>,
    pub cache_control: Option<Knob<OpenAICacheControl>>,
    pub user_field: Option<Knob<OpenAIChatUserField>>,
    pub forced_tool_choice: Option<Knob<OpenAIChatForcedToolChoice>>,
    pub json_schema: Option<Knob<OpenAIChatJsonSchema>>,
    pub token_scoring: Option<Knob<OpenAIChatTokenScoring>>,
    pub reasoning_off: Option<Knob<OpenAIChatReasoningOff>>,
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
    pub token_scoring: Option<Knob<OpenAIChatTokenScoring>>,
    pub reasoning_off: Option<Knob<OpenAIChatReasoningOff>>,
    pub reasoning_efforts: Option<ReasoningEfforts>,
    pub tool_result_media: Option<Knob<ToolResultMedia>>,
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
        token_scoring: None,
        reasoning_off: None,
        reasoning_efforts: None,
        tool_result_media: None,
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
        tool_result_media: None,
        cache_control: None,
        user_field: None,
        forced_tool_choice: None,
        json_schema: None,
        token_scoring: None,
        reasoning_off: None,
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
                token_scoring,
                reasoning_off,
                reasoning_efforts,
                tool_result_media
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
            // The base Chat wire's tool row takes text only (OpenAI's own schema;
            // a 200 with the image not received on gpt-5.4, 2026-09-07).
            tool_result_media: Knob::resolve(self.tool_result_media, ToolResultMedia::Reject),
            cache_control: Knob::resolve(self.cache_control, OpenAICacheControl::OpenAI),
            user_field: Knob::resolve(self.user_field, OpenAIChatUserField::User),
            forced_tool_choice: Knob::resolve(self.forced_tool_choice, SendReject::Send),
            json_schema: Knob::resolve(self.json_schema, SendReject::Send),
            token_scoring: Knob::resolve(self.token_scoring, OpenAIChatTokenScoring::None),
            reasoning_off: Knob::resolve(self.reasoning_off, OpenAIChatReasoningOff::Send),
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
    pub tool_result_media: ToolResultMedia,
    pub cache_control: OpenAICacheControl,
    pub user_field: OpenAIChatUserField,
    pub forced_tool_choice: OpenAIChatForcedToolChoice,
    pub json_schema: OpenAIChatJsonSchema,
    pub token_scoring: OpenAIChatTokenScoring,
    pub reasoning_off: OpenAIChatReasoningOff,
    pub reasoning_efforts: Option<ReasoningEfforts>,
    pub routing: Option<JsonObject>,
    pub extensions: Option<JsonObject>,
}

impl Default for ResolvedOpenAIChatCompat {
    fn default() -> Self {
        OpenAIChatCompat::EMPTY.resolve()
    }
}

/// The server presets (`lm15/compat.py`), generated from lm15-contract
/// tables/providers.json into `src/generated/tables.rs`; the receipt behind
/// each knob is cited at the reference table.
pub const OPENAI_CHAT_PRESETS: &[(&str, OpenAIChatCompat)] =
    crate::generated::tables::OPENAI_CHAT_PRESETS;

/// The roots the presets supply (`lm15/compat.py`), generated likewise.
pub const OPENAI_CHAT_PRESET_BASE_URLS: &[(&str, &str)] =
    crate::generated::tables::OPENAI_CHAT_PRESET_BASE_URLS;

/// The DeepInfra models measured to honour a forced tool choice (survey of
/// 24, 2026-09-26); each id is a prefix, so a suffixed variant (-0731,
/// -Turbo) inherits its entry. Generated from the deepinfra preset's model
/// overrides (the ones that send a forced tool choice).
pub const DEEPINFRA_FORCED_TOOL_CHOICE: &[&str] =
    crate::generated::tables::DEEPINFRA_FORCED_TOOL_CHOICE;

#[cfg(test)]
mod tests {
    use super::*;
    use OpenAIChatMaxTokensField::MaxCompletionTokens;
    use OpenAIChatThinkingFormat as Think;

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
        assert_eq!(OPENAI_CHAT_PRESETS.len(), 19); // + lmstudio (2026-09-11), + four inference hosts (2026-09-26)
        assert_eq!(OPENAI_CHAT_PRESET_BASE_URLS.len(), 16);
        assert_eq!(
            OpenAIChatCompat::preset("ollama")
                .unwrap()
                .resolve()
                .thinking_format,
            Think::ReasoningEffort
        );
    }
}

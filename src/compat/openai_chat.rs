//! OpenAI Chat Completions compat (`lm15/compat.py:300-828`). Data only:
//! the chat dialect (module 4 W3) consults the resolved value, after
//! `for_model` applied the door's per-model overrides.

use super::{
    IncludeOmit, JsonObject, Knob, OpenAICacheControl, ReasoningEfforts, SendReject,
    ToolResultMedia,
};
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

const fn scoring_preset() -> OpenAIChatCompat {
    let mut value = preset(MaxTokens, Think::ReasoningEffort, OpenAICacheControl::None);
    value.token_scoring = Some(Set(OpenAIChatTokenScoring::LogprobTokenIds));
    value
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
    if knobs.token_scoring.is_some() {
        compat.token_scoring = knobs.token_scoring;
    }
    if knobs.reasoning_off.is_some() {
        compat.reasoning_off = knobs.reasoning_off;
    }
    if knobs.tool_result_media.is_some() {
        compat.tool_result_media = knobs.tool_result_media;
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
    // Ollama translates reasoning_effort to its native think control (MAP-13.7).
    (
        "ollama",
        preset(MaxTokens, Think::ReasoningEffort, OpenAICacheControl::None),
    ),
    // LM Studio: ollama's wire policy (lmstudio.ai docs list the same Chat
    // Completions fields) at its own documented address,
    // http://localhost:1234/v1. Until 2026-09-11 the name was an alias of
    // "ollama" and took ollama's port. No live receipt for the policy yet.
    (
        "lmstudio",
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
        with(
            preset(MaxTokens, Think::Deepseek, OpenAICacheControl::None),
            ChatModelOverride {
                tool_result_media: Some(Set(ToolResultMedia::Images)), // MAP-10: image tool results received live 2026-09-07
                ..ChatModelOverride::NONE
            },
            None,
            &[],
        ),
    ),
    ("vllm", scoring_preset()),
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
                tool_result_media: Some(Set(ToolResultMedia::Images)), // MAP-10: image tool results received live 2026-09-07

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
                tool_result_media: Some(Set(ToolResultMedia::Images)), // MAP-10: image tool results received live 2026-09-07

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
    // ─── Open-model inference hosts (changes/2026-09-26-inference-hosts-live.md) ───
    // One policy for the four, each knob receipted live 2026-09-26: the
    // reasoning_effort dial (Fireworks refuses the `reasoning` object);
    // reasoning replayed as reasoning_content (a planted code word was
    // recalled through it; Fireworks refuses `reasoning`);
    // max_completion_tokens and stream usage honoured; caching automatic, so a
    // key or long retention is dropped with a record.
    //
    // DeepInfra: 422 on media in a tool row; a forced tool choice goes only
    // to the 14 models a survey showed honour it, refused elsewhere (MAP-8,
    // ratified 2026-09-26).
    (
        "deepinfra",
        with(
            inference_host(),
            ChatModelOverride {
                thinking_replay: Some(Set(OpenAIChatThinkingReplay::Native)),
                tool_result_media: Some(Set(ToolResultMedia::Reject)),
                forced_tool_choice: Some(Set(SendReject::Reject)),
                ..ChatModelOverride::NONE
            },
            None,
            DEEPINFRA_OVERRIDES,
        ),
    ),
    // Together, gpt-oss: a forced tool choice answers 500 (retryable: refused
    // before the wire); xhigh/max/unknown words run at medium (clamped,
    // recorded); `none` accepted and reasoning still billed (lowest level
    // instead). GLM-5.3 ignores `none`. Media in tool rows: open cell.
    (
        "together",
        with(
            inference_host(),
            ChatModelOverride {
                thinking_replay: Some(Set(OpenAIChatThinkingReplay::Native)),
                tool_result_media: Some(Set(ToolResultMedia::Reject)),
                ..ChatModelOverride::NONE
            },
            None,
            &[
                (
                    "openai/gpt-oss",
                    ChatModelOverride {
                        forced_tool_choice: Some(Set(SendReject::Reject)),
                        reasoning_efforts: Some(&[
                            ReasoningEffort::Low,
                            ReasoningEffort::Medium,
                            ReasoningEffort::High,
                        ]),
                        reasoning_off: Some(Set(OpenAIChatReasoningOff::Lowest)),
                        ..ChatModelOverride::NONE
                    },
                ),
                ("zai-org/GLM-5.3", REASONING_OFF_LOWEST),
            ],
        ),
    ),
    // MAP-10: images read in a tool result (Fireworks GLM-5.3-Flash, Parasail Qwen3-VL-8B).
    (
        "fireworks",
        with(
            inference_host(),
            ChatModelOverride {
                thinking_replay: Some(Set(OpenAIChatThinkingReplay::Native)),
                tool_result_media: Some(Set(ToolResultMedia::Images)),
                ..ChatModelOverride::NONE
            },
            None,
            &[],
        ),
    ),
    (
        "parasail",
        with(
            inference_host(),
            ChatModelOverride {
                thinking_replay: Some(Set(OpenAIChatThinkingReplay::Native)),
                tool_result_media: Some(Set(ToolResultMedia::Images)),
                ..ChatModelOverride::NONE
            },
            None,
            &[],
        ),
    ),
];

const fn inference_host() -> OpenAIChatCompat {
    preset(
        MaxCompletionTokens,
        Think::ReasoningEffort,
        OpenAICacheControl::None,
    )
}

const REASONING_OFF_LOWEST: ChatModelOverride = ChatModelOverride {
    reasoning_off: Some(Set(OpenAIChatReasoningOff::Lowest)),
    ..ChatModelOverride::NONE
};

const FORCED_TOOL_CHOICE_SEND: ChatModelOverride = ChatModelOverride {
    forced_tool_choice: Some(Set(SendReject::Send)),
    ..ChatModelOverride::NONE
};

/// DeepInfra models measured to honour a forced tool choice (survey of 24,
/// 2026-09-26, `research/providers/deepinfra/tool_choice_survey.py`); each id
/// is a prefix, so a suffixed variant (`-0731`, `-Turbo`) inherits its entry.
pub const DEEPINFRA_FORCED_TOOL_CHOICE: &[&str] = &[
    "deepseek-ai/DeepSeek-V3.2",
    "deepseek-ai/DeepSeek-V4-Flash",
    "deepseek-ai/DeepSeek-V4.1-Flash",
    "zai-org/GLM-5.3-Flash",
    "moonshotai/Kimi-K2.6",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "Qwen/Qwen3.6-27B",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "nvidia/NVIDIA-Nemotron-3.5-Lightning",
    "ibm-granite/granite-4.2-8b",
    "XiaomiMiMo/MiMo-V2.6-Flash",
    "tencent/Hy3",
    "google/gemini-3.1-flash-lite",
    "anthropic/claude-haiku-4-5",
];

/// gpt-oss first (reasoning off → lowest), then every surveyed model.
const DEEPINFRA_OVERRIDES: &[(&str, ChatModelOverride)] = &[
    ("openai/gpt-oss", REASONING_OFF_LOWEST),
    ("deepseek-ai/DeepSeek-V3.2", FORCED_TOOL_CHOICE_SEND),
    ("deepseek-ai/DeepSeek-V4-Flash", FORCED_TOOL_CHOICE_SEND),
    ("deepseek-ai/DeepSeek-V4.1-Flash", FORCED_TOOL_CHOICE_SEND),
    ("zai-org/GLM-5.3-Flash", FORCED_TOOL_CHOICE_SEND),
    ("moonshotai/Kimi-K2.6", FORCED_TOOL_CHOICE_SEND),
    (
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        FORCED_TOOL_CHOICE_SEND,
    ),
    ("Qwen/Qwen3.6-27B", FORCED_TOOL_CHOICE_SEND),
    ("Qwen/Qwen3-Next-80B-A3B-Instruct", FORCED_TOOL_CHOICE_SEND),
    (
        "nvidia/NVIDIA-Nemotron-3.5-Lightning",
        FORCED_TOOL_CHOICE_SEND,
    ),
    ("ibm-granite/granite-4.2-8b", FORCED_TOOL_CHOICE_SEND),
    ("XiaomiMiMo/MiMo-V2.6-Flash", FORCED_TOOL_CHOICE_SEND),
    ("tencent/Hy3", FORCED_TOOL_CHOICE_SEND),
    ("google/gemini-3.1-flash-lite", FORCED_TOOL_CHOICE_SEND),
    ("anthropic/claude-haiku-4-5", FORCED_TOOL_CHOICE_SEND),
];

/// `lm15/compat.py:725-744` `OPENAI_CHAT_PRESET_BASE_URLS`.
pub const OPENAI_CHAT_PRESET_BASE_URLS: &[(&str, &str)] = &[
    ("openai", "https://api.openai.com/v1"),
    ("ollama", "http://localhost:11434/v1"),
    ("lmstudio", "http://localhost:1234/v1"), // lmstudio.ai docs (Local Server)
    ("groq", "https://api.groq.com/openai/v1"),
    ("openrouter", "https://openrouter.ai/api/v1"),
    ("xai", "https://api.x.ai/v1"),
    ("vllm", "http://localhost:8000/v1"),
    ("sglang", "http://localhost:30000/v1"),
    ("deepseek", "https://api.deepseek.com"),
    ("zai", "https://api.z.ai/api/paas/v4"),
    ("meta", "https://api.meta.ai/v1"),
    ("moonshotai", "https://api.moonshot.ai/v1"),
    // The open-model inference hosts, each its documented OpenAI-compatible
    // root (DeepInfra: the /v1/openai root, not /v1).
    ("deepinfra", "https://api.deepinfra.com/v1/openai"),
    ("together", "https://api.together.ai/v1"),
    ("fireworks", "https://api.fireworks.ai/inference/v1"),
    ("parasail", "https://api.parasail.io/v1"),
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

//! Compat values: a server's wire quirks as data (`lm15/compat.py`;
//! spec/auth.md AUTH-10 "for the chat dialect an optional compat value";
//! playbooks/api-family.md rule 7: a knob exists only when the wire has no
//! other way, and never a new one here — a case that needs one is a
//! report).
//!
//! One typed struct per dialect ([`AnthropicCompat`],
//! [`OpenAIResponsesCompat`], [`OpenAIChatCompat`]) with `Option<Knob<_>>`
//! fields — `None` inherits, `Knob::Auto` is the dialect's own behaviour,
//! `Knob::Set(v)` is explicit — and a `Resolved*` form the dialect reads.
//! The preset tables are `const` data copied from the reference with
//! file:line citations; a dialect consults a knob at the named point and
//! nowhere else.

mod anthropic;
mod openai_chat;
mod openai_responses;

pub use anthropic::{
    AnthropicCacheControl, AnthropicCompat, AnthropicParallelToolCalls, AnthropicSamplingParams,
    AnthropicStructuredOutput, AnthropicThinkingFormat, AnthropicThinkingReplay,
    ResolvedAnthropicCompat, ANTHROPIC_PRESETS, ANTHROPIC_PRESET_BASE_URLS,
};
pub use openai_chat::{
    ChatModelOverride, OpenAIChatAssistantAfterToolResult, OpenAIChatAssistantReasoningContent,
    OpenAIChatBuiltinTools, OpenAIChatCompat, OpenAIChatForcedToolChoice,
    OpenAIChatInstructionRole, OpenAIChatJsonSchema, OpenAIChatMaxTokensField,
    OpenAIChatStreamUsage, OpenAIChatThinkingFormat, OpenAIChatThinkingReplay, OpenAIChatUserField,
    ResolvedOpenAIChatCompat, OPENAI_CHAT_PRESETS, OPENAI_CHAT_PRESET_BASE_URLS,
};
pub use openai_responses::{
    OpenAIResponsesBuiltinTools, OpenAIResponsesCommentaryPhase, OpenAIResponsesCompat,
    OpenAIResponsesDeveloperRole, OpenAIResponsesEditImageField,
    OpenAIResponsesMaxOutputTokensField, OpenAIResponsesReasoningFormat,
    ResolvedOpenAIResponsesCompat, OPENAI_RESPONSES_PRESETS, OPENAI_RESPONSES_PRESET_BASE_URLS,
};

use crate::types::ReasoningEffort;

/// A knob value in a partial compat: the dialect's own heuristic, or an
/// explicit policy (`lm15/compat.py:9-13`: `"auto"` is explicit and
/// distinct from `None`, which inherits).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Knob<T> {
    Auto,
    Set(T),
}

impl<T: Copy> Knob<T> {
    /// `None` and `Auto` resolve to the dialect default; `Set` wins.
    pub fn resolve(knob: Option<Knob<T>>, default: T) -> T {
        match knob {
            Some(Knob::Set(value)) => value,
            None | Some(Knob::Auto) => default,
        }
    }
}

/// MAP-10 (`lm15/compat.py` `ToolResultMedia`): what a ToolResultPart's media
/// parts may become on the wire. `Native` = images and documents as native
/// blocks inside the result item; `Images` = images only (documents raise);
/// `Reject` = every media part raises before the wire. Measured per preset:
/// lm15-contract/research/tool-result-content/30-model.md.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ToolResultMedia {
    Native,
    Images,
    Reject,
}

impl ToolResultMedia {
    pub fn as_str(self) -> &'static str {
        match self {
            ToolResultMedia::Native => "native",
            ToolResultMedia::Images => "images",
            ToolResultMedia::Reject => "reject",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "native" => Some(ToolResultMedia::Native),
            "images" => Some(ToolResultMedia::Images),
            "reject" => Some(ToolResultMedia::Reject),
            _ => None,
        }
    }

    /// Whether a part kind (`Part::type_name`) may ride inside a result item.
    pub fn admits(self, kind: &str) -> bool {
        match self {
            ToolResultMedia::Native => matches!(kind, "image" | "document"),
            ToolResultMedia::Images => kind == "image",
            ToolResultMedia::Reject => false,
        }
    }
}

/// The two-value knobs shared by the OpenAI dialects (`lm15/compat.py:68-69`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IncludeOmit {
    Include,
    Omit,
}

impl IncludeOmit {
    pub fn as_str(self) -> &'static str {
        match self {
            IncludeOmit::Include => "include",
            IncludeOmit::Omit => "omit",
        }
    }
}

/// `send` sends the field; `reject` raises before the wire (MAP-8 §2: a
/// silent widen is worse than an error).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SendReject {
    Send,
    Reject,
}

impl SendReject {
    pub fn as_str(self) -> &'static str {
        match self {
            SendReject::Send => "send",
            SendReject::Reject => "reject",
        }
    }
}

/// `lm15/compat.py:78` `OpenAICacheControl` (shared by both OpenAI dialects).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpenAICacheControl {
    None,
    OpenAI,
    OpenAIImplicit,
    Anthropic,
}

impl OpenAICacheControl {
    pub fn as_str(self) -> &'static str {
        match self {
            OpenAICacheControl::None => "none",
            OpenAICacheControl::OpenAI => "openai",
            OpenAICacheControl::OpenAIImplicit => "openai_implicit",
            OpenAICacheControl::Anthropic => "anthropic",
        }
    }
}

/// The server's native reasoning-effort levels when it does NOT refuse the
/// others (`lm15/compat.py:362-369`); `off` is never listed.
pub type ReasoningEfforts = &'static [ReasoningEffort];

/// An opaque JSON object a compat carries (`routing`, `extensions`).
pub type JsonObject = serde_json::Map<String, serde_json::Value>;

/// The compat value bound to an adapter: one per dialect, or none for the
/// Gemini dialect (which takes no compat, as in the reference registry).
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Compat {
    #[default]
    None,
    Anthropic(AnthropicCompat),
    OpenAIResponses(OpenAIResponsesCompat),
    OpenAIChat(OpenAIChatCompat),
}

impl Compat {
    pub fn anthropic(&self) -> Option<&AnthropicCompat> {
        match self {
            Compat::Anthropic(c) => Some(c),
            _ => None,
        }
    }

    pub fn openai_responses(&self) -> Option<&OpenAIResponsesCompat> {
        match self {
            Compat::OpenAIResponses(c) => Some(c),
            _ => None,
        }
    }

    pub fn openai_chat(&self) -> Option<&OpenAIChatCompat> {
        match self {
            Compat::OpenAIChat(c) => Some(c),
            _ => None,
        }
    }
}

/// Preset-name normalization and the permanent spelling aliases
/// (`lm15/compat.py:472-489`). One map serves all three tables: a name
/// means the same server in each.
pub fn preset_key(name: &str) -> String {
    let key: String = name
        .to_ascii_lowercase()
        .chars()
        .map(|c| match c {
            '-' | ' ' | '.' => '_',
            other => other,
        })
        .collect();
    match key.as_str() {
        "openai_chat" | "chat" | "chat_completions" | "responses" | "openai_responses" => {
            "openai".into()
        }
        "lmstudio" | "lm_studio" => "ollama".into(),
        "dashscope_qwen" => "qwen".into(),
        "z_ai" => "zai".into(),
        _ => key,
    }
}

fn find<'a, T>(table: &'a [(&'static str, T)], name: &str) -> Option<&'a T> {
    let key = preset_key(name);
    table
        .iter()
        .find(|(preset, _)| *preset == key)
        .map(|(_, value)| value)
}

/// The base URL a preset name supplies (`lm15/compat.py` `*_PRESET_BASE_URLS`).
pub fn preset_base_url(
    table: &'static [(&'static str, &'static str)],
    name: &str,
) -> Option<&'static str> {
    find(table, name).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_keys_normalize_and_alias() {
        assert_eq!(preset_key("LM-Studio"), "ollama");
        assert_eq!(preset_key("z.ai"), "zai");
        assert_eq!(preset_key("openai_chat"), "openai");
        assert_eq!(preset_key("bedrock-mantle"), "bedrock_mantle");
        assert_eq!(preset_key("groq"), "groq");
    }

    #[test]
    fn knob_resolution() {
        assert_eq!(Knob::resolve(None, 1), 1);
        assert_eq!(Knob::resolve(Some(Knob::Auto), 1), 1);
        assert_eq!(Knob::resolve(Some(Knob::Set(2)), 1), 2);
    }

    #[test]
    fn every_preset_name_resolves_to_a_base_url_when_listed() {
        assert_eq!(
            preset_base_url(OPENAI_CHAT_PRESET_BASE_URLS, "LM Studio"),
            Some("http://localhost:11434/v1")
        );
        assert_eq!(
            preset_base_url(ANTHROPIC_PRESET_BASE_URLS, "anthropic"),
            Some("https://api.anthropic.com/v1")
        );
        assert_eq!(
            preset_base_url(OPENAI_RESPONSES_PRESET_BASE_URLS, "bedrock"),
            None
        );
    }
}

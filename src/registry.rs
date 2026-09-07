//! The one table of named providers (mirrors `lm15.registry`, copied as
//! data — port.md rule 2). A provider string names one wire dialect plus,
//! for bound entries, the compat preset of the server it targets.

/// The wire formats lm15 speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dialect {
    OpenaiResponses,
    OpenaiChat,
    Anthropic,
    Gemini,
}

impl Dialect {
    pub fn as_str(self) -> &'static str {
        match self {
            Dialect::OpenaiResponses => "openai-responses",
            Dialect::OpenaiChat => "openai-chat",
            Dialect::Anthropic => "anthropic",
            Dialect::Gemini => "gemini",
        }
    }
}

/// How an entry relates to its dialect adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryKind {
    /// The dialect adapter's own manifest (openai, anthropic, gemini, ...).
    AdapterOwned,
    /// A dialect with an access policy and compat preset bound (groq, ...).
    Bound,
    /// A cloud door (AUTH-10 host) in front of an existing dialect.
    Hosted,
}

/// Everything module 1–2 needs to know about one named provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderDefinition {
    /// Canonical provider string (hyphenated).
    pub id: &'static str,
    pub dialect: Dialect,
    pub kind: EntryKind,
    /// Compat preset name for bound/hosted chat, responses and anthropic entries.
    pub compat: Option<&'static str>,
    /// The key a keyless local server accepts (AUTH-1 last rung).
    pub placeholder_key: Option<&'static str>,
    pub note: &'static str,
}

const fn owned(id: &'static str, dialect: Dialect, note: &'static str) -> ProviderDefinition {
    ProviderDefinition {
        id,
        dialect,
        kind: EntryKind::AdapterOwned,
        compat: None,
        placeholder_key: None,
        note,
    }
}

const fn bound(
    id: &'static str,
    dialect: Dialect,
    compat: &'static str,
    note: &'static str,
) -> ProviderDefinition {
    ProviderDefinition {
        id,
        dialect,
        kind: EntryKind::Bound,
        compat: Some(compat),
        placeholder_key: None,
        note,
    }
}

const fn local(
    id: &'static str,
    placeholder_key: &'static str,
    note: &'static str,
) -> ProviderDefinition {
    ProviderDefinition {
        id,
        dialect: Dialect::OpenaiChat,
        kind: EntryKind::Bound,
        compat: Some(id),
        placeholder_key: Some(placeholder_key),
        note,
    }
}

const fn hosted(
    id: &'static str,
    dialect: Dialect,
    compat: Option<&'static str>,
    note: &'static str,
) -> ProviderDefinition {
    ProviderDefinition {
        id,
        dialect,
        kind: EntryKind::Hosted,
        compat,
        placeholder_key: None,
        note,
    }
}

/// Declaration order is presentation order, as in the reference.
pub const PROVIDERS: &[ProviderDefinition] = &[
    owned("openai", Dialect::OpenaiResponses, "OpenAI Responses API"),
    owned(
        "openai-chat",
        Dialect::OpenaiChat,
        "OpenAI Chat Completions dialect (the de-facto standard other servers speak)",
    ),
    owned("anthropic", Dialect::Anthropic, "Anthropic Messages API"),
    owned("gemini", Dialect::Gemini, "Google Gemini API"),
    owned(
        "xai",
        Dialect::OpenaiChat,
        "xAI Grok (Chat Completions dialect; XAI_API_KEY or subscription OAuth)",
    ),
    owned(
        "claude-code",
        Dialect::Anthropic,
        "Claude subscription through the local `claude` CLI login",
    ),
    owned(
        "openai-codex",
        Dialect::OpenaiResponses,
        "ChatGPT subscription through the local `codex` CLI login",
    ),
    bound(
        "groq",
        Dialect::OpenaiChat,
        "groq",
        "Groq Cloud (Chat Completions dialect)",
    ),
    bound(
        "openrouter",
        Dialect::OpenaiChat,
        "openrouter",
        "OpenRouter (Chat Completions dialect)",
    ),
    bound(
        "deepseek",
        Dialect::OpenaiChat,
        "deepseek",
        "DeepSeek (Chat Completions dialect; thinking mode on by default)",
    ),
    bound(
        "deepseek-anthropic",
        Dialect::Anthropic,
        "deepseek",
        "DeepSeek over the Anthropic Messages wire (same key as `deepseek`; no model listing)",
    ),
    bound(
        "zai",
        Dialect::OpenaiChat,
        "zai",
        "Z.AI GLM (Chat Completions dialect; general endpoint, not the Coding Plan)",
    ),
    bound(
        "moonshotai",
        Dialect::OpenaiChat,
        "moonshotai",
        "Moonshot AI Kimi (Chat Completions dialect)",
    ),
    bound(
        "moonshotai-responses",
        Dialect::OpenaiResponses,
        "moonshotai",
        "Moonshot AI Kimi over the Responses wire (same key as `moonshotai`)",
    ),
    bound(
        "moonshotai-anthropic",
        Dialect::Anthropic,
        "moonshotai",
        "Moonshot AI Kimi over the Anthropic Messages wire (same key as `moonshotai`)",
    ),
    bound(
        "meta",
        Dialect::OpenaiResponses,
        "meta",
        "Meta Model API over the Responses wire",
    ),
    bound(
        "meta-chat",
        Dialect::OpenaiChat,
        "meta",
        "Meta Model API over the Chat Completions wire (same key as `meta`)",
    ),
    bound(
        "meta-anthropic",
        Dialect::Anthropic,
        "meta",
        "Meta Model API over the Anthropic Messages wire (same key as `meta`)",
    ),
    hosted(
        "azure",
        Dialect::OpenaiResponses,
        None,
        "Azure OpenAI v1 Responses wire ({resource}.openai.azure.com)",
    ),
    hosted(
        "azure-chat",
        Dialect::OpenaiChat,
        Some("openai"),
        "Azure OpenAI v1 Chat Completions wire (same resource)",
    ),
    hosted(
        "azure-anthropic",
        Dialect::Anthropic,
        None,
        "Claude in Microsoft Foundry ({resource}.services.ai.azure.com/anthropic)",
    ),
    hosted(
        "aws-anthropic",
        Dialect::Anthropic,
        None,
        "Claude Platform on AWS (Anthropic-operated; SigV4 or ANTHROPIC_AWS_API_KEY)",
    ),
    hosted(
        "bedrock-anthropic",
        Dialect::Anthropic,
        None,
        "Claude in Amazon Bedrock (bedrock-mantle; SigV4 or AWS_BEARER_TOKEN_BEDROCK)",
    ),
    hosted(
        "bedrock-chat",
        Dialect::OpenaiChat,
        Some("bedrock"),
        "Amazon Bedrock over the OpenAI Chat Completions wire (bedrock-runtime /openai/v1)",
    ),
    hosted(
        "bedrock-mantle-chat",
        Dialect::OpenaiChat,
        Some("bedrock-mantle"),
        "Amazon Bedrock Chat Completions on bedrock-mantle (un-versioned ids)",
    ),
    hosted(
        "vertex",
        Dialect::Gemini,
        None,
        "Gemini on Google Cloud (Agent Platform); ADC chain",
    ),
    hosted(
        "vertex-express",
        Dialect::Gemini,
        None,
        "Agent Platform express mode: GOOGLE_API_KEY as ?key=",
    ),
    hosted(
        "vertex-anthropic",
        Dialect::Anthropic,
        None,
        "Claude on Google Cloud (rawPredict; model in the path)",
    ),
    local("ollama", "ollama", "local ollama server (keyless)"),
    local("vllm", "EMPTY", "local vLLM server (keyless)"),
    local("sglang", "EMPTY", "local SGLang server (keyless)"),
];

/// Provider strings are hyphenated; the underscore spelling is a permanent
/// alias (`openai_chat` → `openai-chat`).
pub fn canonical_provider(name: &str) -> String {
    name.replace('_', "-")
}

/// The definition for a provider string in either spelling.
pub fn lookup(name: &str) -> Option<&'static ProviderDefinition> {
    let canonical = canonical_provider(name);
    PROVIDERS.iter().find(|d| d.id == canonical)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_unique_and_hyphenated() {
        for (i, d) in PROVIDERS.iter().enumerate() {
            assert_eq!(d.id, canonical_provider(d.id));
            assert!(
                PROVIDERS[..i].iter().all(|other| other.id != d.id),
                "{}",
                d.id
            );
            if d.kind == EntryKind::Bound {
                assert!(d.compat.is_some(), "{} names its preset", d.id);
            }
            if d.placeholder_key.is_some() {
                assert_eq!(d.kind, EntryKind::Bound);
            }
        }
    }

    #[test]
    fn lookup_accepts_both_spellings() {
        assert_eq!(lookup("openai_chat").unwrap().id, "openai-chat");
        assert_eq!(
            lookup("deepseek-anthropic").unwrap().dialect,
            Dialect::Anthropic
        );
        assert!(lookup("nope").is_none());
        assert_eq!(PROVIDERS.len(), 31);
    }
}

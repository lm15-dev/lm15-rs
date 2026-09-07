//! The one table of named providers (mirrors `lm15.registry`, copied as
//! data — port.md rule 2). A provider string names one wire dialect plus,
//! for bound entries, the compat preset of the server it targets, and
//! [`adapter_for`] binds the three (dialect + policy + compat) into a
//! [`ProviderLM`] the way the reference router does.

use crate::adapter::{LmBuilder, ProviderLM};
use crate::auth::{access_policy, AccessPolicy, CredentialProvider};
use crate::cloud::hosts::HostSettings;
use crate::errors::Lm15Error;
use crate::wire::Clock;

/// The wire formats lm15 speaks (the codec is `wire::Dialect`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DialectId {
    OpenaiResponses,
    OpenaiChat,
    Anthropic,
    Gemini,
}

impl DialectId {
    pub fn as_str(self) -> &'static str {
        match self {
            DialectId::OpenaiResponses => "openai-responses",
            DialectId::OpenaiChat => "openai-chat",
            DialectId::Anthropic => "anthropic",
            DialectId::Gemini => "gemini",
        }
    }

    /// The dialect's own default base URL (`lm15/providers/*.py`
    /// `_DEFAULT_BASE_URL`), used when neither the policy, a host, nor a
    /// preset supplies one.
    pub fn default_base_url(self) -> &'static str {
        match self {
            DialectId::OpenaiResponses | DialectId::OpenaiChat => "https://api.openai.com/v1",
            DialectId::Anthropic => "https://api.anthropic.com/v1",
            DialectId::Gemini => "https://generativelanguage.googleapis.com/v1beta",
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
    pub dialect: DialectId,
    pub kind: EntryKind,
    /// Compat preset name for bound/hosted chat, responses and anthropic
    /// entries, and for `xai` (its constructor binds the `xai` preset,
    /// `lm15/providers/xai.py:70`).
    pub compat: Option<&'static str>,
    /// The key a keyless local server accepts (AUTH-1 last rung).
    pub placeholder_key: Option<&'static str>,
    pub note: &'static str,
}

const fn owned(id: &'static str, dialect: DialectId, note: &'static str) -> ProviderDefinition {
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
    dialect: DialectId,
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
        dialect: DialectId::OpenaiChat,
        kind: EntryKind::Bound,
        compat: Some(id),
        placeholder_key: Some(placeholder_key),
        note,
    }
}

const fn hosted(
    id: &'static str,
    dialect: DialectId,
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
    owned("openai", DialectId::OpenaiResponses, "OpenAI Responses API"),
    owned(
        "openai-chat",
        DialectId::OpenaiChat,
        "OpenAI Chat Completions dialect (the de-facto standard other servers speak)",
    ),
    owned("anthropic", DialectId::Anthropic, "Anthropic Messages API"),
    owned("gemini", DialectId::Gemini, "Google Gemini API"),
    ProviderDefinition {
        compat: Some("xai"),
        ..owned(
            "xai",
            DialectId::OpenaiChat,
            "xAI Grok (Chat Completions dialect; XAI_API_KEY or subscription OAuth)",
        )
    },
    owned(
        "claude-code",
        DialectId::Anthropic,
        "Claude subscription through the local `claude` CLI login",
    ),
    owned(
        "openai-codex",
        DialectId::OpenaiResponses,
        "ChatGPT subscription through the local `codex` CLI login",
    ),
    bound(
        "groq",
        DialectId::OpenaiChat,
        "groq",
        "Groq Cloud (Chat Completions dialect)",
    ),
    bound(
        "openrouter",
        DialectId::OpenaiChat,
        "openrouter",
        "OpenRouter (Chat Completions dialect)",
    ),
    bound(
        "deepseek",
        DialectId::OpenaiChat,
        "deepseek",
        "DeepSeek (Chat Completions dialect; thinking mode on by default)",
    ),
    bound(
        "deepseek-anthropic",
        DialectId::Anthropic,
        "deepseek",
        "DeepSeek over the Anthropic Messages wire (same key as `deepseek`; no model listing)",
    ),
    bound(
        "zai",
        DialectId::OpenaiChat,
        "zai",
        "Z.AI GLM (Chat Completions dialect; general endpoint, not the Coding Plan)",
    ),
    bound(
        "moonshotai",
        DialectId::OpenaiChat,
        "moonshotai",
        "Moonshot AI Kimi (Chat Completions dialect)",
    ),
    bound(
        "moonshotai-responses",
        DialectId::OpenaiResponses,
        "moonshotai",
        "Moonshot AI Kimi over the Responses wire (same key as `moonshotai`)",
    ),
    bound(
        "moonshotai-anthropic",
        DialectId::Anthropic,
        "moonshotai",
        "Moonshot AI Kimi over the Anthropic Messages wire (same key as `moonshotai`)",
    ),
    bound(
        "meta",
        DialectId::OpenaiResponses,
        "meta",
        "Meta Model API over the Responses wire",
    ),
    bound(
        "meta-chat",
        DialectId::OpenaiChat,
        "meta",
        "Meta Model API over the Chat Completions wire (same key as `meta`)",
    ),
    bound(
        "meta-anthropic",
        DialectId::Anthropic,
        "meta",
        "Meta Model API over the Anthropic Messages wire (same key as `meta`)",
    ),
    hosted(
        "azure",
        DialectId::OpenaiResponses,
        None,
        "Azure OpenAI v1 Responses wire ({resource}.openai.azure.com)",
    ),
    hosted(
        "azure-chat",
        DialectId::OpenaiChat,
        Some("openai"),
        "Azure OpenAI v1 Chat Completions wire (same resource)",
    ),
    hosted(
        "azure-anthropic",
        DialectId::Anthropic,
        None,
        "Claude in Microsoft Foundry ({resource}.services.ai.azure.com/anthropic)",
    ),
    hosted(
        "aws-anthropic",
        DialectId::Anthropic,
        None,
        "Claude Platform on AWS (Anthropic-operated; SigV4 or ANTHROPIC_AWS_API_KEY)",
    ),
    hosted(
        "bedrock-anthropic",
        DialectId::Anthropic,
        None,
        "Claude in Amazon Bedrock (bedrock-mantle; SigV4 or AWS_BEARER_TOKEN_BEDROCK)",
    ),
    hosted(
        "bedrock-chat",
        DialectId::OpenaiChat,
        Some("bedrock"),
        "Amazon Bedrock over the OpenAI Chat Completions wire (bedrock-runtime /openai/v1)",
    ),
    hosted(
        "bedrock-mantle-chat",
        DialectId::OpenaiChat,
        Some("bedrock-mantle"),
        "Amazon Bedrock Chat Completions on bedrock-mantle (un-versioned ids)",
    ),
    hosted(
        "vertex",
        DialectId::Gemini,
        None,
        "Gemini on Google Cloud (Agent Platform); ADC chain",
    ),
    hosted(
        "vertex-express",
        DialectId::Gemini,
        None,
        "Agent Platform express mode: GOOGLE_API_KEY as ?key=",
    ),
    hosted(
        "vertex-anthropic",
        DialectId::Anthropic,
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

impl ProviderDefinition {
    /// The access policy this entry binds (`lm15/registry.py`
    /// `ProviderDefinition.access`); every entry has one.
    pub fn access(&self) -> &'static AccessPolicy {
        access_policy(self.id).expect("every registry entry has an access policy")
    }

    /// True when the access policy names a cloud host (AUTH-10).
    pub fn hosted(&self) -> bool {
        self.access().host.is_some()
    }
}

/// The adapter a provider string names, exactly as the router builds it
/// (`lm15/vet.py:81-110` `adapter_for_provider`): the dialect with the
/// entry's access policy and compat preset bound; `base_url` overrides
/// every default; `settings` are the host settings (AUTH-10); `clock` is
/// the time source (a fixed one under the harness). No environment is
/// read.
pub fn adapter_for(
    provider: &str,
    credential: impl CredentialProvider + Send + Sync + 'static,
    base_url: Option<&str>,
    settings: Option<HostSettings>,
    clock: Option<Box<dyn Clock + Send + Sync>>,
) -> Result<ProviderLM, Lm15Error> {
    let definition = lookup(provider).ok_or_else(|| {
        Lm15Error::not_configured(format!(
            "unknown provider {provider:?}; known providers: {}",
            PROVIDERS
                .iter()
                .map(|d| d.id)
                .collect::<Vec<_>>()
                .join(", ")
        ))
    })?;
    let mut builder = LmBuilder::for_entry(definition).api_key(credential);
    if let Some(base_url) = base_url {
        builder = builder.base_url(base_url);
    }
    if let Some(settings) = settings {
        builder = builder.settings(settings);
    }
    if let Some(clock) = clock {
        builder = builder.clock_boxed(clock);
    }
    builder.build()
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
            DialectId::Anthropic
        );
        assert!(lookup("nope").is_none());
        assert_eq!(PROVIDERS.len(), 31);
    }

    /// Every registry entry has a policy of the same id, and the hosted
    /// kind agrees with the policy's host (`lm15/registry.py` rules).
    #[test]
    fn entries_agree_with_the_policy_table() {
        for d in PROVIDERS {
            assert_eq!(d.access().provider, d.id);
            assert_eq!(d.kind == EntryKind::Hosted, d.hosted(), "{}", d.id);
            assert_eq!(d.placeholder_key, d.access().placeholder_key, "{}", d.id);
            if let Some(name) = d.compat {
                let known = match d.dialect {
                    DialectId::Anthropic => crate::compat::AnthropicCompat::preset(name).is_some(),
                    DialectId::OpenaiResponses => {
                        crate::compat::OpenAIResponsesCompat::preset(name).is_some()
                    }
                    DialectId::OpenaiChat => {
                        crate::compat::OpenAIChatCompat::preset(name).is_some()
                    }
                    DialectId::Gemini => false,
                };
                assert!(known, "{}: preset {name}", d.id);
            }
        }
        assert_eq!(crate::auth::ACCESS_POLICIES.len(), PROVIDERS.len());
    }
}

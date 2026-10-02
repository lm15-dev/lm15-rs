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
    Typesafe,
}

impl DialectId {
    pub fn as_str(self) -> &'static str {
        match self {
            DialectId::OpenaiResponses => "openai-responses",
            DialectId::OpenaiChat => "openai-chat",
            DialectId::Anthropic => "anthropic",
            DialectId::Gemini => "gemini",
            DialectId::Typesafe => "typesafe",
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
            DialectId::Typesafe => "https://api.typesafe.ai",
        }
    }

    /// The one preset name that resolves to [`Self::default_base_url`]
    /// (api-family 2026-09-11): any other name must supply its server's
    /// address or be given one.
    pub fn default_preset(self) -> &'static str {
        match self {
            DialectId::OpenaiResponses | DialectId::OpenaiChat => "openai",
            DialectId::Anthropic => "anthropic",
            DialectId::Gemini | DialectId::Typesafe => "",
        }
    }

    /// The dialect's name for a message to the user.
    pub fn wire_name(self) -> &'static str {
        match self {
            DialectId::OpenaiResponses => "Responses",
            DialectId::OpenaiChat => "Chat Completions",
            DialectId::Anthropic => "Messages",
            DialectId::Gemini => "generateContent",
            DialectId::Typesafe => "System One",
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
    /// entries (the registry's own value; see [`ProviderDefinition::preset`]
    /// for the one an adapter-owned class binds itself).
    pub compat: Option<&'static str>,
    /// The key a keyless local server accepts (AUTH-1 last rung).
    pub placeholder_key: Option<&'static str>,
    pub note: &'static str,
}

/// Declaration order is presentation order, as in the reference: the
/// reference's rows (lm15-contract tables/providers.json, generated into
/// `src/generated/tables.rs`). A provider is added there, never here.
pub const PROVIDERS: &[ProviderDefinition] = crate::generated::tables::PROVIDERS;

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
    /// The compat preset the adapter is built with: the row's own, or the
    /// one an adapter-owned class binds in its constructor. Only `xai` does:
    /// the reference's `XaiLM` passes `compat="xai"` (lm15-python
    /// `lm15/providers/xai.py`), adapter code, not a registry value, so it is
    /// written here and not in the generated table.
    pub fn preset(&self) -> Option<&'static str> {
        match (self.compat, self.kind, self.id) {
            (Some(name), _, _) => Some(name),
            (None, EntryKind::AdapterOwned, "xai") => Some("xai"),
            _ => None,
        }
    }

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
    let mut builder = LmBuilder::for_entry(definition)
        .api_key(credential)
        .env(Default::default());
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
        assert_eq!(PROVIDERS.len(), 36); // + the four inference hosts (2026-09-26)
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
                    DialectId::Gemini | DialectId::Typesafe => false,
                };
                assert!(known, "{}: preset {name}", d.id);
            }
        }
        assert_eq!(crate::auth::ACCESS_POLICIES.len(), PROVIDERS.len());
    }
}

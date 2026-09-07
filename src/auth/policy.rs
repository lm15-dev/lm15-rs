//! Access policies as data (spec/auth.md AUTH-10; playbooks/port.md rule 2:
//! "copy tables as data"). The auth-relevant columns of the reference table
//! `lm15/access.py` (provider, credential policy, env keys in declared
//! order, auth schemes in preference order, login hint) and the keyless
//! local servers' placeholder keys from `lm15/registry.py`.
//!
//! Providers are named by their registry id (`lm15/registry.py`), which is
//! the string a model spec uses; `openai_chat` (the access-table spelling)
//! maps to `openai-chat` through [`crate::registry::canonical_provider`]
//! (the one home of that rule; re-exported here for the auth surface).

use super::credential::AuthScheme;
use super::stores::{CLAUDE_CODE_LOGIN_HINT, OPENAI_CODEX_LOGIN_HINT, XAI_LOGIN_HINT};

/// spec/vocabularies.md `CredentialPolicy`; spec/auth.md AUTH-1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CredentialPolicy {
    Key,
    OAuth,
    OAuthUnlessExplicit,
    AwsChain,
    AzureChain,
    GcpChain,
}

impl CredentialPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            CredentialPolicy::Key => "key",
            CredentialPolicy::OAuth => "oauth",
            CredentialPolicy::OAuthUnlessExplicit => "oauth-unless-explicit",
            CredentialPolicy::AwsChain => "aws-chain",
            CredentialPolicy::AzureChain => "azure-chain",
            CredentialPolicy::GcpChain => "gcp-chain",
        }
    }

    /// `aws-chain`, `azure-chain`, `gcp-chain`: module 3b (playbooks/port.md).
    pub fn is_cloud_chain(self) -> bool {
        matches!(
            self,
            CredentialPolicy::AwsChain | CredentialPolicy::AzureChain | CredentialPolicy::GcpChain
        )
    }
}

/// The auth face of an access policy (AUTH-10). Pure data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessPolicy {
    /// Canonical provider string (errors, routing, doctor).
    pub provider: &'static str,
    /// AUTH-1 policy.
    pub credential_policy: CredentialPolicy,
    /// Declared environment keys, in declared order (AUTH-1 rung 2).
    pub env_keys: &'static [&'static str],
    /// The schemes this door accepts, in preference order (AUTH-2 selects).
    pub auth_scheme: &'static [AuthScheme],
    /// Local-server presets only: the key sent when nothing is configured
    /// (AUTH-1 rung 3).
    pub placeholder_key: Option<&'static str>,
    /// Re-login guidance for stored-login policies (AUTH-6).
    pub login_hint: Option<&'static str>,
}

use AuthScheme::{Bearer, QueryKey, SigV4, XApiKey};
use CredentialPolicy::{AwsChain, AzureChain, GcpChain, Key, OAuth, OAuthUnlessExplicit};

const fn policy(
    provider: &'static str,
    credential_policy: CredentialPolicy,
    env_keys: &'static [&'static str],
    auth_scheme: &'static [AuthScheme],
) -> AccessPolicy {
    AccessPolicy {
        provider,
        credential_policy,
        env_keys,
        auth_scheme,
        placeholder_key: None,
        login_hint: None,
    }
}

const META_ENV_KEYS: &[&str] = &["META_API_KEY"];
const MOONSHOTAI_ENV_KEYS: &[&str] = &["MOONSHOTAI_API_KEY", "MOONSHOT_API_KEY"];

/// The table, in the reference's declaration order (`lm15/access.py`).
pub const ACCESS_POLICIES: &[AccessPolicy] = &[
    policy("anthropic", Key, &["ANTHROPIC_API_KEY"], &[XApiKey]),
    AccessPolicy {
        login_hint: Some(CLAUDE_CODE_LOGIN_HINT),
        ..policy("claude-code", OAuth, &[], &[Bearer])
    },
    policy("openai", Key, &["OPENAI_API_KEY"], &[Bearer]),
    AccessPolicy {
        login_hint: Some(OPENAI_CODEX_LOGIN_HINT),
        ..policy("openai-codex", OAuth, &[], &[Bearer])
    },
    policy("openai-chat", Key, &["OPENAI_API_KEY"], &[Bearer]),
    AccessPolicy {
        login_hint: Some(XAI_LOGIN_HINT),
        ..policy("xai", OAuthUnlessExplicit, &["XAI_API_KEY"], &[Bearer])
    },
    // The Gemini dialect renders x-api-key as x-goog-api-key.
    policy(
        "gemini",
        Key,
        &["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        &[XApiKey],
    ),
    policy("meta", Key, META_ENV_KEYS, &[Bearer]),
    policy("groq", Key, &["GROQ_API_KEY"], &[Bearer]),
    policy("openrouter", Key, &["OPENROUTER_API_KEY"], &[Bearer]),
    policy("deepseek", Key, &["DEEPSEEK_API_KEY"], &[Bearer]),
    policy("zai", Key, &["ZAI_API_KEY"], &[Bearer]),
    policy("moonshotai", Key, MOONSHOTAI_ENV_KEYS, &[Bearer]),
    policy("moonshotai-responses", Key, MOONSHOTAI_ENV_KEYS, &[Bearer]),
    policy("meta-chat", Key, META_ENV_KEYS, &[Bearer]),
    policy("deepseek-anthropic", Key, &["DEEPSEEK_API_KEY"], &[XApiKey]),
    policy("meta-anthropic", Key, META_ENV_KEYS, &[Bearer]),
    policy("moonshotai-anthropic", Key, MOONSHOTAI_ENV_KEYS, &[Bearer]),
    // Cloud hosts (module 3b): present as data; explain_auth does not walk them.
    policy(
        "aws-anthropic",
        AwsChain,
        &["ANTHROPIC_AWS_API_KEY"],
        &[SigV4, XApiKey],
    ),
    policy(
        "bedrock-anthropic",
        AwsChain,
        &["AWS_BEARER_TOKEN_BEDROCK"],
        &[SigV4, XApiKey],
    ),
    policy(
        "bedrock-chat",
        AwsChain,
        &["AWS_BEARER_TOKEN_BEDROCK"],
        &[SigV4, Bearer],
    ),
    policy(
        "bedrock-mantle-chat",
        AwsChain,
        &["AWS_BEARER_TOKEN_BEDROCK"],
        &[SigV4, Bearer],
    ),
    policy(
        "azure",
        AzureChain,
        &["AZURE_OPENAI_API_KEY"],
        &[AuthScheme::ApiKey, Bearer],
    ),
    policy(
        "azure-chat",
        AzureChain,
        &["AZURE_OPENAI_API_KEY"],
        &[AuthScheme::ApiKey, Bearer],
    ),
    policy(
        "azure-anthropic",
        AzureChain,
        &["ANTHROPIC_FOUNDRY_API_KEY"],
        &[XApiKey, Bearer],
    ),
    policy("vertex", GcpChain, &[], &[Bearer]),
    policy("vertex-express", Key, &["GOOGLE_API_KEY"], &[QueryKey]),
    policy("vertex-anthropic", GcpChain, &[], &[Bearer]),
    // Keyless local servers (`lm15/registry.py` placeholder_key).
    AccessPolicy {
        placeholder_key: Some("ollama"),
        ..policy("ollama", Key, &[], &[Bearer])
    },
    AccessPolicy {
        placeholder_key: Some("EMPTY"),
        ..policy("vllm", Key, &[], &[Bearer])
    },
    AccessPolicy {
        placeholder_key: Some("EMPTY"),
        ..policy("sglang", Key, &[], &[Bearer])
    },
];

pub use crate::registry::canonical_provider;

/// Every provider in the table, sorted.
pub fn known_providers() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = ACCESS_POLICIES.iter().map(|p| p.provider).collect();
    names.sort_unstable();
    names
}

/// The policy for a provider string (underscore alias accepted).
pub fn access_policy(provider: &str) -> Option<&'static AccessPolicy> {
    let canonical = canonical_provider(provider);
    ACCESS_POLICIES.iter().find(|p| p.provider == canonical)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_strings_are_unique() {
        let mut names = known_providers();
        names.dedup();
        assert_eq!(names.len(), ACCESS_POLICIES.len());
    }

    #[test]
    fn oauth_policies_declare_no_env_keys() {
        // AUTH-1: "An `oauth` manifest declares no environment keys."
        for policy in ACCESS_POLICIES {
            if policy.credential_policy == OAuth {
                assert!(policy.env_keys.is_empty(), "{}", policy.provider);
                assert!(policy.login_hint.is_some(), "{}", policy.provider);
            }
        }
    }

    #[test]
    fn placeholder_presets_declare_no_env_keys() {
        for policy in ACCESS_POLICIES {
            if policy.placeholder_key.is_some() {
                assert!(policy.env_keys.is_empty(), "{}", policy.provider);
            }
        }
    }

    #[test]
    fn underscore_alias_resolves() {
        assert_eq!(
            access_policy("openai_chat").unwrap().provider,
            "openai-chat"
        );
        assert!(access_policy("nope").is_none());
    }
}

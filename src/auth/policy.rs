//! Access policies as data (spec/auth.md AUTH-10; playbooks/port.md rule 2:
//! "copy tables as data"). Every column of the reference table
//! `lm15/access.py` (provider, endpoint surfaces, credential policy, auth
//! modes, env keys in declared order, auth schemes in preference order,
//! static headers, host descriptor, login hint, backend, backend options,
//! system prefix, base URL) and the keyless local servers' placeholder keys
//! from `lm15/registry.py`.
//!
//! Providers are named by their registry id (`lm15/registry.py`), which is
//! the string a model spec uses; `openai_chat` (the access-table spelling)
//! maps to `openai-chat` through [`crate::registry::canonical_provider`]
//! (the one home of that rule; re-exported here for the auth surface).
//!
//! The dialect consults the policy at the named points of AUTH-10 and
//! nowhere else: `supports` at every surface driver, `auth_scheme` in the
//! emit path (`crate::wire::emit`), `headers` in the dialect's header
//! builder (Anthropic joins `anthropic-beta`), `system_prefix` in the
//! payload, `host` in `crate::cloud::hosts`, `backend` in the dialect's
//! stated branches.

use super::credential::AuthScheme;
use super::stores::XAI_LOGIN_HINT;

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

    /// `aws-chain`, `azure-chain`, `gcp-chain`: the cloud chains (`crate::cloud`).
    pub fn is_cloud_chain(self) -> bool {
        matches!(
            self,
            CredentialPolicy::AwsChain | CredentialPolicy::AzureChain | CredentialPolicy::GcpChain
        )
    }
}

/// The endpoint surfaces an access path carries (`lm15/features.py:62-82`
/// `EndpointSupport`; spec/support-matrix.json pins every row).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EndpointSupport {
    pub complete: bool,
    pub stream: bool,
    pub live: bool,
    pub files: bool,
    pub batches: bool,
    pub images: bool,
    pub speech: bool,
    pub video: bool,
    pub responses_api: bool,
    pub models: bool,
    pub caches: bool,
}

impl EndpointSupport {
    /// `complete` and `stream` only (the reference's field defaults).
    pub const CHAT: EndpointSupport = EndpointSupport {
        complete: true,
        stream: true,
        live: false,
        files: false,
        batches: false,
        images: false,
        speech: false,
        video: false,
        responses_api: false,
        models: false,
        caches: false,
    };

    /// `complete`, `stream` and `models`.
    pub const CHAT_MODELS: EndpointSupport = EndpointSupport {
        models: true,
        ..EndpointSupport::CHAT
    };

    /// The surface by its support-matrix name; unknown names are `false`.
    pub fn supports_endpoint(&self, name: &str) -> bool {
        match name {
            "complete" => self.complete,
            "stream" => self.stream,
            "live" => self.live,
            "files" => self.files,
            "batches" => self.batches,
            "images" => self.images,
            "speech" => self.speech,
            "video" => self.video,
            "responses_api" => self.responses_api,
            "models" => self.models,
            "caches" => self.caches,
            _ => false,
        }
    }
}

/// spec/vocabularies.md `ModelPlacement`: where the model goes on a host
/// (`lm15/features.py:96`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelPlacement {
    Body,
    Path,
}

impl ModelPlacement {
    pub fn as_str(self) -> &'static str {
        match self {
            ModelPlacement::Body => "body",
            ModelPlacement::Path => "path",
        }
    }
}

/// spec/vocabularies.md `StreamFraming` (`lm15/features.py:94`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamFraming {
    Sse,
    AwsEventStream,
}

impl StreamFraming {
    pub fn as_str(self) -> &'static str {
        match self {
            StreamFraming::Sse => "sse",
            StreamFraming::AwsEventStream => "aws-event-stream",
        }
    }
}

/// AUTH-10 `host.anthropic_version_in`: the `anthropic-version` header, or
/// an `anthropic_version` body field with this value
/// (`lm15/features.py:124`, `:145-146`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnthropicVersionIn {
    Header,
    Body(&'static str),
}

/// One host setting: its name, the env variables consulted in order when
/// the caller did not pass it, and its default (`None` = required)
/// (`lm15/features.py:100-107` `HostSetting`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostSetting {
    pub name: &'static str,
    pub env: &'static [&'static str],
    pub default: Option<&'static str>,
}

/// How a dialect reaches a cloud door (spec/auth.md AUTH-10 `host`;
/// `lm15/features.py:110-153` `HostSpec`).
///
/// - `base_url`: a template over the settings — `{region}`, `{project}`,
///   `{location}`, `{location_host}` (derived from `location`), `{resource}`.
/// - `paths`: endpoint-path overrides keyed by the dialect's endpoint name
///   (`messages`, `messages/stream`); `{model}` is the request's model.
///   Absent means the dialect's own path under `base_url`.
/// - `model_in`: `Path` removes the model field from the payload.
/// - `required_headers`: `(header name, setting name)` pairs sent on every
///   request from the resolved settings.
/// - `sigv4_service`: the SigV4 credential-scope service name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostSpec {
    pub base_url: &'static str,
    /// Vendor endpoint variables, in precedence order (AUTH-10).
    pub endpoint_env: &'static [&'static str],
    pub settings: &'static [HostSetting],
    pub paths: &'static [(&'static str, &'static str)],
    pub model_in: ModelPlacement,
    pub anthropic_version_in: AnthropicVersionIn,
    pub stream_framing: StreamFraming,
    pub required_headers: &'static [(&'static str, &'static str)],
    pub sigv4_service: Option<&'static str>,
}

impl HostSpec {
    /// A host with the reference's field defaults (`body`, `header`, `sse`).
    pub const fn new(base_url: &'static str) -> HostSpec {
        HostSpec {
            base_url,
            endpoint_env: &[],
            settings: &[],
            paths: &[],
            model_in: ModelPlacement::Body,
            anthropic_version_in: AnthropicVersionIn::Header,
            stream_framing: StreamFraming::Sse,
            required_headers: &[],
            sigv4_service: None,
        }
    }

    /// The path override for an endpoint name, when the host has one.
    pub fn path_for(&self, endpoint: &str) -> Option<&'static str> {
        self.paths
            .iter()
            .find(|(name, _)| *name == endpoint)
            .map(|(_, path)| *path)
    }

    pub fn setting_names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.settings.iter().map(|s| s.name)
    }
}

/// An access policy (AUTH-10; `lm15/features.py:156-312` `AccessPolicy`).
/// Pure data; the dialect consults it at the named points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccessPolicy {
    /// Canonical provider string (errors, routing, doctor).
    pub provider: &'static str,
    /// Endpoint surfaces this access path carries; a dialect that
    /// implements a surface still refuses when the policy does not carry it.
    pub supports: EndpointSupport,
    /// AUTH-1 policy.
    pub credential_policy: CredentialPolicy,
    /// Support-matrix pinned auth-mode names (doctor, docs).
    pub auth_modes: &'static [&'static str],
    /// Support-matrix pinned enterprise variants (docs).
    pub enterprise_variants: &'static [&'static str],
    /// Declared environment keys, in declared order (AUTH-1 rung 2).
    pub env_keys: &'static [&'static str],
    /// The schemes this door accepts, in preference order (AUTH-2 selects).
    pub auth_scheme: &'static [AuthScheme],
    /// Static headers on every request, in order; the dialect merges them
    /// by its stated rule (Anthropic joins `anthropic-beta`).
    pub headers: &'static [(&'static str, &'static str)],
    /// The cloud door, or `None` for the dialect's public API.
    pub host: Option<HostSpec>,
    /// Re-login guidance for stored-login policies (AUTH-6).
    pub login_hint: Option<&'static str>,
    /// Dialect-consulted variant name; `api` is the provider's public API.
    pub backend: &'static str,
    /// String knobs the backend variant needs.
    pub backend_options: &'static [(&'static str, &'static str)],
    /// Text the backend requires first in the system prompt / instructions.
    pub system_prefix: Option<&'static str>,
    /// This access path's default base URL, when it is not the dialect's.
    pub base_url: Option<&'static str>,
    /// Local-server presets only: the key sent when nothing is configured
    /// (AUTH-1 rung 3).
    pub placeholder_key: Option<&'static str>,
    /// The `backend_options` a caller may set on a door without a host
    /// (AUTH-10, amended 2026-09-30): each names a `backend_options` key and
    /// the env variables the router consults for it; its default is the
    /// table's `backend_options` value (`default` stays `None`). The
    /// subscription doors declare `client_version`.
    pub backend_settings: &'static [HostSetting],
}

impl AccessPolicy {
    /// The first header-carrying scheme as the two-value spelling the
    /// dialects consult for an `ApiKey` (`lm15/features.py:291-302`):
    /// `bearer`, or `x-api-key` (which also stands for `api-key`).
    pub fn auth_header(&self) -> AuthScheme {
        for scheme in self.auth_scheme {
            match scheme {
                AuthScheme::Bearer => return AuthScheme::Bearer,
                AuthScheme::XApiKey | AuthScheme::ApiKey => return AuthScheme::XApiKey,
                _ => {}
            }
        }
        AuthScheme::Bearer
    }

    pub fn is_cloud_chain(&self) -> bool {
        self.credential_policy.is_cloud_chain()
    }

    /// The value of a backend option, when declared.
    pub fn backend_option(&self, name: &str) -> Option<&'static str> {
        self.backend_options
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, value)| *value)
    }

    /// The door's backend settings (AUTH-10, amended 2026-09-30): the
    /// caller's value, then `env` (when given — the router passes the
    /// environment, an adapter built by hand does not), then the table's
    /// `backend_options` value. `sources` receives each origin (`explicit`,
    /// `env:<VAR>`, `default`). A name the door does not declare is a
    /// configuration error that lists the names it does: a setting nothing
    /// reads would otherwise be dropped with nothing said.
    pub fn resolve_backend_settings(
        &self,
        given: &crate::cloud::hosts::HostSettings,
        env: Option<&std::collections::BTreeMap<String, String>>,
        mut sources: Option<&mut std::collections::BTreeMap<String, String>>,
    ) -> Result<crate::cloud::hosts::HostSettings, crate::errors::Lm15Error> {
        let known: Vec<&str> = self.backend_settings.iter().map(|s| s.name).collect();
        let mut unknown: Vec<&String> = given
            .keys()
            .filter(|k| !known.contains(&k.as_str()))
            .collect();
        unknown.sort();
        if !unknown.is_empty() {
            let names: Vec<String> = unknown.iter().map(|n| format!("'{n}'")).collect();
            let (hint, fix) = if known.is_empty() {
                (
                    "this door takes no settings".to_string(),
                    format!("Remove the settings entry for {}", self.provider),
                )
            } else {
                (
                    format!("known: {}", known.join(", ")),
                    format!("Pass only {} for {}", known.join(", "), self.provider),
                )
            };
            let mut meta = crate::errors::ErrorMeta::new(format!(
                "{}: unknown setting(s) {}; {hint}\n\n  To fix:\n    - {fix}\n",
                self.provider,
                names.join(", ")
            ));
            meta.provider = Some(self.provider.to_string());
            return Err(crate::errors::Lm15Error::NotConfiguredError(meta));
        }
        let mut out = crate::cloud::hosts::HostSettings::new();
        for setting in self.backend_settings {
            let mut value = given.get(setting.name).filter(|v| !v.is_empty()).cloned();
            let mut origin = value.as_ref().map(|_| "explicit".to_string());
            if value.is_none() {
                if let Some(env) = env {
                    if let Some(var) = setting
                        .env
                        .iter()
                        .find(|var| env.get(**var).is_some_and(|v| !v.is_empty()))
                    {
                        value = env.get(*var).cloned();
                        origin = Some(format!("env:{var}"));
                    }
                }
            }
            let value = value.unwrap_or_else(|| {
                self.backend_option(setting.name)
                    .unwrap_or_default()
                    .to_string()
            });
            if let Some(sources) = sources.as_deref_mut() {
                sources.insert(
                    setting.name.to_string(),
                    origin.unwrap_or_else(|| "default".into()),
                );
            }
            out.insert(setting.name.to_string(), value);
        }
        Ok(out)
    }
}

/// `lm15/access.py` `claude_code_version_guidance`: the claude-code door's
/// minimum-version refusal, with what an lm15 caller changes (AUTH-10
/// backend settings). The server says "run 'claude update'", which does not
/// move the version lm15 claims. Any other message is returned unchanged.
pub fn claude_code_version_guidance(message: &str) -> String {
    const HEAD: &str = "Claude Code ";
    const MIDDLE: &str = " does not support this model; version ";
    const TAIL: &str = " or newer is required";
    if message.contains("\n\n  To fix:") {
        return message.to_string();
    }
    let required = message.find(HEAD).and_then(|start| {
        let rest = &message[start + HEAD.len()..];
        let sent_end = rest.find(char::is_whitespace)?;
        let rest = rest[sent_end..].strip_prefix(MIDDLE)?;
        let required_end = rest.find(char::is_whitespace)?;
        let required = &rest[..required_end];
        (!required.is_empty() && rest[required_end..].starts_with(TAIL) && sent_end > 0)
            .then_some(required)
    });
    match required {
        None => message.to_string(),
        Some(required) => format!(
            "{message}\n\n  To fix:\n    - lm15 sends this version itself; updating Claude Code does not change it\n    - Set the claude-code setting client_version to {required} or newer (or {CLAUDE_CODE_VERSION_ENV}={required})\n"
        ),
    }
}

// ─── The table (`lm15/access.py`) ───────────────────────────────────
//
// Generated from lm15-contract tables/providers.json into
// `src/generated/tables.rs` (playbooks/port.md rule 2: tables are data,
// copied). The names below are this crate's public `auth::*` names; a provider
// added to the table needs none (the registry and `ACCESS_POLICIES` iterate
// the rows). `KIMI_CODE` and `GITHUB_COPILOT` are the managed-login declared
// routes: not in `ACCESS_POLICIES`, no contract wire receipt, no support claim.

use crate::generated::tables;

pub const ANTHROPIC_API: AccessPolicy = tables::ANTHROPIC;
pub const CLAUDE_CODE: AccessPolicy = tables::CLAUDE_CODE;
pub const OPENAI_API: AccessPolicy = tables::OPENAI;
pub const OPENAI_CODEX: AccessPolicy = tables::OPENAI_CODEX;
pub const OPENAI_CHAT_API: AccessPolicy = tables::OPENAI_CHAT;
/// xAI, with this crate's login hint: the table's names the reference's
/// Python function (`lm15.auth.login_xai()`); a Rust caller runs
/// `lm15::auth::login("xai")` (AUTH-9), so the hint names that.
pub const XAI: AccessPolicy = AccessPolicy {
    login_hint: Some(XAI_LOGIN_HINT),
    ..tables::XAI
};
pub const KIMI_CODE: AccessPolicy = tables::KIMI_CODE;
pub const GITHUB_COPILOT: AccessPolicy = tables::GITHUB_COPILOT;
pub const GEMINI_API: AccessPolicy = tables::GEMINI;
pub const META: AccessPolicy = tables::META;
pub const GROQ: AccessPolicy = tables::GROQ;
pub const OPENROUTER: AccessPolicy = tables::OPENROUTER;
pub const DEEPSEEK: AccessPolicy = tables::DEEPSEEK;
pub const ZAI: AccessPolicy = tables::ZAI;
pub const DEEPINFRA: AccessPolicy = tables::DEEPINFRA;
pub const TOGETHER: AccessPolicy = tables::TOGETHER;
pub const FIREWORKS: AccessPolicy = tables::FIREWORKS;
pub const PARASAIL: AccessPolicy = tables::PARASAIL;
pub const MOONSHOTAI: AccessPolicy = tables::MOONSHOTAI;
pub const MOONSHOTAI_RESPONSES: AccessPolicy = tables::MOONSHOTAI_RESPONSES;
pub const META_CHAT: AccessPolicy = tables::META_CHAT;
pub const DEEPSEEK_ANTHROPIC: AccessPolicy = tables::DEEPSEEK_ANTHROPIC;
pub const META_ANTHROPIC: AccessPolicy = tables::META_ANTHROPIC;
pub const MOONSHOTAI_ANTHROPIC: AccessPolicy = tables::MOONSHOTAI_ANTHROPIC;
pub const AWS_ANTHROPIC: AccessPolicy = tables::AWS_ANTHROPIC;
pub const BEDROCK_ANTHROPIC: AccessPolicy = tables::BEDROCK_ANTHROPIC;
pub const BEDROCK_CHAT: AccessPolicy = tables::BEDROCK_CHAT;
pub const BEDROCK_MANTLE_CHAT: AccessPolicy = tables::BEDROCK_MANTLE_CHAT;
pub const AZURE: AccessPolicy = tables::AZURE;
pub const AZURE_CHAT: AccessPolicy = tables::AZURE_CHAT;
pub const AZURE_ANTHROPIC: AccessPolicy = tables::AZURE_ANTHROPIC;
pub const VERTEX: AccessPolicy = tables::VERTEX;
pub const VERTEX_EXPRESS: AccessPolicy = tables::VERTEX_EXPRESS;
pub const VERTEX_ANTHROPIC: AccessPolicy = tables::VERTEX_ANTHROPIC;
pub const OLLAMA: AccessPolicy = tables::OLLAMA;
pub const VLLM: AccessPolicy = tables::VLLM;
pub const SGLANG: AccessPolicy = tables::SGLANG;
pub const TYPESAFE: AccessPolicy = tables::TYPESAFE;

/// The table, in the reference's registry order.
pub const ACCESS_POLICIES: &[AccessPolicy] = tables::ACCESS_POLICIES;

// ─── Values the table carries, by their historic names ──────────────
//
// Computed from the table at compile time, so each is the table's value and
// never a second copy.

const fn str_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

const fn entry(table: &'static [(&'static str, &'static str)], key: &str) -> &'static str {
    let mut i = 0;
    while i < table.len() {
        if str_eq(table[i].0, key) {
            return table[i].1;
        }
        i += 1;
    }
    panic!("a table value this crate names is missing from src/generated/tables.rs")
}

const fn some(value: Option<&'static str>) -> &'static str {
    match value {
        Some(v) => v,
        None => panic!("a table value this crate names is missing from src/generated/tables.rs"),
    }
}

const fn setting_env(settings: &'static [HostSetting], name: &str) -> &'static str {
    let mut i = 0;
    while i < settings.len() {
        if str_eq(settings[i].name, name) && !settings[i].env.is_empty() {
            return settings[i].env[0];
        }
        i += 1;
    }
    panic!("a backend setting this crate names is missing from src/generated/tables.rs")
}

/// `lm15/access.py` `DEFAULT_CLAUDE_CODE_VERSION`: the Claude Code release
/// this door says it is (`user-agent: claude-cli/<version>`). Anthropic's
/// server reads it: a model can require a newer release (claude-opus-5-5
/// refuses anything before 2.1.280, live 2026-09-23 and 2026-09-30); callers
/// move it without a release through the `client_version` setting or
/// LM15_CLAUDE_CODE_VERSION.
pub const DEFAULT_CLAUDE_CODE_VERSION: &str = entry(CLAUDE_CODE.backend_options, "client_version");
/// The router's env fallback for claude-code's `client_version`.
pub const CLAUDE_CODE_VERSION_ENV: &str =
    setting_env(CLAUDE_CODE.backend_settings, "client_version");
/// The router's env fallback for openai-codex's `client_version`.
pub const CODEX_CLIENT_VERSION_ENV: &str =
    setting_env(OPENAI_CODEX.backend_settings, "client_version");
/// `lm15/access.py` `DEFAULT_CLAUDE_CODE_SYSTEM_PROMPT`.
pub const DEFAULT_CLAUDE_CODE_SYSTEM_PROMPT: &str = some(CLAUDE_CODE.system_prefix);
/// `lm15/access.py` `DEFAULT_CODEX_BASE_URL`.
pub const DEFAULT_CODEX_BASE_URL: &str = some(OPENAI_CODEX.base_url);
/// `lm15/access.py` `DEFAULT_CODEX_ORIGINATOR`.
pub const DEFAULT_CODEX_ORIGINATOR: &str = entry(OPENAI_CODEX.headers, "originator");
/// `lm15/access.py` `DEFAULT_CODEX_INSTRUCTIONS`.
pub const DEFAULT_CODEX_INSTRUCTIONS: &str = some(OPENAI_CODEX.system_prefix);
/// `lm15/access.py` `DEFAULT_CODEX_CLIENT_VERSION`.
pub const DEFAULT_CODEX_CLIENT_VERSION: &str =
    entry(OPENAI_CODEX.backend_options, "client_version");
/// `lm15/access.py` `DEFAULT_XAI_BASE_URL`.
pub const DEFAULT_XAI_BASE_URL: &str = some(XAI.base_url);

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

/// AUTH-1 § Shared explicit keys (spec/auth.md, ratified 2026-09-09):
/// which explicit `api_keys` entry serves `provider`. Select an exact
/// provider entry first (either spelling). Without one, select the single
/// configured provider whose declared `env_keys` list is identical to the
/// target's non-empty list, including order — derived from the provider
/// declarations, never a second family-name table. Thus `openai` supplies
/// `openai-chat`, but `gemini` does not supply `vertex-express` (overlapping
/// lists are not identical), and empty lists do not join local servers,
/// OAuth stores or cloud chains.
///
/// Several shared candidates without an exact entry are ambiguous:
/// `Err(candidates)`, never chosen by map order, never resolved by comparing
/// secrets or invoking credential providers. `entries` are the configured
/// provider strings (values are never consulted); the answer is the entry
/// as configured.
pub fn shared_api_key_source<'a>(
    entries: impl IntoIterator<Item = &'a str>,
    provider: &str,
) -> Result<Option<&'a str>, Vec<&'a str>> {
    let target = canonical_provider(provider);
    let entries: Vec<&str> = entries.into_iter().collect();
    let exact: Vec<&str> = entries
        .iter()
        .copied()
        .filter(|e| canonical_provider(e) == target)
        .collect();
    let mut candidates = exact;
    if candidates.is_empty() {
        if let Some(policy) = access_policy(&target) {
            if !policy.env_keys.is_empty() {
                candidates = entries
                    .iter()
                    .copied()
                    .filter(|e| {
                        access_policy(e).is_some_and(|other| other.env_keys == policy.env_keys)
                    })
                    .collect();
            }
        }
    }
    match candidates.len() {
        0 => Ok(None),
        1 => Ok(Some(candidates[0])),
        _ => {
            candidates.sort_unstable();
            Err(candidates)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use AuthScheme::{Bearer, SigV4, XApiKey};
    use CredentialPolicy::OAuth;

    #[test]
    fn provider_strings_are_unique() {
        let mut names = known_providers();
        names.dedup();
        assert_eq!(names.len(), ACCESS_POLICIES.len());
        assert_eq!(ACCESS_POLICIES.len(), 36); // + the four inference hosts (2026-09-26)
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

    /// `lm15/features.py:280-283`: sigv4 needs a host with a service; a
    /// cloud chain needs a host (vertex-express is the stated exception).
    #[test]
    fn sigv4_and_cloud_chains_name_a_host() {
        for policy in ACCESS_POLICIES {
            if policy.auth_scheme.contains(&SigV4) {
                assert!(
                    policy.host.and_then(|h| h.sigv4_service).is_some(),
                    "{}",
                    policy.provider
                );
            }
            if policy.is_cloud_chain() {
                assert!(policy.host.is_some(), "{}", policy.provider);
            }
            assert!(!policy.auth_scheme.is_empty(), "{}", policy.provider);
        }
    }

    /// `lm15/access.py:95`: the user-agent is the versioned CLI string.
    #[test]
    fn claude_code_user_agent_carries_the_version() {
        let ua = CLAUDE_CODE
            .headers
            .iter()
            .find(|(k, _)| *k == "user-agent")
            .map(|(_, v)| *v)
            .unwrap();
        assert_eq!(ua, format!("claude-cli/{DEFAULT_CLAUDE_CODE_VERSION}"));
    }

    #[test]
    fn auth_header_projection() {
        assert_eq!(ANTHROPIC_API.auth_header(), XApiKey);
        assert_eq!(AZURE.auth_header(), XApiKey);
        assert_eq!(BEDROCK_CHAT.auth_header(), Bearer);
        assert_eq!(VERTEX_EXPRESS.auth_header(), Bearer);
        assert_eq!(
            OPENAI_CODEX.backend_option("client_version"),
            Some("0.147.0")
        );
        assert_eq!(
            VERTEX_ANTHROPIC.host.unwrap().path_for("messages/stream"),
            Some("/publishers/anthropic/models/{model}:streamRawPredict")
        );
    }

    /// Every bound policy's base URL is the compat table's URL for its
    /// preset (`lm15/registry.py` rule: one copy of each URL).
    #[test]
    fn bound_base_urls_match_the_compat_tables() {
        assert_eq!(GROQ.base_url, Some("https://api.groq.com/openai/v1"));
        assert_eq!(DEEPSEEK.base_url, Some("https://api.deepseek.com"));
        assert_eq!(
            DEEPSEEK_ANTHROPIC.base_url,
            Some("https://api.deepseek.com/anthropic/v1")
        );
        assert_eq!(
            MOONSHOTAI_RESPONSES.base_url,
            Some("https://api.moonshot.ai/v1")
        );
        assert_eq!(OLLAMA.base_url, Some("http://localhost:11434/v1"));
    }
}

//! `LMRouter` (playbooks/api-family.md § The core loop; the reference's
//! `lm15.router`): a lookup table you can read, not a framework. Three
//! resolution rungs in fixed order, then the AUTH-1 credential chain,
//! then one adapter per provider, built lazily and reused.
//!
//! Rungs (`resolve`):
//!
//! 1. explicit prefix — `"openai:gpt-4.1-mini"`; the string is split on
//!    the FIRST `:`; a head that names a registry provider (either
//!    spelling) routes, anything else is a bare model id (a fine-tune id
//!    like `ft:gpt-4.1:org` needs `openai:ft:gpt-4.1:org`);
//! 2. catalog — `RouterConfig::catalog` entries by id or alias; an exact
//!    id beats an alias; more than one provider, or more than one entry
//!    of one provider, is an error, never a pick by order;
//! 3. built-in rules — [`DEFAULT_RULES`] prefix match, first wins.
//!
//! The reference has a rung 0 (a `provider` attribute on the model value:
//! a `str` subclass from a catalog package). Rust strings carry no
//! attributes; the rung does not exist here (stated in the README). Its
//! catalog rung discovers packages; this port takes the catalog as data.
//!
//! `resolve` is pure: no network, no file reads, no secret values (it
//! records WHICH env var would be read, never the value). `lm` reads the
//! credential: the explicit `api_keys` entry, then — by the provider's
//! declared policy (AUTH-1) — the stored login, the declared env keys in
//! order, a keyless local server's placeholder key.
//!
//! Errors are the ratified vocabulary (spec/vocabularies.md, 2026-09-08):
//! a model string that routes nowhere is `UnknownModelError`
//! (`unknown_model`, carrying `model`); one the catalog offers under more
//! than one provider is `AmbiguousModelError` (`ambiguous_model`, carrying
//! `model` and `providers`); a provider with no credential is a
//! `NotConfiguredError` (the class the reference's `MissingCredentialError`
//! inherits). All three sit under `ConfigurationError`. There is no
//! router-wide class or code. Pinned by `--direction router`.

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::adapter::{EventStream, LmBuilder, ProviderLM};
use crate::auth::policy::shared_api_key_source;
use crate::auth::{
    canonical_provider, AccessPolicy, CredentialPolicy, CredentialProvider, StoredLogin,
};
#[cfg(feature = "native")]
use crate::cloud::chains::{profile_setting, ChainContext, ChainProvider};
use crate::cloud::hosts::{resolve_settings_with_endpoint, HostSettings};
use crate::errors::{ErrorMeta, Lm15Error};
use crate::registry::{lookup, DialectId, EntryKind, ProviderDefinition, PROVIDERS};
use crate::transport::Transport;
use crate::types::{JsonObject, ModelInfo, Request, Response};
use serde_json::Value;

// ─── rules ───────────────────────────────────────────────────────────

/// Maps a model-id prefix to a provider. That is all a rule is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouteRule {
    pub prefix: &'static str,
    pub provider: &'static str,
    /// A short human rationale, surfaced by `Resolution::describe`.
    pub note: &'static str,
}

const fn rule(prefix: &'static str, provider: &'static str, note: &'static str) -> RouteRule {
    RouteRule {
        prefix,
        provider,
        note,
    }
}

/// The complete built-in knowledge of the router (`lm15/router.py`
/// `DEFAULT_RULES`, copied as data — port.md rule 2). First match wins.
/// A convenience, not a registry of truth: a new model family needs a
/// release, a catalog, or the `provider:` prefix.
pub const DEFAULT_RULES: &[RouteRule] = &[
    rule("jev", "typesafe", "TypeSafe System One judgment models"),
    rule("claude-", "anthropic", "Anthropic Claude family"),
    rule(
        "gpt-",
        "openai",
        "OpenAI GPT family (Responses API; use openai-chat: for Chat Completions)",
    ),
    rule("o1", "openai", "OpenAI o1 reasoning family"),
    rule("o3", "openai", "OpenAI o3 reasoning family"),
    rule("o4", "openai", "OpenAI o4 reasoning family"),
    rule("gemini-", "gemini", "Google Gemini family"),
    rule(
        "gemma-",
        "gemini",
        "Google Gemma open models, served by the Gemini API (live /models listing 2026-09-01)",
    ),
    rule(
        "nano-banana",
        "gemini",
        "Google image models on the Gemini API (live /models listing 2026-09-01)",
    ),
    rule(
        "grok-",
        "xai",
        "xAI Grok family (XAI_API_KEY or subscription OAuth)",
    ),
    rule("sora-", "openai", "OpenAI Sora video generation"),
    rule("veo-", "gemini", "Google Veo video generation"),
    rule(
        "chat-latest",
        "openai",
        "OpenAI rolling chat alias (live /models listing 2026-09-01)",
    ),
];

// ─── resolution ──────────────────────────────────────────────────────

/// Which rung answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteSource {
    Prefix,
    Catalog,
    Rule,
}

impl RouteSource {
    pub fn as_str(self) -> &'static str {
        match self {
            RouteSource::Prefix => "prefix",
            RouteSource::Catalog => "catalog",
            RouteSource::Rule => "rule",
        }
    }
}

/// The complete answer to "how did you route this string". `resolve`
/// returning this IS the explain method; there is no separate one.
#[derive(Debug, Clone, PartialEq)]
pub struct Resolution {
    /// The verbatim input string.
    pub requested: String,
    /// The id sent on the wire (prefix stripped; an alias resolved).
    pub model: String,
    /// The canonical provider string.
    pub provider: String,
    /// The adapter's family name (`AnthropicLM`, `OpenAIChatLM`, ...).
    pub adapter: &'static str,
    pub source: RouteSource,
    /// The matching rule when `source` is `Rule`.
    pub rule: Option<RouteRule>,
    /// The env var the key would be read from; `None` for a stored-login
    /// provider, a keyless local preset, or when an explicit `api_keys`
    /// entry overrides env lookup.
    pub env_key: Option<&'static str>,
    /// Catalog metadata when `source` is `Catalog`.
    pub model_info: Option<ModelInfo>,
    /// The compat preset name when routed through a bound registry entry.
    pub compat: Option<&'static str>,
    /// Application declaration, not a receipted built-in provider.
    pub declared: bool,
    pub declared_env_keys: Vec<String>,
    pub declared_compat: Option<crate::compat::Compat>,
    pub declaration_note: Option<String>,
    pub base_url: Option<String>,
    pub credential: Option<String>,
    pub credential_policy: String,
}

impl Resolution {
    /// One paragraph, human-readable.
    pub fn describe(&self) -> String {
        let mut parts = vec![format!(
            "{:?} -> provider {:?} ({})",
            self.requested, self.provider, self.adapter
        )];
        match self.source {
            RouteSource::Prefix => parts.push("via explicit provider prefix".into()),
            RouteSource::Catalog => parts.push("via catalog match".into()),
            RouteSource::Rule => {
                if let Some(rule) = &self.rule {
                    let note = if rule.note.is_empty() {
                        String::new()
                    } else {
                        format!(" — {}", rule.note)
                    };
                    parts.push(format!("via built-in rule prefix={:?}{note}", rule.prefix));
                }
            }
        }
        if self.declared {
            parts.push("declared by RouterConfig (no built-in receipts)".into());
            if let Some(note) = &self.declaration_note {
                if !note.is_empty() {
                    parts.push(note.clone());
                }
            }
            if !self.declared_env_keys.is_empty() {
                parts.push(format!(
                    "declared key variables: {}",
                    self.declared_env_keys.join(", ")
                ));
            }
        }
        if let Some(url) = &self.base_url {
            parts.push(format!("endpoint {url}"));
        }
        if let Some(compat) = self.compat {
            parts.push(format!("compat preset {compat:?}"));
        }
        parts.push(format!("wire model {:?}", self.model));
        if let Some(name) = &self.credential {
            parts.push(format!(
                "named credential {name:?}; default chain is not walked"
            ));
            return format!("{}.", parts.join("; "));
        }
        let definition = lookup(&self.provider);
        let policy = definition.map(|d| d.access().credential_policy);
        match policy {
            Some(CredentialPolicy::OAuthUnlessExplicit) => {
                // resolve() is pure (no file reads): the chain is described,
                // not decided.
                let mut chain = String::from(
                    "key from explicit api_keys, else the stored subscription OAuth credential",
                );
                if let Some(key) = self.env_key {
                    chain.push_str(&format!(", else ${key}"));
                }
                parts.push(chain);
            }
            _ if self.env_key.is_some() => {
                parts.push(format!("key from ${}", self.env_key.unwrap()));
            }
            Some(CredentialPolicy::OAuth) => {
                parts.push("local OAuth credential (no env key)".into());
            }
            _ if definition.is_some_and(|d| d.placeholder_key.is_some()) => {
                parts
                    .push("key from explicit api_keys or the preset's local-server default".into());
            }
            _ => parts.push("key from explicit api_keys".into()),
        }
        format!("{}.", parts.join("; "))
    }
}

impl fmt::Display for Resolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.describe())
    }
}

// ─── config ──────────────────────────────────────────────────────────

type SharedCredentials = Arc<dyn CredentialProvider + Send + Sync>;

/// A provider local to one router. Its compat is an application declaration,
/// never a claim that the built-in provider corpus measured this server.
#[derive(Clone)]
pub struct DeclaredProvider {
    pub id: String,
    pub aliases: Vec<String>,
    pub base_url: String,
    pub env_keys: Vec<String>,
    pub placeholder_key: Option<String>,
    pub compat: crate::compat::Compat,
    pub note: String,
    factory: Option<DeclaredFactory>,
}

type DeclaredFactory = Arc<dyn Fn(LmBuilder) -> Result<ProviderLM, Lm15Error> + Send + Sync>;

impl DeclaredProvider {
    pub fn chat(
        id: impl Into<String>,
        base_url: impl Into<String>,
        compat: crate::compat::OpenAIChatCompat,
    ) -> Self {
        Self::new(
            id.into(),
            base_url.into(),
            crate::compat::Compat::OpenAIChat(compat),
        )
    }
    pub fn responses(
        id: impl Into<String>,
        base_url: impl Into<String>,
        compat: crate::compat::OpenAIResponsesCompat,
    ) -> Self {
        Self::new(
            id.into(),
            base_url.into(),
            crate::compat::Compat::OpenAIResponses(compat),
        )
    }
    pub fn anthropic(
        id: impl Into<String>,
        base_url: impl Into<String>,
        compat: crate::compat::AnthropicCompat,
    ) -> Self {
        Self::new(
            id.into(),
            base_url.into(),
            crate::compat::Compat::Anthropic(compat),
        )
    }
    fn new(id: String, base_url: String, compat: crate::compat::Compat) -> Self {
        Self {
            id: canonical_provider(&id),
            base_url,
            compat,
            aliases: Vec::new(),
            env_keys: Vec::new(),
            placeholder_key: None,
            note: String::new(),
            factory: None,
        }
    }
    pub fn aliases(mut self, aliases: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.aliases = aliases
            .into_iter()
            .map(|s| canonical_provider(&s.into()))
            .collect();
        self
    }
    pub fn env_keys(mut self, keys: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.env_keys = keys.into_iter().map(Into::into).collect();
        self
    }
    pub fn placeholder_key(mut self, key: impl Into<String>) -> Self {
        self.placeholder_key = Some(key.into());
        self
    }
    pub fn factory(
        mut self,
        factory: impl Fn(LmBuilder) -> Result<ProviderLM, Lm15Error> + Send + Sync + 'static,
    ) -> Self {
        self.factory = Some(Arc::new(factory));
        self
    }
    fn template(&self) -> &'static ProviderDefinition {
        lookup(match &self.compat {
            crate::compat::Compat::OpenAIChat(_) => "openai-chat",
            crate::compat::Compat::OpenAIResponses(_) => "openai",
            crate::compat::Compat::Anthropic(_) => "anthropic",
            crate::compat::Compat::None => "openai-chat",
        })
        .expect("built-in dialect declaration")
    }
}

impl fmt::Debug for DeclaredProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeclaredProvider")
            .field("id", &self.id)
            .field("aliases", &self.aliases)
            .field("base_url_supplied", &!self.base_url.is_empty())
            .field("env_keys", &self.env_keys)
            .field("compat", &self.compat)
            .field("factory", &self.factory.is_some())
            .finish_non_exhaustive()
    }
}

/// Everything the router consults. All explicit, nothing discovered
/// behind your back.
#[derive(Clone)]
pub struct RouterConfig {
    rules: Vec<RouteRule>,
    env: Option<BTreeMap<String, String>>,
    api_keys: BTreeMap<String, SharedCredentials>,
    base_urls: BTreeMap<String, String>,
    settings: BTreeMap<String, HostSettings>,
    transport: Option<Arc<dyn Transport>>,
    catalog: Vec<ModelInfo>,
    credentials_path: Option<PathBuf>,
    providers: Vec<DeclaredProvider>,
    configuration_errors: Vec<String>,
    adaptations: crate::adaptation::AdaptationPolicy,
    credentials: BTreeMap<String, String>,
    timeouts: crate::transport::Timeouts,
    max_connections: usize,
    budget_explicit: bool,
    #[cfg(feature = "native")]
    auth: Option<crate::login::Auth>,
}

impl Default for RouterConfig {
    fn default() -> Self {
        RouterConfig {
            rules: DEFAULT_RULES.to_vec(),
            env: None,
            api_keys: BTreeMap::new(),
            base_urls: BTreeMap::new(),
            settings: BTreeMap::new(),
            transport: None,
            catalog: Vec::new(),
            credentials_path: None,
            providers: Vec::new(),
            configuration_errors: Vec::new(),
            adaptations: crate::adaptation::AdaptationPolicy::Note,
            credentials: BTreeMap::new(),
            timeouts: crate::transport::Timeouts::default(),
            max_connections: crate::transport::DEFAULT_MAX_CONNECTIONS,
            budget_explicit: false,
            #[cfg(feature = "native")]
            auth: None,
        }
    }
}

impl RouterConfig {
    pub fn new() -> Self {
        RouterConfig::default()
    }

    pub fn providers(mut self, providers: impl IntoIterator<Item = DeclaredProvider>) -> Self {
        self.providers = providers.into_iter().collect();
        self
    }

    pub fn adaptations(mut self, policy: crate::adaptation::AdaptationPolicy) -> Self {
        self.adaptations = policy;
        self
    }

    /// Managed authentication (AUTH-15 mode B): an [`Auth`](crate::login::Auth)
    /// whose saved connections supply the credential when no explicit
    /// `api_key` / named `credential` does. With it, environment keys, other
    /// tools' login files and the machine's cloud identity are never
    /// consulted: a missing, expired, rejected or signed-out connection is a
    /// typed `AuthOperationError`, never a silent switch to a metered key.
    /// Keyless local servers still work without a connection. It also routes
    /// the connection-only providers (`kimi-code`, `github-copilot`).
    #[cfg(feature = "native")]
    pub fn auth(mut self, auth: crate::login::Auth) -> Self {
        self.auth = Some(auth);
        self
    }

    #[cfg(feature = "native")]
    pub fn managed_auth(&self) -> Option<&crate::login::Auth> {
        self.auth.as_ref()
    }

    pub fn credential(mut self, provider: &str, name: impl Into<String>) -> Self {
        self.credentials
            .insert(canonical_provider(provider), name.into());
        self
    }
    pub fn timeouts(mut self, timeouts: crate::transport::Timeouts) -> Self {
        self.timeouts = timeouts;
        self.budget_explicit = true;
        self
    }
    pub fn max_connections(mut self, maximum: usize) -> Self {
        self.max_connections = maximum;
        self.budget_explicit = true;
        self
    }

    fn canonical(&self, provider: &str) -> String {
        let name = canonical_provider(provider);
        self.providers
            .iter()
            .find(|d| d.id == name || d.aliases.contains(&name))
            .map(|d| d.id.clone())
            .unwrap_or(name)
    }

    fn declared(&self, provider: &str) -> Option<&DeclaredProvider> {
        let name = self.canonical(provider);
        self.providers.iter().find(|d| d.id == name)
    }

    /// The prefix rules, replacing [`DEFAULT_RULES`].
    pub fn rules(mut self, rules: impl IntoIterator<Item = RouteRule>) -> Self {
        self.rules = rules.into_iter().collect();
        self
    }

    /// The environment to read (hermetic tests); the process environment
    /// at lookup time otherwise.
    pub fn env<K: Into<String>, V: Into<String>>(
        mut self,
        env: impl IntoIterator<Item = (K, V)>,
    ) -> Self {
        self.env = Some(env.into_iter().map(|(k, v)| (k.into(), v.into())).collect());
        self
    }

    /// An explicit credential for a provider (either spelling); beats
    /// every other rung (AUTH-1). A string is the `ApiKey` shorthand; any
    /// `CredentialProvider` is invoked per request. The entry also serves a
    /// sibling provider whose declared env-key list is identical (AUTH-1 §
    /// Shared explicit keys, 2026-09-09): `openai` supplies `openai-chat`.
    pub fn api_key(
        mut self,
        provider: &str,
        credential: impl CredentialProvider + Send + Sync + 'static,
    ) -> Self {
        if self
            .api_keys
            .insert(canonical_provider(provider), Arc::new(credential))
            .is_some()
        {
            self.configuration_errors.push(format!(
                "duplicate api_key entry for {provider:?} (including underscore aliases)"
            ));
        }
        self
    }

    /// The trusted endpoint root for this provider, including cloud doors.
    /// A cloud door retains its policy and appends its rendered door path.
    pub fn base_url(mut self, provider: &str, url: impl Into<String>) -> Self {
        self.base_urls
            .insert(canonical_provider(provider), url.into());
        self
    }

    /// Host settings for a cloud door (AUTH-10): `{"bedrock-anthropic":
    /// {"region": "us-east-1"}}`. A setting not given here is read from
    /// its env variables in order, then its default; `region` and
    /// `resource` have none and raise.
    pub fn settings(mut self, provider: &str, settings: HostSettings) -> Self {
        self.settings.insert(canonical_provider(provider), settings);
        self
    }

    /// One host setting.
    pub fn setting(mut self, provider: &str, name: &str, value: &str) -> Self {
        self.settings
            .entry(canonical_provider(provider))
            .or_default()
            .insert(name.to_string(), value.to_string());
        self
    }

    /// The transport every adapter the router builds sends through.
    pub fn transport(mut self, transport: impl Transport + 'static) -> Self {
        self.transport = Some(Arc::new(transport));
        self
    }

    pub fn transport_shared(mut self, transport: Arc<dyn Transport>) -> Self {
        self.transport = Some(transport);
        self
    }

    /// The model catalog for rung 2 (the reference discovers one from
    /// installed packages; here it is data).
    pub fn catalog(mut self, catalog: impl IntoIterator<Item = ModelInfo>) -> Self {
        self.catalog = catalog.into_iter().collect();
        self
    }

    /// One stored-login file for every stored-login provider (tests);
    /// the AUTH-8 paths otherwise.
    pub fn credentials_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.credentials_path = Some(path.into());
        self
    }

    fn env_value(&self, key: &str) -> Option<String> {
        match &self.env {
            Some(map) => map.get(key).cloned(),
            None => std::env::var(key).ok(),
        }
        .filter(|v| !v.is_empty())
    }

    /// The whole environment as a map (host settings resolve against one).
    fn env_map(&self) -> BTreeMap<String, String> {
        match &self.env {
            Some(map) => map.clone(),
            None => std::env::vars().collect(),
        }
    }

    /// The explicit entry that serves `provider` (AUTH-1 § Shared explicit
    /// keys): exact, else the one sibling with an identical env-key
    /// declaration. Ambiguity is a `NotConfiguredError`, never a choice by
    /// map order.
    fn explicit_credential(&self, provider: &str) -> Result<Option<SharedCredentials>, Lm15Error> {
        if let Some(value) = self.api_keys.get(&self.canonical(provider)) {
            return Ok(Some(value.clone()));
        }
        let declared_keys = self.declared(provider).map(|d| d.env_keys.clone());
        let keys = declared_keys.unwrap_or_else(|| {
            lookup(provider)
                .map(|d| d.access().env_keys.iter().map(|s| s.to_string()).collect())
                .unwrap_or_default()
        });
        if !keys.is_empty() {
            let candidates: Vec<_> = self
                .api_keys
                .iter()
                .filter(|(name, _)| {
                    let other = self
                        .declared(name)
                        .map(|d| d.env_keys.clone())
                        .unwrap_or_else(|| {
                            lookup(name)
                                .map(|d| {
                                    d.access().env_keys.iter().map(|s| s.to_string()).collect()
                                })
                                .unwrap_or_default()
                        });
                    other == keys
                })
                .collect();
            if candidates.len() == 1 {
                return Ok(Some(candidates[0].1.clone()));
            }
            if candidates.len() > 1 {
                return Err(not_configured(format!(
                    "ambiguous shared api_keys for {provider:?}; supply an exact entry"
                )));
            }
        }
        match shared_api_key_source(self.api_keys.keys().map(String::as_str), provider) {
            Ok(None) => Ok(None),
            Ok(Some(entry)) => Ok(self.api_keys.get(entry).cloned()),
            Err(candidates) => Err(not_configured(format!(
                "RouterConfig api_keys: ambiguous credentials for {provider:?} from {}; \
                 supply one entry under {provider:?} or keep only one shared entry",
                candidates
                    .iter()
                    .map(|c| format!("{c:?}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ))),
        }
    }

    /// The explicit `base_urls` entry for `provider` (exact provider only;
    /// a URL is never shared the way a key is).
    fn explicit_base_url(&self, provider: &str) -> Option<&str> {
        self.base_urls.get(provider).map(String::as_str)
    }

    /// Every provider string this config is keyed by (`api_key`,
    /// `base_url`, `settings`) must name a routable provider: an entry that
    /// matches nothing is otherwise silently ignored, and the request goes
    /// out on whatever the environment holds — the wrong account, with
    /// nothing said (AUTH-1). The check runs when a router is built.
    fn check_provider_keyed(&self) -> Result<(), Lm15Error> {
        if let Some(error) = self.configuration_errors.first() {
            return Err(not_configured(error.clone()));
        }
        for endpoint in self.base_urls.values() {
            crate::cloud::hosts::join_endpoint(endpoint, "")?;
        }
        let mut names = std::collections::BTreeSet::new();
        for d in &self.providers {
            if matches!(d.compat, crate::compat::Compat::None)
                || (d.placeholder_key.is_some() && !d.env_keys.is_empty())
            {
                return Err(not_configured(format!(
                    "{}: declare a dialect compat and either env keys or a keyless placeholder",
                    d.id
                )));
            }
            for name in std::iter::once(&d.id).chain(d.aliases.iter()) {
                if name.is_empty()
                    || name.contains([':', '/', ' '])
                    || lookup(name).is_some()
                    || !names.insert(name.clone())
                {
                    return Err(not_configured(format!("declared provider id/alias {name:?} is invalid or collides with another provider")));
                }
            }
            crate::cloud::hosts::join_endpoint(&d.base_url, "")?;
        }
        for (provider, name) in &self.credentials {
            if !matches!(
                name.as_str(),
                "platform" | "workload" | "environment" | "cli"
            ) || !lookup(provider).is_some_and(|d| d.access().is_cloud_chain())
            {
                return Err(not_configured(format!("{provider}: named credential {name:?} requires a cloud chain and one of platform, workload, environment, cli")));
            }
            if self.explicit_credential(provider)?.is_some() {
                return Err(not_configured(format!(
                    "{provider}: credential and api_key are mutually exclusive"
                )));
            }
        }
        let known: Vec<&str> = crate::registry::PROVIDERS.iter().map(|d| d.id).collect();
        for (field, keys) in [
            ("api_key", self.api_keys.keys().collect::<Vec<_>>()),
            ("base_url", self.base_urls.keys().collect::<Vec<_>>()),
            ("settings", self.settings.keys().collect::<Vec<_>>()),
        ] {
            for key in keys {
                if lookup(key).is_some() || self.declared(key).is_some() {
                    continue;
                }
                let close = known
                    .iter()
                    .copied()
                    .filter(|k| similarity(k, key) >= 0.6)
                    .max_by(|a, b| similarity(a, key).total_cmp(&similarity(b, key)));
                let hint = close
                    .map(|c| format!(" Did you mean {c:?}?"))
                    .unwrap_or_default();
                return Err(not_configured(format!(
                    "RouterConfig::{field}: {key:?} is not a provider lm15 routes to.{hint} \
                     router.resolve(model).provider (or resolve_openai_chat) names the one a \
                     model string uses; known: {}",
                    known.join(", ")
                )));
            }
        }
        Ok(())
    }
}

/// difflib-style ratio: 2·LCS / (|a| + |b|).
fn similarity(a: &str, b: &str) -> f64 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let mut dp = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            dp[i][j] = if a[i - 1] == b[j - 1] {
                dp[i - 1][j - 1] + 1
            } else {
                dp[i - 1][j].max(dp[i][j - 1])
            };
        }
    }
    (2 * dp[a.len()][b.len()]) as f64 / (a.len() + b.len()) as f64
}

fn not_configured(message: String) -> Lm15Error {
    Lm15Error::NotConfiguredError(ErrorMeta::new(message))
}

impl fmt::Debug for RouterConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // AUTH-5: credential values never render; the env map may hold
        // key values, so only its presence is shown.
        f.debug_struct("RouterConfig")
            .field("rules", &self.rules.len())
            .field(
                "env",
                &self.env.as_ref().map(|m| format!("<{} vars>", m.len())),
            )
            .field("api_keys", &self.api_keys.keys().collect::<Vec<_>>())
            .field(
                "base_url_providers",
                &self.base_urls.keys().collect::<Vec<_>>(),
            )
            .field("settings", &self.settings)
            .field("catalog", &self.catalog.len())
            .field("credentials_path", &self.credentials_path)
            .finish_non_exhaustive()
    }
}

// ─── the router ──────────────────────────────────────────────────────

/// Routes model strings to provider adapters. Config is immutable; the
/// only state is one adapter per provider, built lazily and reused.
pub struct LMRouter {
    config: RouterConfig,
    lms: Mutex<BTreeMap<String, Arc<ProviderLM>>>,
}

impl Default for LMRouter {
    fn default() -> Self {
        LMRouter::new()
    }
}

impl LMRouter {
    pub fn new() -> Self {
        LMRouter::with_config(RouterConfig::default()).expect("an empty config names no provider")
    }

    /// A router over `config`. Every provider string the config is keyed
    /// by must name a routable provider; a near miss is named
    /// (`NotConfiguredError`) rather than silently ignored.
    pub fn with_config(mut config: RouterConfig) -> Result<Self, Lm15Error> {
        #[cfg(feature = "native")]
        if config.auth.is_some() {
            // A managed router also routes the connection-only doors: declared
            // providers, added only here because only a managed Auth can hold
            // their credential.
            for declared in crate::login::declared_providers() {
                if !config
                    .providers
                    .iter()
                    .any(|d| canonical_provider(&d.id) == declared.id)
                {
                    config.providers.push(declared);
                }
            }
        }
        for definition in &mut config.providers {
            definition.id = canonical_provider(&definition.id);
            definition.aliases = definition
                .aliases
                .iter()
                .map(|name| canonical_provider(name))
                .collect();
        }
        config.check_provider_keyed()?;
        if config.transport.is_some() && config.budget_explicit {
            return Err(not_configured("custom transport and router connection budgets are mutually exclusive; configure the transport itself".into()));
        }
        let mut keys = BTreeMap::new();
        for (name, credential) in std::mem::take(&mut config.api_keys) {
            let name = config.canonical(&name);
            if keys.insert(name.clone(), credential).is_some() {
                return Err(not_configured(format!(
                    "duplicate api_key aliases for {name:?}"
                )));
            }
        }
        config.api_keys = keys;
        config.catalog = std::mem::take(&mut config.catalog)
            .into_iter()
            .map(|mut info| {
                info.provider = config.canonical(&info.provider);
                info
            })
            .collect();
        config.base_urls = std::mem::take(&mut config.base_urls)
            .into_iter()
            .map(|(k, v)| (config.canonical(&k), v))
            .collect();
        config.settings = std::mem::take(&mut config.settings)
            .into_iter()
            .map(|(k, v)| (config.canonical(&k), v))
            .collect();
        #[cfg(feature = "native")]
        if config.transport.is_none() {
            config.transport = Some(Arc::new(
                crate::transport::HttpTransport::builder()
                    .timeouts(config.timeouts)
                    .max_connections(config.max_connections)
                    .build()?,
            ));
        }
        Ok(LMRouter {
            config,
            lms: Mutex::new(BTreeMap::new()),
        })
    }

    pub fn config(&self) -> &RouterConfig {
        &self.config
    }

    /// Pure lookup: no network, no file reads, no secret values.
    pub fn resolve(&self, model: &str) -> Result<Resolution, Lm15Error> {
        resolve(model, &self.config)
    }

    /// `resolve`, then construct-or-reuse the provider adapter. The
    /// adapter is an ordinary `ProviderLM`: keep it and configure it
    /// yourself when the router's defaults are not yours.
    pub fn lm(&self, model: &str) -> Result<Arc<ProviderLM>, Lm15Error> {
        let resolution = self.resolve(model)?;
        self.lm_for(&resolution)
    }

    fn lm_for(&self, resolution: &Resolution) -> Result<Arc<ProviderLM>, Lm15Error> {
        let mut lms = self
            .lms
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(lm) = lms.get(&resolution.provider) {
            return Ok(Arc::clone(lm));
        }
        let lm = Arc::new(build_lm(&resolution.provider, &self.config)?);
        lms.insert(resolution.provider.clone(), Arc::clone(&lm));
        Ok(lm)
    }

    pub fn plan(&self, request: &Request) -> Result<Vec<crate::adaptation::Adaptation>, Lm15Error> {
        let resolution = self.resolve(&request.model)?;
        // Planning uses a parse-only binding: no credential resolution, files,
        // commands, metadata endpoints or provider transport are consulted.
        let mut builder = if let Some(d) = self.config.declared(&resolution.provider) {
            LmBuilder::for_entry(d.template())
                .provider_name(&d.id)
                .provider_aliases(&d.aliases)
                .compat(d.compat.clone())
                .base_url(&d.base_url)
        } else {
            LmBuilder::for_entry(lookup(&resolution.provider).expect("resolved provider"))
        };
        builder = builder
            .api_key("plan-only")
            .adaptations(self.config.adaptations);
        builder.plan(&routed(request, &resolution))
    }

    pub fn explain_auth(&self, model: &str) -> Result<crate::auth::Report, Lm15Error> {
        let resolution = self.resolve(model)?;
        if let Some(d) = self.config.declared(&resolution.provider) {
            let explicit = self.config.explicit_credential(&d.id)?;
            let chosen = d
                .env_keys
                .iter()
                .find(|k| self.config.env_value(k).is_some());
            let (kind, label, configured) = if let Some(credential) = explicit {
                let source = credential
                    .source()
                    .unwrap_or_else(crate::auth::CredentialSource::callable);
                (source.kind, source.label, true)
            } else if let Some(key) = chosen {
                (format!("env:{key}"), format!("env ${key}"), true)
            } else if d.placeholder_key.is_some() {
                (
                    "placeholder".into(),
                    "declared local-server placeholder".into(),
                    true,
                )
            } else {
                (
                    "api_keys".into(),
                    "no declared credential present".into(),
                    false,
                )
            };
            return Ok(crate::auth::Report {
                provider: d.id.clone(),
                steps: vec![crate::auth::Step {
                    kind,
                    source: label,
                    detail: "application declaration; no built-in provider receipts".into(),
                    state: if configured {
                        crate::auth::StepState::Selected
                    } else {
                        crate::auth::StepState::Absent
                    },
                }],
                configured,
                settings: Vec::new(),
                named_credential: None,
                base_url: resolution.base_url,
                endpoint_source: Some("RouterConfig declaration".into()),
            });
        }
        let options = crate::auth::ExplainOptions {
            env: Some(self.config.env_map().into_iter().collect()),
            api_key_providers: self.config.api_keys.keys().cloned().collect(),
            callable_providers: self
                .config
                .api_keys
                .iter()
                .filter(|(_, c)| c.source().is_some_and(|s| s.kind == "callable"))
                .map(|(k, _)| k.clone())
                .collect(),
            credentials_path: self.config.credentials_path.clone(),
            settings: self.config.settings.get(&resolution.provider).cloned(),
            credential: self.config.credentials.get(&resolution.provider).cloned(),
            base_url: self
                .config
                .explicit_base_url(&resolution.provider)
                .map(str::to_string),
            #[cfg(feature = "native")]
            auth: self.config.auth.clone(),
            ..Default::default()
        };
        crate::auth::explain_auth(&resolution.provider, &options).map_err(Lm15Error::from)
    }

    pub fn doctor(&self, model: &str) -> Result<String, Lm15Error> {
        Ok(format!(
            "{}\n{}",
            self.resolve(model)?.describe(),
            self.explain_auth(model)?.describe()
        ))
    }

    pub async fn complete(&self, request: &Request) -> Result<Response, Lm15Error> {
        let resolution = self.resolve(&request.model)?;
        let lm = self.lm_for(&resolution)?;
        lm.complete(&routed(request, &resolution)).await
    }

    /// The canonical events of one streamed call (see
    /// `ProviderLM::stream`). A routing or credential failure is the
    /// stream's one `Err` item.
    pub fn stream(&self, request: &Request) -> EventStream {
        let resolution = match self.resolve(&request.model) {
            Ok(resolution) => resolution,
            Err(err) => return EventStream::failed(String::new(), err),
        };
        match self.lm_for(&resolution) {
            Ok(lm) => lm.stream(&routed(request, &resolution)),
            Err(err) => EventStream::failed(resolution.provider, err),
        }
    }
}

// ─── the OpenAI-shaped door (api-family § Ingest) ────────────────────

/// litellm's routing prefixes (`<provider>/<model>`) that name a door lm15
/// has, copied as data. A prefix absent here is refused by name — never
/// routed by rule — because litellm's own rule is "a leading known provider
/// name is the provider", and an unknown one is an error there too. Where
/// litellm's name covers two lm15 doors (bedrock, vertex_ai: Anthropic or
/// not, by model) it is left out: choosing would be a guess.
pub const LITELLM_PROVIDER_PREFIXES: &[(&str, &str)] = &[
    ("openai", "openai-chat"),
    ("anthropic", "anthropic"),
    ("gemini", "gemini"),
    ("groq", "groq"),
    ("openrouter", "openrouter"),
    ("deepseek", "deepseek"),
    ("xai", "xai"),
    ("ollama", "ollama"),
    ("ollama_chat", "ollama"),
    ("hosted_vllm", "vllm"),
    ("moonshot", "moonshotai"),
    ("azure", "azure-chat"),
];

/// Keyword arguments of `create()` / `completion()` that configure the
/// CLIENT, not the request: refused with the lm15 place they belong.
const CLIENT_KEYWORDS: &[(&str, &str)] = &[
    (
        "api_key",
        "RouterConfig::api_key(provider, key) or the environment",
    ),
    ("api_base", "RouterConfig::base_url(provider, url)"),
    ("base_url", "RouterConfig::base_url(provider, url)"),
    ("timeout", "RouterConfig::transport(...)"),
    (
        "num_retries",
        "your own retry loop over Lm15Error::is_retryable (lm15 never retries)",
    ),
    (
        "max_retries",
        "your own retry loop over Lm15Error::is_retryable (lm15 never retries)",
    ),
    ("headers", "RouterConfig::transport(...)"),
    ("extra_headers", "RouterConfig::transport(...)"),
    (
        "extra_body",
        "config.extensions on the Request (build it with request_from_openai_chat and edit)",
    ),
    ("extra_query", "RouterConfig::transport(...)"),
    (
        "cache",
        "your own cache keyed on the Request (lm15 has no response cache)",
    ),
    (
        "caching",
        "your own cache keyed on the Request (lm15 has no response cache)",
    ),
    ("mock_response", "a scripted Transport"),
    (
        "drop_params",
        "nothing: lm15 refuses what it cannot carry instead of dropping it",
    ),
    ("custom_llm_provider", "the model string's prefix"),
];

/// The lm15 model string for a model string written for the OpenAI SDK or
/// litellm (`playbooks/api-family.md` § Ingest): an lm15 string
/// (`provider:model`) is left alone; litellm's `provider/model` maps its
/// prefix through [`LITELLM_PROVIDER_PREFIXES`] (only the first segment; a
/// model id may contain slashes itself: `groq/openai/gpt-oss-20b`); a bare
/// name routes by lm15's rules, except that OpenAI's models go to the Chat
/// Completions door (`openai-chat:`) — the endpoint both libraries were
/// using — not the Responses API (see [`LMRouter::resolve_openai_chat`]).
pub fn openai_chat_model_string(model: &str) -> Result<String, Lm15Error> {
    if model.contains(':') {
        return Ok(model.to_string());
    }
    let Some((head, rest)) = model.split_once('/') else {
        return Ok(model.to_string());
    };
    if rest.is_empty() {
        return Ok(model.to_string());
    }
    match LITELLM_PROVIDER_PREFIXES
        .iter()
        .find(|(prefix, _)| *prefix == head)
    {
        Some((_, provider)) => Ok(format!("{provider}:{rest}")),
        None => {
            let mut known: Vec<&str> = LITELLM_PROVIDER_PREFIXES.iter().map(|(p, _)| *p).collect();
            known.sort_unstable();
            Err(unknown_model(
                format!(
                    "could not read {model:?} as a litellm model string: {head:?} is not a provider \
                     prefix lm15 has a door for (known: {}); write it as lm15's provider:model instead",
                    known.join(", ")
                ),
                model,
            ))
        }
    }
}

/// `(model, messages, kwargs)` → the Chat Completions body, after refusing
/// the client keywords by name.
fn split_openai_chat_call(
    model: &str,
    messages: &Value,
    kwargs: &JsonObject,
) -> Result<Value, Lm15Error> {
    for (key, where_) in CLIENT_KEYWORDS {
        if kwargs.contains_key(*key) {
            return Err(not_configured(format!(
                "{key:?} configures the client, not the request; in lm15 it lives in {where_}"
            )));
        }
    }
    let mut body = JsonObject::new();
    body.insert("model".into(), Value::String(model.to_string()));
    body.insert("messages".into(), messages.clone());
    for (key, value) in kwargs {
        body.insert(key.clone(), value.clone());
    }
    Ok(Value::Object(body))
}

impl LMRouter {
    /// [`Self::resolve`] for the OpenAI-shaped door: `model` is read by
    /// [`openai_chat_model_string`], and a bare OpenAI name goes to Chat
    /// Completions (`openai-chat`), the endpoint the OpenAI SDK and litellm
    /// were using. Like `resolve`: no network, no credential invocation, no
    /// secret values.
    pub fn resolve_openai_chat(&self, model: &str) -> Result<Resolution, Lm15Error> {
        let normalized = match model.split_once('/') {
            Some((head, rest)) if self.config.declared(head).is_some() => {
                format!("{}:{rest}", self.config.canonical(head))
            }
            _ => openai_chat_model_string(model)?,
        };
        let resolution = self.resolve(&normalized)?;
        if resolution.source == RouteSource::Rule && resolution.provider == "openai" {
            return self.resolve(&format!("openai-chat:{}", resolution.model));
        }
        Ok(resolution)
    }

    /// The Request behind [`Self::complete_from_openai_chat`], and the LM it
    /// routes to. `model` may be written for the OpenAI SDK, for litellm, or
    /// for lm15; `messages` is the call's `messages` array; `kwargs` its
    /// other keywords. The body is read with the destination door's own
    /// spellings when it speaks the Chat Completions wire, else with
    /// OpenAI's (MAP-12: every key maps, passes through, or is refused by
    /// name). Client keywords (`api_key`, `timeout`, …) are refused with the
    /// `RouterConfig` place named.
    pub fn request_from_openai_chat(
        &self,
        model: &str,
        messages: &Value,
        kwargs: &JsonObject,
    ) -> Result<(Request, Arc<ProviderLM>), Lm15Error> {
        let resolution = self.resolve_openai_chat(model)?;
        let body = split_openai_chat_call(&resolution.requested, messages, kwargs)?;
        let lm = self.lm_for(&resolution)?;
        let request = if lm.dialect() == DialectId::OpenaiChat {
            lm.request_from_openai_chat(&body)?
        } else {
            crate::dialects::openai_chat::ingest::request_from_openai_chat(&body, None)?
        };
        Ok((routed(&request, &resolution), lm))
    }

    /// `client.chat.completions.create(model=, messages=, ...)` or
    /// `litellm.completion(model=, messages=, ...)` — the same call, answered
    /// by lm15 as a canonical [`Response`]. See
    /// [`Self::request_from_openai_chat`] for how the pieces are read;
    /// [`Self::stream_from_openai_chat`] is the streaming twin (wrap it in
    /// [`crate::ResponseStream`] for the assembled answer).
    pub async fn complete_from_openai_chat(
        &self,
        model: &str,
        messages: &Value,
        kwargs: &JsonObject,
    ) -> Result<Response, Lm15Error> {
        let (request, lm) = self.request_from_openai_chat(model, messages, kwargs)?;
        lm.complete(&request).await
    }

    /// The streaming twin of [`Self::complete_from_openai_chat`]: typed lm15
    /// stream events, not OpenAI-shaped chunks.
    pub fn stream_from_openai_chat(
        &self,
        model: &str,
        messages: &Value,
        kwargs: &JsonObject,
    ) -> EventStream {
        match self.request_from_openai_chat(model, messages, kwargs) {
            Ok((request, lm)) => lm.stream(&request),
            Err(err) => EventStream::failed(String::new(), err),
        }
    }
}

impl fmt::Debug for LMRouter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let built: Vec<String> = self
            .lms
            .lock()
            .map(|lms| lms.keys().cloned().collect())
            .unwrap_or_default();
        f.debug_struct("LMRouter")
            .field("config", &self.config)
            .field("built", &built)
            .finish()
    }
}

/// The request with the wire model in place of the routed string.
fn routed(request: &Request, resolution: &Resolution) -> Request {
    if request.model == resolution.model {
        return request.clone();
    }
    Request {
        model: resolution.model.clone(),
        ..request.clone()
    }
}

// ─── resolution ──────────────────────────────────────────────────────

fn unknown_model(message: String, model: &str) -> Lm15Error {
    Lm15Error::unknown_model(message, model)
}

fn ambiguous_model(message: String, model: &str, providers: &[&str]) -> Lm15Error {
    Lm15Error::ambiguous_model(
        message,
        model,
        providers.iter().map(|p| p.to_string()).collect(),
    )
}

fn known_providers() -> String {
    PROVIDERS
        .iter()
        .map(|d| d.id)
        .collect::<Vec<_>>()
        .join(", ")
}

/// The adapter's family name for a registry entry (`Resolution.adapter`).
fn adapter_name(definition: &ProviderDefinition) -> &'static str {
    match (definition.kind, definition.id) {
        (EntryKind::AdapterOwned, "xai") => "XaiLM",
        (EntryKind::AdapterOwned, "claude-code") => "ClaudeCodeLM",
        (EntryKind::AdapterOwned, "openai-codex") => "OpenAICodexLM",
        _ => match definition.dialect {
            DialectId::OpenaiResponses => "OpenAILM",
            DialectId::OpenaiChat => "OpenAIChatLM",
            DialectId::Anthropic => "AnthropicLM",
            DialectId::Gemini => "GeminiLM",
            DialectId::Typesafe => "TypeSafeLM",
        },
    }
}

fn resolution(
    requested: &str,
    wire_model: String,
    definition: &'static ProviderDefinition,
    source: RouteSource,
    config: &RouterConfig,
    rule: Option<RouteRule>,
    model_info: Option<ModelInfo>,
) -> Resolution {
    Resolution {
        requested: requested.to_string(),
        model: wire_model,
        provider: definition.id.to_string(),
        adapter: adapter_name(definition),
        source,
        rule,
        env_key: env_key_for(definition, config),
        model_info,
        compat: match definition.kind {
            EntryKind::AdapterOwned => None,
            EntryKind::Bound | EntryKind::Hosted => definition.compat,
        },
        declared: false,
        declared_env_keys: Vec::new(),
        declared_compat: None,
        declaration_note: None,
        base_url: config.explicit_base_url(definition.id).map(str::to_string),
        credential: config.credentials.get(definition.id).cloned(),
        credential_policy: definition.access().credential_policy.as_str().into(),
    }
}

fn declared_resolution(
    requested: &str,
    model: &str,
    d: &DeclaredProvider,
    source: RouteSource,
    rule: Option<RouteRule>,
    model_info: Option<ModelInfo>,
    config: &RouterConfig,
) -> Resolution {
    Resolution {
        requested: requested.into(),
        model: model.into(),
        provider: d.id.clone(),
        adapter: adapter_name(d.template()),
        source,
        rule,
        env_key: None,
        model_info,
        compat: None,
        declared: true,
        declared_env_keys: d.env_keys.clone(),
        declared_compat: Some(d.compat.clone()),
        declaration_note: Some(d.note.clone()),
        base_url: Some(
            config
                .explicit_base_url(&d.id)
                .unwrap_or(&d.base_url)
                .into(),
        ),
        credential: None,
        credential_policy: "key".into(),
    }
}

/// WHICH env var `lm` would read (never the value): `None` under an
/// explicit entry, for a provider with no declared keys, else the first
/// set key, else the first declared.
fn env_key_for(definition: &ProviderDefinition, config: &RouterConfig) -> Option<&'static str> {
    if config.credentials.contains_key(definition.id) {
        return None;
    }
    if matches!(
        config.explicit_credential(definition.id),
        Ok(Some(_)) | Err(_)
    ) {
        return None;
    }
    let env_keys = definition.access().env_keys;
    env_keys
        .iter()
        .copied()
        .find(|key| config.env_value(key).is_some())
        .or_else(|| env_keys.first().copied())
}

fn resolve(model: &str, config: &RouterConfig) -> Result<Resolution, Lm15Error> {
    if model.is_empty() {
        return Err(unknown_model(
            "model must be a non-empty string".into(),
            model,
        ));
    }

    // Rung 1: explicit provider prefix (split on the FIRST colon).
    if let Some((head, rest)) = model.split_once(':') {
        if let Some(definition) = config.declared(head) {
            if !rest.is_empty() {
                return Ok(declared_resolution(
                    model,
                    rest,
                    definition,
                    RouteSource::Prefix,
                    None,
                    None,
                    config,
                ));
            }
        }
        if let Some(definition) = lookup(head) {
            if !rest.is_empty() {
                return Ok(resolution(
                    model,
                    rest.to_string(),
                    definition,
                    RouteSource::Prefix,
                    config,
                    None,
                    None,
                ));
            }
        }
    }

    // Rung 2: the catalog.
    if !config.catalog.is_empty() {
        let matches: Vec<&ModelInfo> = config
            .catalog
            .iter()
            .filter(|info| info.id == model || info.aliases.iter().any(|a| a == model))
            .collect();
        let mut providers: Vec<&str> = Vec::new();
        for info in &matches {
            if !providers.contains(&info.provider.as_str()) {
                providers.push(&info.provider);
            }
        }
        if providers.len() > 1 {
            let options = providers
                .iter()
                .map(|p| format!("\"{p}:{model}\""))
                .collect::<Vec<_>>()
                .join(" or ");
            return Err(ambiguous_model(
                format!(
                    "model {model:?} is offered by multiple providers: {}. Fix: use the explicit \
                     form, e.g. model {:?} — options: {options}.",
                    providers.join(", "),
                    format!("{}:{model}", providers[0]),
                ),
                model,
                &providers,
            ));
        }
        if !matches.is_empty() {
            // An exact id beats an alias; an alias never shadows an entry
            // whose canonical id IS the requested string.
            let exact: Vec<&ModelInfo> =
                matches.iter().copied().filter(|i| i.id == model).collect();
            let narrowed = if exact.is_empty() { &matches } else { &exact };
            if narrowed.len() > 1 {
                let ids = narrowed
                    .iter()
                    .map(|i| i.id.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(ambiguous_model(
                    format!(
                        "model {model:?} matches multiple catalog entries ({ids}) under provider \
                         {:?}. Fix: request a canonical id directly.",
                        narrowed[0].provider
                    ),
                    model,
                    &providers,
                ));
            }
            let info = narrowed[0];
            if let Some(definition) = config.declared(&info.provider) {
                return Ok(declared_resolution(
                    model,
                    &info.id,
                    definition,
                    RouteSource::Catalog,
                    None,
                    Some(info.clone()),
                    config,
                ));
            }
            let Some(definition) = lookup(&info.provider) else {
                return Err(unknown_model(
                    format!(
                        "model {model:?} resolved in the catalog to provider {:?}, but lm15 has no \
                         adapter or compat preset for it. Known providers: {}. Construct a \
                         provider adapter directly (OpenAIChatLM with a base_url) for an \
                         OpenAI-compatible server.",
                        info.provider,
                        known_providers()
                    ),
                    model,
                ));
            };
            let wire_model = if info.id == model {
                model.to_string()
            } else {
                info.id.clone()
            };
            return Ok(resolution(
                model,
                wire_model,
                definition,
                RouteSource::Catalog,
                config,
                None,
                Some(info.clone()),
            ));
        }
    }

    // Rung 3: built-in prefix rules, first match wins.
    for rule in &config.rules {
        if model.starts_with(rule.prefix) {
            if let Some(definition) = config.declared(rule.provider) {
                return Ok(declared_resolution(
                    model,
                    model,
                    definition,
                    RouteSource::Rule,
                    Some(*rule),
                    None,
                    config,
                ));
            }
            let Some(definition) = lookup(rule.provider) else {
                return Err(unknown_model(
                    format!(
                        "rule {rule:?} names provider {:?}, which has no adapter. Known providers: {}.",
                        rule.provider,
                        known_providers()
                    ),
                    model,
                ));
            };
            return Ok(resolution(
                model,
                model.to_string(),
                definition,
                RouteSource::Rule,
                config,
                Some(*rule),
                None,
            ));
        }
    }

    let mut hints = Vec::new();
    if let Some((head, rest)) = model.split_once(':') {
        let head = canonical_provider(head);
        if let Some(close) = closest_provider(&head) {
            hints.push(format!("Did you mean \"{close}:{rest}\"?"));
        }
    }
    hints.push(format!(
        "Use an explicit provider prefix — \"provider:{model}\" with provider one of: {}.",
        known_providers()
    ));
    if config.catalog.is_empty() {
        hints.push("Or pass a model catalog: RouterConfig::new().catalog(models).".into());
    }
    Err(unknown_model(
        format!(
            "could not route model {model:?}: no provider prefix, {}, and none of the {} built-in \
             rules matched. {}",
            if config.catalog.is_empty() {
                "no catalog supplied"
            } else {
                "no catalog match"
            },
            config.rules.len(),
            hints.join(" ")
        ),
        model,
    ))
}

/// A registry provider within edit distance of `head` (the reference's
/// `difflib.get_close_matches(cutoff=0.75)`, approximated by a normalized
/// Levenshtein similarity).
fn closest_provider(head: &str) -> Option<&'static str> {
    let mut best: Option<(f64, &'static str)> = None;
    for definition in PROVIDERS {
        let distance = levenshtein(head, definition.id);
        let longest = head.chars().count().max(definition.id.chars().count());
        if longest == 0 {
            continue;
        }
        let similarity = 1.0 - distance as f64 / longest as f64;
        if similarity >= 0.75 && best.is_none_or(|(s, _)| similarity > s) {
            best = Some((similarity, definition.id));
        }
    }
    best.map(|(_, id)| id)
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1];
        for (j, cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            cur.push((prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost));
        }
        prev = cur;
    }
    prev[b.len()]
}

// ─── construction (the AUTH-1 chain) ─────────────────────────────────

fn missing_credential(policy: &AccessPolicy, what: &str) -> Lm15Error {
    let provider = policy.provider;
    let mut meta = ErrorMeta::new(format!(
        "no {what} found for provider {provider:?}. Set {} in the environment, or pass \
         RouterConfig::new().api_key({provider:?}, ...).",
        if policy.env_keys.is_empty() {
            "an explicit credential".to_string()
        } else {
            policy.env_keys.join(" or ")
        }
    ));
    meta.provider = Some(provider.to_string());
    Lm15Error::NotConfiguredError(meta)
}

/// The stored login of `provider` with AUTH-3 refresh enabled: the
/// router's transport (or the shared one) for the token endpoint, the
/// AUTH-8 lock directory from the router's env. Without a HOME (or
/// `LM15_LOCK_DIR`) there is no lock directory and the login stays
/// read-only: an expired token is then the typed `AuthError`, never an
/// unlocked refresh.
fn refreshing_login(provider: &str, config: &RouterConfig) -> Result<StoredLogin, Lm15Error> {
    let env = |key: &str| config.env_value(key);
    let login = StoredLogin::for_provider(provider, &env, config.credentials_path.as_deref());
    let Some(lock_dir) = crate::auth::lock_dir(&env) else {
        return Ok(login);
    };
    let transport: Arc<dyn Transport> = match &config.transport {
        Some(transport) => Arc::clone(transport),
        None => crate::transport::default_transport()?,
    };
    Ok(login.refreshing(transport, lock_dir))
}

/// The adapter of `provider` over the AUTH-1 chain against the process
/// environment: what the family's `OpenAILM()` / `OpenAILM::new()` does
/// (api-family § Providers, direct). No explicit credential; a stored
/// login, the declared env keys in order, a keyless server's
/// placeholder; the shared transport. Anything more (an explicit key, a
/// base URL, settings, a transport) is the builder.
pub(crate) fn provider_from_environment(provider: &str) -> Result<ProviderLM, Lm15Error> {
    let canonical = canonical_provider(provider);
    if lookup(&canonical).is_none() {
        return Err(crate::auth::AuthError::UnknownProvider {
            provider: provider.to_string(),
        }
        .into());
    }
    build_lm(&canonical, &RouterConfig::default())
}

fn build_lm(provider: &str, config: &RouterConfig) -> Result<ProviderLM, Lm15Error> {
    #[cfg(feature = "native")]
    let managed = match &config.auth {
        Some(auth)
            if config.explicit_credential(provider)?.is_none()
                && !config.credentials.contains_key(&config.canonical(provider)) =>
        {
            let placeholder = match config.declared(provider) {
                Some(d) => d.placeholder_key.clone(),
                None => lookup(provider)
                    .and_then(|d| d.placeholder_key.or(d.access().placeholder_key))
                    .map(str::to_string),
            };
            let hosted = config.declared(provider).is_none()
                && lookup(provider).is_some_and(|d| d.access().host.is_some());
            Some(crate::login::managed_route(
                auth,
                &config.canonical(provider),
                placeholder,
                hosted,
            )?)
        }
        _ => None,
    };
    if let Some(d) = config.declared(provider) {
        #[cfg(feature = "native")]
        let managed_credential = managed.as_ref().and_then(|m| m.credential.clone());
        #[cfg(not(feature = "native"))]
        let managed_credential: Option<SharedCredentials> = None;
        let credentials = match config.explicit_credential(&d.id)? {
            Some(c) => c,
            None if managed_credential.is_some() => managed_credential.clone().expect("checked"),
            None => d
                .env_keys
                .iter()
                .find_map(|key| config.env_value(key))
                .or_else(|| d.placeholder_key.clone())
                .map(|key| Arc::new(key) as SharedCredentials)
                .ok_or_else(|| {
                    not_configured(format!(
                        "{}: no explicit api_key or declared environment credential ({})",
                        d.id,
                        d.env_keys.join(", ")
                    ))
                })?,
        };
        #[allow(unused_mut)]
        let mut base_url = config
            .explicit_base_url(&d.id)
            .unwrap_or(&d.base_url)
            .to_string();
        #[cfg(feature = "native")]
        if let Some(from_connection) = managed
            .as_ref()
            .and_then(|m| m.base_url.clone())
            .filter(|_| config.explicit_base_url(&d.id).is_none())
        {
            base_url = from_connection;
        }
        let mut builder = LmBuilder::for_entry(d.template())
            .provider_name(&d.id)
            .provider_aliases(&d.aliases)
            .compat(d.compat.clone())
            .base_url(&base_url)
            .api_key(credentials)
            .adaptations(config.adaptations);
        if let Some(t) = &config.transport {
            builder = builder.transport_shared(t.clone());
        }
        return match &d.factory {
            Some(factory) => factory(builder),
            None => builder.build(),
        };
    }
    let definition = lookup(provider).expect("a resolution names a registry entry");
    let policy = definition.access();
    let provider = definition.id;
    let env = |key: &str| config.env_value(key);
    let mut builder = LmBuilder::for_entry(definition).adaptations(config.adaptations);
    if let Some(transport) = &config.transport {
        builder = builder.transport_shared(Arc::clone(transport));
    }
    #[cfg(feature = "native")]
    if let Some(route) = &managed {
        if let Some(account_id) = &route.account_id {
            builder = builder.account_id(account_id);
        }
        if let (Some(url), None) = (&route.base_url, config.explicit_base_url(provider)) {
            if policy.host.is_none() {
                builder = builder.base_url(url);
            }
        }
    }
    #[cfg(feature = "native")]
    let managed_named = managed.as_ref().and_then(|m| m.named.clone());
    let is_managed = config_is_managed(config);

    if policy.credential_policy == CredentialPolicy::OAuth && !is_managed {
        // The stored login owns the provider (AUTH-1): validated now for
        // the typed, re-login-guided error, then re-read per request and
        // refreshed before a request when expired (AUTH-3).
        let login = refreshing_login(provider, config)?;
        let initial = login.read()?;
        if let Some(account_id) = initial.account_id() {
            builder = builder.account_id(account_id);
        }
        return builder.api_key(login).build();
    }

    let env_map = config.env_map();
    let endpoint = config
        .explicit_base_url(provider)
        .map(str::to_string)
        .or_else(|| {
            policy.host.as_ref().and_then(|host| {
                crate::cloud::hosts::endpoint_from_env(host, &env_map)
                    .map(|(_, value)| value.to_string())
            })
        });
    if let Some(url) = &endpoint {
        builder = builder.base_url(url);
    }
    builder = builder.env(env_map.clone());

    let mut credential: Option<Box<dyn CredentialProvider + Send + Sync>> = config
        .explicit_credential(provider)?
        .map(|c| Box::new(c) as Box<dyn CredentialProvider + Send + Sync>);
    #[cfg(feature = "native")]
    if credential.is_none() {
        if let Some(shared) = managed.as_ref().and_then(|m| m.credential.clone()) {
            credential = Some(Box::new(shared));
        }
    }

    #[cfg(feature = "native")]
    if let Some(host) = &policy.host {
        // A cloud door (AUTH-10): settings from config, then env, then the
        // cloud's own profile (AWS region, GCP project), then defaults.
        // The credential: the explicit entry, or — for a cloud chain
        // policy — the chain's caching provider (AUTH-1/AUTH-2/AUTH-3),
        // or — for a `key` policy — the declared env keys.
        let env_map = config.env_map();
        let now = crate::auth::time_now();
        let transport: Arc<dyn Transport> = match &config.transport {
            Some(transport) => Arc::clone(transport),
            None => crate::transport::default_transport()?,
        };
        let mut ctx = ChainContext::online(env_map.clone(), transport, now);
        let mut given = config.settings.get(provider).cloned().unwrap_or_default();
        if policy.is_cloud_chain() {
            for setting in host.settings {
                let from_config = given.get(setting.name).is_some_and(|v| !v.is_empty());
                let from_env = setting
                    .env
                    .iter()
                    .any(|var| env_map.get(*var).is_some_and(|v| !v.is_empty()));
                if !from_config && !from_env {
                    if let Some(value) = profile_setting(policy, &ctx, setting.name) {
                        given.insert(setting.name.to_string(), value);
                    }
                }
            }
        }
        let settings = resolve_settings_with_endpoint(
            Some(host),
            &given,
            Some(&env_map),
            provider,
            endpoint.as_deref(),
        )?;
        builder = builder.settings(settings.clone());
        let named = managed_named.as_ref().or(config.credentials.get(provider));
        if credential.is_none() && is_managed && named.is_none() {
            return Err(crate::errors::AuthOperation::error(
                format!("{provider}: no credential for this cloud door under a managed Auth"),
                "login_required",
                "resolution",
                "not_committed",
                "select_connection",
            ));
        }
        if credential.is_none() && policy.is_cloud_chain() {
            ctx.settings = settings;
            credential = Some(match named {
                Some(name) => Box::new(ChainProvider::named(policy, ctx, name)?),
                None => Box::new(ChainProvider::new(policy, ctx)),
            });
        }
    }
    #[cfg(not(feature = "native"))]
    if let Some(host) = &policy.host {
        // The wire codec build has no cloud profile files, CLIs or metadata
        // endpoints: settings come from config and env; the credential must
        // be explicit (a cloud chain cannot run here).
        let env_map = config.env_map();
        let given = config.settings.get(provider).cloned().unwrap_or_default();
        let settings = resolve_settings_with_endpoint(
            Some(host),
            &given,
            Some(&env_map),
            provider,
            endpoint.as_deref(),
        )?;
        builder = builder.settings(settings);
        if credential.is_none() && policy.is_cloud_chain() {
            return Err(Lm15Error::NotConfiguredError(ErrorMeta::new(format!(
                "{provider}: the {} credential chain is not available in this build (no `native` feature: no profile files, CLIs or metadata endpoints); pass an explicit credential",
                policy.credential_policy.as_str()
            ))));
        }
    }

    if credential.is_none() && is_managed {
        // Managed mode never falls through to an environment key, a CLI file
        // or a placeholder the connection step did not choose (AUTH-15).
        return Err(crate::errors::AuthOperation::error(
            format!("{provider}: no saved connection in this scope; sign in with Auth::login or connect()"),
            "login_required", "resolution", "not_committed", "restart_login",
        ));
    }

    if credential.is_none() && policy.credential_policy == CredentialPolicy::OAuthUnlessExplicit {
        // A usable stored subscription login outranks ambient env keys: it
        // spends no money per token (AUTH-1).
        // An unusable or signed-out one BLOCKS them (R3, ratified
        // 2026-09-22): a failed subscription is never silently replaced by
        // a metered key.
        let login = refreshing_login(provider, config)?;
        match login.state() {
            crate::auth::StoredState::Usable => credential = Some(Box::new(login)),
            state @ (crate::auth::StoredState::Unusable | crate::auth::StoredState::LoggedOut) => {
                let what = if state == crate::auth::StoredState::LoggedOut {
                    "was signed out"
                } else {
                    "is expired and cannot be renewed"
                };
                let present = policy
                    .env_keys
                    .iter()
                    .find(|key| config.env_value(key).is_some())
                    .map(|key| format!("${key} is set but is used only when passed explicitly: "))
                    .unwrap_or_default();
                let hint = policy.login_hint.unwrap_or("sign in");
                let mut meta = ErrorMeta::new(format!(
                    "the {provider:?} subscription login {what}. {present}sign in again ({hint}), \
                     or pass the key deliberately with RouterConfig::new().api_key({provider:?}, ...)."
                ));
                meta.provider = Some(provider.to_string());
                return Err(Lm15Error::NotConfiguredError(meta));
            }
            crate::auth::StoredState::Absent => {}
        }
    }

    if credential.is_none() {
        credential = policy.env_keys.iter().find_map(|key| {
            config.env_value(key).map(|value| {
                Box::new(crate::auth::SourcedCredential {
                    provider: value,
                    source: crate::auth::CredentialSource::environment(key),
                }) as Box<dyn CredentialProvider + Send + Sync>
            })
        });
    }

    if credential.is_none() {
        if let Some(placeholder) = policy.placeholder_key {
            credential = Some(Box::new(placeholder.to_string()));
        }
    }

    let Some(credential) = credential else {
        if policy.credential_policy == CredentialPolicy::OAuthUnlessExplicit {
            // Nothing anywhere: the typed error with the login hint.
            let login =
                StoredLogin::for_provider(provider, &env, config.credentials_path.as_deref());
            return Err(login.read().map(|_| ()).unwrap_err().into());
        }
        return Err(missing_credential(policy, "API key"));
    };
    builder.api_key(credential).build()
}

fn config_is_managed(config: &RouterConfig) -> bool {
    #[cfg(feature = "native")]
    {
        config.auth.is_some()
    }
    #[cfg(not(feature = "native"))]
    {
        let _ = config;
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ModelOrigin;

    fn hermetic(env: &[(&str, &str)]) -> RouterConfig {
        RouterConfig::new().env(env.iter().copied())
    }

    #[test]
    fn an_explicit_key_serves_its_sibling_and_ambiguity_is_refused() {
        // spec/auth.md § Shared explicit keys (ratified 2026-09-09).
        assert_eq!(
            shared_api_key_source(["openai"], "openai-chat"),
            Ok(Some("openai"))
        );
        assert_eq!(
            shared_api_key_source(["openai_chat"], "openai"),
            Ok(Some("openai_chat"))
        );
        assert_eq!(
            shared_api_key_source(["vertex-express"], "gemini"),
            Ok(None)
        ); // overlapping, not identical
        assert_eq!(shared_api_key_source(["ollama"], "vllm"), Ok(None)); // empty env lists never share
        assert_eq!(
            shared_api_key_source(["meta", "meta-chat"], "meta-anthropic"),
            Err(vec!["meta", "meta-chat"])
        );
        assert_eq!(
            shared_api_key_source(["meta", "meta-chat", "meta-anthropic"], "meta-anthropic"),
            Ok(Some("meta-anthropic"))
        );
        let router = LMRouter::with_config(hermetic(&[]).api_key("openai", "k")).unwrap();
        let r = router.resolve("openai-chat:gpt-4.1-mini").unwrap();
        assert_eq!(r.env_key, None, "the shared entry is the explicit rung");
        assert_eq!(
            router.lm("openai-chat:gpt-4.1-mini").unwrap().provider(),
            "openai-chat"
        );
        let router =
            LMRouter::with_config(hermetic(&[]).api_key("meta", "a").api_key("meta-chat", "b"))
                .unwrap();
        let err = router.lm("meta-anthropic:m").unwrap_err();
        assert_eq!(err.class_name(), "NotConfiguredError");
        assert!(err.message().contains("ambiguous"), "{err}");
        // A provider string this config is keyed by must route somewhere.
        let err = LMRouter::with_config(hermetic(&[]).api_key("opnai", "k")).unwrap_err();
        assert_eq!(err.class_name(), "NotConfiguredError");
        assert!(err.message().contains("Did you mean \"openai\""), "{err}");
        let err = LMRouter::with_config(hermetic(&[]).base_url("nope", "http://x")).unwrap_err();
        assert_eq!(err.class_name(), "NotConfiguredError");
        // base_url: the exact provider's adapter, never a cloud door.
        let router = LMRouter::with_config(
            hermetic(&[])
                .api_key("openai-chat", "k")
                .base_url("openai_chat", "http://h/v1"),
        )
        .unwrap();
        assert_eq!(
            router.lm("openai-chat:m").unwrap().base_url(),
            "http://h/v1"
        );
        let router = LMRouter::with_config(
            hermetic(&[("AWS_REGION", "eu-west-1")])
                .api_key("bedrock-chat", "bearer")
                .base_url("bedrock-chat", "http://h/v1"),
        )
        .unwrap();
        let lm = router.lm("bedrock-chat:m").unwrap();
        assert_eq!(lm.base_url(), "http://h/v1/openai/v1");
        assert_eq!(lm.settings()["region"], "eu-west-1");
    }

    #[test]
    fn the_openai_shaped_door_reads_the_other_libraries_model_strings() {
        assert_eq!(
            openai_chat_model_string("groq/openai/gpt-oss-20b").unwrap(),
            "groq:openai/gpt-oss-20b"
        );
        assert_eq!(
            openai_chat_model_string("openai-chat:gpt-4o-mini").unwrap(),
            "openai-chat:gpt-4o-mini"
        );
        assert_eq!(
            openai_chat_model_string("gpt-4o-mini").unwrap(),
            "gpt-4o-mini"
        );
        assert_eq!(
            openai_chat_model_string("bedrock/anthropic.claude")
                .unwrap_err()
                .class_name(),
            "UnknownModelError"
        );
        let router = LMRouter::with_config(
            hermetic(&[])
                .api_key("openai", "k")
                .api_key("anthropic", "k"),
        )
        .unwrap();
        assert_eq!(
            router.resolve_openai_chat("gpt-4o-mini").unwrap().provider,
            "openai-chat"
        );
        assert_eq!(
            router
                .resolve_openai_chat("anthropic/claude-sonnet-4-5")
                .unwrap()
                .provider,
            "anthropic"
        );
        let messages = serde_json::json!([{"role": "user", "content": "hi"}]);
        let mut kwargs = JsonObject::new();
        kwargs.insert("api_key".into(), Value::String("x".into()));
        let err = router
            .request_from_openai_chat("gpt-4o-mini", &messages, &kwargs)
            .unwrap_err();
        assert!(err.message().contains("configures the client"), "{err}");
        let mut kwargs = JsonObject::new();
        kwargs.insert("max_completion_tokens".into(), Value::from(5));
        let (request, lm) = router
            .request_from_openai_chat("gpt-4o-mini", &messages, &kwargs)
            .unwrap();
        assert_eq!(lm.provider(), "openai-chat");
        assert_eq!(request.model, "gpt-4o-mini");
        assert_eq!(request.config.max_tokens, Some(5));
        // An Anthropic destination reads the same body with OpenAI's spellings.
        let (request, lm) = router
            .request_from_openai_chat("anthropic/claude-sonnet-4-5", &messages, &kwargs)
            .unwrap();
        assert_eq!(lm.provider(), "anthropic");
        assert_eq!(request.model, "claude-sonnet-4-5");
    }

    #[test]
    fn prefix_rung_splits_on_the_first_colon_and_accepts_both_spellings() {
        let router = LMRouter::with_config(hermetic(&[("OPENAI_API_KEY", "k")])).unwrap();
        let r = router.resolve("openai:ft:gpt-4.1:org").unwrap();
        assert_eq!(r.source, RouteSource::Prefix);
        assert_eq!(r.provider, "openai");
        assert_eq!(r.model, "ft:gpt-4.1:org");
        assert_eq!(r.adapter, "OpenAILM");
        assert_eq!(r.env_key, Some("OPENAI_API_KEY"));
        let r = router.resolve("openai_chat:gpt-4.1-mini").unwrap();
        assert_eq!(r.provider, "openai-chat");
        assert_eq!(r.adapter, "OpenAIChatLM");
        let r = router.resolve("groq:llama").unwrap();
        assert_eq!(r.adapter, "OpenAIChatLM");
        assert_eq!(r.compat, Some("groq"));
        assert_eq!(r.env_key, Some("GROQ_API_KEY"));
        // An empty remainder is not a prefix route.
        let err = router.resolve("openai:").unwrap_err();
        assert_eq!(err.class_name(), "UnknownModelError");
        assert_eq!(err.model(), Some("openai:"));
        // An unknown head is a bare model id: rules still apply.
        let r = router.resolve("gpt-4.1:custom").unwrap();
        assert_eq!(r.source, RouteSource::Rule);
        assert_eq!(r.model, "gpt-4.1:custom");
    }

    #[test]
    fn rule_rung_first_match_wins_and_reports_the_rule() {
        let router = LMRouter::with_config(hermetic(&[])).unwrap();
        let r = router.resolve("claude-haiku-4-5").unwrap();
        assert_eq!(r.provider, "anthropic");
        assert_eq!(r.rule.unwrap().prefix, "claude-");
        assert_eq!(r.env_key, Some("ANTHROPIC_API_KEY"));
        assert!(r.describe().contains("built-in rule"), "{}", r.describe());
        let r = router.resolve("grok-4").unwrap();
        assert_eq!(r.provider, "xai");
        assert_eq!(r.adapter, "XaiLM");
        assert!(
            r.describe().contains("stored subscription OAuth"),
            "{}",
            r.describe()
        );
        let r = router.resolve("gemini-2.5-flash").unwrap();
        assert_eq!(r.env_key, Some("GEMINI_API_KEY"));
        let router = LMRouter::with_config(hermetic(&[("GOOGLE_API_KEY", "k")])).unwrap();
        assert_eq!(
            router.resolve("gemini-2.5-flash").unwrap().env_key,
            Some("GOOGLE_API_KEY")
        );
    }

    #[test]
    fn unroutable_strings_explain_themselves() {
        let router = LMRouter::with_config(hermetic(&[])).unwrap();
        let err = router.resolve("").unwrap_err();
        assert_eq!(err.class_name(), "UnknownModelError");
        assert_eq!(err.code().as_str(), "unknown_model");
        assert!(err.is_a(crate::errors::ErrorClass::ConfigurationError));
        assert!(!err.is_a(crate::errors::ErrorClass::NotConfiguredError));
        let err = router.resolve("llama-3").unwrap_err();
        assert!(err.message().contains("could not route"), "{err}");
        assert!(err.message().contains("no catalog supplied"), "{err}");
        let err = router.resolve("anthropc:claude-x").unwrap_err();
        assert!(
            err.message()
                .contains("Did you mean \"anthropic:claude-x\""),
            "{err}"
        );
    }

    fn info(id: &str, provider: &str, aliases: &[&str]) -> ModelInfo {
        ModelInfo {
            id: id.into(),
            provider: provider.into(),
            api_family: "chat".into(),
            aliases: aliases.iter().map(|a| a.to_string()).collect(),
            origin: ModelOrigin::default(),
            inference: None,
            extensions: None,
        }
    }

    #[test]
    fn catalog_rung_exact_beats_alias_and_ambiguity_is_an_error() {
        let config = hermetic(&[]).catalog(vec![
            info("llama-3.3-70b-versatile", "groq", &["llama-70b"]),
            info("deepseek-chat", "deepseek", &["ds"]),
            info("shared", "groq", &[]),
            info("shared", "deepseek", &[]),
            info("alias-only", "deepseek", &["deepseek-chat"]),
            info("nowhere-1", "nowhere", &[]),
        ]);
        let router = LMRouter::with_config(config).unwrap();
        let r = router.resolve("llama-70b").unwrap();
        assert_eq!(r.source, RouteSource::Catalog);
        assert_eq!(r.model, "llama-3.3-70b-versatile");
        assert_eq!(r.provider, "groq");
        assert!(r.model_info.is_some());
        // The exact id wins over another entry's alias under the same
        // provider (across providers it is ambiguous, alias or not).
        let r = router.resolve("deepseek-chat").unwrap();
        assert_eq!(r.provider, "deepseek");
        assert_eq!(r.model, "deepseek-chat");
        let err = router.resolve("shared").unwrap_err();
        assert!(err.message().contains("multiple providers"), "{err}");
        assert_eq!(err.class_name(), "AmbiguousModelError");
        assert_eq!(err.code().as_str(), "ambiguous_model");
        assert_eq!(err.model(), Some("shared"));
        assert_eq!(
            err.candidate_providers(),
            Some(&["groq".to_string(), "deepseek".to_string()][..])
        );
        let err = router.resolve("nowhere-1").unwrap_err();
        assert!(err.message().contains("no adapter"), "{err}");
        assert_eq!(err.class_name(), "UnknownModelError");
        // A prefix still beats the catalog.
        assert_eq!(
            router.resolve("openai:shared").unwrap().source,
            RouteSource::Prefix
        );
    }

    #[test]
    fn credentials_follow_auth_1_for_key_providers() {
        // Explicit beats env; env keys in declared order; placeholder last.
        let config = hermetic(&[("GEMINI_API_KEY", ""), ("GOOGLE_API_KEY", "g")])
            .api_key("openai_chat", "explicit");
        let router = LMRouter::with_config(config).unwrap();
        assert_eq!(router.resolve("openai-chat:m").unwrap().env_key, None);
        let lm = router.lm("openai-chat:m").unwrap();
        let built = lm
            .build_request(
                &Request::new("m", vec![crate::Message::user("hi").unwrap()]).unwrap(),
                false,
            )
            .unwrap();
        assert_eq!(built.header("authorization"), Some("Bearer explicit"));
        let lm = router.lm("gemini-2.5-flash").unwrap();
        let built = lm
            .build_request(
                &Request::new(
                    "gemini-2.5-flash",
                    vec![crate::Message::user("hi").unwrap()],
                )
                .unwrap(),
                false,
            )
            .unwrap();
        assert_eq!(built.header("x-goog-api-key"), Some("g"));
        let lm = router.lm("ollama:llama3").unwrap();
        let built = lm
            .build_request(
                &Request::new("llama3", vec![crate::Message::user("hi").unwrap()]).unwrap(),
                false,
            )
            .unwrap();
        assert_eq!(built.header("authorization"), Some("Bearer ollama"));
        // Nothing anywhere: the typed error naming the env keys.
        let err = router.lm("anthropic:claude-x").unwrap_err();
        assert_eq!(err.class_name(), "NotConfiguredError");
        assert_eq!(err.provider(), Some("anthropic"));
        assert!(err.message().contains("ANTHROPIC_API_KEY"), "{err}");
        // One adapter per provider, reused.
        let a = router.lm("groq:a");
        assert!(a.is_err());
        let x = router.lm("openai-chat:x").unwrap();
        let y = router.lm("openai_chat:y").unwrap();
        assert!(Arc::ptr_eq(&x, &y));
    }

    #[test]
    fn hosted_doors_resolve_settings_and_cloud_chains_walk() {
        // A cloud chain with nothing configured builds the adapter (the
        // chain resolves at the first request, asynchronously) …
        let router = LMRouter::with_config(hermetic(&[
            ("AWS_REGION", "eu-west-1"),
            ("AWS_EC2_METADATA_DISABLED", "true"),
            ("HOME", "/nonexistent"),
        ]))
        .unwrap();
        let lm = router.lm("bedrock-chat:m").unwrap();
        let request = Request::new("m", vec![crate::Message::user("hi").unwrap()]).unwrap();
        // … and a synchronous build before `prepare` says so, typed.
        let err = lm.build_request(&request, false).unwrap_err();
        assert_eq!(err.class_name(), "NotConfiguredError");
        // It names the step to take (the wording since the credential
        // provider learned expiry: "not been prepared or has expired").
        assert!(err.message().contains("call prepare"), "{err}");
        // Static env keys: the chain selects them and signs (module 3b).
        let router = LMRouter::with_config(hermetic(&[
            ("AWS_REGION", "eu-west-1"),
            ("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE"),
            (
                "AWS_SECRET_ACCESS_KEY",
                "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            ),
            ("HOME", "/nonexistent"),
        ]))
        .unwrap();
        let lm = router.lm("bedrock-chat:m").unwrap();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let err = lm.complete(&request).await.unwrap_err();
            // The chain resolved (no auth error); the wire refused the
            // connection or the region: anything but "not resolved".
            assert!(!err.message().contains("not been resolved"), "{err}");
        });
        let built = lm.build_request(&request, false).unwrap();
        assert!(built
            .header("authorization")
            .unwrap()
            .starts_with("AWS4-HMAC-SHA256"));
        // Rung 0 (an explicit entry) works without 3b.
        let router = LMRouter::with_config(
            hermetic(&[("AWS_REGION", "eu-west-1")]).api_key("bedrock-chat", "bearer"),
        )
        .unwrap();
        let lm = router.lm("bedrock-chat:m").unwrap();
        assert_eq!(lm.settings()["region"], "eu-west-1");
        assert_eq!(
            lm.base_url(),
            "https://bedrock-runtime.eu-west-1.amazonaws.com/openai/v1"
        );
        // A missing required setting is the typed error naming it (the
        // AWS profile is the last fallback before that: HOME is pinned).
        let router = LMRouter::with_config(
            hermetic(&[("HOME", "/nonexistent")]).api_key("bedrock-chat", "bearer"),
        )
        .unwrap();
        let err = router.lm("bedrock-chat:m").unwrap_err();
        assert_eq!(err.class_name(), "NotConfiguredError");
        assert!(err.message().contains("AWS_REGION"), "{err}");
        // Config settings beat env.
        let router = LMRouter::with_config(
            hermetic(&[("AWS_REGION", "eu-west-1")])
                .api_key("bedrock-chat", "bearer")
                .setting("bedrock-chat", "region", "us-east-1"),
        )
        .unwrap();
        assert_eq!(
            router.lm("bedrock-chat:m").unwrap().settings()["region"],
            "us-east-1"
        );
        // A `key` policy on a host: env key + settings.
        let router = LMRouter::with_config(hermetic(&[("GOOGLE_API_KEY", "g")])).unwrap();
        let lm = router.lm("vertex-express:gemini-2.5-flash").unwrap();
        assert!(
            lm.base_url().contains("aiplatform.googleapis.com"),
            "{}",
            lm.base_url()
        );
    }

    fn write_login(name: &str, body: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "lm15-router-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, body).unwrap();
        path
    }

    fn now_ms() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64
    }

    #[test]
    fn oauth_providers_use_the_stored_login_only() {
        let fresh = now_ms() + 3_600_000;
        let path = write_login(
            "claude.json",
            &format!(
                r#"{{"claudeAiOauth":{{"accessToken":"tok","refreshToken":"r","expiresAt":{fresh}}}}}"#
            ),
        );
        let router = LMRouter::with_config(
            hermetic(&[("ANTHROPIC_API_KEY", "ambient")]).credentials_path(&path),
        )
        .unwrap();
        let r = router.resolve("claude-code:claude-x").unwrap();
        assert_eq!(r.env_key, None);
        assert!(
            r.describe().contains("local OAuth credential"),
            "{}",
            r.describe()
        );
        let lm = router.lm("claude-code:claude-x").unwrap();
        let request = Request::new("claude-x", vec![crate::Message::user("hi").unwrap()]).unwrap();
        let built = lm.build_request(&request, false).unwrap();
        assert_eq!(built.header("authorization"), Some("Bearer tok"));
        // Never the env key, even when the login is gone (AUTH-1:
        // stored-credential-owns-provider).
        std::fs::remove_file(&path).unwrap();
        let err = lm.build_request(&request, false).unwrap_err();
        assert_eq!(err.class_name(), "NotConfiguredError");
        assert!(err.message().contains("/login"), "{err}");
        let router = LMRouter::with_config(
            hermetic(&[("ANTHROPIC_API_KEY", "ambient")]).credentials_path(&path),
        )
        .unwrap();
        let err = router.lm("claude-code:claude-x").unwrap_err();
        assert_eq!(err.class_name(), "NotConfiguredError");
        assert_eq!(err.provider(), Some("claude-code"));
    }

    #[test]
    fn codex_login_binds_the_account_id() {
        let path = write_login(
            "codex.json",
            r#"{"tokens":{"access_token":"a.b.c","refresh_token":"r","account_id":"acct-file"},"last_refresh":"2020-01-01T00:00:00Z"}"#,
        );
        let router = LMRouter::with_config(hermetic(&[]).credentials_path(&path)).unwrap();
        let lm = router.lm("openai-codex:gpt-5-codex").unwrap();
        let request =
            Request::new("gpt-5-codex", vec![crate::Message::user("hi").unwrap()]).unwrap();
        let built = lm.build_request(&request, false).unwrap();
        assert_eq!(built.header("chatgpt-account-id"), Some("acct-file"));
        assert_eq!(built.header("authorization"), Some("Bearer a.b.c"));
    }

    #[test]
    fn oauth_unless_explicit_walks_the_chain_in_order() {
        let fresh = now_ms() + 3_600_000;
        let expired = now_ms() - 1;
        let usable = write_login(
            "xai.json",
            &format!(r#"{{"xai":{{"type":"oauth","access":"stored","expires":{fresh}}}}}"#),
        );
        let request = Request::new("grok-4", vec![crate::Message::user("hi").unwrap()]).unwrap();
        let bearer = |router: &LMRouter| {
            router
                .lm("grok-4")
                .unwrap()
                .build_request(&request, false)
                .unwrap()
                .header("authorization")
                .unwrap()
                .to_string()
        };
        // Explicit beats the stored login.
        let router = LMRouter::with_config(
            hermetic(&[("XAI_API_KEY", "env")])
                .api_key("xai", "explicit")
                .credentials_path(&usable),
        )
        .unwrap();
        assert_eq!(bearer(&router), "Bearer explicit");
        // The stored login beats the env key (it spends no money).
        let router =
            LMRouter::with_config(hermetic(&[("XAI_API_KEY", "env")]).credentials_path(&usable))
                .unwrap();
        assert_eq!(bearer(&router), "Bearer stored");
        // An unusable login (expired, no refresh) BLOCKS the env key (R3,
        // ratified 2026-09-22): a failed subscription is never silently
        // replaced by a metered key. Passing the key explicitly still works.
        let unusable = write_login(
            "xai.json",
            &format!(r#"{{"xai":{{"type":"oauth","access":"stale","expires":{expired}}}}}"#),
        );
        let router = LMRouter::with_config(
            hermetic(&[("XAI_API_KEY", "SECRET-SENTINEL-DO-NOT-PRINT")])
                .credentials_path(&unusable),
        )
        .unwrap();
        let err = router.lm("grok-4").unwrap_err();
        assert_eq!(err.class_name(), "NotConfiguredError");
        assert!(
            err.message().contains("expired and cannot be renewed"),
            "{err}"
        );
        assert!(err.message().contains("$XAI_API_KEY is set"), "{err}");
        assert!(
            !format!("{err:?}").contains("SECRET-SENTINEL"),
            "the key's value never prints"
        );
        let router = LMRouter::with_config(
            hermetic(&[("XAI_API_KEY", "env")])
                .api_key("xai", "explicit")
                .credentials_path(&unusable),
        )
        .unwrap();
        assert_eq!(bearer(&router), "Bearer explicit");
        // A signed-out marker in lm15's own store blocks it too, across restarts.
        let signed_out = write_login(
            "xai.json",
            r#"{"_lm15":{"slots":{"xai":{"logged_out":true}}}}"#,
        );
        let router = LMRouter::with_config(
            hermetic(&[("XAI_API_KEY", "env")]).credentials_path(&signed_out),
        )
        .unwrap();
        let err = router.lm("grok-4").unwrap_err();
        assert!(err.message().contains("was signed out"), "{err}");
        // A usable-but-expired login (refresh token present) is selected by
        // AUTH-1 (never the env key). `build_request` by hand skips the
        // adapter's `prepare` step that refreshes it (AUTH-3, tested end to
        // end in tests/login_refresh.rs), so the sync path is the typed
        // refusal naming that step — loud, never a silent fall back.
        let refreshable = write_login(
            "xai.json",
            &format!(
                r#"{{"xai":{{"type":"oauth","access":"stale","expires":{expired},"refresh":"r"}}}}"#
            ),
        );
        let router = LMRouter::with_config(
            hermetic(&[("XAI_API_KEY", "env")]).credentials_path(&refreshable),
        )
        .unwrap();
        let err = router
            .lm("grok-4")
            .unwrap()
            .build_request(&request, false)
            .unwrap_err();
        assert_eq!(err.class_name(), "AuthError");
        assert!(err.message().contains("refresh"), "{err}");
        // Nothing anywhere: the login hint.
        let router =
            LMRouter::with_config(hermetic(&[]).credentials_path("/nonexistent/xai.json")).unwrap();
        let err = router.lm("grok-4").unwrap_err();
        assert_eq!(err.class_name(), "NotConfiguredError");
        assert!(err.message().contains("login"), "{err}");
    }

    #[test]
    fn config_debug_never_renders_a_credential() {
        let config = hermetic(&[("OPENAI_API_KEY", "SECRET-SENTINEL-DO-NOT-PRINT")])
            .api_key("openai", "SECRET-SENTINEL-DO-NOT-PRINT");
        let text = format!("{config:?}");
        assert!(!text.contains("SENTINEL"), "{text}");
        assert!(text.contains("\"openai\""), "{text}");
        let router = LMRouter::with_config(config).unwrap();
        assert!(!format!("{router:?}").contains("SENTINEL"));
    }
}

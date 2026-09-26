//! AUTH-7: explain how a provider's credential would resolve. No secrets,
//! no network, no writes.
//!
//! [`explain_auth`] walks exactly the AUTH-1 chain of the provider's policy
//! (`key`, `oauth`, `oauth-unless-explicit`) and reports every rung; the
//! cloud chains (`aws-chain`, `azure-chain`, `gcp-chain`) are walked
//! offline by `crate::cloud::chains` (`unprobed` for a network rung).
//!
//! Purity note, stated because it is a real trade-off: the walk tests env
//! vars for presence, so secret values do transit process memory. They are
//! never stored on the report, never included in `describe()`, and never
//! part of any `Debug` output.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};

use super::error::AuthError;
use super::policy::{access_policy, canonical_provider, CredentialPolicy};
use super::stores::{
    read_claude_code_credential, read_codex_cli_credential, read_xai_credential,
    LocalOAuthCredential,
};

/// spec/vocabularies.md `AuthStepState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StepState {
    /// This rung supplies the credential.
    Selected,
    /// Usable, but an earlier rung wins.
    Shadowed,
    /// Nothing here.
    Absent,
    /// A network or subprocess rung the offline doctor did not contact.
    Unprobed,
}

impl StepState {
    pub fn as_str(self) -> &'static str {
        match self {
            StepState::Selected => "selected",
            StepState::Shadowed => "shadowed",
            StepState::Absent => "absent",
            StepState::Unprobed => "unprobed",
        }
    }
}

/// One rung of the chain. `kind` is the language-neutral vocabulary of the
/// fixture (`api_keys`, `env:<VAR>`, `placeholder`, `oauth-file`); `source`
/// and `detail` are human text with no secret material by construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    pub kind: String,
    pub source: String,
    pub detail: String,
    pub state: StepState,
}

impl Step {
    fn new(
        kind: impl Into<String>,
        source: impl Into<String>,
        detail: impl Into<String>,
        state: StepState,
    ) -> Self {
        Step {
            kind: kind.into(),
            source: source.into(),
            detail: detail.into(),
            state,
        }
    }

    pub fn describe(&self) -> String {
        let marker = match self.state {
            StepState::Selected => "=> ",
            StepState::Shadowed => " ~ ",
            StepState::Absent => " - ",
            StepState::Unprobed => " ? ",
        };
        format!("{marker}{}: {}", self.source, self.detail)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Report {
    pub provider: String,
    pub steps: Vec<Step>,
    /// True when a rung is selected OR unprobed (PROTOCOL.md `explain_auth`).
    pub configured: bool,
    /// Resolved host settings by name and value (AUTH-7); empty for the
    /// core policies, which have no host.
    pub settings: Vec<(String, String)>,
    /// A deterministic cloud identity, rather than the default chain.
    pub named_credential: Option<String>,
    /// The resolved door URL (never an unvalidated endpoint).
    pub base_url: Option<String>,
    /// `base_urls`, `env $VARIABLE`, or `template`.
    pub endpoint_source: Option<String>,
    /// Where each setting came from (AUTH-10 `from`, amended 2026-09-26):
    /// `explicit`, `env:<VAR>`, `adc-env`, `gcloud-config`, `adc-file`,
    /// `metadata`, `aws-profile`, `default`; `unprobed:metadata` when only
    /// the metadata server could answer; `missing`.
    pub setting_sources: Vec<(String, String)>,
}

fn setting_from(origin: &str) -> String {
    if let Some(var) = origin.strip_prefix("env:") {
        return format!("env ${var}");
    }
    match origin {
        "explicit" => "settings",
        "adc-env" => "the GOOGLE_APPLICATION_CREDENTIALS file",
        "gcloud-config" => "gcloud's active configuration",
        "adc-file" => "the gcloud application default credentials file",
        "metadata" | "unprobed:metadata" => "the Google Cloud metadata server",
        "aws-profile" => "the active AWS profile",
        other => other,
    }
    .to_string()
}

impl Report {
    pub fn selected(&self) -> Option<&Step> {
        self.steps
            .iter()
            .find(|step| step.state == StepState::Selected)
    }

    pub fn describe(&self) -> String {
        let mut lines = vec![format!("auth for provider {:?}:", self.provider)];
        if let Some(name) = &self.named_credential {
            lines.push(format!(
                "  named credential: {name} (default chain is not walked)"
            ));
            #[cfg(feature = "native")]
            if let Some(policy) = access_policy(&self.provider) {
                if let Ok(meaning) = crate::cloud::chains::named_meaning(policy, name) {
                    lines.push(format!("  identity: {meaning}"));
                }
            }
        }
        lines.extend(
            self.steps
                .iter()
                .map(|step| format!("  {}", step.describe())),
        );
        let unprobed: Vec<&str> = self
            .steps
            .iter()
            .filter(|step| step.state == StepState::Unprobed)
            .map(|step| step.source.as_str())
            .collect();
        match self.selected() {
            Some(step) => {
                lines.push(format!("  configured: yes — {}", step.source));
                if !unprobed.is_empty() {
                    lines.push(format!(
                        "  note: {} run first at request time and may win",
                        unprobed.join(", ")
                    ));
                }
            }
            None if self.configured => lines.push(format!(
                "  configured: probably — {} (unprobed offline)",
                unprobed.join(", ")
            )),
            None => lines.push("  configured: no".into()),
        }
        for (name, value) in &self.settings {
            let origin = self
                .setting_sources
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, o)| o.as_str())
                .filter(|o| *o != "missing");
            match origin {
                Some(origin) => lines.push(format!(
                    "  setting {name}: {value} (from {})",
                    setting_from(origin)
                )),
                None => lines.push(format!("  setting {name}: {value}")),
            }
        }
        for (name, origin) in &self.setting_sources {
            if origin.starts_with("unprobed:") {
                lines.push(format!(
                    "  setting {name}: not found offline; {} is asked at request time",
                    setting_from(origin)
                ));
            }
        }
        if let Some(url) = &self.base_url {
            let source = self.endpoint_source.as_deref().unwrap_or("template");
            lines.push(format!("  base url: {url} (from {source})"));
        }
        lines.join("\n")
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.describe())
    }
}

/// Inputs for [`explain_auth`]. Every input is the caller's: with `env`
/// set, the process environment is never consulted (PROTOCOL.md
/// `explain_auth`: "the shim must never consult its real process
/// environment"); `None` means the process environment.
///
/// `api_key_providers` lists providers with an explicit `api_keys` entry —
/// presence only; values are never consulted by explain.
///
/// `credentials_path` overrides the provider's stored-login file (the
/// borrowed Claude Code / Codex file, or the lm15-owned store for xai).
/// Without it the AUTH-8 default paths are derived from `env` (`HOME`,
/// `LM15_CREDENTIALS_PATH`, `XDG_CONFIG_HOME`).
#[derive(Default, Clone)]
pub struct ExplainOptions {
    pub env: Option<HashMap<String, String>>,
    pub api_key_providers: Vec<String>,
    pub credentials_path: Option<PathBuf>,
    /// Cloud chains: a filesystem overlay (`{path: content}`) the walk
    /// reads instead of the real one (the harness's sandbox HOME).
    pub files: Option<HashMap<String, String>>,
    /// Cloud doors: the caller's host settings (AUTH-10); env and the
    /// cloud profile fill the rest for the report.
    pub settings: Option<crate::cloud::hosts::HostSettings>,
    /// Select only this named cloud identity (platform/workload/environment/cli).
    pub credential: Option<String>,
    /// Explicit endpoint override, ahead of the host's vendor variables.
    pub base_url: Option<String>,
    /// Explicit entries supplied by callables. Their presence is reported,
    /// but no callable is invoked or inspected. May overlap api_key_providers.
    pub callable_providers: Vec<String>,
    /// A managed `Auth` (AUTH-15 mode B): the walk is the managed router's —
    /// explicit entry, named identity, the saved connection; environment keys
    /// shown and not consulted. Store reads only, no renewal.
    #[cfg(feature = "native")]
    pub auth: Option<crate::login::Auth>,
}

impl fmt::Debug for ExplainOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExplainOptions")
            .field("environment_supplied", &self.env.is_some())
            .field("explicit_entries", &self.api_key_providers.len())
            .field("callable_entries", &self.callable_providers.len())
            .field("files_supplied", &self.files.is_some())
            .field("settings_supplied", &self.settings.is_some())
            .field("named_credential_supplied", &self.credential.is_some())
            .field("endpoint_supplied", &self.base_url.is_some())
            .finish_non_exhaustive()
    }
}

impl ExplainOptions {
    fn explicit_providers(&self) -> Vec<&str> {
        let mut entries: Vec<_> = self.api_key_providers.iter().map(String::as_str).collect();
        for name in &self.callable_providers {
            if !entries.contains(&name.as_str()) {
                entries.push(name);
            }
        }
        entries
    }

    fn explicit_source<'a>(&'a self, canonical: &str) -> Result<Option<&'a str>, AuthError> {
        super::policy::shared_api_key_source(self.explicit_providers(), canonical).map_err(
            |candidates| {
                AuthError::not_configured(
                    canonical,
                    format!("ambiguous explicit credentials for {canonical:?} from {candidates:?}"),
                    format!("supply one entry under {canonical:?} or keep only one shared entry"),
                )
            },
        )
    }

    fn explicit_detail(&self, entry: &str) -> &'static str {
        if self
            .callable_providers
            .iter()
            .any(|name| canonical_provider(name) == canonical_provider(entry))
        {
            "application-supplied callable (identity not inspected)"
        } else {
            "provided (value never shown)"
        }
    }

    fn environment(&self) -> std::collections::BTreeMap<String, String> {
        match &self.env {
            Some(map) => map.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            None => std::env::vars().collect(),
        }
    }

    fn env_value(&self, key: &str) -> Option<String> {
        match &self.env {
            Some(map) => map.get(key).cloned(),
            None => std::env::var(key).ok(),
        }
        .filter(|value| !value.is_empty())
    }

    /// The files an `oauth` / `oauth-unless-explicit` provider reads, in
    /// order — the same files the router's [`super::StoredLogin`] reads
    /// (AUTH-7: the doctor walks exactly the chain real construction walks).
    fn stored_login_paths(&self, provider: &str) -> Vec<PathBuf> {
        super::login::stored_login_paths(
            provider,
            &|key| self.env_value(key),
            self.credentials_path.as_deref(),
        )
    }
}

/// Walks the AUTH-1 chain (or only the named identity's rungs) and reports
/// every step (AUTH-7). No network, subprocesses, or writes. File reads
/// inspect stored logins and cloud configuration; overlays remain offline.
///
/// Unknown providers, duplicate explicit entries, and invalid or conflicting
/// named credentials return typed configuration errors before any file walk.
pub fn explain_auth(provider: &str, options: &ExplainOptions) -> Result<Report, AuthError> {
    let policy = access_policy(provider).ok_or_else(|| AuthError::UnknownProvider {
        provider: provider.to_string(),
    })?;
    let canonical = policy.provider.to_string();
    validate_explicit_entries(options)?;
    #[cfg(feature = "native")]
    if let Some(auth) = &options.auth {
        return explain_managed(policy, &canonical, auth, options);
    }
    if let Some(name) = &options.credential {
        if !policy.credential_policy.is_cloud_chain() {
            return Err(AuthError::not_configured(
                &canonical,
                "named credentials require a cloud identity policy; this is not a cloud door",
                "remove credential or choose a cloud provider",
            ));
        }
        #[cfg(feature = "native")]
        crate::cloud::chains::named_meaning(policy, name)?;
        #[cfg(not(feature = "native"))]
        if !matches!(
            name.as_str(),
            "platform" | "workload" | "environment" | "cli"
        ) {
            return Err(AuthError::not_configured(
                &canonical,
                "unknown named credential",
                "choose platform, workload, environment, or cli",
            ));
        }
        if options.explicit_source(&canonical)?.is_some() {
            return Err(AuthError::not_configured(
                &canonical,
                "both api_keys and credentials select this provider's identity",
                "supply either credential or api_keys, not both",
            ));
        }
    }

    if policy.credential_policy.is_cloud_chain() {
        let mut report = explain_cloud(policy, &canonical, options)?;
        annotate_jwt(&mut report, policy, options);
        return Ok(report);
    }

    if policy.credential_policy == CredentialPolicy::OAuth {
        // The chain below never runs (AUTH-1 stored-credential-owns-provider).
        let step = oauth_file_step(&canonical, &options.stored_login_paths(&canonical), false);
        let configured = step.state == StepState::Selected;
        return Ok(Report {
            provider: canonical,
            steps: vec![step],
            configured,
            settings: Vec::new(),
            named_credential: None,
            base_url: None,
            endpoint_source: None,
            setting_sources: Vec::new(),
        });
    }

    let mut steps = Vec::with_capacity(3 + policy.env_keys.len());
    let mut selected = false;

    // AUTH-1 § Shared explicit keys, AUTH-7: the same selection the router
    // makes; the source configuration key is named when it differs from
    // the target. Ambiguity is the router's NotConfiguredError, here too.
    let explicit = super::policy::shared_api_key_source(options.explicit_providers(), &canonical)
        .map_err(|candidates| AuthError::NotConfigured {
        provider: Some(canonical.clone()),
        message: format!(
            "ambiguous explicit credentials for {canonical:?} from {}",
            candidates
                .iter()
                .map(|c| format!("{c:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        hint: Some(format!(
            "supply one entry under {canonical:?} or keep only one shared entry"
        )),
    })?;
    if let Some(entry) = explicit {
        let mut source = "explicit api_keys entry".to_string();
        if canonical_provider(entry) != canonical {
            source.push_str(&format!(" (via {entry:?}, shared env-key declarations)"));
        }
        steps.push(Step::new(
            "api_keys",
            source,
            options.explicit_detail(entry),
            StepState::Selected,
        ));
        selected = true;
    } else {
        steps.push(Step::new(
            "api_keys",
            "explicit api_keys entry",
            "not provided",
            StepState::Absent,
        ));
    }

    // An unusable or signed-out subscription login BLOCKS the env keys (R3,
    // ratified 2026-09-22): they show as shadowed, and nothing is selected.
    let mut blocked = false;
    if policy.credential_policy == CredentialPolicy::OAuthUnlessExplicit {
        // The stored subscription login outranks env keys (AUTH-1): it
        // spends no money per token. Only the explicit entry can shadow it.
        let paths = options.stored_login_paths(&canonical);
        let mut step = oauth_file_step(&canonical, &paths, selected);
        match crate::auth::StoredLogin::at(&canonical, paths).state() {
            crate::auth::StoredState::LoggedOut if !selected => {
                step = Step::new(
                    "oauth-file",
                    step.source.clone(),
                    "signed out (marker present)",
                    StepState::Absent,
                );
                blocked = true;
            }
            crate::auth::StoredState::Unusable if !selected => blocked = true,
            _ => {}
        }
        selected = selected || step.state == StepState::Selected;
        steps.push(step);
    }

    for key in policy.env_keys {
        let kind = format!("env:{key}");
        let source = format!("env ${key}");
        if options.env_value(key).is_some() {
            let state = if selected || blocked {
                StepState::Shadowed
            } else {
                StepState::Selected
            };
            let detail = if blocked && !selected {
                "set, blocked by the failed/signed-out subscription (pass it explicitly to use it)"
            } else {
                "set (value never shown)"
            };
            steps.push(Step::new(kind, source, detail, state));
            selected = selected || !blocked;
        } else {
            steps.push(Step::new(kind, source, "not set", StepState::Absent));
        }
    }

    if policy.placeholder_key.is_some() {
        let state = if selected {
            StepState::Shadowed
        } else {
            StepState::Selected
        };
        steps.push(Step::new(
            "placeholder",
            "local-server placeholder key",
            format!("preset default for keyless {canonical} servers"),
            state,
        ));
        selected = true;
    }

    let (settings, base_url, endpoint_source, setting_sources) =
        host_report(policy, options, &options.environment(), |_| None);
    let mut report = Report {
        provider: canonical,
        steps,
        configured: selected,
        settings,
        named_credential: None,
        base_url,
        endpoint_source,
        setting_sources,
    };
    annotate_jwt(&mut report, policy, options);
    Ok(report)
}

fn annotate_jwt(report: &mut Report, policy: &super::AccessPolicy, options: &ExplainOptions) {
    if !policy.auth_scheme.contains(&super::AuthScheme::Bearer)
        || !policy
            .auth_scheme
            .iter()
            .any(|s| matches!(s, super::AuthScheme::ApiKey | super::AuthScheme::XApiKey))
    {
        return;
    }
    for step in &mut report.steps {
        if let Some(key) = step.kind.strip_prefix("env:") {
            if let Some(shape) = options
                .env_value(key)
                .and_then(|value| super::looks_like_access_token(&value))
            {
                step.detail.push_str(&format!("; sent as bearer ({shape})"));
            }
        }
    }
}

fn explicit_source_label(canonical: &str, entry: Option<&str>) -> String {
    let mut source = "explicit api_keys entry".to_string();
    if let Some(entry) = entry.filter(|entry| canonical_provider(entry) != canonical) {
        source.push_str(&format!(" (via {entry:?}, shared env-key declarations)"));
    }
    source
}

fn validate_explicit_entries(options: &ExplainOptions) -> Result<(), AuthError> {
    let mut seen = HashMap::new();
    for entry in options.api_key_providers.iter().chain(
        options
            .callable_providers
            .iter()
            .filter(|name| !options.api_key_providers.contains(name)),
    ) {
        let canonical = canonical_provider(entry);
        if let Some(previous) = seen.insert(canonical.clone(), entry) {
            return Err(AuthError::not_configured(
                canonical,
                format!("duplicate explicit credential entries {previous:?} and {entry:?}"),
                "keep one spelling for each provider",
            ));
        }
    }
    Ok(())
}

/// The host half of a report: settings by name and value, the door URL and
/// where it came from, and where each setting came from (AUTH-10 `from`).
type HostReport = (
    Vec<(String, String)>,
    Option<String>,
    Option<String>,
    Vec<(String, String)>,
);

/// Resolve endpoints without retaining raw endpoint input on an error: an
/// invalid URL may contain userinfo, a query token, or other secret material.
fn host_report(
    policy: &crate::auth::AccessPolicy,
    options: &ExplainOptions,
    env: &std::collections::BTreeMap<String, String>,
    profile: impl Fn(&str) -> Option<crate::cloud::hosts::ProfileValue>,
) -> HostReport {
    use crate::cloud::hosts::{
        endpoint_from_env, resolve_base_url, resolve_settings_traced, ProfileValue, SettingsTrace,
    };
    let Some(host) = &policy.host else {
        return (Vec::new(), None, None, Vec::new());
    };
    let (endpoint, source) = if let Some(endpoint) = options.base_url.as_deref() {
        (Some(endpoint), "base_urls".to_string())
    } else if let Some((variable, endpoint)) = endpoint_from_env(host, env) {
        (Some(endpoint), format!("env ${variable}"))
    } else {
        (None, "template".to_string())
    };
    let mut given = options.settings.clone().unwrap_or_default();
    let mut trace = SettingsTrace {
        collect: true,
        ..SettingsTrace::default()
    };
    for setting in host.settings {
        if given.get(setting.name).is_none_or(String::is_empty)
            && !setting
                .env
                .iter()
                .any(|key| env.get(*key).is_some_and(|v| !v.is_empty()))
        {
            match profile(setting.name) {
                Some(ProfileValue::Found(value, from)) => {
                    given.insert(setting.name.to_string(), value);
                    trace.injected.insert(setting.name.to_string(), from);
                }
                Some(ProfileValue::Metadata) if setting.default.is_none() => {
                    trace.pending.insert(setting.name.to_string());
                }
                _ => {}
            }
        }
    }
    let resolved = match resolve_settings_traced(
        Some(host),
        &given,
        Some(env),
        policy.provider,
        endpoint,
        &mut trace,
    ) {
        Ok(settings) => settings,
        Err(error) => {
            return (
                // Host errors are value-free. Unknown setting names are caller
                // input, so suppress those rather than copying them into a report.
                vec![(
                    "error".into(),
                    if matches!(&error, crate::errors::Lm15Error::ConfigurationError(_)) {
                        "unknown host setting (name and value not shown)".into()
                    } else {
                        error.message().replace('"', "'")
                    },
                )],
                None,
                None,
                Vec::new(),
            );
        }
    };
    let sources: Vec<(String, String)> = trace.sources.clone().into_iter().collect();
    let mut settings: Vec<_> = resolved.clone().into_iter().collect();
    if let Some(problem) = trace.problems.first() {
        settings.push(("error".into(), problem.message().replace('"', "'")));
        return (settings, None, None, sources);
    }
    if !trace.pending.is_empty() {
        return (settings, None, None, sources);
    }
    match resolve_base_url(host, &resolved, endpoint) {
        Ok(url) => (settings, Some(url), Some(source), sources),
        Err(error) => {
            settings.push(("error".into(), error.message().to_string()));
            (settings, None, None, sources)
        }
    }
}

/// The cloud-chain walk (module 3b): `cloud::chains::explain` over an
/// offline context built from the options, plus the resolved host
/// settings (explicit, env, the cloud profile, defaults) for the report.
#[cfg(not(feature = "native"))]
fn explain_cloud(
    policy: &'static crate::auth::AccessPolicy,
    canonical: &str,
    options: &ExplainOptions,
) -> Result<Report, AuthError> {
    // The wire codec build: no profile files, CLIs or metadata endpoints to
    // walk. The report says so, rung by rung absent, as the router refuses.
    let entry = options.explicit_source(canonical)?;
    let explicit = entry.is_some();
    let (settings, base_url, endpoint_source, setting_sources) =
        host_report(policy, options, &options.environment(), |_| None);
    let steps = vec![
        Step {
            kind: "api_keys".into(),
            source: explicit_source_label(canonical, entry),
            detail: entry
                .map(|entry| options.explicit_detail(entry))
                .unwrap_or("not provided")
                .into(),
            state: if explicit {
                StepState::Selected
            } else {
                StepState::Absent
            },
        },
        Step {
            kind: options
                .credential
                .as_deref()
                .unwrap_or(policy.credential_policy.as_str())
                .into(),
            source: format!(
                "{} (profile files, CLIs, metadata endpoints)",
                options
                    .credential
                    .as_deref()
                    .unwrap_or(policy.credential_policy.as_str())
            ),
            detail: "not available in this build (no `native` feature)".into(),
            state: StepState::Absent,
        },
    ];
    Ok(Report {
        provider: canonical.to_string(),
        steps,
        configured: explicit,
        settings,
        named_credential: options.credential.clone(),
        base_url,
        endpoint_source,
        setting_sources,
    })
}

#[cfg(feature = "native")]
fn explain_cloud(
    policy: &'static crate::auth::AccessPolicy,
    canonical: &str,
    options: &ExplainOptions,
) -> Result<Report, AuthError> {
    use crate::cloud::chains::{self, ChainContext};
    let mut env = options.environment();
    // ChainContext's general constructor may fall back to the process HOME.
    // The doctor's supplied environment is authoritative, including absence.
    if options.env.is_some() && env.get("HOME").is_none_or(String::is_empty) {
        env.insert("HOME".into(), "/".into());
    }
    let mut ctx = ChainContext::offline(env.clone(), crate::auth::time_now());
    if let Some(files) = &options.files {
        ctx = ctx.with_files(files.iter().map(|(k, v)| (k.clone(), v.clone())).collect());
    }
    let entry = options.explicit_source(canonical)?;
    let explicit = entry.is_some();
    let (settings, base_url, endpoint_source, setting_sources) =
        host_report(policy, options, &env, |name| {
            chains::profile_setting(policy, &ctx, name)
        });
    ctx.settings = settings
        .iter()
        .filter(|(name, _)| name != "error")
        .cloned()
        .collect();
    let (mut steps, configured) = match options.credential.as_deref() {
        Some(name) => chains::explain_named(policy, &ctx, explicit, name)?,
        None => chains::explain(policy, &ctx, explicit)?,
    };
    if let Some(entry) = entry {
        if let Some(step) = steps.iter_mut().find(|step| step.kind == "api_keys") {
            step.source = explicit_source_label(canonical, Some(entry));
            step.detail = options.explicit_detail(entry).into();
        }
    }
    Ok(Report {
        provider: canonical.to_string(),
        steps: steps
            .into_iter()
            .map(|s| Step {
                kind: s.kind,
                source: s.source,
                detail: s.detail,
                state: match s.state {
                    "selected" => StepState::Selected,
                    "shadowed" => StepState::Shadowed,
                    "unprobed" => StepState::Unprobed,
                    _ => StepState::Absent,
                },
            })
            .collect(),
        configured,
        settings,
        named_credential: options.credential.clone(),
        base_url,
        endpoint_source,
        setting_sources,
    })
}

/// The stored-login rung. `shadowed`: an earlier rung already won, so a
/// usable login is reported `shadowed` instead of `selected`.
fn oauth_file_step(provider: &str, paths: &[PathBuf], shadowed: bool) -> Step {
    let reader: fn(&Path) -> Result<LocalOAuthCredential, AuthError> = match provider {
        "claude-code" => read_claude_code_credential,
        "openai-codex" => read_codex_cli_credential,
        _ => read_xai_credential,
    };
    let found = paths
        .iter()
        .find_map(|path| reader(path).ok().map(|credential| (credential, path)));
    match found {
        None => {
            let checked = if paths.is_empty() {
                "(no HOME to derive the AUTH-8 path from)".to_string()
            } else {
                paths
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(" or ")
            };
            Step::new(
                "oauth-file",
                format!("local OAuth credential {checked}"),
                "missing or unreadable",
                StepState::Absent,
            )
        }
        Some((credential, path)) => {
            let source = format!("local OAuth credential {}", path.display());
            let detail = expiry_detail(&credential);
            let state = if !credential.usable() {
                StepState::Absent
            } else if shadowed {
                StepState::Shadowed
            } else {
                StepState::Selected
            };
            Step::new("oauth-file", source, detail, state)
        }
    }
}

fn expiry_detail(credential: &LocalOAuthCredential) -> String {
    let remaining_ms = match credential.remaining_ms() {
        Ok(None) => return "no recorded expiry".into(),
        // AUTH-8: a value the clock arithmetic cannot use is reported, not
        // read as fresh; the rung is `absent` because `usable()` is false.
        Err(_) => return "malformed expiry (out of range)".into(),
        Ok(Some(remaining)) => remaining,
    };
    if remaining_ms <= 0 {
        return if credential.has_refresh_token() {
            "expired, refresh token present".into()
        } else {
            "expired, NO refresh token".into()
        };
    }
    let minutes = remaining_ms / 60_000;
    let (hours, minutes) = (minutes / 60, minutes % 60);
    if hours > 0 {
        format!("fresh, expires in {hours}h {minutes:02}m")
    } else {
        format!("fresh, expires in {minutes}m")
    }
}

/// AUTH-15 mode B, rung by rung: the explicit entry, the named cloud
/// identity, the scope's saved connection; environment keys shown and
/// marked not consulted. Store reads only, no renewal (AUTH-7).
#[cfg(feature = "native")]
fn explain_managed(
    policy: &super::policy::AccessPolicy,
    canonical: &str,
    auth: &crate::login::Auth,
    options: &ExplainOptions,
) -> Result<Report, AuthError> {
    let mut steps = Vec::new();
    let mut selected = false;
    match options.explicit_source(canonical)? {
        Some(entry) => {
            steps.push(Step::new(
                "api_keys",
                explicit_source_label(canonical, Some(entry)),
                "provided (value never shown)",
                StepState::Selected,
            ));
            selected = true;
        }
        None => steps.push(Step::new(
            "api_keys",
            "explicit api_keys entry",
            "not provided",
            StepState::Absent,
        )),
    }
    if let Some(named) = &options.credential {
        steps.push(Step::new(
            "named_cloud",
            format!("named credential \"{named}\""),
            "explicit",
            if selected {
                StepState::Shadowed
            } else {
                StepState::Selected
            },
        ));
        selected = true;
    }
    let status = auth
        .status(canonical)
        .map_err(|e| AuthError::Lm15(Box::new(e)))?;
    match &status.connection {
        Some(connection) => {
            let expires = status
                .expires_at
                .as_ref()
                .map(|e| format!(", expires {e}"))
                .unwrap_or_default();
            let detail = format!("{} ({}{expires})", connection.label, status.usability);
            let state = if selected {
                StepState::Shadowed
            } else if status.ready() {
                StepState::Selected
            } else {
                StepState::Absent
            };
            steps.push(Step::new(
                "connection",
                format!("saved connection {}", connection.id),
                detail,
                state,
            ));
            selected = selected || state == StepState::Selected;
        }
        None => {
            let detail = if status.logged_out {
                "signed out (marker present)"
            } else {
                "none saved in this scope"
            };
            steps.push(Step::new(
                "connection",
                format!("saved connection in {}", auth.store().description()),
                detail,
                StepState::Absent,
            ));
        }
    }
    for key in policy.env_keys {
        let set = options.env_value(key).is_some_and(|v| !v.is_empty());
        steps.push(if set {
            Step::new(
                format!("env:{key}"),
                format!("env ${key}"),
                "set, not consulted under a managed Auth (pass it explicitly to use it)",
                StepState::Shadowed,
            )
        } else {
            Step::new(
                format!("env:{key}"),
                format!("env ${key}"),
                "not set",
                StepState::Absent,
            )
        });
    }
    let placeholder = crate::registry::lookup(canonical)
        .and_then(|d| d.placeholder_key)
        .or(policy.placeholder_key);
    if placeholder.is_some() && !status.logged_out {
        steps.push(Step::new(
            "placeholder",
            "local-server placeholder key",
            format!("preset default for keyless {canonical} servers"),
            if selected {
                StepState::Shadowed
            } else {
                StepState::Selected
            },
        ));
        selected = true;
    }
    Ok(Report {
        provider: canonical.to_string(),
        steps,
        configured: selected,
        settings: Vec::new(),
        named_credential: None,
        base_url: None,
        endpoint_source: None,
        setting_sources: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hermetic(env: &[(&str, &str)]) -> ExplainOptions {
        ExplainOptions {
            env: Some(
                env.iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect(),
            ),
            ..Default::default()
        }
    }

    #[test]
    fn underscore_alias_is_accepted() {
        let report = explain_auth("openai_chat", &hermetic(&[])).unwrap();
        assert_eq!(report.provider, "openai-chat");
    }

    #[test]
    fn unknown_provider_names_known_ones() {
        let error = explain_auth("nope", &hermetic(&[])).unwrap_err();
        assert_eq!(error.code(), "not_configured");
        assert!(error.to_string().contains("anthropic"));
    }

    #[test]
    fn cloud_chain_providers_walk_their_chain_offline() {
        // Module 3b: the walk is the cloud SDK's order; nothing configured
        // and the metadata rungs disabled is "not configured", with every
        // rung reported.
        let report = explain_auth(
            "bedrock-chat",
            &hermetic(&[
                ("AWS_REGION", "us-east-1"),
                ("AWS_EC2_METADATA_DISABLED", "true"),
                ("HOME", "/nonexistent"),
            ]),
        )
        .unwrap();
        assert!(!report.configured, "{}", report.describe());
        assert_eq!(report.steps[1].kind, "env:AWS_BEARER_TOKEN_BEDROCK");
        assert_eq!(report.steps.last().unwrap().kind, "imds");
        assert!(report
            .settings
            .iter()
            .any(|(k, v)| k == "region" && v == "us-east-1"));
        // A static key pair is selected; IMDS behind it is shadowed.
        let report = explain_auth(
            "bedrock-chat",
            &hermetic(&[
                ("AWS_REGION", "us-east-1"),
                ("AWS_ACCESS_KEY_ID", "AKID"),
                ("AWS_SECRET_ACCESS_KEY", "SECRET-SENTINEL-DO-NOT-PRINT"),
                ("HOME", "/nonexistent"),
            ]),
        )
        .unwrap();
        assert!(report.configured);
        assert_eq!(report.selected().unwrap().kind, "env:AWS_ACCESS_KEY_ID");
        assert_eq!(report.steps.last().unwrap().state, StepState::Shadowed);
        assert!(!report.describe().contains("SENTINEL"));
        // vertex-express is a hosted door with the ordinary key chain (core).
        assert!(explain_auth("vertex-express", &hermetic(&[])).is_ok());
    }

    #[test]
    fn gemini_env_key_order_first_wins() {
        let report = explain_auth(
            "gemini",
            &hermetic(&[("GEMINI_API_KEY", "a"), ("GOOGLE_API_KEY", "b")]),
        )
        .unwrap();
        assert_eq!(report.selected().unwrap().kind, "env:GEMINI_API_KEY");
        assert_eq!(report.steps[2].kind, "env:GOOGLE_API_KEY");
        assert_eq!(report.steps[2].state, StepState::Shadowed);
    }

    #[test]
    fn empty_env_value_is_absent() {
        let report = explain_auth("groq", &hermetic(&[("GROQ_API_KEY", "")])).unwrap();
        assert!(!report.configured);
        assert_eq!(report.steps[1].state, StepState::Absent);
    }

    #[test]
    fn stored_login_paths_follow_auth_8() {
        let options = hermetic(&[("HOME", "/h")]);
        assert_eq!(
            options.stored_login_paths("claude-code"),
            [PathBuf::from("/h/.claude/.credentials.json")]
        );
        assert_eq!(
            options.stored_login_paths("openai-codex"),
            [PathBuf::from("/h/.codex/auth.json")]
        );
        assert_eq!(
            options.stored_login_paths("xai"),
            [
                PathBuf::from("/h/.config/lm15/credentials.json"),
                PathBuf::from("/h/.pi/agent/auth.json")
            ]
        );
        let xdg = hermetic(&[("HOME", "/h"), ("XDG_CONFIG_HOME", "/x")]);
        assert_eq!(
            xdg.stored_login_paths("xai")[0],
            PathBuf::from("/x/lm15/credentials.json")
        );
        let explicit = hermetic(&[("HOME", "/h"), ("LM15_CREDENTIALS_PATH", "/s.json")]);
        assert_eq!(
            explicit.stored_login_paths("xai")[0],
            PathBuf::from("/s.json")
        );
        // No HOME, no override: nothing to read; the rung is absent, not a panic.
        let report = explain_auth("claude-code", &hermetic(&[])).unwrap();
        assert_eq!(report.steps[0].state, StepState::Absent);
        assert!(!report.configured);
    }

    #[test]
    fn describe_carries_every_step_and_the_verdict() {
        let report = explain_auth("ollama", &hermetic(&[])).unwrap();
        let text = report.describe();
        assert!(text.contains("auth for provider \"ollama\":"));
        assert!(text.contains("=> local-server placeholder key"));
        assert!(text.ends_with("configured: yes — local-server placeholder key"));
        assert_eq!(text, report.to_string());
    }

    /// F6 (review 2026-09-07): an expiry the `i64` clock arithmetic cannot
    /// use is `absent` with a "malformed" detail — never fresh, never a
    /// panic (AUTH-6, AUTH-8).
    #[test]
    fn out_of_range_expiry_is_absent_and_malformed() {
        let dir = std::env::temp_dir().join(format!("lm15-doctor-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // xai store: expires = i64::MIN.
        let xai = dir.join("xai.json");
        std::fs::write(
            &xai,
            r#"{"xai": {"type": "oauth", "access": "t", "expires": -9223372036854775808, "refresh": "r"}}"#,
        )
        .unwrap();
        // codex: JWT exp = i64::MAX, so exp * 1000 overflows.
        let codex = dir.join("codex.json");
        std::fs::write(
            &codex,
            r#"{"tokens": {"access_token": "h.eyJleHAiOjkyMjMzNzIwMzY4NTQ3NzU4MDd9.s", "refresh_token": "r"}}"#,
        )
        .unwrap();
        for (provider, path) in [("xai", &xai), ("openai-codex", &codex)] {
            let mut options = hermetic(&[]);
            options.credentials_path = Some(path.clone());
            let report = explain_auth(provider, &options).unwrap();
            let step = report
                .steps
                .iter()
                .find(|s| s.kind == "oauth-file")
                .unwrap_or_else(|| panic!("{provider}: no oauth-file rung"));
            assert_eq!(step.state, StepState::Absent, "{provider}: {step:?}");
            assert!(
                step.detail.contains("malformed"),
                "{provider}: {}",
                step.detail
            );
            assert!(
                !step.detail.contains("fresh"),
                "{provider}: {}",
                step.detail
            );
            assert!(!report.configured, "{provider}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

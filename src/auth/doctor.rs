//! AUTH-7: explain how a provider's credential would resolve. No secrets,
//! no network, no writes.
//!
//! [`explain_auth`] walks exactly the AUTH-1 chain of the provider's policy
//! (`key`, `oauth`, `oauth-unless-explicit`) and reports every rung; the
//! cloud chains (module 3b) are answered with [`AuthError::NotImplemented`].
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
    /// A configured network or subprocess rung of a cloud chain the offline
    /// doctor did not contact (module 3b; never produced by this port yet).
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
}

impl Report {
    pub fn selected(&self) -> Option<&Step> {
        self.steps
            .iter()
            .find(|step| step.state == StepState::Selected)
    }

    pub fn describe(&self) -> String {
        let mut lines = vec![format!("auth for provider {:?}:", self.provider)];
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
            lines.push(format!("  setting {name}: {value}"));
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
#[derive(Debug, Default, Clone)]
pub struct ExplainOptions {
    pub env: Option<HashMap<String, String>>,
    pub api_key_providers: Vec<String>,
    pub credentials_path: Option<PathBuf>,
}

impl ExplainOptions {
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

/// Walks the AUTH-1 chain and reports every rung (AUTH-7). No network I/O,
/// no writes; file reads are limited to the stored-login file.
///
/// Errors: an unknown provider ([`AuthError::UnknownProvider`]) or a
/// cloud-chain provider ([`AuthError::NotImplemented`], module 3b).
pub fn explain_auth(provider: &str, options: &ExplainOptions) -> Result<Report, AuthError> {
    let policy = access_policy(provider).ok_or_else(|| AuthError::UnknownProvider {
        provider: provider.to_string(),
    })?;
    let canonical = policy.provider.to_string();

    if policy.credential_policy.is_cloud_chain() {
        return Err(AuthError::NotImplemented {
            provider: canonical,
            policy: policy.credential_policy,
        });
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
        });
    }

    let mut steps = Vec::with_capacity(3 + policy.env_keys.len());
    let mut selected = false;

    let explicit = options
        .api_key_providers
        .iter()
        .any(|name| canonical_provider(name) == canonical);
    if explicit {
        steps.push(Step::new(
            "api_keys",
            "explicit api_keys entry",
            "provided (value never shown)",
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

    if policy.credential_policy == CredentialPolicy::OAuthUnlessExplicit {
        // The stored subscription login outranks env keys (AUTH-1): it
        // spends no money per token. Only the explicit entry can shadow it.
        let step = oauth_file_step(
            &canonical,
            &options.stored_login_paths(&canonical),
            selected,
        );
        selected = selected || step.state == StepState::Selected;
        steps.push(step);
    }

    for key in policy.env_keys {
        let kind = format!("env:{key}");
        let source = format!("env ${key}");
        if options.env_value(key).is_some() {
            let state = if selected {
                StepState::Shadowed
            } else {
                StepState::Selected
            };
            steps.push(Step::new(kind, source, "set (value never shown)", state));
            selected = true;
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

    Ok(Report {
        provider: canonical,
        steps,
        configured: selected,
        settings: Vec::new(),
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
    fn cloud_chain_providers_answer_not_implemented_naming_module_3b() {
        for provider in ["bedrock-chat", "azure", "vertex-anthropic"] {
            let error =
                explain_auth(provider, &hermetic(&[("AWS_REGION", "us-east-1")])).unwrap_err();
            assert!(
                matches!(error, AuthError::NotImplemented { .. }),
                "{provider}: {error:?}"
            );
            assert_eq!(error.class_name(), "NotConfiguredError");
            assert!(error.to_string().contains("module 3b"), "{error}");
        }
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

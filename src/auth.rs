//! lm15-contract auth surface (spec/auth.md, ratified 2026-08-31):
//! credential providers (AUTH-2), the resolution chain (AUTH-1), the explain
//! report (AUTH-7), and read-side borrowed CLI credentials (AUTH-8).
//!
//! Secrecy invariant (AUTH-5): no secret value is stored on a [`Report`],
//! rendered by `describe`, or emitted by any `Debug`/`Display` impl here.
//!
//! Not yet implemented in this port (stated, not absorbed): the AUTH-3/4
//! write side (locked double-checked refresh, atomic 0600 writes) and the
//! AUTH-9 login primitives. This port currently reads credentials only.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

/// A token inside this window counts as expired (AUTH-3 skew).
const REFRESH_SKEW_MS: i64 = 5 * 60 * 1000;

// ─── credential providers (AUTH-2) ───────────────────────────────────

/// Supplies a credential value per request. Adapters must call it at
/// request-build time and never cache the value.
pub trait CredentialProvider {
    fn token(&self) -> Result<String, AuthError>;
}

/// A fixed credential value. `Debug` is redacted.
pub struct StaticCredential(String);

impl StaticCredential {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
}

impl CredentialProvider for StaticCredential {
    fn token(&self) -> Result<String, AuthError> {
        Ok(self.0.clone())
    }
}

impl fmt::Debug for StaticCredential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("StaticCredential(redacted)")
    }
}

/// Adapts a closure to [`CredentialProvider`].
pub struct FnCredential<F: Fn() -> Result<String, AuthError>>(pub F);

impl<F: Fn() -> Result<String, AuthError>> CredentialProvider for FnCredential<F> {
    fn token(&self) -> Result<String, AuthError> {
        (self.0)()
    }
}

// ─── errors (AUTH-6) ─────────────────────────────────────────────────

#[derive(Debug)]
pub enum AuthError {
    /// The provider string names nothing in the built-in table.
    UnknownProvider { provider: String },
    /// No usable credential source (missing/unreadable/malformed file).
    /// The hint names the fix; it never contains token material.
    NotConfigured { provider: String, message: String, hint: String },
}

impl fmt::Display for AuthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AuthError::UnknownProvider { provider } => write!(
                f,
                "unknown provider {provider:?}; known providers: {}",
                known_providers().join(", ")
            ),
            AuthError::NotConfigured { provider, message, hint } => {
                write!(f, "{provider}: {message}\n\n  To fix:\n    - {hint}\n")
            }
        }
    }
}

impl std::error::Error for AuthError {}

// ─── provider table (mirrors the reference router) ───────────────────

struct ProviderSpec {
    name: &'static str,
    env_keys: &'static [&'static str],
    default_key: Option<&'static str>,
    oauth_file: Option<&'static str>,
}

const PROVIDERS: &[ProviderSpec] = &[
    ProviderSpec { name: "openai", env_keys: &["OPENAI_API_KEY"], default_key: None, oauth_file: None },
    ProviderSpec { name: "openai-chat", env_keys: &["OPENAI_API_KEY"], default_key: None, oauth_file: None },
    ProviderSpec { name: "anthropic", env_keys: &["ANTHROPIC_API_KEY"], default_key: None, oauth_file: None },
    ProviderSpec { name: "gemini", env_keys: &["GEMINI_API_KEY", "GOOGLE_API_KEY"], default_key: None, oauth_file: None },
    ProviderSpec { name: "groq", env_keys: &["GROQ_API_KEY"], default_key: None, oauth_file: None },
    ProviderSpec { name: "openrouter", env_keys: &["OPENROUTER_API_KEY"], default_key: None, oauth_file: None },
    ProviderSpec { name: "deepseek", env_keys: &["DEEPSEEK_API_KEY"], default_key: None, oauth_file: None },
    ProviderSpec { name: "zai", env_keys: &["ZAI_API_KEY"], default_key: None, oauth_file: None },
    ProviderSpec { name: "ollama", env_keys: &[], default_key: Some("ollama"), oauth_file: None },
    ProviderSpec { name: "vllm", env_keys: &[], default_key: Some("EMPTY"), oauth_file: None },
    ProviderSpec { name: "sglang", env_keys: &[], default_key: Some("EMPTY"), oauth_file: None },
    ProviderSpec { name: "claude-code", env_keys: &[], default_key: None, oauth_file: Some("claude-code") },
    ProviderSpec { name: "openai-codex", env_keys: &[], default_key: None, oauth_file: Some("openai-codex") },
];

/// Maps the permanent underscore alias to the hyphenated provider string.
pub fn canonical_provider(name: &str) -> String {
    name.replace('_', "-")
}

/// Every provider in the built-in table, sorted.
pub fn known_providers() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = PROVIDERS.iter().map(|spec| spec.name).collect();
    names.sort_unstable();
    names
}

fn provider_spec(canonical: &str) -> Option<&'static ProviderSpec> {
    PROVIDERS.iter().find(|spec| spec.name == canonical)
}

// ─── explain report (AUTH-7) ─────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepState {
    /// This rung supplies the credential.
    Selected,
    /// Usable, but an earlier rung wins.
    Shadowed,
    /// Nothing here.
    Absent,
}

impl StepState {
    pub fn as_str(self) -> &'static str {
        match self {
            StepState::Selected => "selected",
            StepState::Shadowed => "shadowed",
            StepState::Absent => "absent",
        }
    }
}

/// One rung of the chain. `kind` uses the contract vocabulary
/// (`api_keys`, `env:<KEY>`, `placeholder`, `oauth-file`); `detail` is
/// human text with no secret material by construction.
#[derive(Debug, Clone)]
pub struct Step {
    pub kind: String,
    pub detail: String,
    pub state: StepState,
}

#[derive(Debug, Clone)]
pub struct Report {
    pub provider: String,
    pub steps: Vec<Step>,
    pub configured: bool,
}

impl Report {
    pub fn selected(&self) -> Option<&Step> {
        self.steps.iter().find(|step| step.state == StepState::Selected)
    }

    pub fn describe(&self) -> String {
        let mut out = format!("auth for provider {:?}:\n", self.provider);
        for step in &self.steps {
            let marker = match step.state {
                StepState::Selected => "=> ",
                StepState::Shadowed => " ~ ",
                StepState::Absent => " - ",
            };
            out.push_str(&format!("  {marker}{}: {}\n", step.kind, step.detail));
        }
        match self.selected() {
            Some(step) => out.push_str(&format!("  configured: yes — {}", step.kind)),
            None => out.push_str("  configured: no"),
        }
        out
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.describe())
    }
}

/// Options for [`explain_auth`]. `env: None` means the process environment.
/// `api_key_providers` lists providers with explicit credentials — presence
/// only, values are never consulted by explain.
#[derive(Default)]
pub struct ExplainOptions {
    pub env: Option<HashMap<String, String>>,
    pub api_key_providers: Vec<String>,
    pub claude_credentials_path: Option<PathBuf>,
    pub codex_auth_path: Option<PathBuf>,
}

fn env_value(options: &ExplainOptions, key: &str) -> Option<String> {
    match &options.env {
        Some(map) => map.get(key).cloned(),
        None => std::env::var(key).ok(),
    }
}

/// Walks the AUTH-1 chain and reports every rung (AUTH-7). No network I/O.
/// Env values are tested for presence only and never retained — that
/// presence check is the one stated purity trade-off.
pub fn explain_auth(provider: &str, options: &ExplainOptions) -> Result<Report, AuthError> {
    let canonical = canonical_provider(provider);
    let spec = provider_spec(&canonical)
        .ok_or_else(|| AuthError::UnknownProvider { provider: provider.to_string() })?;

    if let Some(oauth_provider) = spec.oauth_file {
        let step = oauth_file_step(oauth_provider, options);
        let configured = step.state == StepState::Selected;
        return Ok(Report { provider: canonical, steps: vec![step], configured });
    }

    let mut steps = Vec::with_capacity(2 + spec.env_keys.len());
    let mut selected = false;

    if options.api_key_providers.iter().any(|name| name == &canonical) {
        steps.push(Step {
            kind: "api_keys".into(),
            detail: "provided (value never shown)".into(),
            state: StepState::Selected,
        });
        selected = true;
    } else {
        steps.push(Step { kind: "api_keys".into(), detail: "not provided".into(), state: StepState::Absent });
    }

    for key in spec.env_keys {
        let kind = format!("env:{key}");
        if env_value(options, key).filter(|value| !value.is_empty()).is_some() {
            let state = if selected { StepState::Shadowed } else { StepState::Selected };
            steps.push(Step { kind, detail: "set (value never shown)".into(), state });
            selected = true;
        } else {
            steps.push(Step { kind, detail: "not set".into(), state: StepState::Absent });
        }
    }

    if spec.default_key.is_some() {
        let state = if selected { StepState::Shadowed } else { StepState::Selected };
        steps.push(Step {
            kind: "placeholder".into(),
            detail: format!("preset default for keyless {canonical} servers"),
            state,
        });
        selected = true;
    }

    Ok(Report { provider: canonical, steps, configured: selected })
}

fn oauth_file_step(provider: &'static str, options: &ExplainOptions) -> Step {
    let credential = match provider {
        "claude-code" => {
            let path = options
                .claude_credentials_path
                .clone()
                .or_else(default_claude_credentials_path);
            path.and_then(|p| read_claude_code_credential(&p).ok())
        }
        _ => {
            let path = options.codex_auth_path.clone().or_else(default_codex_auth_path);
            path.and_then(|p| read_codex_cli_credential(&p).ok())
        }
    };
    let absent = |detail: &str| Step {
        kind: "oauth-file".into(),
        detail: detail.into(),
        state: StepState::Absent,
    };
    match credential {
        None => absent("missing or unreadable"),
        Some(credential) if credential.expired() => {
            if credential.has_refresh_token() {
                Step {
                    kind: "oauth-file".into(),
                    detail: "expired, refresh token present".into(),
                    state: StepState::Selected,
                }
            } else {
                absent("expired, NO refresh token")
            }
        }
        Some(_) => Step { kind: "oauth-file".into(), detail: "fresh".into(), state: StepState::Selected },
    }
}

// ─── borrowed CLI credentials, read side (AUTH-8) ────────────────────

/// A locally stored OAuth credential. Token fields are private and `Debug`
/// is redacted (AUTH-5); use the accessors.
pub struct LocalOAuthCredential {
    access_token: String,
    refresh_token: Option<String>,
    expires_at_ms: Option<i64>,
    account_id: Option<String>,
}

impl LocalOAuthCredential {
    pub fn access_token(&self) -> &str {
        &self.access_token
    }

    pub fn has_refresh_token(&self) -> bool {
        self.refresh_token.is_some()
    }

    pub fn account_id(&self) -> Option<&str> {
        self.account_id.as_deref()
    }

    pub fn expired(&self) -> bool {
        match self.expires_at_ms {
            Some(expires) => now_ms() >= expires,
            None => false,
        }
    }
}

impl fmt::Debug for LocalOAuthCredential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LocalOAuthCredential(expires_at_ms={:?}, refresh={}, redacted)",
            self.expires_at_ms,
            self.has_refresh_token()
        )
    }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

/// `~/.claude/.credentials.json` (AUTH-8).
pub fn default_claude_credentials_path() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".claude/.credentials.json"))
}

/// `~/.codex/auth.json` (AUTH-8).
pub fn default_codex_auth_path() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".codex/auth.json"))
}

fn not_configured(provider: &str, message: String, hint: &str) -> AuthError {
    AuthError::NotConfigured { provider: provider.into(), message, hint: hint.into() }
}

const CLAUDE_HINT: &str = "Log in again: run `claude` and use /login (Claude subscription auth)";
const CODEX_HINT: &str = "Log in again: run `codex login` (ChatGPT subscription auth)";

fn read_json_object(path: &Path, provider: &str, hint: &str) -> Result<Value, AuthError> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| not_configured(provider, format!("could not read {}: {error}", path.display()), hint))?;
    let value: Value = serde_json::from_str(&text)
        .map_err(|_| not_configured(provider, format!("{} is not valid JSON", path.display()), hint))?;
    if !value.is_object() {
        return Err(not_configured(provider, format!("{} has an unexpected shape", path.display()), hint));
    }
    Ok(value)
}

fn string_field(object: &Value, key: &str) -> Option<String> {
    object.get(key)?.as_str().filter(|s| !s.is_empty()).map(str::to_string)
}

/// Read-only loader for the Claude Code CLI credential file.
pub fn read_claude_code_credential(path: &Path) -> Result<LocalOAuthCredential, AuthError> {
    let data = read_json_object(path, "claude-code", CLAUDE_HINT)?;
    let oauth = data.get("claudeAiOauth").filter(|v| v.is_object()).ok_or_else(|| {
        not_configured("claude-code", format!("{} has no claudeAiOauth section", path.display()), CLAUDE_HINT)
    })?;
    let access_token = string_field(oauth, "accessToken").ok_or_else(|| {
        not_configured("claude-code", format!("{} has no access token", path.display()), CLAUDE_HINT)
    })?;
    Ok(LocalOAuthCredential {
        access_token,
        refresh_token: string_field(oauth, "refreshToken"),
        expires_at_ms: oauth.get("expiresAt").and_then(Value::as_i64),
        account_id: None,
    })
}

/// Read-only loader for the OpenAI Codex CLI auth file; expiry comes from
/// the access token's JWT `exp` claim minus the AUTH-3 skew.
pub fn read_codex_cli_credential(path: &Path) -> Result<LocalOAuthCredential, AuthError> {
    let data = read_json_object(path, "openai-codex", CODEX_HINT)?;
    let tokens = data.get("tokens").filter(|v| v.is_object()).ok_or_else(|| {
        not_configured("openai-codex", format!("{} has no tokens section", path.display()), CODEX_HINT)
    })?;
    let access_token = string_field(tokens, "access_token").ok_or_else(|| {
        not_configured("openai-codex", format!("{} has no access token", path.display()), CODEX_HINT)
    })?;
    let payload = jwt_payload(&access_token);
    let expires_at_ms = payload
        .as_ref()
        .and_then(|p| p.get("exp"))
        .and_then(Value::as_i64)
        .map(|exp| exp * 1000 - REFRESH_SKEW_MS);
    let account_id = string_field(tokens, "account_id").or_else(|| {
        payload
            .as_ref()
            .and_then(|p| p.get("https://api.openai.com/auth"))
            .and_then(|claim| claim.get("chatgpt_account_id"))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    });
    Ok(LocalOAuthCredential {
        access_token,
        refresh_token: string_field(tokens, "refresh_token"),
        expires_at_ms,
        account_id,
    })
}

fn jwt_payload(token: &str) -> Option<Value> {
    let mut parts = token.split('.');
    let (_header, payload, _sig) = (parts.next()?, parts.next()?, parts.next()?);
    if parts.next().is_some() {
        return None;
    }
    let decoded = base64url_decode(payload)?;
    serde_json::from_slice(&decoded).ok()
}

/// Minimal base64url (no padding) decoder — kept local to avoid a
/// dependency for one call site.
fn base64url_decode(input: &str) -> Option<Vec<u8>> {
    fn value_of(byte: u8) -> Option<u32> {
        match byte {
            b'A'..=b'Z' => Some((byte - b'A') as u32),
            b'a'..=b'z' => Some((byte - b'a') as u32 + 26),
            b'0'..=b'9' => Some((byte - b'0') as u32 + 52),
            b'-' => Some(62),
            b'_' => Some(63),
            _ => None,
        }
    }
    let bytes = input.trim_end_matches('=').as_bytes();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    for chunk in bytes.chunks(4) {
        if chunk.len() == 1 {
            return None;
        }
        let mut accumulator: u32 = 0;
        for &byte in chunk {
            accumulator = (accumulator << 6) | value_of(byte)?;
        }
        accumulator <<= 6 * (4 - chunk.len()) as u32;
        let produced = chunk.len() - 1;
        out.extend_from_slice(&accumulator.to_be_bytes()[1..1 + produced]);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SENTINEL: &str = "SECRET-SENTINEL-DO-NOT-PRINT";

    #[test]
    fn base64url_roundtrips_json() {
        let decoded = base64url_decode("eyJleHAiOjF9").unwrap(); // {"exp":1}
        assert_eq!(decoded, b"{\"exp\":1}");
    }

    #[test]
    fn static_credential_debug_is_redacted() {
        let credential = StaticCredential::new(SENTINEL);
        assert!(!format!("{credential:?}").contains(SENTINEL));
        assert_eq!(credential.token().unwrap(), SENTINEL);
    }

    #[test]
    fn fn_credential_is_invoked_per_call() {
        use std::cell::Cell;
        let calls = Cell::new(0);
        let provider = FnCredential(|| {
            calls.set(calls.get() + 1);
            Ok(format!("token-{}", calls.get()))
        });
        assert_ne!(provider.token().unwrap(), provider.token().unwrap());
    }

    #[test]
    fn underscore_alias_is_accepted() {
        let report = explain_auth("openai_chat", &ExplainOptions {
            env: Some(HashMap::new()),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(report.provider, "openai-chat");
    }

    #[test]
    fn unknown_provider_names_known_ones() {
        let error = explain_auth("nope", &ExplainOptions::default()).unwrap_err();
        assert!(error.to_string().contains("anthropic"));
    }

    #[test]
    fn gemini_env_key_order_first_wins() {
        let env = HashMap::from([
            ("GEMINI_API_KEY".to_string(), "a".to_string()),
            ("GOOGLE_API_KEY".to_string(), "b".to_string()),
        ]);
        let report = explain_auth("gemini", &ExplainOptions { env: Some(env), ..Default::default() }).unwrap();
        assert_eq!(report.selected().unwrap().kind, "env:GEMINI_API_KEY");
        assert_eq!(report.steps[2].kind, "env:GOOGLE_API_KEY");
        assert_eq!(report.steps[2].state, StepState::Shadowed);
    }
}

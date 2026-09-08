//! A stored local login as a [`CredentialProvider`] (AUTH-1 `oauth` /
//! `oauth-unless-explicit`; the reference's `_CREDENTIAL_LOADERS` and
//! `get_*_access_token`): the file is read on every request, so a token
//! the owning CLI refreshed is picked up by a long-lived client. Nothing
//! is cached here.
//!
//! Refresh (AUTH-3) runs in [`CredentialProvider::prepare`] — the async
//! step the adapter awaits before building a request, the same door the
//! cloud chains use — when the login was built with a transport
//! ([`StoredLogin::refreshing`]; the router does this). The sequence is
//! the reference's: read; if expired with a refresh token, take the
//! AUTH-4 lock on the source file, re-read (a sibling may have refreshed
//! while we waited — using its result instead of repeating the refresh
//! is what keeps a rotated refresh token from being spent twice), then
//! POST the refresh while holding the lock and write the file back
//! atomically. A login built without a transport answers the typed
//! `AuthError` with the re-login hint — loud, never a silent fall back to
//! an environment key (AUTH-1, stored-credential-owns-provider).

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use super::credential::{Credential, CredentialProvider};
use super::error::AuthError;
use super::lock::{write_private_json_atomic, FileLock, DEFAULT_LOCK_TIMEOUT};
use super::refresh::{credential_from_token_response, merged_file, refresh_request, LoginProvider};
use super::stores::{
    claude_credentials_path, codex_auth_path, pi_agent_auth_path, read_claude_code_credential,
    read_codex_cli_credential, read_xai_credential, LocalOAuthCredential, CLAUDE_CODE_LOGIN_HINT,
    OPENAI_CODEX_LOGIN_HINT, XAI_LOGIN_HINT,
};
use crate::transport::{BoxFuture, Transport};

/// The files a stored-login provider reads, in order (AUTH-8), derived
/// from `env` (`HOME`, `LM15_CREDENTIALS_PATH`, `XDG_CONFIG_HOME`) — or
/// the one explicit `credentials_path`. Empty when no `HOME` is set.
pub fn stored_login_paths(
    provider: &str,
    env: &dyn Fn(&str) -> Option<String>,
    credentials_path: Option<&Path>,
) -> Vec<PathBuf> {
    if let Some(path) = credentials_path {
        return vec![path.to_path_buf()];
    }
    let value = |key: &str| env(key).filter(|v| !v.is_empty());
    let home = value("HOME").map(PathBuf::from);
    match provider {
        "claude-code" => home
            .map(|h| claude_credentials_path(&h))
            .into_iter()
            .collect(),
        "openai-codex" => home.map(|h| codex_auth_path(&h)).into_iter().collect(),
        _ => {
            // AUTH-8: `$LM15_CREDENTIALS_PATH`, else
            // `$XDG_CONFIG_HOME/lm15/credentials.json`, else
            // `~/.config/lm15/credentials.json`; then the Pi agent store.
            let mut paths = Vec::new();
            let store = match value("LM15_CREDENTIALS_PATH") {
                Some(path) => Some(PathBuf::from(path)),
                None => match value("XDG_CONFIG_HOME") {
                    Some(config_home) => Some(PathBuf::from(config_home)),
                    None => home.as_ref().map(|h| h.join(".config")),
                }
                .map(|base| base.join("lm15").join("credentials.json")),
            };
            paths.extend(store);
            paths.extend(home.as_ref().map(|h| pi_agent_auth_path(h)));
            paths
        }
    }
}

type Reader = fn(&Path) -> Result<LocalOAuthCredential, AuthError>;

/// The reader and login hint of a stored-login provider.
fn reader_for(provider: &str) -> (Reader, &'static str) {
    match provider {
        "claude-code" => (read_claude_code_credential, CLAUDE_CODE_LOGIN_HINT),
        "openai-codex" => (read_codex_cli_credential, OPENAI_CODEX_LOGIN_HINT),
        _ => (read_xai_credential, XAI_LOGIN_HINT),
    }
}

/// What a refreshing login needs beyond the file: the transport for the
/// token endpoint and the lm15-owned lock directory (AUTH-4/AUTH-8).
#[derive(Clone)]
struct Refresher {
    transport: Arc<dyn Transport>,
    lock_dir: PathBuf,
    lock_timeout: Duration,
}

/// A stored local login, read per request; refreshed before a request
/// when built with [`StoredLogin::refreshing`].
#[derive(Clone)]
pub struct StoredLogin {
    provider: String,
    paths: Vec<PathBuf>,
    refresher: Option<Refresher>,
}

impl StoredLogin {
    /// The login of `provider` at the AUTH-8 paths derived from `env`, or
    /// at `credentials_path`.
    pub fn for_provider(
        provider: &str,
        env: &dyn Fn(&str) -> Option<String>,
        credentials_path: Option<&Path>,
    ) -> Self {
        StoredLogin {
            provider: provider.to_string(),
            paths: stored_login_paths(provider, env, credentials_path),
            refresher: None,
        }
    }

    /// The login of `provider` at exactly these files, first readable wins.
    pub fn at(provider: &str, paths: Vec<PathBuf>) -> Self {
        StoredLogin {
            provider: provider.to_string(),
            paths,
            refresher: None,
        }
    }

    /// Enable AUTH-3 refresh: an expired login with a refresh token is
    /// refreshed through `transport` in `prepare()`, under the AUTH-4
    /// lock in `lock_dir`, and written back to the file it came from.
    pub fn refreshing(mut self, transport: Arc<dyn Transport>, lock_dir: PathBuf) -> Self {
        self.refresher = Some(Refresher {
            transport,
            lock_dir,
            lock_timeout: DEFAULT_LOCK_TIMEOUT,
        });
        self
    }

    /// How long `prepare()` waits for a sibling process's refresh before
    /// [`AuthError::LockTimeout`] (default 60 s, the reference's).
    pub fn lock_timeout(mut self, timeout: Duration) -> Self {
        if let Some(refresher) = &mut self.refresher {
            refresher.lock_timeout = timeout;
        }
        self
    }

    /// Whether this login refreshes (`refreshing` was called).
    pub fn refreshes(&self) -> bool {
        self.refresher.is_some()
    }

    pub fn provider(&self) -> &str {
        &self.provider
    }

    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    /// Read the file now (the first readable path). A missing or malformed
    /// file is the reader's `NotConfigured` error with the login hint.
    pub fn read(&self) -> Result<LocalOAuthCredential, AuthError> {
        self.read_with_source().map(|(credential, _)| credential)
    }

    /// `read`, plus the path it came from (the file a refresh writes back
    /// to; AUTH-8: xAI rotates refresh tokens, so the source must get it).
    pub fn read_with_source(&self) -> Result<(LocalOAuthCredential, &Path), AuthError> {
        let (reader, hint) = reader_for(&self.provider);
        let mut last = None;
        for path in &self.paths {
            match reader(path) {
                Ok(credential) => return Ok((credential, path)),
                Err(err) => last = Some(err),
            }
        }
        Err(last.unwrap_or_else(|| {
            AuthError::not_configured(
                self.provider.clone(),
                "no HOME to derive the stored-login path from (spec/auth.md AUTH-8)",
                hint,
            )
        }))
    }

    fn expired_error(&self, credential: &LocalOAuthCredential) -> AuthError {
        let (_, hint) = reader_for(&self.provider);
        AuthError::Expired {
            provider: self.provider.clone(),
            refreshable: credential.has_refresh_token(),
            refresh_configured: self.refreshes(),
            hint: hint.to_string(),
        }
    }

    /// AUTH-3: refresh the stored login if it is expired, under the AUTH-4
    /// lock with the double-checked re-read; a no-op when it is fresh.
    /// Returns the credential as it stands afterwards.
    pub async fn refresh_if_expired(&self) -> Result<LocalOAuthCredential, AuthError> {
        let (credential, source) = self.read_with_source()?;
        if !credential.expired() {
            return Ok(credential);
        }
        let Some(refresher) = &self.refresher else {
            return Err(self.expired_error(&credential));
        };
        if !credential.has_refresh_token() {
            return Err(self.expired_error(&credential));
        }
        let provider =
            LoginProvider::from_id(&self.provider).ok_or_else(|| AuthError::UnknownProvider {
                provider: self.provider.clone(),
            })?;
        let (reader, _) = reader_for(&self.provider);

        let _lock = FileLock::acquire(&refresher.lock_dir, source, refresher.lock_timeout).await?;
        // Double-checked: a sibling may have refreshed (and rotated the
        // refresh token) while we waited for the lock.
        let current = reader(source)?;
        if !current.expired() {
            return Ok(current);
        }
        let Some(refresh_token) = current.refresh_token() else {
            return Err(self.expired_error(&current));
        };
        let refreshed = self
            .post_refresh(provider, refresher, refresh_token)
            .await?;
        // The file as it stands under the lock, with only the credential
        // fields replaced (foreign fields survive).
        let existing = std::fs::read_to_string(source)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok());
        write_private_json_atomic(source, &merged_file(provider, existing, &refreshed))?;
        Ok(refreshed)
    }

    async fn post_refresh(
        &self,
        provider: LoginProvider,
        refresher: &Refresher,
        refresh_token: &str,
    ) -> Result<LocalOAuthCredential, AuthError> {
        let failed = |message: String| AuthError::Rejected {
            provider: Some(self.provider.clone()),
            message,
            hint: Some(provider.login_hint().to_string()),
        };
        let request = refresh_request(provider, refresh_token);
        let response = refresher
            .transport
            .send(request)
            .await
            .map_err(|err| failed(format!("token refresh failed: {}", err.message())))?;
        let status = response.status;
        let body = response
            .read()
            .await
            .map_err(|err| failed(format!("token refresh failed: {}", err.message())))?;
        if !(200..300).contains(&status) {
            // AUTH-5: the body is a bearer-equivalent (or names one); only
            // the status is reported.
            return Err(failed(format!(
                "the stored login is expired and the refresh attempt failed (HTTP {status})"
            )));
        }
        let body: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&body)
            .map_err(|_| failed("token refresh response is not a JSON object".into()))?;
        credential_from_token_response(provider, &body, Some(refresh_token))
    }

    /// AUTH-1's offline probe: is a usable login (fresh, or expired with a
    /// refresh token) stored? Reads the file, never the network.
    pub fn is_usable(&self) -> bool {
        self.read().is_ok_and(|credential| credential.usable())
    }
}

impl CredentialProvider for StoredLogin {
    fn credential(&self) -> Result<Credential, AuthError> {
        let credential = self.read()?;
        if credential.expired() {
            return Err(self.expired_error(&credential));
        }
        // Unix seconds for the credential value (AUTH-2); the file records
        // milliseconds.
        let expires_at = credential.expires_at_ms().map(|ms| ms.div_euclid(1000));
        Credential::bearer_token(credential.access_token(), expires_at)
    }

    /// The AUTH-3 refresh, when enabled; `credential()` then reads the
    /// refreshed file.
    fn prepare(&self) -> Option<BoxFuture<'_, Result<(), AuthError>>> {
        self.refresher.as_ref()?;
        Some(Box::pin(async move {
            self.refresh_if_expired().await.map(|_| ())
        }))
    }
}

impl fmt::Debug for StoredLogin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // AUTH-5: paths are not secrets; contents never appear.
        f.debug_struct("StoredLogin")
            .field("provider", &self.provider)
            .field("paths", &self.paths)
            .field("refreshes", &self.refreshes())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |key: &str| map.get(key).cloned()
    }

    fn write(dir: &Path, name: &str, body: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
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
    fn paths_follow_auth_8() {
        let e = env(&[("HOME", "/h")]);
        assert_eq!(
            stored_login_paths("claude-code", &e, None),
            vec![PathBuf::from("/h/.claude/.credentials.json")]
        );
        assert_eq!(
            stored_login_paths("openai-codex", &e, None),
            vec![PathBuf::from("/h/.codex/auth.json")]
        );
        assert_eq!(
            stored_login_paths("xai", &e, None),
            vec![
                PathBuf::from("/h/.config/lm15/credentials.json"),
                PathBuf::from("/h/.pi/agent/auth.json")
            ]
        );
        let e = env(&[("HOME", "/h"), ("XDG_CONFIG_HOME", "/x")]);
        assert_eq!(
            stored_login_paths("xai", &e, None)[0],
            PathBuf::from("/x/lm15/credentials.json")
        );
        let e = env(&[("HOME", "/h"), ("LM15_CREDENTIALS_PATH", "/c.json")]);
        assert_eq!(
            stored_login_paths("xai", &e, None)[0],
            PathBuf::from("/c.json")
        );
        assert!(stored_login_paths("claude-code", &env(&[]), None).is_empty());
        assert_eq!(
            stored_login_paths("claude-code", &env(&[]), Some(Path::new("/explicit"))),
            vec![PathBuf::from("/explicit")]
        );
    }

    #[test]
    fn fresh_login_is_a_bearer_token_read_per_call() {
        let dir = tempdir();
        let expires = now_ms() + 3_600_000;
        let path = write(
            &dir,
            ".claude/.credentials.json",
            &format!(
                r#"{{"claudeAiOauth":{{"accessToken":"tok-1","refreshToken":"r","expiresAt":{expires}}}}}"#
            ),
        );
        let login = StoredLogin::at("claude-code", vec![path.clone()]);
        assert!(login.is_usable());
        let credential = login.credential().unwrap();
        assert_eq!(
            credential,
            Credential::bearer_token("tok-1", Some(expires.div_euclid(1000))).unwrap()
        );
        // The CLI rotates the token; the next call sees it.
        std::fs::write(
            &path,
            format!(
                r#"{{"claudeAiOauth":{{"accessToken":"tok-2","refreshToken":"r","expiresAt":{expires}}}}}"#
            ),
        )
        .unwrap();
        assert!(matches!(
            login.credential().unwrap(),
            Credential::BearerToken { value, .. } if value == "tok-2"
        ));
        assert!(!format!("{login:?}").contains("tok-"));
    }

    #[test]
    fn expired_login_is_the_auth_error_never_a_fallback() {
        let dir = tempdir();
        let expired = now_ms() - 1;
        let path = write(
            &dir,
            "codex.json",
            r#"{"tokens":{"access_token":"a.b.c","refresh_token":"r","account_id":"acct"},"last_refresh":"2020-01-01T00:00:00Z"}"#,
        );
        // A Codex file without an exp claim has no recorded expiry: fresh.
        let login = StoredLogin::at("openai-codex", vec![path]);
        assert!(login.credential().is_ok());

        let path = write(
            &dir,
            ".claude/.credentials.json",
            &format!(
                r#"{{"claudeAiOauth":{{"accessToken":"SECRET-SENTINEL-DO-NOT-PRINT","refreshToken":"r","expiresAt":{expired}}}}}"#
            ),
        );
        let login = StoredLogin::at("claude-code", vec![path.clone()]);
        // Usable by AUTH-1 (a refresh token is present) — so the router
        // selects it — but this login was built without a transport, so
        // nothing refreshes it: loud, typed.
        assert!(login.is_usable());
        assert!(!login.refreshes());
        let err = login.credential().unwrap_err();
        assert_eq!(err.class_name(), "AuthError");
        assert_eq!(err.code(), "auth");
        assert!(matches!(
            err,
            AuthError::Expired {
                refreshable: true,
                ..
            }
        ));
        let text = err.to_string();
        assert!(text.contains("/login"), "{text}");
        assert!(!text.contains("SENTINEL"));
        let lm15: crate::errors::Lm15Error = err.into();
        assert_eq!(lm15.class_name(), "AuthError");

        std::fs::write(
            &path,
            format!(r#"{{"claudeAiOauth":{{"accessToken":"x","expiresAt":{expired}}}}}"#),
        )
        .unwrap();
        assert!(!login.is_usable());
        assert!(matches!(
            login.credential().unwrap_err(),
            AuthError::Expired {
                refreshable: false,
                ..
            }
        ));
    }

    #[test]
    fn missing_file_is_not_configured_with_the_hint() {
        let login = StoredLogin::at("claude-code", vec![PathBuf::from("/nonexistent/x.json")]);
        assert!(!login.is_usable());
        let err = login.credential().unwrap_err();
        assert_eq!(err.class_name(), "NotConfiguredError");
        assert!(err.to_string().contains("/login"));
        let login = StoredLogin::at("claude-code", vec![]);
        assert!(login.credential().unwrap_err().to_string().contains("HOME"));
    }

    fn tempdir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "lm15-login-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}

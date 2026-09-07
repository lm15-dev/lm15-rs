//! AUTH-8 read side: the borrowed CLI files (`~/.claude/.credentials.json`,
//! `~/.codex/auth.json`), the lm15-owned store's xAI entry, and the Pi agent
//! store (`~/.pi/agent/auth.json`, same xAI entry format).
//!
//! These formats are wire-fact-like: owned by foreign tools, revalidated
//! against reality, never "cleaned". This port never writes them (the
//! AUTH-3/4 write side is not implemented; stated in the README).

use std::fmt;
use std::path::{Path, PathBuf};

use serde_json::Value;

use super::error::AuthError;
use super::time::now_ms;

/// AUTH-3 skew, in milliseconds (the borrowed files store milliseconds).
const REFRESH_SKEW_MS: i64 = 5 * 60 * 1000;

pub const CLAUDE_CODE_LOGIN_HINT: &str =
    "Log in again: run `claude` and use /login (Claude subscription auth)";
pub const OPENAI_CODEX_LOGIN_HINT: &str =
    "Log in again: run `codex login` (ChatGPT subscription auth)";
/// AUTH-9 names the uniform `login(provider)` door; this port does not ship
/// it yet (stated in the README), so the hint names the reference's flow.
pub const XAI_LOGIN_HINT: &str =
    "Log in again: run lm15's xAI login (`login(\"xai\")`, spec/auth.md AUTH-9; SuperGrok / X Premium subscription auth)";

/// The recorded expiry of a borrowed file, as read (AUTH-8: revalidated
/// against reality, never "cleaned").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Expiry {
    /// The file records no expiry.
    Unrecorded,
    /// Unix milliseconds, as the file records it.
    AtMs(i64),
    /// The file records a value the clock arithmetic cannot use (out of
    /// the `i64` millisecond range). Never fresh, never usable.
    Malformed,
}

/// A locally stored OAuth credential. Token fields are private and `Debug`
/// is redacted (AUTH-5); use the accessors.
#[derive(Clone)]
pub struct LocalOAuthCredential {
    access_token: String,
    refresh_token: Option<String>,
    expiry: Expiry,
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

    /// The recorded expiry.
    pub fn expiry(&self) -> Expiry {
        self.expiry
    }

    /// Unix milliseconds, as the file records it; `None` when unrecorded
    /// or malformed.
    pub fn expires_at_ms(&self) -> Option<i64> {
        match self.expiry {
            Expiry::AtMs(ms) => Some(ms),
            Expiry::Unrecorded | Expiry::Malformed => None,
        }
    }

    /// Milliseconds until the recorded expiry at the file's own clock
    /// (`expiresAt - now`; `<= 0` is expired). `Err(Malformed)` when the
    /// file records a value the subtraction cannot represent.
    pub fn remaining_ms(&self) -> Result<Option<i64>, Expiry> {
        match self.expiry {
            Expiry::Unrecorded => Ok(None),
            Expiry::Malformed => Err(Expiry::Malformed),
            Expiry::AtMs(expires) => expires
                .checked_sub(now_ms())
                .map(Some)
                .ok_or(Expiry::Malformed),
        }
    }

    /// The file's own clock: `now >= expiresAt`. A malformed expiry counts
    /// as expired.
    pub fn expired(&self) -> bool {
        match self.remaining_ms() {
            Ok(Some(remaining)) => remaining <= 0,
            Ok(None) => false,
            Err(_) => true,
        }
    }

    /// Fresh, or expired with a refresh token to refresh it at request
    /// time (AUTH-1 `oauth-unless-explicit`: "usable"). A malformed expiry
    /// is never usable.
    pub fn usable(&self) -> bool {
        match self.remaining_ms() {
            Ok(None) => true,
            Ok(Some(remaining)) => remaining > 0 || self.has_refresh_token(),
            Err(_) => false,
        }
    }
}

impl fmt::Debug for LocalOAuthCredential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LocalOAuthCredential(expiry={:?}, refresh={}, <redacted>)",
            self.expiry,
            self.has_refresh_token()
        )
    }
}

/// `~/.claude/.credentials.json` under `home` (AUTH-8).
pub(crate) fn claude_credentials_path(home: &Path) -> PathBuf {
    home.join(".claude/.credentials.json")
}

/// `~/.codex/auth.json` under `home` (AUTH-8).
pub(crate) fn codex_auth_path(home: &Path) -> PathBuf {
    home.join(".codex/auth.json")
}

/// `~/.pi/agent/auth.json` under `home` (AUTH-8).
pub(crate) fn pi_agent_auth_path(home: &Path) -> PathBuf {
    home.join(".pi/agent/auth.json")
}

fn read_json_object(path: &Path, provider: &str, hint: &str) -> Result<Value, AuthError> {
    let text = std::fs::read_to_string(path).map_err(|error| {
        let message = if error.kind() == std::io::ErrorKind::NotFound {
            format!("No credentials file at {}.", path.display())
        } else {
            format!(
                "Could not read credentials file at {}: {}",
                path.display(),
                error.kind()
            )
        };
        AuthError::not_configured(provider, message, hint)
    })?;
    let value: Value = serde_json::from_str(&text).map_err(|_| {
        AuthError::not_configured(
            provider,
            format!("Credentials file at {} is not valid JSON.", path.display()),
            hint,
        )
    })?;
    if !value.is_object() {
        return Err(AuthError::not_configured(
            provider,
            format!(
                "Credentials file at {} has an unexpected shape.",
                path.display()
            ),
            hint,
        ));
    }
    Ok(value)
}

fn string_field(object: &Value, key: &str) -> Option<String> {
    object
        .get(key)?
        .as_str()
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// The Claude Code CLI file: `{"claudeAiOauth": {accessToken, expiresAt (ms), refreshToken?}}`.
pub fn read_claude_code_credential(path: &Path) -> Result<LocalOAuthCredential, AuthError> {
    let data = read_json_object(path, "claude-code", CLAUDE_CODE_LOGIN_HINT)?;
    let oauth = data
        .get("claudeAiOauth")
        .filter(|v| v.is_object())
        .ok_or_else(|| {
            AuthError::not_configured(
                "claude-code",
                format!(
                    "Credentials file at {} has no claudeAiOauth section.",
                    path.display()
                ),
                CLAUDE_CODE_LOGIN_HINT,
            )
        })?;
    let access_token = string_field(oauth, "accessToken").ok_or_else(|| {
        AuthError::not_configured(
            "claude-code",
            format!(
                "Credentials file at {} has no access token.",
                path.display()
            ),
            CLAUDE_CODE_LOGIN_HINT,
        )
    })?;
    Ok(LocalOAuthCredential {
        access_token,
        refresh_token: string_field(oauth, "refreshToken"),
        expiry: expiry_ms(oauth.get("expiresAt")),
        account_id: None,
    })
}

/// The OpenAI Codex CLI file: `{"tokens": {access_token, refresh_token?, account_id?}}`;
/// expiry comes from the access token's JWT `exp` claim minus the AUTH-3 skew.
pub fn read_codex_cli_credential(path: &Path) -> Result<LocalOAuthCredential, AuthError> {
    let data = read_json_object(path, "openai-codex", OPENAI_CODEX_LOGIN_HINT)?;
    let tokens = data
        .get("tokens")
        .filter(|v| v.is_object())
        .ok_or_else(|| {
            AuthError::not_configured(
                "openai-codex",
                format!(
                    "Credentials file at {} has no tokens section.",
                    path.display()
                ),
                OPENAI_CODEX_LOGIN_HINT,
            )
        })?;
    let access_token = string_field(tokens, "access_token").ok_or_else(|| {
        AuthError::not_configured(
            "openai-codex",
            format!(
                "Credentials file at {} has no access token.",
                path.display()
            ),
            OPENAI_CODEX_LOGIN_HINT,
        )
    })?;
    let payload = jwt_payload(&access_token);
    let expiry = match payload
        .as_ref()
        .and_then(|p| p.get("exp"))
        .and_then(number_ms)
    {
        None => Expiry::Unrecorded,
        Some(exp) => exp
            .checked_mul(1000)
            .and_then(|ms| ms.checked_sub(REFRESH_SKEW_MS))
            .map_or(Expiry::Malformed, Expiry::AtMs),
    };
    let account_id = string_field(tokens, "account_id").or_else(|| {
        payload
            .as_ref()
            .and_then(|p| p.get("https://api.openai.com/auth"))
            .and_then(|claim| string_field(claim, "chatgpt_account_id"))
    });
    Ok(LocalOAuthCredential {
        access_token,
        refresh_token: string_field(tokens, "refresh_token"),
        expiry,
        account_id,
    })
}

/// The xAI entry of the lm15-owned store or the Pi agent store (AUTH-8):
/// `{"xai": {"type": "oauth", "access", "expires" (ms), "refresh"?}}`.
pub fn read_xai_credential(path: &Path) -> Result<LocalOAuthCredential, AuthError> {
    let data = read_json_object(path, "xai", XAI_LOGIN_HINT)?;
    let entry = data.get("xai").filter(|v| v.is_object()).ok_or_else(|| {
        AuthError::not_configured(
            "xai",
            format!("Credentials file at {} has no xai entry.", path.display()),
            XAI_LOGIN_HINT,
        )
    })?;
    let access_token = string_field(entry, "access").ok_or_else(|| {
        AuthError::not_configured(
            "xai",
            format!(
                "Credentials file at {} has no access token.",
                path.display()
            ),
            XAI_LOGIN_HINT,
        )
    })?;
    Ok(LocalOAuthCredential {
        access_token,
        refresh_token: string_field(entry, "refresh"),
        expiry: expiry_ms(entry.get("expires")),
        account_id: None,
    })
}

/// A number: an integer, or a float truncated as the reference does
/// (`int(expires)`). Anything else is not a number.
fn number_ms(value: &Value) -> Option<i64> {
    value.as_i64().or_else(|| value.as_f64().map(|f| f as i64))
}

/// A recorded millisecond expiry: absent or non-numeric is unrecorded (as
/// the reference reads it); a number `now` cannot be subtracted from is
/// malformed (the reference has unbounded ints; `i64` does not).
fn expiry_ms(value: Option<&Value>) -> Expiry {
    match value.and_then(number_ms) {
        None => Expiry::Unrecorded,
        Some(ms) if ms.checked_sub(now_ms()).is_some() => Expiry::AtMs(ms),
        Some(_) => Expiry::Malformed,
    }
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

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("lm15-stores-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn base64url_roundtrips_json() {
        let decoded = base64url_decode("eyJleHAiOjF9").unwrap(); // {"exp":1}
        assert_eq!(decoded, b"{\"exp\":1}");
    }

    #[test]
    fn xai_store_entry_is_read_and_redacted() {
        let dir = scratch("xai");
        let path = dir.join("credentials.json");
        std::fs::write(
            &path,
            format!(r#"{{"xai": {{"type": "oauth", "access": "{SENTINEL}", "expires": 1, "refresh": "{SENTINEL}"}}}}"#),
        )
        .unwrap();
        let credential = read_xai_credential(&path).unwrap();
        assert_eq!(credential.access_token(), SENTINEL);
        assert!(credential.expired());
        assert!(credential.usable());
        assert!(!format!("{credential:?}").contains(SENTINEL));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_and_malformed_files_are_typed_errors() {
        let dir = scratch("bad");
        let missing = read_xai_credential(&dir.join("nope.json")).unwrap_err();
        assert_eq!(missing.code(), "not_configured");
        assert_eq!(missing.provider(), Some("xai"));
        assert!(missing.to_string().contains(XAI_LOGIN_HINT));
        let path = dir.join("bad.json");
        std::fs::write(&path, "[1, 2]").unwrap();
        assert!(read_claude_code_credential(&path).is_err());
        std::fs::write(
            &path,
            format!(r#"{{"claudeAiOauth": {{"accessToken": "{SENTINEL}"}}}}"#),
        )
        .unwrap();
        let credential = read_claude_code_credential(&path).unwrap();
        assert!(!credential.expired());
        assert_eq!(credential.expires_at_ms(), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn codex_expiry_comes_from_the_jwt_with_skew() {
        let dir = scratch("codex");
        let path = dir.join("auth.json");
        // header.{"exp":1}.sig
        std::fs::write(
            &path,
            r#"{"tokens": {"access_token": "eyJhbGciOiJub25lIn0.eyJleHAiOjF9.sig"}}"#,
        )
        .unwrap();
        let credential = read_codex_cli_credential(&path).unwrap();
        assert_eq!(credential.expires_at_ms(), Some(1000 - REFRESH_SKEW_MS));
        assert_eq!(credential.expiry(), Expiry::AtMs(1000 - REFRESH_SKEW_MS));
        assert!(credential.expired());
        assert!(!credential.usable());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// F6: `checked_mul` / `checked_sub` on foreign numbers.
    #[test]
    fn out_of_range_expiry_reads_as_malformed_never_usable() {
        let dir = scratch("range");
        let path = dir.join("claude.json");
        std::fs::write(
            &path,
            r#"{"claudeAiOauth": {"accessToken": "t", "expiresAt": -9223372036854775808, "refreshToken": "r"}}"#,
        )
        .unwrap();
        let credential = read_claude_code_credential(&path).unwrap();
        assert_eq!(credential.expiry(), Expiry::Malformed);
        assert_eq!(credential.expires_at_ms(), None);
        assert!(credential.expired());
        assert!(!credential.usable());
        // A large but representable value stays a plain timestamp.
        std::fs::write(
            &path,
            r#"{"claudeAiOauth": {"accessToken": "t", "expiresAt": 9223372036854775807}}"#,
        )
        .unwrap();
        let credential = read_claude_code_credential(&path).unwrap();
        assert_eq!(credential.expiry(), Expiry::AtMs(i64::MAX));
        assert!(!credential.expired());
        // codex: exp * 1000 overflows.
        let codex = dir.join("codex.json");
        std::fs::write(
            &codex,
            r#"{"tokens": {"access_token": "h.eyJleHAiOjkyMjMzNzIwMzY4NTQ3NzU4MDd9.s"}}"#,
        )
        .unwrap();
        let credential = read_codex_cli_credential(&codex).unwrap();
        assert_eq!(credential.expiry(), Expiry::Malformed);
        assert!(!credential.usable());
        // A non-numeric expiry is unrecorded, as the reference reads it.
        std::fs::write(
            &path,
            r#"{"claudeAiOauth": {"accessToken": "t", "expiresAt": "soon"}}"#,
        )
        .unwrap();
        let credential = read_claude_code_credential(&path).unwrap();
        assert_eq!(credential.expiry(), Expiry::Unrecorded);
        assert!(credential.usable());
        let _ = std::fs::remove_dir_all(&dir);
    }
}

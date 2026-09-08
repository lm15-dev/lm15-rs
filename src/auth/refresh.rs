//! AUTH-3 refresh, per stored-login provider: the token-endpoint request
//! (copied as data from the reference's `lm15/auth.py`
//! `refresh_*_credential`), the response mapping, and the write-back into
//! the file the credential came from (AUTH-8: the borrowed CLI files keep
//! their own shape; the xAI entry goes back to whichever store it was
//! read from, because xAI rotates refresh tokens).
//!
//! Secrecy (AUTH-5): request and response bodies are bearer-equivalents
//! and never reach an error message; a refusal names the HTTP status.

use serde_json::{json, Map, Value};

use super::error::AuthError;
use super::stores::{
    extract_chatgpt_account_id, jwt_expires_at_ms, Expiry, LocalOAuthCredential,
    CLAUDE_CODE_LOGIN_HINT, OPENAI_CODEX_LOGIN_HINT, XAI_LOGIN_HINT,
};
use super::time::{format_rfc3339, now_ms, now_unix};
use crate::wire::TransportRequest;

/// AUTH-3 skew, in milliseconds, subtracted at write time (the reference
/// records `expires_in` minus the skew).
pub(crate) const REFRESH_SKEW_MS: i64 = 5 * 60 * 1000;

pub const CLAUDE_CODE_CLIENT_ID: &str = "9d1c250a-e61b-44d5-88ed-5944d1962f5e";
pub const CLAUDE_CODE_TOKEN_URL: &str = "https://platform.claude.com/v1/oauth/token";
pub const OPENAI_CODEX_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
pub const OPENAI_CODEX_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
pub const XAI_CLIENT_ID: &str = "b1a00492-073a-47ea-816f-4c329264a828";
pub const XAI_TOKEN_URL: &str = "https://auth.x.ai/oauth2/token";
pub const XAI_DEVICE_CODE_URL: &str = "https://auth.x.ai/oauth2/device/code";
pub const XAI_OAUTH_SCOPE: &str = "openid profile email offline_access grok-cli:access api:access";
/// xAI may omit `expires_in`; the reference assumes an hour.
const XAI_DEFAULT_TOKEN_LIFETIME_S: i64 = 3600;

/// The stored-login providers this module refreshes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoginProvider {
    ClaudeCode,
    OpenAICodex,
    Xai,
}

impl LoginProvider {
    pub fn from_id(provider: &str) -> Option<Self> {
        match provider {
            "claude-code" => Some(LoginProvider::ClaudeCode),
            "openai-codex" => Some(LoginProvider::OpenAICodex),
            "xai" => Some(LoginProvider::Xai),
            _ => None,
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            LoginProvider::ClaudeCode => "claude-code",
            LoginProvider::OpenAICodex => "openai-codex",
            LoginProvider::Xai => "xai",
        }
    }

    pub fn login_hint(self) -> &'static str {
        match self {
            LoginProvider::ClaudeCode => CLAUDE_CODE_LOGIN_HINT,
            LoginProvider::OpenAICodex => OPENAI_CODEX_LOGIN_HINT,
            LoginProvider::Xai => XAI_LOGIN_HINT,
        }
    }
}

/// A form body, `application/x-www-form-urlencoded`, in pair order.
pub(crate) fn form_body(pairs: &[(&str, &str)]) -> Vec<u8> {
    let encoded: Vec<String> = pairs
        .iter()
        .map(|(k, v)| format!("{}={}", form_encode(k), form_encode(v)))
        .collect();
    encoded.join("&").into_bytes()
}

/// `urllib.parse.urlencode` (quote_plus): unreserved bytes verbatim,
/// space as `+`, everything else percent-encoded.
fn form_encode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

pub(crate) fn post_form(url: &str, pairs: &[(&str, &str)]) -> TransportRequest {
    TransportRequest {
        method: "POST".into(),
        url: url.into(),
        params: Vec::new(),
        headers: vec![
            (
                "content-type".into(),
                "application/x-www-form-urlencoded".into(),
            ),
            ("accept".into(), "application/json".into()),
        ],
        body: None,
        raw: Some(form_body(pairs)),
        read_timeout: Some(std::time::Duration::from_secs(30)),
    }
}

fn post_json(url: &str, body: Value) -> TransportRequest {
    TransportRequest {
        method: "POST".into(),
        url: url.into(),
        params: Vec::new(),
        headers: vec![
            ("content-type".into(), "application/json".into()),
            ("accept".into(), "application/json".into()),
        ],
        body: Some(body),
        raw: None,
        read_timeout: Some(std::time::Duration::from_secs(30)),
    }
}

/// The refresh request for `provider` (`refresh_*_credential`).
pub fn refresh_request(provider: LoginProvider, refresh_token: &str) -> TransportRequest {
    match provider {
        LoginProvider::ClaudeCode => post_json(
            CLAUDE_CODE_TOKEN_URL,
            json!({
                "grant_type": "refresh_token",
                "client_id": CLAUDE_CODE_CLIENT_ID,
                "refresh_token": refresh_token,
            }),
        ),
        LoginProvider::OpenAICodex => post_form(
            OPENAI_CODEX_TOKEN_URL,
            &[
                ("grant_type", "refresh_token"),
                ("refresh_token", refresh_token),
                ("client_id", OPENAI_CODEX_CLIENT_ID),
            ],
        ),
        LoginProvider::Xai => post_form(
            XAI_TOKEN_URL,
            &[
                ("grant_type", "refresh_token"),
                ("client_id", XAI_CLIENT_ID),
                ("refresh_token", refresh_token),
            ],
        ),
    }
}

fn str_field<'a>(body: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    body.get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
}

fn seconds_field(body: &Map<String, Value>, key: &str) -> Option<f64> {
    match body.get(key)? {
        Value::Number(n) => n.as_f64(),
        _ => None,
    }
}

fn malformed(provider: LoginProvider, what: &str) -> AuthError {
    AuthError::Rejected {
        provider: Some(provider.id().into()),
        message: format!("token response is missing {what}"),
        hint: Some(provider.login_hint().into()),
    }
}

/// The credential a 2xx token response yields (`refresh_*_credential` /
/// `_xai_credential_from_token_response`). `previous_refresh` stands in
/// when the server does not rotate the refresh token (Codex, xAI); Claude
/// Code requires all three fields.
pub fn credential_from_token_response(
    provider: LoginProvider,
    body: &Map<String, Value>,
    previous_refresh: Option<&str>,
) -> Result<LocalOAuthCredential, AuthError> {
    let access =
        str_field(body, "access_token").ok_or_else(|| malformed(provider, "access_token"))?;
    match provider {
        LoginProvider::ClaudeCode => {
            let refresh = str_field(body, "refresh_token")
                .ok_or_else(|| malformed(provider, "refresh_token"))?;
            let expires_in = seconds_field(body, "expires_in")
                .ok_or_else(|| malformed(provider, "expires_in"))?;
            Ok(LocalOAuthCredential::new(
                access,
                Some(refresh),
                Expiry::AtMs(now_ms() + (expires_in * 1000.0) as i64 - REFRESH_SKEW_MS),
                None,
            ))
        }
        LoginProvider::OpenAICodex => {
            let refresh = str_field(body, "refresh_token")
                .or(previous_refresh)
                .ok_or_else(|| malformed(provider, "refresh_token"))?;
            Ok(LocalOAuthCredential::new(
                access,
                Some(refresh),
                jwt_expires_at_ms(access),
                extract_chatgpt_account_id(access),
            ))
        }
        LoginProvider::Xai => {
            let refresh = str_field(body, "refresh_token").or(previous_refresh);
            let lifetime_s = seconds_field(body, "expires_in")
                .filter(|s| *s > 0.0)
                .unwrap_or(XAI_DEFAULT_TOKEN_LIFETIME_S as f64);
            Ok(LocalOAuthCredential::new(
                access,
                refresh,
                Expiry::AtMs(now_ms() + (lifetime_s * 1000.0) as i64 - REFRESH_SKEW_MS),
                None,
            ))
        }
    }
}

/// The file content after writing `credential` into `current` (the file
/// as read under the lock, or an empty object): the reference's
/// `_write_*_credential_unlocked` / `_xai_credential_to_entry`. Foreign
/// fields are kept; only the credential fields change.
pub fn merged_file(
    provider: LoginProvider,
    current: Option<Value>,
    credential: &LocalOAuthCredential,
) -> Value {
    let mut data = match current {
        Some(Value::Object(map)) => map,
        _ => Map::new(),
    };
    match provider {
        LoginProvider::ClaudeCode => {
            let mut section = take_object(&mut data, "claudeAiOauth");
            section.insert("accessToken".into(), credential.access_token().into());
            if let Some(refresh) = credential.refresh_token() {
                section.insert("refreshToken".into(), refresh.into());
            }
            if let Some(ms) = credential.expires_at_ms() {
                section.insert("expiresAt".into(), ms.into());
            }
            data.insert("claudeAiOauth".into(), Value::Object(section));
        }
        LoginProvider::OpenAICodex => {
            let mut tokens = take_object(&mut data, "tokens");
            tokens.insert("access_token".into(), credential.access_token().into());
            if let Some(refresh) = credential.refresh_token() {
                tokens.insert("refresh_token".into(), refresh.into());
            }
            if let Some(account_id) = credential.account_id() {
                tokens.insert("account_id".into(), account_id.into());
            }
            // `id_token` survives untouched inside `tokens` (the reference
            // re-reads and re-writes it; keeping the object does the same).
            data.insert("tokens".into(), Value::Object(tokens));
            data.entry("auth_mode").or_insert_with(|| "chatgpt".into());
            data.insert("last_refresh".into(), format_rfc3339(now_unix()).into());
        }
        LoginProvider::Xai => {
            let mut entry = take_object(&mut data, "xai");
            entry.insert("type".into(), "oauth".into());
            entry.insert("access".into(), credential.access_token().into());
            if let Some(refresh) = credential.refresh_token() {
                entry.insert("refresh".into(), refresh.into());
            }
            if let Some(ms) = credential.expires_at_ms() {
                entry.insert("expires".into(), ms.into());
            }
            data.insert("xai".into(), Value::Object(entry));
        }
    }
    Value::Object(data)
}

fn take_object(data: &mut Map<String, Value>, key: &str) -> Map<String, Value> {
    match data.remove(key) {
        Some(Value::Object(map)) => map,
        _ => Map::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refresh_requests_copy_the_reference() {
        let r = refresh_request(LoginProvider::ClaudeCode, "rt");
        assert_eq!(r.url, CLAUDE_CODE_TOKEN_URL);
        assert_eq!(
            r.body.unwrap(),
            json!({"grant_type": "refresh_token", "client_id": CLAUDE_CODE_CLIENT_ID, "refresh_token": "rt"})
        );
        let r = refresh_request(LoginProvider::OpenAICodex, "r t");
        assert_eq!(
            String::from_utf8(r.raw.unwrap()).unwrap(),
            format!(
                "grant_type=refresh_token&refresh_token=r+t&client_id={OPENAI_CODEX_CLIENT_ID}"
            )
        );
        let r = refresh_request(LoginProvider::Xai, "a/b");
        assert_eq!(
            String::from_utf8(r.raw.unwrap()).unwrap(),
            format!("grant_type=refresh_token&client_id={XAI_CLIENT_ID}&refresh_token=a%2Fb")
        );
        assert!(r
            .headers
            .iter()
            .any(|(k, v)| k == "content-type" && v == "application/x-www-form-urlencoded"));
    }

    #[test]
    fn token_responses_map_like_the_reference() {
        let body: Map<String, Value> = serde_json::from_value(
            json!({"access_token": "a", "refresh_token": "r", "expires_in": 3600}),
        )
        .unwrap();
        let c = credential_from_token_response(LoginProvider::ClaudeCode, &body, None).unwrap();
        assert_eq!(c.access_token(), "a");
        assert_eq!(c.refresh_token(), Some("r"));
        let ms = c.expires_at_ms().unwrap();
        let expected = now_ms() + 3_600_000 - REFRESH_SKEW_MS;
        assert!((ms - expected).abs() < 2000, "{ms} vs {expected}");

        // Claude Code requires every field.
        let short: Map<String, Value> =
            serde_json::from_value(json!({"access_token": "a", "refresh_token": "r"})).unwrap();
        let err =
            credential_from_token_response(LoginProvider::ClaudeCode, &short, None).unwrap_err();
        assert_eq!(err.class_name(), "AuthError");
        assert!(err.to_string().contains("expires_in"));

        // Codex keeps the previous refresh token when none is rotated.
        let body: Map<String, Value> =
            serde_json::from_value(json!({"access_token": "x.y.z"})).unwrap();
        let c = credential_from_token_response(LoginProvider::OpenAICodex, &body, Some("prev"))
            .unwrap();
        assert_eq!(c.refresh_token(), Some("prev"));
        assert_eq!(c.expiry(), Expiry::Unrecorded);

        // xAI defaults the lifetime to an hour and may omit the refresh.
        let body: Map<String, Value> =
            serde_json::from_value(json!({"access_token": "a"})).unwrap();
        let c = credential_from_token_response(LoginProvider::Xai, &body, Some("prev")).unwrap();
        assert_eq!(c.refresh_token(), Some("prev"));
        let ms = c.expires_at_ms().unwrap();
        assert!((ms - (now_ms() + 3_600_000 - REFRESH_SKEW_MS)).abs() < 2000);
        let c = credential_from_token_response(LoginProvider::Xai, &body, None).unwrap();
        assert_eq!(c.refresh_token(), None);
    }

    #[test]
    fn write_back_keeps_foreign_fields() {
        let cred = LocalOAuthCredential::new("A", Some("R"), Expiry::AtMs(5), None);
        let out = merged_file(
            LoginProvider::ClaudeCode,
            Some(
                json!({"claudeAiOauth": {"accessToken": "old", "scopes": ["x"], "subscriptionType": "max"}, "other": 1}),
            ),
            &cred,
        );
        assert_eq!(
            out,
            json!({"claudeAiOauth": {"accessToken": "A", "scopes": ["x"], "subscriptionType": "max", "refreshToken": "R", "expiresAt": 5}, "other": 1})
        );

        let cred =
            LocalOAuthCredential::new("A", Some("R"), Expiry::Unrecorded, Some("acct".into()));
        let out = merged_file(
            LoginProvider::OpenAICodex,
            Some(
                json!({"tokens": {"access_token": "old", "id_token": "id"}, "OPENAI_API_KEY": null}),
            ),
            &cred,
        );
        let obj = out.as_object().unwrap();
        assert_eq!(
            obj["tokens"],
            json!({"access_token": "A", "id_token": "id", "refresh_token": "R", "account_id": "acct"})
        );
        assert_eq!(obj["auth_mode"], "chatgpt");
        assert!(obj["last_refresh"].as_str().unwrap().ends_with('Z'));
        assert!(obj.contains_key("OPENAI_API_KEY"));

        let cred = LocalOAuthCredential::new("A", None, Expiry::AtMs(7), None);
        let out = merged_file(LoginProvider::Xai, None, &cred);
        assert_eq!(
            out,
            json!({"xai": {"type": "oauth", "access": "A", "expires": 7}})
        );
        let out = merged_file(
            LoginProvider::Xai,
            Some(
                json!({"xai": {"type": "oauth", "access": "old", "refresh": "keep"}, "anthropic": {"type": "api"}}),
            ),
            &cred,
        );
        // A credential without a refresh token leaves the stored one.
        assert_eq!(
            out,
            json!({"xai": {"type": "oauth", "access": "A", "refresh": "keep", "expires": 7}, "anthropic": {"type": "api"}})
        );
    }
}

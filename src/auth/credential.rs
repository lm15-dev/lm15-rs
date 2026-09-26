//! AUTH-2: a credential is a closed sum, not a string.
//!
//! | Kind | Canonical JSON |
//! |---|---|
//! | `ApiKey` | `{"kind":"api_key","value"}` |
//! | `BearerToken` | `{"kind":"bearer_token","value","expires_at"?}` |
//! | `AwsCredentials` | `{"kind":"aws","access_key_id","secret_access_key","session_token"?,"expires_at"?}` |
//!
//! `expires_at` is RFC 3339 on the wire and Unix seconds here; absent means
//! non-expiring. Absent fields are omitted, never null (docs/serde-rules.md).
//! The wire form is [`crate::Canonical`] (`from_json` / `to_json`), in
//! `crate::serde`; this is the one `Credential` type of the crate, also
//! exported as `lm15::Credential`.
//!
//! Secrecy (AUTH-5): `Debug` and `Display` never show values; `to_json` is
//! the only way out and its callers are the ones sending the value on the
//! wire or writing a fixture on purpose.

use std::fmt;

use super::error::AuthError;
use super::time::{format_rfc3339, now_unix};

/// AUTH-3: a token inside this window counts as expired.
pub const EXPIRY_SKEW_SECONDS: i64 = 300;

/// spec/vocabularies.md `CredentialKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CredentialKind {
    ApiKey,
    BearerToken,
    Aws,
}

impl CredentialKind {
    pub fn as_str(self) -> &'static str {
        match self {
            CredentialKind::ApiKey => "api_key",
            CredentialKind::BearerToken => "bearer_token",
            CredentialKind::Aws => "aws",
        }
    }

    /// The schemes this kind may travel under, in the KIND's preference
    /// order (spec/auth.md AUTH-2, D1).
    pub fn accepted_schemes(self) -> &'static [AuthScheme] {
        match self {
            CredentialKind::ApiKey => &[
                AuthScheme::Bearer,
                AuthScheme::XApiKey,
                AuthScheme::ApiKey,
                AuthScheme::QueryKey,
            ],
            CredentialKind::BearerToken => &[AuthScheme::Bearer, AuthScheme::XApiKey],
            CredentialKind::Aws => &[AuthScheme::SigV4],
        }
    }
}

/// spec/vocabularies.md `AuthScheme`: how a credential travels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AuthScheme {
    /// `Authorization: Bearer …`
    Bearer,
    /// `x-api-key: …` (the Gemini dialect spells it `x-goog-api-key`)
    XApiKey,
    /// `api-key: …`
    ApiKey,
    /// `?key=…`
    QueryKey,
    /// `Authorization: AWS4-HMAC-SHA256 …` (module 3b)
    SigV4,
}

impl AuthScheme {
    pub fn as_str(self) -> &'static str {
        match self {
            AuthScheme::Bearer => "bearer",
            AuthScheme::XApiKey => "x-api-key",
            AuthScheme::ApiKey => "api-key",
            AuthScheme::QueryKey => "query-key",
            AuthScheme::SigV4 => "sigv4",
        }
    }
}

impl fmt::Display for AuthScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An AUTH-2 credential value. Fields are public so adapters can send them;
/// `Debug`/`Display` redact them (AUTH-5).
#[derive(Clone, PartialEq, Eq)]
pub enum Credential {
    ApiKey {
        value: String,
    },
    BearerToken {
        value: String,
        /// Unix seconds; `None` means non-expiring.
        expires_at: Option<i64>,
    },
    AwsCredentials {
        access_key_id: String,
        secret_access_key: String,
        session_token: Option<String>,
        /// Unix seconds; `None` means long-term.
        expires_at: Option<i64>,
    },
}

impl Credential {
    /// The `ApiKey` shorthand. Empty values are rejected.
    pub fn api_key(value: impl Into<String>) -> Result<Self, AuthError> {
        let value = value.into();
        non_empty("ApiKey.value", &value)?;
        Ok(Credential::ApiKey { value })
    }

    pub fn bearer_token(
        value: impl Into<String>,
        expires_at: Option<i64>,
    ) -> Result<Self, AuthError> {
        let value = value.into();
        non_empty("BearerToken.value", &value)?;
        Ok(Credential::BearerToken { value, expires_at })
    }

    pub fn aws(
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
        session_token: Option<String>,
        expires_at: Option<i64>,
    ) -> Result<Self, AuthError> {
        let access_key_id = access_key_id.into();
        let secret_access_key = secret_access_key.into();
        non_empty("AwsCredentials.access_key_id", &access_key_id)?;
        non_empty("AwsCredentials.secret_access_key", &secret_access_key)?;
        if let Some(token) = &session_token {
            non_empty("AwsCredentials.session_token", token)?;
        }
        Ok(Credential::AwsCredentials {
            access_key_id,
            secret_access_key,
            session_token,
            expires_at,
        })
    }

    pub fn kind(&self) -> CredentialKind {
        match self {
            Credential::ApiKey { .. } => CredentialKind::ApiKey,
            Credential::BearerToken { .. } => CredentialKind::BearerToken,
            Credential::AwsCredentials { .. } => CredentialKind::Aws,
        }
    }

    /// `expires_at` in Unix seconds, when the kind carries one.
    pub fn expires_at(&self) -> Option<i64> {
        match self {
            Credential::ApiKey { .. } => None,
            Credential::BearerToken { expires_at, .. }
            | Credential::AwsCredentials { expires_at, .. } => *expires_at,
        }
    }

    /// AUTH-3: expired, or inside the five-minute skew window, at `now`
    /// (Unix seconds). A credential without expiry never expires.
    pub fn is_expired_at(&self, now: i64) -> bool {
        match self.expires_at() {
            Some(expires) => expires - now <= EXPIRY_SKEW_SECONDS,
            None => false,
        }
    }

    pub fn is_expired(&self) -> bool {
        self.is_expired_at(now_unix())
    }
}

fn non_empty(field: &str, value: &str) -> Result<(), AuthError> {
    if value.is_empty() {
        return Err(invalid(&format!("{field} must be a non-empty string")));
    }
    Ok(())
}

fn invalid(message: &str) -> AuthError {
    AuthError::NotConfigured {
        provider: None,
        message: message.to_string(),
        hint: None,
    }
}

impl fmt::Debug for Credential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tail = |expires: Option<i64>| match expires {
            Some(unix) => format!(", expires_at={}", format_rfc3339(unix)),
            None => String::new(),
        };
        match self {
            Credential::ApiKey { .. } => f.write_str("ApiKey(<redacted>)"),
            Credential::BearerToken { expires_at, .. } => {
                write!(f, "BearerToken(<redacted>{})", tail(*expires_at))
            }
            Credential::AwsCredentials { expires_at, .. } => {
                write!(f, "AwsCredentials(<redacted>{})", tail(*expires_at))
            }
        }
    }
}

impl fmt::Display for Credential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

/// A plain string reads as an `ApiKey` (AUTH-2 shorthand). An empty string
/// is rejected here exactly as `"".credential()` rejects it.
impl TryFrom<&str> for Credential {
    type Error = AuthError;

    fn try_from(value: &str) -> Result<Self, AuthError> {
        Credential::api_key(value)
    }
}

impl TryFrom<String> for Credential {
    type Error = AuthError;

    fn try_from(value: String) -> Result<Self, AuthError> {
        Credential::api_key(value)
    }
}

/// Scheme selection (spec/auth.md AUTH-2, changes/2026-09-06-decisions.md D1):
///
/// - `ApiKey`: the policy's first header-carrying scheme in POLICY order;
/// - `BearerToken`: `bearer` if the policy lists it, else `x-api-key` if it
///   lists it — the TOKEN's order, not the policy's;
/// - `AwsCredentials`: `sigv4` only.
///
/// Anything else is `NotConfiguredError` naming the schemes the kind
/// accepts and the schemes the door offers.
pub fn select_scheme(
    policy_schemes: &[AuthScheme],
    credential: &Credential,
) -> Result<AuthScheme, AuthError> {
    let kind = credential.kind();
    let accepted = kind.accepted_schemes();
    let found = match kind {
        CredentialKind::BearerToken => accepted.iter().find(|s| policy_schemes.contains(s)),
        CredentialKind::ApiKey | CredentialKind::Aws => {
            policy_schemes.iter().find(|s| accepted.contains(s))
        }
    };
    let found = found.copied();
    if matches!(found, Some(AuthScheme::ApiKey | AuthScheme::XApiKey))
        && policy_schemes.contains(&AuthScheme::Bearer)
        && matches!(credential, Credential::ApiKey { value } if looks_like_access_token(value).is_some())
    {
        return Ok(AuthScheme::Bearer);
    }
    found.ok_or_else(|| {
        let join = |schemes: &[AuthScheme]| {
            schemes
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join("/")
        };
        AuthError::NotConfigured {
            provider: None,
            message: format!(
                "a {} credential cannot travel under {}; it accepts {}",
                kind.as_str(),
                join(policy_schemes),
                join(accepted)
            ),
            hint: None,
        }
    })
}

/// A JWS compact token, identified only on hybrid key/bearer doors (AUTH-2).
pub fn is_jwt(value: &str) -> bool {
    let segments: Vec<_> = value.split('.').collect();
    if segments.len() != 3
        || segments.iter().any(|s| {
            s.is_empty()
                || !s
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        })
    {
        return false;
    }
    super::stores::base64url_decode(segments[0])
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .is_some_and(|v| v.as_object().is_some_and(|o| o.contains_key("alg")))
}

/// The token shape a plain string has, if any (AUTH-2, amended 2026-09-19
/// and 2026-09-26): `"JWT"` (JWS compact) or `"Google access token"`
/// (`ya29.`, what every Google token endpoint issues). No key any door
/// issues has either shape; on a key-first door that also takes bearer
/// (`vertex`, the Azure doors) such a string travels as bearer.
pub fn looks_like_access_token(value: &str) -> Option<&'static str> {
    if value.starts_with("ya29.") {
        Some("Google access token")
    } else if is_jwt(value) {
        Some("JWT")
    } else {
        None
    }
}

/// Secret-free identity provenance (AUTH-1). This is never credential JSON.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CredentialSource {
    pub kind: String,
    pub label: String,
    pub named: Option<String>,
    pub expires_at: Option<i64>,
}

impl CredentialSource {
    /// The chain's fixture rung name (`kind` on doctor steps).
    pub fn rung(&self) -> &str {
        &self.kind
    }
    pub fn explicit() -> Self {
        Self {
            kind: "api_keys".into(),
            label: "an explicit api_key".into(),
            named: None,
            expires_at: None,
        }
    }
    pub fn callable() -> Self {
        Self {
            kind: "callable".into(),
            label: "an application-supplied callable (identity not inspected)".into(),
            named: None,
            expires_at: None,
        }
    }
    pub fn environment(name: &str) -> Self {
        Self {
            kind: format!("env:{name}"),
            label: format!("env ${name}"),
            named: None,
            expires_at: None,
        }
    }
    pub fn describe(&self) -> String {
        let mut text = format!("{} ({})", self.label, self.kind);
        if let Some(name) = &self.named {
            text.push_str(&format!("; named credential {name:?}"));
        }
        if let Some(expiry) = self.expires_at {
            text.push_str(&format!("; expires {}", format_rfc3339(expiry)));
        }
        text
    }
}

/// Attach a known source to a provider without examining or caching its values.
pub struct SourcedCredential<P> {
    pub provider: P,
    pub source: CredentialSource,
}
impl<P: CredentialProvider> CredentialProvider for SourcedCredential<P> {
    fn credential(&self) -> Result<Credential, AuthError> {
        self.provider.credential()
    }
    fn source(&self) -> Option<CredentialSource> {
        Some(self.source.clone())
    }
    fn prepare(&self) -> Option<crate::transport::BoxFuture<'_, Result<(), AuthError>>> {
        self.provider.prepare()
    }
}
impl<P> fmt::Debug for SourcedCredential<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SourcedCredential")
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

// ─── credential providers (AUTH-2) ───────────────────────────────────

/// Supplies a credential value per request (the single-method interface
/// of playbooks/port.md § Idioms). The adapter invokes it at request-build
/// time and never caches the result; caching belongs to the provider.
pub trait CredentialProvider {
    fn credential(&self) -> Result<Credential, AuthError>;

    /// Resolve the value and its provenance together. Stateful providers may
    /// override this to snapshot both under one lock.
    fn credential_with_source(&self) -> Result<(Credential, Option<CredentialSource>), AuthError> {
        let value = self.credential()?;
        let mut source = self.source();
        if let Some(source) = source.as_mut() {
            source.expires_at = value.expires_at();
        }
        Ok((value, source))
    }

    /// Identity provenance only; never invokes the provider or inspects its secret.
    fn source(&self) -> Option<CredentialSource> {
        Some(CredentialSource::callable())
    }

    /// An asynchronous step the adapter awaits before building a request:
    /// a cloud chain resolves or refreshes its cached credential here
    /// (AUTH-3), over the network, without blocking a runtime worker.
    /// `None` (the default) means nothing to do; `credential()` then
    /// answers from memory or a file.
    fn prepare(&self) -> Option<crate::transport::BoxFuture<'_, Result<(), AuthError>>> {
        None
    }
}

/// A value is its own provider.
impl CredentialProvider for Credential {
    fn source(&self) -> Option<CredentialSource> {
        Some(CredentialSource {
            expires_at: self.expires_at(),
            ..CredentialSource::explicit()
        })
    }
    fn credential(&self) -> Result<Credential, AuthError> {
        Ok(self.clone())
    }
}

/// A string is the `ApiKey` shorthand.
impl CredentialProvider for str {
    fn source(&self) -> Option<CredentialSource> {
        Some(CredentialSource::explicit())
    }
    fn credential(&self) -> Result<Credential, AuthError> {
        Credential::api_key(self)
    }
}

impl CredentialProvider for String {
    fn source(&self) -> Option<CredentialSource> {
        Some(CredentialSource::explicit())
    }
    fn credential(&self) -> Result<Credential, AuthError> {
        Credential::api_key(self.as_str())
    }
}

impl<T: CredentialProvider + ?Sized> CredentialProvider for &T {
    fn credential_with_source(&self) -> Result<(Credential, Option<CredentialSource>), AuthError> {
        (**self).credential_with_source()
    }
    fn source(&self) -> Option<CredentialSource> {
        (**self).source()
    }
    fn prepare(&self) -> Option<crate::transport::BoxFuture<'_, Result<(), AuthError>>> {
        (**self).prepare()
    }
    fn credential(&self) -> Result<Credential, AuthError> {
        (**self).credential()
    }
}

impl<T: CredentialProvider + ?Sized> CredentialProvider for Box<T> {
    fn credential_with_source(&self) -> Result<(Credential, Option<CredentialSource>), AuthError> {
        (**self).credential_with_source()
    }
    fn source(&self) -> Option<CredentialSource> {
        (**self).source()
    }
    fn credential(&self) -> Result<Credential, AuthError> {
        (**self).credential()
    }
    fn prepare(&self) -> Option<crate::transport::BoxFuture<'_, Result<(), AuthError>>> {
        (**self).prepare()
    }
}

impl<T: CredentialProvider + ?Sized> CredentialProvider for std::sync::Arc<T> {
    fn credential_with_source(&self) -> Result<(Credential, Option<CredentialSource>), AuthError> {
        (**self).credential_with_source()
    }
    fn source(&self) -> Option<CredentialSource> {
        (**self).source()
    }
    fn credential(&self) -> Result<Credential, AuthError> {
        (**self).credential()
    }
    fn prepare(&self) -> Option<crate::transport::BoxFuture<'_, Result<(), AuthError>>> {
        (**self).prepare()
    }
}

/// A fixed credential value. `Debug` is redacted.
#[derive(Clone)]
pub struct StaticCredential(Credential);

impl StaticCredential {
    /// Resolves `source` once and holds the value: a [`Credential`], or a
    /// string as the `ApiKey` shorthand (an empty string is rejected).
    pub fn new(source: impl CredentialProvider) -> Result<Self, AuthError> {
        source.credential().map(Self)
    }
}

impl CredentialProvider for StaticCredential {
    fn source(&self) -> Option<CredentialSource> {
        self.0.source()
    }
    fn credential(&self) -> Result<Credential, AuthError> {
        Ok(self.0.clone())
    }
}

impl fmt::Debug for StaticCredential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StaticCredential({:?})", self.0)
    }
}

/// Adapts a closure to [`CredentialProvider`]. The closure runs on every
/// call; a chain provider that caches does so inside the closure (AUTH-3).
pub struct FnCredential<F: Fn() -> Result<Credential, AuthError>>(pub F);

impl<F: Fn() -> Result<Credential, AuthError>> CredentialProvider for FnCredential<F> {
    fn credential(&self) -> Result<Credential, AuthError> {
        (self.0)()
    }
}

impl<F: Fn() -> Result<Credential, AuthError>> fmt::Debug for FnCredential<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("FnCredential(<closure>)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Canonical;
    use serde_json::{json, Value};

    const SENTINEL: &str = "SECRET-SENTINEL-DO-NOT-PRINT";

    /// The `credential` kind vectors of lm15-contract `serde/canonical.json`
    /// (hand-authored 2026-09-03; values are fake markers, not secrets).
    fn vectors() -> Vec<(&'static str, Value)> {
        vec![
            (
                "credential.api_key",
                json!({"kind": "api_key", "value": "KEY-VECTOR-NOT-A-SECRET"}),
            ),
            (
                "credential.bearer_token.minimal",
                json!({"kind": "bearer_token", "value": "TOKEN-VECTOR-NOT-A-SECRET"}),
            ),
            (
                "credential.bearer_token.expiring",
                json!({"kind": "bearer_token", "value": "TOKEN-VECTOR-NOT-A-SECRET", "expires_at": "2026-09-03T13:00:00Z"}),
            ),
            (
                "credential.aws.long_term",
                json!({"kind": "aws", "access_key_id": "AKIDEXAMPLE", "secret_access_key": "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"}),
            ),
            (
                "credential.aws.session",
                json!({"kind": "aws", "access_key_id": "AKIDEXAMPLE", "secret_access_key": "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY", "session_token": "SESSION-VECTOR-NOT-A-SECRET", "expires_at": "2026-09-03T18:00:00Z"}),
            ),
        ]
    }

    #[test]
    fn canonical_vectors_roundtrip_exactly() {
        for (id, value) in vectors() {
            let parsed = Credential::from_json(&value).unwrap_or_else(|e| panic!("{id}: {e}"));
            assert_eq!(parsed.to_json(), value, "{id}");
        }
    }

    #[test]
    fn absent_is_omitted_never_null() {
        let token = Credential::bearer_token("t", None).unwrap();
        assert!(token.to_json().get("expires_at").is_none());
        let aws = Credential::aws("id", "secret", None, None).unwrap();
        assert!(aws.to_json().get("session_token").is_none());
        assert!(aws.to_json().get("expires_at").is_none());
        // Null in is read as absent; anything else non-string is rejected.
        assert!(Credential::from_json(
            &json!({"kind": "bearer_token", "value": "t", "expires_at": null})
        )
        .is_ok());
        assert!(Credential::from_json(
            &json!({"kind": "bearer_token", "value": "t", "expires_at": 5})
        )
        .is_err());
    }

    #[test]
    fn constructors_reject_empty_and_unknown() {
        assert!(Credential::api_key("").is_err());
        assert!(Credential::bearer_token("", None).is_err());
        assert!(Credential::aws("", "s", None, None).is_err());
        assert!(Credential::aws("a", "s", Some(String::new()), None).is_err());
        assert!(Credential::from_json(&json!({"kind": "password", "value": "x"})).is_err());
        assert!(Credential::from_json(
            &json!({"kind": "bearer_token", "value": "t", "expires_at": "yesterday"})
        )
        .is_err());
    }

    #[test]
    fn debug_and_display_redact_every_kind() {
        let credentials = [
            Credential::api_key(SENTINEL).unwrap(),
            Credential::bearer_token(SENTINEL, Some(1_788_440_400)).unwrap(),
            Credential::aws("AKIDEXAMPLE", SENTINEL, Some(SENTINEL.into()), None).unwrap(),
        ];
        for credential in &credentials {
            for rendering in [format!("{credential:?}"), format!("{credential}")] {
                assert!(!rendering.contains(SENTINEL), "{rendering}");
            }
        }
        assert_eq!(
            format!("{:?}", credentials[1]),
            "BearerToken(<redacted>, expires_at=2026-09-03T13:00:00Z)"
        );
        let provider = StaticCredential::new(SENTINEL).unwrap();
        assert!(!format!("{provider:?}").contains(SENTINEL));
    }

    #[test]
    fn expiry_skew_is_five_minutes() {
        let token = Credential::bearer_token("t", Some(1_000)).unwrap();
        assert!(!token.is_expired_at(1_000 - 301));
        assert!(token.is_expired_at(1_000 - 300));
        assert!(token.is_expired_at(2_000));
        assert!(!Credential::api_key("k").unwrap().is_expired_at(i64::MAX));
    }

    #[test]
    fn string_shorthand_is_api_key() {
        assert_eq!(
            "k".credential().unwrap(),
            Credential::ApiKey { value: "k".into() }
        );
        assert_eq!(
            String::from("k").credential().unwrap().kind(),
            CredentialKind::ApiKey
        );
        // N3: every string door rejects the empty string.
        assert!("".credential().is_err());
        assert!(Credential::try_from("").is_err());
        assert!(StaticCredential::new("").is_err());
        assert_eq!(
            Credential::try_from("k").unwrap().kind(),
            CredentialKind::ApiKey
        );
    }

    #[test]
    fn fn_credential_is_invoked_per_call() {
        use std::cell::Cell;
        let calls = Cell::new(0);
        let provider = FnCredential(|| {
            calls.set(calls.get() + 1);
            Credential::api_key(format!("token-{}", calls.get()))
        });
        assert_ne!(
            provider.credential().unwrap(),
            provider.credential().unwrap()
        );
        assert_eq!(calls.get(), 2);
    }

    // D1, spec/auth.md AUTH-2 scheme selection.
    use AuthScheme::{ApiKey as ApiKeyHeader, Bearer, QueryKey, SigV4, XApiKey};

    fn api_key() -> Credential {
        Credential::api_key("k").unwrap()
    }

    fn bearer() -> Credential {
        Credential::bearer_token("t", None).unwrap()
    }

    fn aws() -> Credential {
        Credential::aws("a", "s", None, None).unwrap()
    }

    #[test]
    fn api_key_takes_the_first_header_scheme_in_policy_order() {
        assert_eq!(select_scheme(&[XApiKey], &api_key()).unwrap(), XApiKey);
        assert_eq!(
            select_scheme(&[ApiKeyHeader, Bearer], &api_key()).unwrap(),
            ApiKeyHeader
        );
        assert_eq!(
            select_scheme(&[SigV4, XApiKey], &api_key()).unwrap(),
            XApiKey
        );
        assert_eq!(select_scheme(&[QueryKey], &api_key()).unwrap(), QueryKey);
        assert!(select_scheme(&[SigV4], &api_key()).is_err());
    }

    #[test]
    fn bearer_token_prefers_bearer_then_x_api_key() {
        // azure-anthropic lists (x-api-key, bearer): the token still goes as bearer.
        assert_eq!(
            select_scheme(&[XApiKey, Bearer], &bearer()).unwrap(),
            Bearer
        );
        // bedrock-anthropic lists (sigv4, x-api-key): the short-term key goes in the key header.
        assert_eq!(
            select_scheme(&[SigV4, XApiKey], &bearer()).unwrap(),
            XApiKey
        );
        // api-key and query-key never carry a token.
        let error = select_scheme(&[ApiKeyHeader, QueryKey], &bearer()).unwrap_err();
        assert_eq!(error.code(), "not_configured");
        assert!(error.to_string().contains("bearer/x-api-key"), "{error}");
        assert!(error.to_string().contains("api-key/query-key"), "{error}");
    }

    #[test]
    fn aws_credentials_sign_only() {
        assert_eq!(select_scheme(&[SigV4, Bearer], &aws()).unwrap(), SigV4);
        assert!(select_scheme(&[Bearer, XApiKey], &aws()).is_err());
        let error = select_scheme(&[XApiKey], &aws()).unwrap_err();
        assert!(error.to_string().contains("sigv4"), "{error}");
    }
}

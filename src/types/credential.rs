//! AUTH-2 credential values (the `credential` serde kind). Debug output
//! never shows a secret (AUTH-5).

use std::fmt;

use super::json::{non_empty, normalize_rfc3339, opt_non_empty, VResult};

/// `{"kind": "api_key", "value"}`.
#[derive(Clone, PartialEq, Eq)]
pub struct ApiKey {
    pub value: String,
}

/// `{"kind": "bearer_token", "value", "expires_at"?}`; `expires_at` is
/// RFC 3339 UTC, normalized to `YYYY-MM-DDTHH:MM:SSZ`.
#[derive(Clone, PartialEq, Eq)]
pub struct BearerToken {
    pub value: String,
    pub expires_at: Option<String>,
}

/// `{"kind": "aws", "access_key_id", "secret_access_key", "session_token"?, "expires_at"?}`.
#[derive(Clone, PartialEq, Eq)]
pub struct AwsCredentials {
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: Option<String>,
    pub expires_at: Option<String>,
}

/// The credential sum type (vocabularies.md § CredentialKind).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Credential {
    ApiKey(ApiKey),
    BearerToken(BearerToken),
    AwsCredentials(AwsCredentials),
}

impl Credential {
    pub fn kind(&self) -> &'static str {
        match self {
            Credential::ApiKey(_) => "api_key",
            Credential::BearerToken(_) => "bearer_token",
            Credential::AwsCredentials(_) => "aws",
        }
    }

    /// Normalize `expires_at` to the canonical UTC form.
    pub fn normalized(self) -> VResult<Self> {
        Ok(match self {
            Credential::BearerToken(mut t) => {
                t.expires_at = t.expires_at.as_deref().map(normalize_rfc3339).transpose()?;
                Credential::BearerToken(t)
            }
            Credential::AwsCredentials(mut c) => {
                c.expires_at = c.expires_at.as_deref().map(normalize_rfc3339).transpose()?;
                Credential::AwsCredentials(c)
            }
            other => other,
        })
    }

    pub fn validate(&self) -> VResult<()> {
        match self {
            Credential::ApiKey(k) => non_empty(&k.value, "ApiKey.value"),
            Credential::BearerToken(t) => {
                non_empty(&t.value, "BearerToken.value")?;
                t.expires_at
                    .as_deref()
                    .map(normalize_rfc3339)
                    .transpose()
                    .map(|_| ())
            }
            Credential::AwsCredentials(c) => {
                non_empty(&c.access_key_id, "AwsCredentials.access_key_id")?;
                non_empty(&c.secret_access_key, "AwsCredentials.secret_access_key")?;
                opt_non_empty(c.session_token.as_ref(), "AwsCredentials.session_token")?;
                c.expires_at
                    .as_deref()
                    .map(normalize_rfc3339)
                    .transpose()
                    .map(|_| ())
            }
        }
    }
}

impl From<&str> for Credential {
    /// A plain string is the `ApiKey` shorthand.
    fn from(value: &str) -> Self {
        Credential::ApiKey(ApiKey {
            value: value.to_string(),
        })
    }
}

impl fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ApiKey(<redacted>)")
    }
}

impl fmt::Debug for BearerToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.expires_at {
            Some(exp) => write!(f, "BearerToken(<redacted>, expires_at={exp})"),
            None => f.write_str("BearerToken(<redacted>)"),
        }
    }
}

impl fmt::Debug for AwsCredentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AwsCredentials(access_key_id={}, <redacted>",
            self.access_key_id
        )?;
        if let Some(exp) = &self.expires_at {
            write!(f, ", expires_at={exp}")?;
        }
        f.write_str(")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_never_prints_secrets() {
        let token = Credential::BearerToken(BearerToken {
            value: "SECRET-VALUE".into(),
            expires_at: None,
        });
        assert!(!format!("{token:?}").contains("SECRET"));
        let aws = AwsCredentials {
            access_key_id: "AKID".into(),
            secret_access_key: "SECRET-VALUE".into(),
            session_token: Some("SECRET-TOKEN".into()),
            expires_at: None,
        };
        let shown = format!("{aws:?}");
        assert!(shown.contains("AKID") && !shown.contains("SECRET"));
    }
}

//! Auth errors (AUTH-6). The class NAME and the ErrorCode are the family
//! (playbooks/api-family.md § Errors); the mechanism is this enum.
//!
//! Messages never carry token material (AUTH-5): every field is a provider
//! id, a policy name, a scheme name or a fix hint.

use std::fmt;

use super::policy::known_providers;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthError {
    /// The provider string names nothing in the access-policy table.
    /// Class `NotConfiguredError`, code `not_configured`.
    UnknownProvider { provider: String },
    /// No usable credential source (missing/unreadable/malformed file, no
    /// key, or a credential kind the door's schemes cannot carry). The
    /// hint names the fix. Class `NotConfiguredError`, code `not_configured`.
    NotConfigured {
        provider: Option<String>,
        message: String,
        hint: Option<String>,
    },
    /// A stored login that is expired and cannot be sent (AUTH-6:
    /// expired-and-unrefreshable → the `AuthError` class, same hint
    /// discipline). `refreshable`: the file holds a refresh token.
    /// `refresh_configured`: this `StoredLogin` can refresh
    /// (`StoredLogin::refreshing`) — the refresh runs in the adapter's
    /// async `prepare` step, which a hand-built request skips.
    /// Class `AuthError`, code `auth`.
    Expired {
        provider: String,
        refreshable: bool,
        refresh_configured: bool,
        hint: String,
    },
    /// The credential-file lock could not be taken within the timeout
    /// (AUTH-4): another lm15 process is refreshing the same credential.
    /// Local and transient — deliberately NOT an auth failure (AUTH-6:
    /// nothing is wrong with the credential). The reference raises a
    /// non-lm15 builtin `TimeoutError`; this port's one error enum maps it
    /// to class `TimeoutError`, code `timeout` (retryable) with no
    /// provider — stated in the README.
    LockTimeout {
        path: String,
        lock_path: String,
        timeout_secs: u64,
    },
    /// AUTH-9: the device code expired before the user approved it — a
    /// typed error distinct from denial. Class `AuthError`, code `auth`.
    DeviceCodeExpired { provider: String },
    /// A credential source that answered and was refused, or answered
    /// something unusable (an STS 403, a token exchange 400, a malformed
    /// credential_process output): AUTH-6 "provider-rejected". Class
    /// `AuthError`, code `auth`.
    Rejected {
        provider: Option<String>,
        message: String,
        hint: Option<String>,
    },
}

impl AuthError {
    /// The ErrorCode literal (spec/vocabularies.md ErrorCode).
    pub fn code(&self) -> &'static str {
        match self {
            AuthError::Expired { .. }
            | AuthError::Rejected { .. }
            | AuthError::DeviceCodeExpired { .. } => "auth",
            AuthError::LockTimeout { .. } => "timeout",
            _ => "not_configured",
        }
    }

    /// The canonical class name (spec/vocabularies.md ErrorCode table).
    pub fn class_name(&self) -> &'static str {
        match self {
            AuthError::Expired { .. }
            | AuthError::Rejected { .. }
            | AuthError::DeviceCodeExpired { .. } => "AuthError",
            AuthError::LockTimeout { .. } => "TimeoutError",
            _ => "NotConfiguredError",
        }
    }

    /// The provider this error is about, when known.
    pub fn provider(&self) -> Option<&str> {
        match self {
            AuthError::UnknownProvider { provider }
            | AuthError::Expired { provider, .. }
            | AuthError::DeviceCodeExpired { provider } => Some(provider),
            AuthError::NotConfigured { provider, .. } | AuthError::Rejected { provider, .. } => {
                provider.as_deref()
            }
            AuthError::LockTimeout { .. } => None,
        }
    }

    pub(crate) fn not_configured(
        provider: impl Into<String>,
        message: impl Into<String>,
        hint: impl Into<String>,
    ) -> Self {
        AuthError::NotConfigured {
            provider: Some(provider.into()),
            message: message.into(),
            hint: Some(hint.into()),
        }
    }
}

impl fmt::Display for AuthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AuthError::UnknownProvider { provider } => write!(
                f,
                "unknown provider {provider:?}; known providers: {}",
                known_providers().join(", ")
            ),
            AuthError::NotConfigured {
                provider,
                message,
                hint,
            } => {
                if let Some(provider) = provider {
                    write!(f, "{provider}: ")?;
                }
                f.write_str(message)?;
                if let Some(hint) = hint {
                    write!(f, "\n\n  To fix:\n    - {hint}\n")?;
                }
                Ok(())
            }
            AuthError::Rejected {
                provider,
                message,
                hint,
            } => {
                if let Some(provider) = provider {
                    write!(f, "{provider}: ")?;
                }
                f.write_str(message)?;
                if let Some(hint) = hint {
                    write!(f, "\n\n  To fix:\n    - {hint}\n")?;
                }
                Ok(())
            }
            AuthError::Expired {
                provider,
                refreshable,
                refresh_configured,
                hint,
            } => {
                if *refreshable && *refresh_configured {
                    write!(
                        f,
                        "{provider}: the stored login is expired; it holds a refresh token, and \
                         the refresh runs in the adapter's async `prepare` step before \
                         `complete` / `stream` — a request built by hand skips it. Send \
                         through the adapter, or refresh with the provider's own tool\n\n  \
                         To fix:\n    - {hint}\n"
                    )
                } else if *refreshable {
                    write!(
                        f,
                        "{provider}: the stored login is expired; it holds a refresh token, \
                         but this StoredLogin was built without a refresh transport \
                         (StoredLogin::refreshing; the router enables it when HOME or \
                         LM15_LOCK_DIR gives it a lock directory) — build through the \
                         router, or refresh with the provider's own tool\n\n  \
                         To fix:\n    - {hint}\n"
                    )
                } else {
                    write!(
                        f,
                        "{provider}: the stored login is expired and has no refresh token\n\n  \
                         To fix:\n    - {hint}\n"
                    )
                }
            }
            AuthError::LockTimeout {
                path,
                lock_path,
                timeout_secs,
            } => write!(
                f,
                "Could not lock credential file {path} within {timeout_secs}s (lock file: \
                 {lock_path}). Another process may be refreshing the same credential; retry, \
                 or remove a stale lock only if you are certain no other process holds it."
            ),
            AuthError::DeviceCodeExpired { provider } => write!(
                f,
                "{provider}: device authorization expired before it was approved. Start the \
                 login again."
            ),
        }
    }
}

impl std::error::Error for AuthError {}

impl From<AuthError> for crate::errors::Lm15Error {
    /// AUTH-6: a missing / unreadable / unimplemented source is a
    /// `NotConfiguredError`; an expired login is an `AuthError`. The
    /// message is the redacted rendering, the provider is carried when
    /// known.
    fn from(err: AuthError) -> Self {
        let mut meta = crate::errors::ErrorMeta::new(err.to_string());
        meta.provider = err.provider().map(str::to_string);
        match err {
            AuthError::Expired { .. }
            | AuthError::Rejected { .. }
            | AuthError::DeviceCodeExpired { .. } => crate::errors::Lm15Error::AuthError(meta),
            AuthError::LockTimeout { .. } => crate::errors::Lm15Error::TimeoutError(meta),
            _ => crate::errors::Lm15Error::NotConfiguredError(meta),
        }
    }
}

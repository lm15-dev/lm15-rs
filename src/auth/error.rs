//! Auth errors (AUTH-6). The class NAME and the ErrorCode are the family
//! (playbooks/api-family.md § Errors); the mechanism is this enum.
//!
//! Messages never carry token material (AUTH-5): every field is a provider
//! id, a policy name, a scheme name or a fix hint.

use std::fmt;

use super::policy::{known_providers, CredentialPolicy};

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
    /// A cloud-chain policy (module 3b) this port does not implement.
    /// Class `NotConfiguredError`, code `not_configured`
    /// (playbooks/port.md: "a port without 3b answers `NotConfiguredError`").
    NotImplemented {
        provider: String,
        policy: CredentialPolicy,
    },
}

impl AuthError {
    /// The ErrorCode literal (spec/vocabularies.md ErrorCode).
    pub fn code(&self) -> &'static str {
        "not_configured"
    }

    /// The canonical class name (spec/vocabularies.md ErrorCode table).
    pub fn class_name(&self) -> &'static str {
        "NotConfiguredError"
    }

    /// The provider this error is about, when known.
    pub fn provider(&self) -> Option<&str> {
        match self {
            AuthError::UnknownProvider { provider }
            | AuthError::NotImplemented { provider, .. } => Some(provider),
            AuthError::NotConfigured { provider, .. } => provider.as_deref(),
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
            AuthError::NotImplemented { provider, policy } => write!(
                f,
                "{provider}: credential policy {} is a cloud chain (spec/auth.md AUTH-1); \
                 this port does not implement module 3b (playbooks/port.md) and cannot \
                 resolve or explain cloud-chain credentials — pass an explicit credential \
                 to a port with module 3b, or use a non-cloud provider",
                policy.as_str()
            ),
        }
    }
}

impl std::error::Error for AuthError {}

impl From<AuthError> for crate::errors::Lm15Error {
    /// Every auth error is a `NotConfiguredError` (AUTH-6); the message is
    /// the redacted rendering, the provider is carried when known.
    fn from(err: AuthError) -> Self {
        let mut meta = crate::errors::ErrorMeta::new(err.to_string());
        meta.provider = err.provider().map(str::to_string);
        crate::errors::Lm15Error::NotConfiguredError(meta)
    }
}

//! Canonical lm15 error hierarchy (spec/vocabularies.md "ErrorCode").
//!
//! Rust has no class inheritance; the hierarchy SHAPE is replicated as one
//! enum whose variants map bidirectionally to the canonical class names and
//! ErrorCode literals. Retryable set: rate_limit, timeout, server, transport.

use thiserror::Error;

/// Shared error metadata (every canonical error class carries these).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ErrorMeta {
    pub provider: Option<String>,
    pub provider_code: Option<String>,
    pub status: Option<u16>,
    pub request_id: Option<String>,
    /// Float-typed per the Number rule (int coerces).
    pub retry_after: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum Lm15Error {
    #[error("{message}")]
    Transport { message: String, meta: ErrorMeta },
    #[error("{message}")]
    NotConfigured { message: String, meta: ErrorMeta },
    #[error("{message}")]
    UnsupportedFeature { message: String, meta: ErrorMeta },
    #[error("{message}")]
    Auth { message: String, meta: ErrorMeta },
    #[error("{message}")]
    Billing { message: String, meta: ErrorMeta },
    #[error("{message}")]
    RateLimit { message: String, meta: ErrorMeta },
    #[error("{message}")]
    InvalidRequest { message: String, meta: ErrorMeta },
    #[error("{message}")]
    ContextLength { message: String, meta: ErrorMeta },
    #[error("{message}")]
    UnsupportedModel { message: String, meta: ErrorMeta },
    #[error("{message}")]
    Timeout { message: String, meta: ErrorMeta },
    #[error("{message}")]
    Server { message: String, meta: ErrorMeta },
    #[error("{message}")]
    Provider { message: String, meta: ErrorMeta },
}

impl Lm15Error {
    /// Canonical class name (the vet protocol's `error.type`).
    pub fn class_name(&self) -> &'static str {
        match self {
            Lm15Error::Transport { .. } => "TransportError",
            Lm15Error::NotConfigured { .. } => "NotConfiguredError",
            Lm15Error::UnsupportedFeature { .. } => "UnsupportedFeatureError",
            Lm15Error::Auth { .. } => "AuthError",
            Lm15Error::Billing { .. } => "BillingError",
            Lm15Error::RateLimit { .. } => "RateLimitError",
            Lm15Error::InvalidRequest { .. } => "InvalidRequestError",
            Lm15Error::ContextLength { .. } => "ContextLengthError",
            Lm15Error::UnsupportedModel { .. } => "UnsupportedModelError",
            Lm15Error::Timeout { .. } => "TimeoutError",
            Lm15Error::Server { .. } => "ServerError",
            Lm15Error::Provider { .. } => "ProviderError",
        }
    }

    /// Canonical ErrorCode literal.
    pub fn code(&self) -> &'static str {
        match self {
            Lm15Error::Transport { .. } => "transport",
            Lm15Error::NotConfigured { .. } => "not_configured",
            Lm15Error::UnsupportedFeature { .. } => "unsupported_feature",
            Lm15Error::Auth { .. } => "auth",
            Lm15Error::Billing { .. } => "billing",
            Lm15Error::RateLimit { .. } => "rate_limit",
            Lm15Error::InvalidRequest { .. } => "invalid_request",
            Lm15Error::ContextLength { .. } => "context_length",
            Lm15Error::UnsupportedModel { .. } => "unsupported_model",
            Lm15Error::Timeout { .. } => "timeout",
            Lm15Error::Server { .. } => "server",
            Lm15Error::Provider { .. } => "provider",
        }
    }

    /// Shared metadata, regardless of variant.
    pub fn meta(&self) -> &ErrorMeta {
        match self {
            Lm15Error::Transport { meta, .. }
            | Lm15Error::NotConfigured { meta, .. }
            | Lm15Error::UnsupportedFeature { meta, .. }
            | Lm15Error::Auth { meta, .. }
            | Lm15Error::Billing { meta, .. }
            | Lm15Error::RateLimit { meta, .. }
            | Lm15Error::InvalidRequest { meta, .. }
            | Lm15Error::ContextLength { meta, .. }
            | Lm15Error::UnsupportedModel { meta, .. }
            | Lm15Error::Timeout { meta, .. }
            | Lm15Error::Server { meta, .. }
            | Lm15Error::Provider { meta, .. } => meta,
        }
    }

    pub fn retryable(&self) -> bool {
        matches!(
            self,
            Lm15Error::RateLimit { .. }
                | Lm15Error::Timeout { .. }
                | Lm15Error::Server { .. }
                | Lm15Error::Transport { .. }
        )
    }
}

/// Error class selector used by the normalizers (mirrors the canonical
/// class hierarchy leaves that provider mapping can produce).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    Auth,
    Billing,
    RateLimit,
    InvalidRequest,
    ContextLength,
    UnsupportedModel,
    Timeout,
    Server,
    Provider,
}

impl ErrorClass {
    pub fn build(self, message: String, meta: ErrorMeta) -> Lm15Error {
        match self {
            ErrorClass::Auth => Lm15Error::Auth { message, meta },
            ErrorClass::Billing => Lm15Error::Billing { message, meta },
            ErrorClass::RateLimit => Lm15Error::RateLimit { message, meta },
            ErrorClass::InvalidRequest => Lm15Error::InvalidRequest { message, meta },
            ErrorClass::ContextLength => Lm15Error::ContextLength { message, meta },
            ErrorClass::UnsupportedModel => Lm15Error::UnsupportedModel { message, meta },
            ErrorClass::Timeout => Lm15Error::Timeout { message, meta },
            ErrorClass::Server => Lm15Error::Server { message, meta },
            ErrorClass::Provider => Lm15Error::Provider { message, meta },
        }
    }
}

/// HTTP status -> typed error fallback (spec/vocabularies.md ErrorCode table;
/// reference: lm15.errors.map_http_error).
pub fn map_http_error(status: u16, message: String, meta: ErrorMeta) -> Lm15Error {
    let class = match status {
        401 | 403 => ErrorClass::Auth,
        402 => ErrorClass::Billing,
        408 | 504 => ErrorClass::Timeout,
        429 => ErrorClass::RateLimit,
        400 | 404 | 409 | 413 | 422 => ErrorClass::InvalidRequest,
        500..=599 => ErrorClass::Server,
        _ => ErrorClass::Provider,
    };
    class.build(message, meta)
}

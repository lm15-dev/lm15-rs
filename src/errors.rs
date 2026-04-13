//! Error hierarchy for lm15.

use std::fmt;

/// Base error type for all lm15 errors.
#[derive(Debug)]
pub enum LM15Error {
    Transport(String),
    Auth(String),
    Billing(String),
    RateLimit(String),
    InvalidRequest(String),
    ContextLength(String),
    Timeout(String),
    Server(String),
    UnsupportedModel(String),
    UnsupportedFeature(String),
    Provider(String),
}

impl fmt::Display for LM15Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transport(m) => write!(f, "TransportError: {m}"),
            Self::Auth(m) => write!(f, "AuthError: {m}"),
            Self::Billing(m) => write!(f, "BillingError: {m}"),
            Self::RateLimit(m) => write!(f, "RateLimitError: {m}"),
            Self::InvalidRequest(m) => write!(f, "InvalidRequestError: {m}"),
            Self::ContextLength(m) => write!(f, "ContextLengthError: {m}"),
            Self::Timeout(m) => write!(f, "TimeoutError: {m}"),
            Self::Server(m) => write!(f, "ServerError: {m}"),
            Self::UnsupportedModel(m) => write!(f, "UnsupportedModelError: {m}"),
            Self::UnsupportedFeature(m) => write!(f, "UnsupportedFeatureError: {m}"),
            Self::Provider(m) => write!(f, "ProviderError: {m}"),
        }
    }
}

impl std::error::Error for LM15Error {}

/// Map an HTTP status code to a typed error.
pub fn map_http_error(status: u16, message: &str) -> LM15Error {
    match status {
        401 | 403 => LM15Error::Auth(message.into()),
        402 => LM15Error::Billing(message.into()),
        429 => LM15Error::RateLimit(message.into()),
        408 | 504 => LM15Error::Timeout(message.into()),
        400 | 404 | 409 | 413 | 422 => LM15Error::InvalidRequest(message.into()),
        500..=599 => LM15Error::Server(message.into()),
        _ => LM15Error::Provider(message.into()),
    }
}

/// Canonical error code string.
pub fn canonical_error_code(err: &LM15Error) -> &'static str {
    match err {
        LM15Error::ContextLength(_) => "context_length",
        LM15Error::Auth(_) => "auth",
        LM15Error::Billing(_) => "billing",
        LM15Error::RateLimit(_) => "rate_limit",
        LM15Error::InvalidRequest(_) => "invalid_request",
        LM15Error::Timeout(_) => "timeout",
        LM15Error::Server(_) => "server",
        LM15Error::Transport(_) => "provider",
        _ => "provider",
    }
}

/// Construct a typed error from a canonical code string.
pub fn error_for_code(code: &str, message: &str) -> LM15Error {
    match code {
        "auth" => LM15Error::Auth(message.into()),
        "billing" => LM15Error::Billing(message.into()),
        "rate_limit" => LM15Error::RateLimit(message.into()),
        "invalid_request" => LM15Error::InvalidRequest(message.into()),
        "context_length" => LM15Error::ContextLength(message.into()),
        "timeout" => LM15Error::Timeout(message.into()),
        "server" => LM15Error::Server(message.into()),
        _ => LM15Error::Provider(message.into()),
    }
}

/// Whether an error is transient (worth retrying).
pub fn is_transient(err: &LM15Error) -> bool {
    matches!(err, LM15Error::RateLimit(_) | LM15Error::Timeout(_) | LM15Error::Server(_) | LM15Error::Transport(_))
}

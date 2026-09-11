//! Canonical error taxonomy (spec/vocabularies.md § ErrorCode).
//!
//! The class hierarchy SHAPE is replicated by [`ErrorClass`] (a class knows
//! its parent); the mechanism is a Rust enum, [`Lm15Error`], with one
//! variant per class (playbooks/api-family.md § Errors). Messages are never
//! pinned; class and ErrorCode are.
//!
//! ```text
//! LM15Error
//! ├── TransportError
//! ├── LockTimeoutError       (the credential-file lock could not be taken; local, transient)
//! ├── StreamAssemblyError
//! ├── ConfigurationError
//! │   ├── NotConfiguredError
//! │   ├── UnknownModelError      (the router: a model string that routes nowhere)
//! │   └── AmbiguousModelError    (the router: a catalog match under more than one provider)
//! ├── CapabilityError
//! │   └── UnsupportedFeatureError
//! └── ProviderError
//!     ├── AuthError
//!     ├── BillingError
//!     ├── RateLimitError
//!     ├── InvalidRequestError
//!     │   ├── ContextLengthError
//!     │   └── UnsupportedModelError
//!     ├── TimeoutError
//!     └── ServerError
//! ```

use std::fmt;

use serde_json::Value;

use crate::registry::{lookup, DialectId};
use crate::types::{Response, ValidationError};

/// The closed ErrorCode vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    Auth,
    Billing,
    RateLimit,
    InvalidRequest,
    ContextLength,
    Timeout,
    Server,
    UnsupportedModel,
    UnsupportedFeature,
    NotConfigured,
    UnknownModel,
    AmbiguousModel,
    Transport,
    LockTimeout,
    StreamAssembly,
    Provider,
}

impl ErrorCode {
    pub const ALL: &'static [ErrorCode] = &[
        ErrorCode::Auth,
        ErrorCode::Billing,
        ErrorCode::RateLimit,
        ErrorCode::InvalidRequest,
        ErrorCode::ContextLength,
        ErrorCode::Timeout,
        ErrorCode::Server,
        ErrorCode::UnsupportedModel,
        ErrorCode::UnsupportedFeature,
        ErrorCode::NotConfigured,
        ErrorCode::UnknownModel,
        ErrorCode::AmbiguousModel,
        ErrorCode::Transport,
        ErrorCode::LockTimeout,
        ErrorCode::StreamAssembly,
        ErrorCode::Provider,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            ErrorCode::Auth => "auth",
            ErrorCode::Billing => "billing",
            ErrorCode::RateLimit => "rate_limit",
            ErrorCode::InvalidRequest => "invalid_request",
            ErrorCode::ContextLength => "context_length",
            ErrorCode::Timeout => "timeout",
            ErrorCode::Server => "server",
            ErrorCode::UnsupportedModel => "unsupported_model",
            ErrorCode::UnsupportedFeature => "unsupported_feature",
            ErrorCode::NotConfigured => "not_configured",
            ErrorCode::UnknownModel => "unknown_model",
            ErrorCode::AmbiguousModel => "ambiguous_model",
            ErrorCode::Transport => "transport",
            ErrorCode::LockTimeout => "lock_timeout",
            ErrorCode::StreamAssembly => "stream_assembly",
            ErrorCode::Provider => "provider",
        }
    }

    pub fn parse(value: &str) -> Result<Self, ValidationError> {
        ErrorCode::ALL
            .iter()
            .copied()
            .find(|c| c.as_str() == value)
            .ok_or_else(|| ValidationError::value(format!("unsupported error code: {value}")))
    }

    /// The canonical class for a code (`error_class_for_code`).
    pub fn class(self) -> ErrorClass {
        match self {
            ErrorCode::Auth => ErrorClass::AuthError,
            ErrorCode::Billing => ErrorClass::BillingError,
            ErrorCode::RateLimit => ErrorClass::RateLimitError,
            ErrorCode::InvalidRequest => ErrorClass::InvalidRequestError,
            ErrorCode::ContextLength => ErrorClass::ContextLengthError,
            ErrorCode::Timeout => ErrorClass::TimeoutError,
            ErrorCode::Server => ErrorClass::ServerError,
            ErrorCode::UnsupportedModel => ErrorClass::UnsupportedModelError,
            ErrorCode::UnsupportedFeature => ErrorClass::UnsupportedFeatureError,
            ErrorCode::NotConfigured => ErrorClass::NotConfiguredError,
            ErrorCode::UnknownModel => ErrorClass::UnknownModelError,
            ErrorCode::AmbiguousModel => ErrorClass::AmbiguousModelError,
            ErrorCode::Transport => ErrorClass::TransportError,
            ErrorCode::LockTimeout => ErrorClass::LockTimeoutError,
            ErrorCode::StreamAssembly => ErrorClass::StreamAssemblyError,
            ErrorCode::Provider => ErrorClass::ProviderError,
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The error class hierarchy; `parent()` gives the tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorClass {
    LM15Error,
    TransportError,
    LockTimeoutError,
    StreamAssemblyError,
    ConfigurationError,
    NotConfiguredError,
    UnknownModelError,
    AmbiguousModelError,
    CapabilityError,
    UnsupportedFeatureError,
    ProviderError,
    AuthError,
    BillingError,
    RateLimitError,
    InvalidRequestError,
    ContextLengthError,
    UnsupportedModelError,
    TimeoutError,
    ServerError,
}

impl ErrorClass {
    pub fn name(self) -> &'static str {
        match self {
            ErrorClass::LM15Error => "LM15Error",
            ErrorClass::TransportError => "TransportError",
            ErrorClass::LockTimeoutError => "LockTimeoutError",
            ErrorClass::StreamAssemblyError => "StreamAssemblyError",
            ErrorClass::ConfigurationError => "ConfigurationError",
            ErrorClass::NotConfiguredError => "NotConfiguredError",
            ErrorClass::UnknownModelError => "UnknownModelError",
            ErrorClass::AmbiguousModelError => "AmbiguousModelError",
            ErrorClass::CapabilityError => "CapabilityError",
            ErrorClass::UnsupportedFeatureError => "UnsupportedFeatureError",
            ErrorClass::ProviderError => "ProviderError",
            ErrorClass::AuthError => "AuthError",
            ErrorClass::BillingError => "BillingError",
            ErrorClass::RateLimitError => "RateLimitError",
            ErrorClass::InvalidRequestError => "InvalidRequestError",
            ErrorClass::ContextLengthError => "ContextLengthError",
            ErrorClass::UnsupportedModelError => "UnsupportedModelError",
            ErrorClass::TimeoutError => "TimeoutError",
            ErrorClass::ServerError => "ServerError",
        }
    }

    /// The parent class; `None` for the root.
    pub fn parent(self) -> Option<ErrorClass> {
        Some(match self {
            ErrorClass::LM15Error => return None,
            ErrorClass::TransportError
            | ErrorClass::LockTimeoutError
            | ErrorClass::StreamAssemblyError
            | ErrorClass::ConfigurationError
            | ErrorClass::CapabilityError
            | ErrorClass::ProviderError => ErrorClass::LM15Error,
            ErrorClass::NotConfiguredError
            | ErrorClass::UnknownModelError
            | ErrorClass::AmbiguousModelError => ErrorClass::ConfigurationError,
            ErrorClass::UnsupportedFeatureError => ErrorClass::CapabilityError,
            ErrorClass::AuthError
            | ErrorClass::BillingError
            | ErrorClass::RateLimitError
            | ErrorClass::InvalidRequestError
            | ErrorClass::TimeoutError
            | ErrorClass::ServerError => ErrorClass::ProviderError,
            ErrorClass::ContextLengthError | ErrorClass::UnsupportedModelError => {
                ErrorClass::InvalidRequestError
            }
        })
    }

    /// `issubclass`: true when `self` is `ancestor` or descends from it.
    pub fn is_a(self, ancestor: ErrorClass) -> bool {
        let mut current = Some(self);
        while let Some(class) = current {
            if class == ancestor {
                return true;
            }
            current = class.parent();
        }
        false
    }

    /// Canonical code, most-specific class first; the root falls back to
    /// `provider`.
    pub fn code(self) -> ErrorCode {
        match self {
            ErrorClass::ContextLengthError => ErrorCode::ContextLength,
            ErrorClass::UnsupportedModelError => ErrorCode::UnsupportedModel,
            ErrorClass::AuthError => ErrorCode::Auth,
            ErrorClass::BillingError => ErrorCode::Billing,
            ErrorClass::RateLimitError => ErrorCode::RateLimit,
            ErrorClass::InvalidRequestError => ErrorCode::InvalidRequest,
            ErrorClass::TimeoutError => ErrorCode::Timeout,
            ErrorClass::ServerError => ErrorCode::Server,
            ErrorClass::UnsupportedFeatureError | ErrorClass::CapabilityError => {
                ErrorCode::UnsupportedFeature
            }
            ErrorClass::NotConfiguredError | ErrorClass::ConfigurationError => {
                ErrorCode::NotConfigured
            }
            ErrorClass::UnknownModelError => ErrorCode::UnknownModel,
            ErrorClass::AmbiguousModelError => ErrorCode::AmbiguousModel,
            ErrorClass::TransportError => ErrorCode::Transport,
            ErrorClass::LockTimeoutError => ErrorCode::LockTimeout,
            ErrorClass::StreamAssemblyError => ErrorCode::StreamAssembly,
            ErrorClass::ProviderError | ErrorClass::LM15Error => ErrorCode::Provider,
        }
    }
}

/// Metadata every error class carries (vocabularies.md § ErrorCode).
/// `retry_after` is float-typed under the Number rule.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ErrorMeta {
    pub message: String,
    pub provider: Option<String>,
    pub provider_code: Option<String>,
    pub status: Option<u16>,
    pub request_id: Option<String>,
    pub retry_after: Option<f64>,
}

impl ErrorMeta {
    pub fn new(message: impl Into<String>) -> Self {
        ErrorMeta {
            message: message.into(),
            ..Default::default()
        }
    }
}

/// What a stream assembly refusal salvaged (MAP-9).
#[derive(Debug, Clone, PartialEq)]
pub struct StreamAssembly {
    pub meta: ErrorMeta,
    pub partial: Option<Box<Response>>,
    pub part_index: Option<u64>,
}

/// The router found no provider for a model string (`unknown_model`):
/// no routable prefix, no catalog match, no rule. Local and pre-network —
/// a provider's own "no such model" reply is `UnsupportedModelError`.
#[derive(Debug, Clone, PartialEq)]
pub struct UnknownModel {
    pub meta: ErrorMeta,
    /// The string as requested.
    pub model: String,
}

/// The catalog matched a model string under more than one provider, or
/// under more than one entry of one provider (`ambiguous_model`). The fix
/// is an explicit `provider:` prefix.
#[derive(Debug, Clone, PartialEq)]
pub struct AmbiguousModel {
    pub meta: ErrorMeta,
    /// The string as requested.
    pub model: String,
    /// Every candidate provider, catalog order, deduplicated.
    pub providers: Vec<String>,
}

/// The credential-file lock could not be taken within the timeout
/// (`lock_timeout`, spec/auth.md AUTH-4/6): another lm15 process is
/// refreshing the same credential. Local and transient; retryable.
#[derive(Debug, Clone, PartialEq)]
pub struct LockTimeout {
    pub meta: ErrorMeta,
    /// The guarded credential file.
    pub path: String,
    /// The lock file in the lm15-owned lock directory.
    pub lock_path: String,
}

/// The lm15 error: one variant per class, same names as the family.
// Every class name ends in `Error` by contract (api-family: "variants,
// same names"), so the variant-name lint does not apply.
#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, PartialEq)]
pub enum Lm15Error {
    TransportError(ErrorMeta),
    LockTimeoutError(LockTimeout),
    StreamAssemblyError(StreamAssembly),
    ConfigurationError(ErrorMeta),
    NotConfiguredError(ErrorMeta),
    UnknownModelError(UnknownModel),
    AmbiguousModelError(AmbiguousModel),
    CapabilityError(ErrorMeta),
    UnsupportedFeatureError(ErrorMeta),
    ProviderError(ErrorMeta),
    AuthError(ErrorMeta),
    BillingError(ErrorMeta),
    RateLimitError(ErrorMeta),
    InvalidRequestError(ErrorMeta),
    ContextLengthError(ErrorMeta),
    UnsupportedModelError(ErrorMeta),
    TimeoutError(ErrorMeta),
    ServerError(ErrorMeta),
}

impl Lm15Error {
    /// Build the error of `class` with `meta`; base `LM15Error` is not
    /// instantiable and maps to `ProviderError` (the code fallback).
    pub fn of_class(class: ErrorClass, meta: ErrorMeta) -> Lm15Error {
        match class {
            ErrorClass::TransportError => Lm15Error::TransportError(meta),
            ErrorClass::LockTimeoutError => Lm15Error::LockTimeoutError(LockTimeout {
                meta,
                path: String::new(),
                lock_path: String::new(),
            }),
            ErrorClass::StreamAssemblyError => Lm15Error::StreamAssemblyError(StreamAssembly {
                meta,
                partial: None,
                part_index: None,
            }),
            ErrorClass::ConfigurationError => Lm15Error::ConfigurationError(meta),
            ErrorClass::NotConfiguredError => Lm15Error::NotConfiguredError(meta),
            ErrorClass::UnknownModelError => Lm15Error::UnknownModelError(UnknownModel {
                meta,
                model: String::new(),
            }),
            ErrorClass::AmbiguousModelError => Lm15Error::AmbiguousModelError(AmbiguousModel {
                meta,
                model: String::new(),
                providers: Vec::new(),
            }),
            ErrorClass::CapabilityError => Lm15Error::CapabilityError(meta),
            ErrorClass::UnsupportedFeatureError => Lm15Error::UnsupportedFeatureError(meta),
            ErrorClass::ProviderError | ErrorClass::LM15Error => Lm15Error::ProviderError(meta),
            ErrorClass::AuthError => Lm15Error::AuthError(meta),
            ErrorClass::BillingError => Lm15Error::BillingError(meta),
            ErrorClass::RateLimitError => Lm15Error::RateLimitError(meta),
            ErrorClass::InvalidRequestError => Lm15Error::InvalidRequestError(meta),
            ErrorClass::ContextLengthError => Lm15Error::ContextLengthError(meta),
            ErrorClass::UnsupportedModelError => Lm15Error::UnsupportedModelError(meta),
            ErrorClass::TimeoutError => Lm15Error::TimeoutError(meta),
            ErrorClass::ServerError => Lm15Error::ServerError(meta),
        }
    }

    pub fn class(&self) -> ErrorClass {
        match self {
            Lm15Error::TransportError(_) => ErrorClass::TransportError,
            Lm15Error::LockTimeoutError(_) => ErrorClass::LockTimeoutError,
            Lm15Error::StreamAssemblyError(_) => ErrorClass::StreamAssemblyError,
            Lm15Error::ConfigurationError(_) => ErrorClass::ConfigurationError,
            Lm15Error::NotConfiguredError(_) => ErrorClass::NotConfiguredError,
            Lm15Error::UnknownModelError(_) => ErrorClass::UnknownModelError,
            Lm15Error::AmbiguousModelError(_) => ErrorClass::AmbiguousModelError,
            Lm15Error::CapabilityError(_) => ErrorClass::CapabilityError,
            Lm15Error::UnsupportedFeatureError(_) => ErrorClass::UnsupportedFeatureError,
            Lm15Error::ProviderError(_) => ErrorClass::ProviderError,
            Lm15Error::AuthError(_) => ErrorClass::AuthError,
            Lm15Error::BillingError(_) => ErrorClass::BillingError,
            Lm15Error::RateLimitError(_) => ErrorClass::RateLimitError,
            Lm15Error::InvalidRequestError(_) => ErrorClass::InvalidRequestError,
            Lm15Error::ContextLengthError(_) => ErrorClass::ContextLengthError,
            Lm15Error::UnsupportedModelError(_) => ErrorClass::UnsupportedModelError,
            Lm15Error::TimeoutError(_) => ErrorClass::TimeoutError,
            Lm15Error::ServerError(_) => ErrorClass::ServerError,
        }
    }

    /// The canonical class name (`error.type` on the vet protocol).
    pub fn class_name(&self) -> &'static str {
        self.class().name()
    }

    /// The ErrorCode literal.
    pub fn code(&self) -> ErrorCode {
        self.class().code()
    }

    /// `isinstance` against the hierarchy: `err.is_a(ErrorClass::ProviderError)`.
    pub fn is_a(&self, ancestor: ErrorClass) -> bool {
        self.class().is_a(ancestor)
    }

    /// RETRYABLE_ERRORS: RateLimitError, TimeoutError, ServerError,
    /// TransportError, LockTimeoutError.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Lm15Error::RateLimitError(_)
                | Lm15Error::TimeoutError(_)
                | Lm15Error::ServerError(_)
                | Lm15Error::TransportError(_)
                | Lm15Error::LockTimeoutError(_)
        )
    }

    pub fn meta(&self) -> &ErrorMeta {
        match self {
            Lm15Error::StreamAssemblyError(s) => &s.meta,
            Lm15Error::LockTimeoutError(l) => &l.meta,
            Lm15Error::UnknownModelError(u) => &u.meta,
            Lm15Error::AmbiguousModelError(a) => &a.meta,
            Lm15Error::TransportError(m)
            | Lm15Error::ConfigurationError(m)
            | Lm15Error::NotConfiguredError(m)
            | Lm15Error::CapabilityError(m)
            | Lm15Error::UnsupportedFeatureError(m)
            | Lm15Error::ProviderError(m)
            | Lm15Error::AuthError(m)
            | Lm15Error::BillingError(m)
            | Lm15Error::RateLimitError(m)
            | Lm15Error::InvalidRequestError(m)
            | Lm15Error::ContextLengthError(m)
            | Lm15Error::UnsupportedModelError(m)
            | Lm15Error::TimeoutError(m)
            | Lm15Error::ServerError(m) => m,
        }
    }

    pub fn meta_mut(&mut self) -> &mut ErrorMeta {
        match self {
            Lm15Error::StreamAssemblyError(s) => &mut s.meta,
            Lm15Error::LockTimeoutError(l) => &mut l.meta,
            Lm15Error::UnknownModelError(u) => &mut u.meta,
            Lm15Error::AmbiguousModelError(a) => &mut a.meta,
            Lm15Error::TransportError(m)
            | Lm15Error::ConfigurationError(m)
            | Lm15Error::NotConfiguredError(m)
            | Lm15Error::CapabilityError(m)
            | Lm15Error::UnsupportedFeatureError(m)
            | Lm15Error::ProviderError(m)
            | Lm15Error::AuthError(m)
            | Lm15Error::BillingError(m)
            | Lm15Error::RateLimitError(m)
            | Lm15Error::InvalidRequestError(m)
            | Lm15Error::ContextLengthError(m)
            | Lm15Error::UnsupportedModelError(m)
            | Lm15Error::TimeoutError(m)
            | Lm15Error::ServerError(m) => m,
        }
    }

    pub fn message(&self) -> &str {
        &self.meta().message
    }

    pub fn provider(&self) -> Option<&str> {
        self.meta().provider.as_deref()
    }

    pub fn provider_code(&self) -> Option<&str> {
        self.meta().provider_code.as_deref()
    }

    pub fn status(&self) -> Option<u16> {
        self.meta().status
    }

    pub fn request_id(&self) -> Option<&str> {
        self.meta().request_id.as_deref()
    }

    pub fn retry_after(&self) -> Option<f64> {
        self.meta().retry_after
    }

    /// `StreamAssemblyError.partial`: the Response assembled without the
    /// offending call.
    pub fn partial(&self) -> Option<&Response> {
        match self {
            Lm15Error::StreamAssemblyError(s) => s.partial.as_deref(),
            _ => None,
        }
    }

    /// `UnknownModelError.model` / `AmbiguousModelError.model`: the model
    /// string as requested.
    pub fn model(&self) -> Option<&str> {
        match self {
            Lm15Error::UnknownModelError(u) => Some(&u.model),
            Lm15Error::AmbiguousModelError(a) => Some(&a.model),
            _ => None,
        }
    }

    /// `LockTimeoutError.path` / `.lock_path`: the guarded file and its lock.
    pub fn lock_paths(&self) -> Option<(&str, &str)> {
        match self {
            Lm15Error::LockTimeoutError(l) => Some((&l.path, &l.lock_path)),
            _ => None,
        }
    }

    /// `AmbiguousModelError.providers`: every candidate provider.
    pub fn candidate_providers(&self) -> Option<&[String]> {
        match self {
            Lm15Error::AmbiguousModelError(a) => Some(&a.providers),
            _ => None,
        }
    }

    /// A `stream_assembly` failure: a stream that cannot become a Response
    /// without inventing a fact (MAP-9 / MAP-3); `partial` is what did
    /// assemble.
    pub fn stream_assembly(
        message: impl Into<String>,
        partial: Option<Response>,
        part_index: Option<u64>,
    ) -> Lm15Error {
        Lm15Error::StreamAssemblyError(StreamAssembly {
            meta: ErrorMeta::new(message),
            partial: partial.map(Box::new),
            part_index,
        })
    }

    pub fn unknown_model(message: impl Into<String>, model: impl Into<String>) -> Lm15Error {
        Lm15Error::UnknownModelError(UnknownModel {
            meta: ErrorMeta::new(message),
            model: model.into(),
        })
    }

    pub fn ambiguous_model(
        message: impl Into<String>,
        model: impl Into<String>,
        providers: Vec<String>,
    ) -> Lm15Error {
        Lm15Error::AmbiguousModelError(AmbiguousModel {
            meta: ErrorMeta::new(message),
            model: model.into(),
            providers,
        })
    }

    pub fn unsupported_feature(message: impl Into<String>) -> Lm15Error {
        Lm15Error::UnsupportedFeatureError(ErrorMeta::new(message))
    }

    pub fn not_configured(message: impl Into<String>) -> Lm15Error {
        Lm15Error::NotConfiguredError(ErrorMeta::new(message))
    }
}

impl fmt::Display for Lm15Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let meta = self.meta();
        let base = if meta.message.is_empty() {
            self.code().as_str()
        } else {
            meta.message.as_str()
        };
        write!(f, "{}: {base}", self.class_name())?;
        if self.is_a(ErrorClass::ProviderError) {
            let mut context: Vec<String> = Vec::new();
            if let Some(provider) = &meta.provider {
                context.push(provider.clone());
            }
            if let Some(status) = meta.status {
                context.push(format!("HTTP {status}"));
            }
            if let Some(request_id) = &meta.request_id {
                context.push(format!("request {request_id}"));
            }
            if !context.is_empty() {
                write!(f, " ({})", context.join(", "))?;
            }
        }
        Ok(())
    }
}

impl std::error::Error for Lm15Error {}

// ─── HTTP status → class ─────────────────────────────────────────────

/// HTTP mapping: 401/403 auth, 402 billing, 408/504 timeout, 429 rate
/// limit, 400/404/409/413/422 invalid request, 5xx server, else provider.
pub fn map_http_error(status: u16, meta: ErrorMeta) -> Lm15Error {
    let meta = ErrorMeta {
        status: Some(status),
        ..meta
    };
    let class = match status {
        401 | 403 => ErrorClass::AuthError,
        402 => ErrorClass::BillingError,
        408 | 504 => ErrorClass::TimeoutError,
        429 => ErrorClass::RateLimitError,
        400 | 404 | 409 | 413 | 422 => ErrorClass::InvalidRequestError,
        500..=599 => ErrorClass::ServerError,
        _ => ErrorClass::ProviderError,
    };
    Lm15Error::of_class(class, meta)
}

// ─── Provider error normalization (tables copied as data) ────────────

/// Normalize a provider's HTTP error into an [`Lm15Error`]. The provider
/// string selects the dialect through the registry; the per-dialect
/// tables below are the reference's, copied as data (port.md rule 2).
pub fn normalize_error(
    provider: &str,
    status: u16,
    body_text: &str,
) -> Result<Lm15Error, ValidationError> {
    let definition = lookup(provider)
        .ok_or_else(|| ValidationError::value(format!("unknown provider: {provider}")))?;
    let ctx = Context {
        provider: definition.id,
    };
    Ok(match definition.dialect {
        DialectId::OpenaiResponses => {
            normalize_openai_shape(&ctx, status, body_text, definition.id == "openai-codex")
        }
        DialectId::OpenaiChat => {
            if definition.id == "xai" {
                normalize_openai_shape(&ctx, status, &xai_refold(body_text), false)
            } else {
                normalize_openai_shape(&ctx, status, body_text, false)
            }
        }
        DialectId::Anthropic => normalize_anthropic(&ctx, status, body_text),
        DialectId::Gemini => normalize_gemini(&ctx, status, body_text),
    })
}

struct Context {
    provider: &'static str,
}

impl Context {
    fn meta(
        &self,
        message: String,
        provider_code: Option<String>,
        request_id: Option<String>,
    ) -> ErrorMeta {
        ErrorMeta {
            message,
            provider: Some(self.provider.to_string()),
            provider_code: provider_code.filter(|c| !c.is_empty()),
            status: None,
            request_id: request_id.filter(|r| !r.is_empty()),
            retry_after: None,
        }
    }

    fn error(
        &self,
        class: ErrorClass,
        status: u16,
        message: String,
        provider_code: Option<String>,
        request_id: Option<String>,
    ) -> Lm15Error {
        let meta = ErrorMeta {
            status: Some(status),
            ..self.meta(message, provider_code, request_id)
        };
        Lm15Error::of_class(class, meta)
    }

    fn http(
        &self,
        status: u16,
        message: String,
        provider_code: Option<String>,
        request_id: Option<String>,
    ) -> Lm15Error {
        map_http_error(status, self.meta(message, provider_code, request_id))
    }
}

/// Python `str(value) or ""` for a JSON scalar used as a code/type.
fn scalar_text(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Bool(true)) => "True".to_string(),
        Some(Value::Bool(false)) => String::new(),
        Some(Value::Number(n)) => {
            if n.as_f64() == Some(0.0) {
                String::new()
            } else {
                n.to_string()
            }
        }
        Some(other) => other.to_string(),
    }
}

fn message_text(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
    }
}

fn fallback_message(status: u16, body: &str) -> String {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return format!("HTTP {status}");
    }
    trimmed.chars().take(500).collect()
}

const MODEL_ERROR_MARKERS: &[&str] = &[
    "not found",
    "does not exist",
    "not exist",
    "not supported",
    "unsupported",
    "not available",
    "unknown",
];

pub(crate) fn is_model_error(text: &str) -> bool {
    let lowered = text.to_lowercase();
    lowered.contains("model") && MODEL_ERROR_MARKERS.iter().any(|m| lowered.contains(m))
}

// OpenAI envelope family: `{"error": {"message", "type", "code"}}`. Shared
// by the Responses and Chat Completions dialects and every compat server.

const OPENAI_MODEL_ERROR_CODES: &[&str] = &[
    "model_not_found",
    "model_not_available",
    "unsupported_model",
    "DeploymentNotFound",
];
const OPENAI_BILLING_CODES: &[&str] = &["insufficient_quota", "1113"];
const OPENAI_BILLING_TYPES: &[&str] = &["insufficient_quota", "exceeded_current_quota_error"];

fn normalize_openai_shape(ctx: &Context, status: u16, body: &str, codex: bool) -> Lm15Error {
    if codex {
        if let Some(err) = normalize_detail_error(ctx, status, body) {
            return err;
        }
    }
    let Ok(data) = serde_json::from_str::<Value>(body) else {
        return ctx.http(status, fallback_message(status, body), None, None);
    };
    let err = data.get("error");
    let (mut msg, code, err_type) = match err {
        Some(Value::Object(e)) => (
            message_text(e.get("message")),
            scalar_text(e.get("code")),
            scalar_text(e.get("type")),
        ),
        Some(other) if data.is_object() => {
            (message_text(Some(other)), String::new(), String::new())
        }
        _ => (String::new(), String::new(), String::new()),
    };
    let provider_code = if !code.is_empty() {
        Some(code.clone())
    } else if !err_type.is_empty() {
        Some(err_type.clone())
    } else {
        None
    };
    if code == "context_length_exceeded" {
        return ctx.error(
            ErrorClass::ContextLengthError,
            status,
            msg,
            provider_code,
            None,
        );
    }
    if OPENAI_MODEL_ERROR_CODES.contains(&code.as_str())
        || (status == 404 && is_model_error(&format!("{msg} {code} {err_type}")))
    {
        return ctx.error(
            ErrorClass::UnsupportedModelError,
            status,
            msg,
            provider_code,
            None,
        );
    }
    // Billing before rate-limit: both can ride HTTP 429, only one is retryable.
    if OPENAI_BILLING_CODES.contains(&code.as_str())
        || OPENAI_BILLING_TYPES.contains(&err_type.as_str())
    {
        return ctx.error(ErrorClass::BillingError, status, msg, provider_code, None);
    }
    if code == "invalid_api_key" || err_type == "authentication_error" {
        return ctx.error(ErrorClass::AuthError, status, msg, provider_code, None);
    }
    if code == "rate_limit_exceeded" || err_type == "rate_limit_error" {
        return ctx.error(ErrorClass::RateLimitError, status, msg, provider_code, None);
    }
    if !code.is_empty() && !msg.contains(&code) {
        msg = format!("{msg} ({code})");
    }
    ctx.http(status, msg, provider_code, None)
}

/// The ChatGPT Codex backend answers `{"detail": "..."}` outside the
/// OpenAI envelope.
fn normalize_detail_error(ctx: &Context, status: u16, body: &str) -> Option<Lm15Error> {
    let data: Value = serde_json::from_str(body).ok()?;
    let detail = data.get("detail")?.as_str()?.trim();
    if detail.is_empty() {
        return None;
    }
    if is_model_error(detail) {
        return Some(ctx.error(
            ErrorClass::UnsupportedModelError,
            status,
            detail.to_string(),
            None,
            None,
        ));
    }
    Some(ctx.http(status, detail.to_string(), None, None))
}

/// xAI's own envelope is `{"code": str, "error": str}`; refold it into the
/// OpenAI shape so the wire code survives as `provider_code`.
fn xai_refold(body: &str) -> String {
    if let Ok(Value::Object(data)) = serde_json::from_str::<Value>(body) {
        if let Some(Value::String(message)) = data.get("error") {
            let mut inner = serde_json::Map::new();
            inner.insert("message".to_string(), Value::String(message.clone()));
            inner.insert(
                "code".to_string(),
                data.get("code").cloned().unwrap_or(Value::Null),
            );
            let mut outer = serde_json::Map::new();
            outer.insert("error".to_string(), Value::Object(inner));
            return Value::Object(outer).to_string();
        }
    }
    body.to_string()
}

// Anthropic Messages envelope: `{"error": {"type", "message"}, "request_id"}`.

pub(crate) const ANTHROPIC_ERROR_TYPES: &[(&str, ErrorClass)] = &[
    ("authentication_error", ErrorClass::AuthError),
    ("permission_error", ErrorClass::AuthError),
    ("billing_error", ErrorClass::BillingError),
    ("rate_limit_error", ErrorClass::RateLimitError),
    ("request_too_large", ErrorClass::InvalidRequestError),
    ("not_found_error", ErrorClass::InvalidRequestError),
    ("resource_not_found_error", ErrorClass::InvalidRequestError),
    ("DeploymentNotFound", ErrorClass::UnsupportedModelError),
    ("invalid_authentication_error", ErrorClass::AuthError),
    ("invalid_request_error", ErrorClass::InvalidRequestError),
    ("api_error", ErrorClass::ServerError),
    ("overloaded_error", ErrorClass::ServerError),
    ("timeout_error", ErrorClass::TimeoutError),
];

pub(crate) fn anthropic_is_context_length(msg: &str) -> bool {
    let l = msg.to_lowercase();
    l.contains("prompt is too long")
        || l.contains("too many tokens")
        || l.contains("context window")
        || l.contains("context length")
        || (l.contains("token") && (l.contains("limit") || l.contains("exceed")))
}

fn normalize_anthropic(ctx: &Context, status: u16, body: &str) -> Lm15Error {
    let data = match serde_json::from_str::<Value>(body) {
        Ok(Value::Object(data)) => data,
        // Valid JSON that is not an object carries no envelope.
        Ok(_) => return ctx.http(status, String::new(), None, None),
        Err(_) => return ctx.http(status, fallback_message(status, body), None, None),
    };
    // Anthropic nests under `error`; Azure Foundry's gateway uses top-level
    // {code, message} before a deployment is reached. Both on one wire.
    let (mut msg, err_type) = match data.get("error") {
        Some(Value::Object(e)) => (message_text(e.get("message")), {
            let t = scalar_text(e.get("type"));
            if t.is_empty() {
                scalar_text(e.get("code"))
            } else {
                t
            }
        }),
        Some(Value::String(s)) => (s.clone(), String::new()),
        _ => (message_text(data.get("message")), {
            let t = scalar_text(data.get("type"));
            if t.is_empty() {
                scalar_text(data.get("code"))
            } else {
                t
            }
        }),
    };
    let request_id = Some(scalar_text(data.get("request_id")));
    let provider_code = Some(err_type.clone());
    if anthropic_is_context_length(&msg) {
        return ctx.error(
            ErrorClass::ContextLengthError,
            status,
            msg,
            provider_code,
            request_id,
        );
    }
    if err_type == "DeploymentNotFound"
        || ((err_type == "not_found_error" || err_type == "resource_not_found_error")
            && is_model_error(&msg))
    {
        return ctx.error(
            ErrorClass::UnsupportedModelError,
            status,
            msg,
            provider_code,
            request_id,
        );
    }
    if let Some((_, class)) = ANTHROPIC_ERROR_TYPES.iter().find(|(t, _)| *t == err_type) {
        return ctx.error(*class, status, msg, provider_code, request_id);
    }
    if !err_type.is_empty() && !msg.contains(&err_type) {
        msg = format!("{msg} ({err_type})");
    }
    ctx.http(status, msg, provider_code, request_id)
}

// Gemini envelope: `{"error": {"status", "message"}}`.

pub(crate) const GEMINI_ERROR_STATUSES: &[(&str, ErrorClass)] = &[
    ("INVALID_ARGUMENT", ErrorClass::InvalidRequestError),
    ("FAILED_PRECONDITION", ErrorClass::BillingError),
    ("PERMISSION_DENIED", ErrorClass::AuthError),
    ("UNAUTHENTICATED", ErrorClass::AuthError),
    ("NOT_FOUND", ErrorClass::InvalidRequestError),
    ("RESOURCE_EXHAUSTED", ErrorClass::RateLimitError),
    ("INTERNAL", ErrorClass::ServerError),
    ("UNAVAILABLE", ErrorClass::ServerError),
    ("DEADLINE_EXCEEDED", ErrorClass::TimeoutError),
];

pub(crate) fn gemini_is_context_length(msg: &str) -> bool {
    let l = msg.to_lowercase();
    (l.contains("token") && (l.contains("limit") || l.contains("exceed")))
        || l.contains("too long")
        || l.contains("context is too long")
        || l.contains("context length")
}

fn normalize_gemini(ctx: &Context, status: u16, body: &str) -> Lm15Error {
    let Ok(data) = serde_json::from_str::<Value>(body) else {
        return ctx.http(status, fallback_message(status, body), None, None);
    };
    let (mut msg, err_status) = match data.get("error") {
        Some(Value::Object(e)) => (message_text(e.get("message")), scalar_text(e.get("status"))),
        Some(other) if data.is_object() => (message_text(Some(other)), String::new()),
        _ => (String::new(), String::new()),
    };
    let provider_code = Some(err_status.clone());
    if gemini_is_context_length(&msg) {
        return ctx.error(
            ErrorClass::ContextLengthError,
            status,
            msg,
            provider_code,
            None,
        );
    }
    if err_status == "NOT_FOUND" && is_model_error(&msg) {
        return ctx.error(
            ErrorClass::UnsupportedModelError,
            status,
            msg,
            provider_code,
            None,
        );
    }
    if let Some((_, class)) = GEMINI_ERROR_STATUSES.iter().find(|(s, _)| *s == err_status) {
        return ctx.error(*class, status, msg, provider_code, None);
    }
    if !err_status.is_empty() && !msg.contains(&err_status) {
        msg = format!("{msg} ({err_status})");
    }
    ctx.http(status, msg, provider_code, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hierarchy_shape_matches_vocabularies() {
        use ErrorClass::*;
        assert_eq!(ContextLengthError.parent(), Some(InvalidRequestError));
        assert_eq!(UnsupportedModelError.parent(), Some(InvalidRequestError));
        assert_eq!(InvalidRequestError.parent(), Some(ProviderError));
        assert_eq!(NotConfiguredError.parent(), Some(ConfigurationError));
        assert_eq!(UnknownModelError.parent(), Some(ConfigurationError));
        assert_eq!(AmbiguousModelError.parent(), Some(ConfigurationError));
        assert!(!UnknownModelError.is_a(NotConfiguredError));
        assert_eq!(UnknownModelError.code(), ErrorCode::UnknownModel);
        assert_eq!(AmbiguousModelError.code(), ErrorCode::AmbiguousModel);
        assert_eq!(UnsupportedFeatureError.parent(), Some(CapabilityError));
        assert_eq!(TransportError.parent(), Some(LM15Error));
        assert_eq!(LockTimeoutError.parent(), Some(LM15Error));
        assert!(!LockTimeoutError.is_a(ProviderError));
        assert!(!LockTimeoutError.is_a(AuthError));
        assert_eq!(LockTimeoutError.code(), ErrorCode::LockTimeout);
        assert_eq!(StreamAssemblyError.parent(), Some(LM15Error));
        assert_eq!(LM15Error.parent(), None);
        assert!(ContextLengthError.is_a(ProviderError));
        assert!(!TransportError.is_a(ProviderError));
        for class in [
            TransportError,
            LockTimeoutError,
            StreamAssemblyError,
            ConfigurationError,
            NotConfiguredError,
            UnknownModelError,
            AmbiguousModelError,
            CapabilityError,
            UnsupportedFeatureError,
            ProviderError,
            AuthError,
            BillingError,
            RateLimitError,
            InvalidRequestError,
            ContextLengthError,
            UnsupportedModelError,
            TimeoutError,
            ServerError,
        ] {
            assert!(class.is_a(LM15Error));
        }
    }

    #[test]
    fn codes_are_bidirectional_with_leaf_classes() {
        for code in ErrorCode::ALL {
            assert_eq!(code.class().code(), *code);
            assert_eq!(ErrorCode::parse(code.as_str()).unwrap(), *code);
        }
        assert!(ErrorCode::parse("boom").is_err());
        assert_eq!(ErrorClass::LM15Error.code(), ErrorCode::Provider);
    }

    #[test]
    fn retryable_set() {
        let meta = ErrorMeta::new("x");
        assert!(Lm15Error::RateLimitError(meta.clone()).is_retryable());
        assert!(Lm15Error::TimeoutError(meta.clone()).is_retryable());
        assert!(Lm15Error::ServerError(meta.clone()).is_retryable());
        assert!(Lm15Error::TransportError(meta.clone()).is_retryable());
        assert!(Lm15Error::of_class(ErrorClass::LockTimeoutError, meta.clone()).is_retryable());
        assert!(!Lm15Error::AuthError(meta.clone()).is_retryable());
        assert!(!Lm15Error::InvalidRequestError(meta).is_retryable());
    }

    #[test]
    fn http_mapping() {
        let classes: Vec<ErrorClass> = [401u16, 402, 408, 429, 404, 503, 418]
            .iter()
            .map(|s| map_http_error(*s, ErrorMeta::new("m")).class())
            .collect();
        assert_eq!(
            classes,
            vec![
                ErrorClass::AuthError,
                ErrorClass::BillingError,
                ErrorClass::TimeoutError,
                ErrorClass::RateLimitError,
                ErrorClass::InvalidRequestError,
                ErrorClass::ServerError,
                ErrorClass::ProviderError,
            ]
        );
    }

    #[test]
    fn unknown_provider_is_a_value_error() {
        assert!(normalize_error("nope", 500, "{}").is_err());
        let err = normalize_error("openai_chat", 500, "not json").unwrap();
        assert_eq!(err.class(), ErrorClass::ServerError);
        assert_eq!(err.message(), "not json");
        let empty = normalize_error("gemini", 502, "   ").unwrap();
        assert_eq!(empty.message(), "HTTP 502");
    }
}

//! The Anthropic Messages dialect, request side (module 4 W1): `POST
//! /messages` for every door that speaks the wire — api.anthropic.com, the
//! Claude Code login, DeepSeek / Meta / Moonshot over their Anthropic
//! endpoints (compat presets), and the cloud hosts (`emit` rewrites the
//! URL, the model placement and the version field; this dialect never
//! sees a credential).
//!
//! Layout: [`tables`] is the data copied from the reference; [`parts`]
//! renders messages into content blocks; [`body`] builds the JSON body in
//! the reference's key order. Refusals (MAP-5..8) are `Lm15Error` values
//! raised before any wire, with the provider string of the binding.

mod body;
mod parts;
mod tables;

pub use body::RESERVED_EXTENSION_KEYS;
pub use parts::{
    citation_text, CONTINUATION_PROVIDER, DEVELOPER_PREFIX, KIND_REDACTED_THINKING,
    KIND_THINKING_SIGNATURE,
};
pub use tables::{
    anthropic_adaptive_class, builtin_tool_type, effort_thinking_budget, ADAPTIVE_CLASS_MARKERS,
    API_VERSION, BETA_HEADER, BUILTIN_TOOL_TYPES, CODE_EXECUTION_BETA, DEFAULT_VISIBLE_TOKENS,
    EFFORT_THINKING_BUDGETS,
};

use serde_json::Value;

use crate::auth::AccessPolicy;
use crate::compat::{AnthropicCompat, ResolvedAnthropicCompat};
use crate::errors::{ErrorMeta, Lm15Error};
use crate::registry::DialectId;
use crate::types::{Request, Tool};
use crate::wire::{BuildContext, Dialect, WireRequest};

/// The dialect value; stateless (everything per binding is in the
/// [`BuildContext`]).
#[derive(Debug, Clone, Copy, Default)]
pub struct Anthropic;

/// The one instance the dialect table points at.
pub static ANTHROPIC: Anthropic = Anthropic;

/// The endpoint name a host path override is keyed by (AUTH-10).
pub const ENDPOINT: &str = "messages";

impl Dialect for Anthropic {
    fn dialect(&self) -> DialectId {
        DialectId::Anthropic
    }

    fn build(
        &self,
        request: &Request,
        stream: bool,
        cx: &BuildContext<'_>,
    ) -> Result<WireRequest, Lm15Error> {
        let compat = resolved_compat(cx);
        let body = body::payload(request, stream, cx, &compat)?;
        let mut wire = WireRequest::post(format!("/{ENDPOINT}"), Value::Object(body));
        wire.headers = headers(request, cx.policy);
        wire.endpoint = Some(ENDPOINT);
        wire.model = Some(cx.model.to_string());
        Ok(wire)
    }
}

/// The binding's compat, resolved; a binding without one (or with another
/// dialect's) gets the dialect defaults.
fn resolved_compat(cx: &BuildContext<'_>) -> ResolvedAnthropicCompat {
    cx.compat
        .anthropic()
        .map(AnthropicCompat::resolve)
        .unwrap_or_default()
}

/// The dialect headers (`lm15/providers/anthropic.py:386-404`):
/// `anthropic-version`, the policy's static headers in their order, and
/// one `anthropic-beta` joining the policy's betas with the dialect's own
/// (`code-execution-2025-05-22` when a `code_execution` builtin is
/// offered) — comma-separated, first occurrence wins, no duplicates.
/// `content-type` and the credential are `emit`'s.
pub fn headers(request: &Request, policy: &AccessPolicy) -> Vec<(String, String)> {
    let mut headers = vec![("anthropic-version".to_string(), API_VERSION.to_string())];
    let mut betas: Vec<&str> = Vec::new();
    let mut add_beta = |beta: &'static str| {
        let beta = beta.trim();
        if !beta.is_empty() && !betas.contains(&beta) {
            betas.push(beta);
        }
    };
    for (name, value) in policy.headers {
        if name.eq_ignore_ascii_case(BETA_HEADER) {
            value.split(',').for_each(&mut add_beta);
        } else {
            headers.push((name.to_string(), value.to_string()));
        }
    }
    for beta in dialect_betas(request) {
        add_beta(beta);
    }
    if !betas.is_empty() {
        headers.push((BETA_HEADER.to_string(), betas.join(",")));
    }
    headers
}

/// The betas this request needs on its own.
fn dialect_betas(request: &Request) -> Vec<&'static str> {
    let mut betas = Vec::new();
    if request
        .tools
        .iter()
        .any(|tool| matches!(tool, Tool::Builtin(b) if b.name == "code_execution"))
    {
        betas.push(CODE_EXECUTION_BETA);
    }
    betas
}

/// The refusal constructors of this dialect: every error names the
/// binding's provider and carries the pinned class (port.md rule 3).
pub(crate) struct Refuse<'a> {
    pub provider: &'a str,
}

impl Refuse<'_> {
    fn meta(&self, message: impl AsRef<str>) -> ErrorMeta {
        let mut meta = ErrorMeta::new(format!("{}: {}", self.provider, message.as_ref()));
        meta.provider = Some(self.provider.to_string());
        meta
    }

    /// `UnsupportedFeatureError` / `unsupported_feature`.
    pub fn feature(&self, message: impl AsRef<str>) -> Lm15Error {
        Lm15Error::UnsupportedFeatureError(self.meta(message))
    }

    /// `UnsupportedModelError` / `unsupported_model`.
    pub fn model(&self, message: impl AsRef<str>) -> Lm15Error {
        Lm15Error::UnsupportedModelError(self.meta(message))
    }

    /// `InvalidRequestError` / `invalid_request`: the request names
    /// something this process cannot read (a media `path`).
    pub fn invalid_request(&self, message: impl AsRef<str>) -> Lm15Error {
        Lm15Error::InvalidRequestError(self.meta(message))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::{ANTHROPIC_API, CLAUDE_CODE};
    use crate::types::{BuiltinTool, Message};

    fn request(tools: Vec<Tool>) -> Request {
        let mut request =
            Request::new("claude-sonnet-4-5", vec![Message::user("hi").unwrap()]).unwrap();
        request.tools = tools;
        request
    }

    fn builtin(name: &str) -> Tool {
        Tool::Builtin(BuiltinTool::new(name, None).unwrap())
    }

    #[test]
    fn beta_header_joins_policy_and_dialect_betas_without_duplicates() {
        assert_eq!(
            headers(&request(vec![]), &ANTHROPIC_API),
            vec![("anthropic-version".to_string(), "2023-06-01".to_string())]
        );
        let with_code = headers(&request(vec![builtin("code_execution")]), &ANTHROPIC_API);
        assert_eq!(
            with_code[1],
            (
                "anthropic-beta".to_string(),
                CODE_EXECUTION_BETA.to_string()
            )
        );
        let claude_code = headers(&request(vec![builtin("code_execution")]), &CLAUDE_CODE);
        let beta = claude_code
            .iter()
            .find(|(k, _)| k == "anthropic-beta")
            .map(|(_, v)| v.as_str());
        assert_eq!(
            beta,
            Some("claude-code-20250219,oauth-2025-04-20,code-execution-2025-05-22")
        );
        assert!(claude_code.iter().any(|(k, v)| k == "x-app" && v == "cli"));
        // The policy header keeps its position; one anthropic-beta only.
        assert_eq!(
            claude_code
                .iter()
                .filter(|(k, _)| k == "anthropic-beta")
                .count(),
            1
        );

        // A beta the policy already lists is not repeated.
        let policy = AccessPolicy {
            headers: &[("anthropic-beta", "code-execution-2025-05-22,x-beta")],
            ..ANTHROPIC_API
        };
        let joined = headers(&request(vec![builtin("code_execution")]), &policy);
        assert_eq!(joined[1].1, "code-execution-2025-05-22,x-beta");
    }
}

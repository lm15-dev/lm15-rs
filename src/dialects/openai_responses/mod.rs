//! The OpenAI Responses dialect, request side (module 4 W2):
//! `POST /responses` for the `openai`, `openai-codex`, `azure`, `meta` and
//! `moonshotai-responses` policies. The reference is
//! `lm15/providers/openai.py` (`_payload`, `_build_input`,
//! `_tool_choice_payload`, the cache helpers); tables are copied as data
//! with citations, the mapping rules are MAP-5..8.
//!
//! Bodies are `serde_json::Value` objects in the reference's insertion
//! order (`serde_json` `preserve_order`): the harness compares JSON
//! structurally, but a SigV4 door hashes the bytes, so the order is kept
//! the same on every door.
//!
//! What `emit` does and this dialect does not: the credential header, the
//! base URL, Azure's host rewrite, `content-type`, and the `chatgpt-codex`
//! backend's `chatgpt-account-id` header (the account id bound to the
//! adapter, else the token's claim).

mod cache;
mod input;
mod payload;
pub mod response;
mod tools;

use serde_json::Value;

use crate::errors::{ErrorMeta, Lm15Error};
use crate::registry::DialectId;
use crate::sse::SseEvent;
use crate::types::{Request, Response, StreamEvent};
use crate::wire::{apply_static_headers, BuildContext, Dialect, WireRequest};

pub use cache::model_has_cache_options;
pub use tools::builtin_type;

/// spec/auth.md AUTH-10: the backend value the dialect branches on
/// (`lm15/providers/openai.py:391` `CODEX_BACKEND`).
pub use crate::wire::CODEX_BACKEND;

/// The dialect value.
pub struct OpenAIResponses;

/// The one instance the registry wires.
pub static OPENAI_RESPONSES: OpenAIResponses = OpenAIResponses;

impl Dialect for OpenAIResponses {
    fn dialect(&self) -> DialectId {
        DialectId::OpenaiResponses
    }

    fn build(
        &self,
        request: &Request,
        stream: bool,
        cx: &BuildContext<'_>,
    ) -> Result<WireRequest, Lm15Error> {
        let compat = payload::resolve_compat(cx.provider, request, cx.compat.openai_responses())?;
        let body = payload::build_payload(request, stream, cx, &compat)?;
        let mut wire = WireRequest::post("/responses", Value::Object(body));
        // `openai.py:541-549` `_headers`: content type, then the policy's
        // static headers (Codex: `OpenAI-Beta`, `originator`).
        wire.headers
            .push(("Content-Type".into(), "application/json".into()));
        apply_static_headers(&mut wire.headers, cx.policy);
        wire.endpoint = Some("responses");
        wire.model = Some(cx.model.to_string());
        Ok(wire)
    }

    fn parse_response(
        &self,
        request: &Request,
        cx: &BuildContext<'_>,
        body: &[u8],
    ) -> Result<Response, Lm15Error> {
        response::parse_response(cx.provider, request, body)
    }

    fn parse_stream_event(
        &self,
        request: &Request,
        cx: &BuildContext<'_>,
        event: &SseEvent,
        out: &mut Vec<StreamEvent>,
    ) -> Result<(), Lm15Error> {
        response::parse_stream_event(cx.provider, request, event, out)
    }
}

/// A MAP-5..8 refusal: `UnsupportedFeatureError` naming the provider.
pub(crate) fn unsupported(provider: &str, message: impl Into<String>) -> Lm15Error {
    let mut meta = ErrorMeta::new(format!("{provider}: {}", message.into()));
    meta.provider = Some(provider.to_string());
    Lm15Error::UnsupportedFeatureError(meta)
}

/// A caller-side request fault that is not a capability gap (an
/// unreadable media path).
pub(crate) fn invalid_request(provider: &str, message: impl Into<String>) -> Lm15Error {
    let mut meta = ErrorMeta::new(format!("{provider}: {}", message.into()));
    meta.provider = Some(provider.to_string());
    Lm15Error::InvalidRequestError(meta)
}

//! The OpenAI Chat Completions dialect, request side (module 4 W3): the
//! wire OpenAI's legacy endpoint and most compatible servers speak
//! (Groq, vLLM, SGLang, ollama, DeepSeek, Z.AI, Moonshot, Meta, xAI, the
//! Azure and Bedrock chat doors). A server's quirks arrive as the
//! [`OpenAIChatCompat`] in the [`BuildContext`]; the dialect consults each
//! knob at one named point and nowhere else.
//!
//! `POST {base_url}/chat/completions` with a JSON body built in the
//! reference's key order (`lm15/providers/openai_chat.py:377-560`); the
//! credential, the host rewrites and SigV4 are `wire::emit`'s.
//!
//! The dialect consults the bound access policy at two points: its
//! static headers (AUTH-10) and, for the `xai` policy, xAI's pinned
//! refusal table (MAP-5, MAP-8; `lm15/providers/xai.py:77-125`) — a
//! provider fact bound to the door, not a compat knob (rule 7).

mod cache;
mod messages;
mod payload;
pub mod response;
mod text;

use serde_json::Value;

use self::text::unsupported;
use crate::compat::{OpenAIChatCompat, ResolvedOpenAIChatCompat};
use crate::errors::Lm15Error;
use crate::registry::DialectId;
use crate::sse::SseEvent;
use crate::types::{Request, Response, StreamEvent, ToolChoiceMode};
use crate::wire::{apply_static_headers, BuildContext, Dialect, WireRequest};

/// The dialect value.
#[derive(Debug, Clone, Copy, Default)]
pub struct OpenAIChat;

/// The one instance the dialect table points at.
pub static OPENAI_CHAT: OpenAIChat = OpenAIChat;

impl Dialect for OpenAIChat {
    fn dialect(&self) -> DialectId {
        DialectId::OpenaiChat
    }

    fn build(
        &self,
        request: &Request,
        stream: bool,
        cx: &BuildContext<'_>,
    ) -> Result<WireRequest, Lm15Error> {
        let compat = resolve_compat(cx, cx.model);
        if cx.policy.provider == "xai" {
            xai_refusals(request, cx.provider)?;
        }
        let body = payload::build_payload(request, stream, cx.model, &compat, cx.provider)?;
        let mut wire = WireRequest::post("/chat/completions", Value::Object(body));
        // `_headers` (`openai_chat.py:197-203`): content type, then the
        // policy's static headers; the credential is emit's.
        wire.headers
            .push(("Content-Type".into(), "application/json".into()));
        apply_static_headers(&mut wire.headers, cx.policy);
        wire.endpoint = Some("chat/completions");
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

/// The compat this request reads: the binding's `OpenAIChatCompat` (the
/// empty partial when the binding carries another kind) with the door's
/// per-model overrides applied (`openai_chat.py:369-375`).
fn resolve_compat(cx: &BuildContext<'_>, model: &str) -> ResolvedOpenAIChatCompat {
    let partial = cx.compat.openai_chat().unwrap_or(&OpenAIChatCompat::EMPTY);
    if partial.model_overrides.is_empty() {
        partial.resolve()
    } else {
        partial.for_model(model).resolve()
    }
}

/// xAI's refusal table (`lm15/providers/xai.py:77-125`), each cell a
/// live-measured silent no-op on api.x.ai: reasoning off (MAP-5),
/// logprobs (docs.x.ai: ignored on grok-4.20+), allowlist subsets other
/// than one forced name (MAP-8 rule 1), a forced tool next to
/// `response_format` (MAP-8 rule 3).
fn xai_refusals(request: &Request, provider: &str) -> Result<(), Lm15Error> {
    let config = &request.config;
    if config.reasoning.as_ref().is_some_and(|r| r.is_off()) {
        return Err(unsupported(
            provider,
            "reasoning cannot be disabled — Grok reasoning models have no off switch, and xAI \
             silently ignores disable fields on the wire. Omit the reasoning config, or pick a \
             non-reasoning Grok model.",
        ));
    }
    if config.logprobs.is_some() {
        return Err(unsupported(
            provider,
            "config.logprobs is not supported — grok-4.20 and newer silently ignore \
             logprobs/top_logprobs on the wire (docs.x.ai, verified live 2026-09-01). OpenAI \
             and Gemini carry logprobs.",
        ));
    }
    if let Some(choice) = &config.tool_choice {
        let forced_one = choice.allowed.len() == 1 && choice.mode == ToolChoiceMode::Required;
        if !choice.allowed.is_empty() && !forced_one {
            return Err(unsupported(
                provider,
                "tool_choice.allowed subsets are silently ignored by api.x.ai (verified live \
                 2026-09-02); force a single tool with mode='required', or send only the \
                 allowed tools in Request.tools",
            ));
        }
        if choice.mode == ToolChoiceMode::Required && config.response_format.is_some() {
            return Err(unsupported(
                provider,
                "a forced tool (mode='required') cannot be combined with response_format — \
                 api.x.ai returns JSON text and drops the call (verified live 2026-09-02)",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;

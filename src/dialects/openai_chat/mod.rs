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
pub mod ingest;
mod messages;
mod payload;
pub mod response;
mod text;

/// [`response::response_from_openai_chat`] under the family's provider
/// name, `openai-chat` (MAP-12 rule 9; the module function of
/// `playbooks/api-family.md` § Ingest).
pub fn response_from_openai_chat(
    body: &serde_json::Value,
    model: Option<&str>,
    choice: Option<usize>,
) -> Result<Response, Lm15Error> {
    response::response_from_openai_chat("openai-chat", body, model, choice)
}

use serde_json::Value;

use crate::compat::{OpenAIChatCompat, ResolvedOpenAIChatCompat};
use crate::errors::Lm15Error;
use crate::registry::DialectId;
use crate::sse::SseEvent;
use crate::types::{ModelInfo, Request, Response, StreamEvent, ToolChoiceMode};
use crate::wire::{
    apply_static_headers, model_infos_from_entries, BuildContext, Dialect, Surfaces, WireRequest,
};

/// The dialect value.
#[derive(Debug, Clone, Copy, Default)]
pub struct OpenAIChat;

/// The one instance the dialect table points at.
pub static OPENAI_CHAT: OpenAIChat = OpenAIChat;

pub mod generation;
pub mod video;

use crate::dialects::openai_responses::files;

impl Surfaces for OpenAIChat {
    fn video_submit_request(
        &self,
        cx: &BuildContext<'_>,
        request: &crate::types::VideoGenerationRequest,
    ) -> Result<WireRequest, Lm15Error> {
        video::submit_request(cx, request)
    }
    fn video_job(
        &self,
        cx: &BuildContext<'_>,
        body: &[u8],
        video_id: Option<&str>,
    ) -> Result<crate::types::VideoJobInfo, Lm15Error> {
        video::job_from_body(cx, body, video_id)
    }
    fn video_status_request(
        &self,
        cx: &BuildContext<'_>,
        video_id: &str,
    ) -> Result<WireRequest, Lm15Error> {
        Ok(video::status_request(cx, video_id))
    }
    fn video_result_fetch(
        &self,
        _cx: &BuildContext<'_>,
        _status_body: &serde_json::Map<String, Value>,
    ) -> Result<Option<WireRequest>, Lm15Error> {
        Ok(None) // the terminal body carries a public URL
    }
    fn video_part(
        &self,
        cx: &BuildContext<'_>,
        status_body: &serde_json::Map<String, Value>,
        _fetched: crate::wire::Fetched<'_>,
    ) -> Result<crate::types::VideoPart, Lm15Error> {
        video::part(cx, status_body)
    }
    fn video_list_request(
        &self,
        cx: &BuildContext<'_>,
        _limit: u64,
        _model: Option<&str>,
    ) -> Result<WireRequest, Lm15Error> {
        Err(video::list_unsupported(cx))
    }

    fn image_generate_request(
        &self,
        cx: &BuildContext<'_>,
        request: &crate::types::ImageGenerationRequest,
    ) -> Result<WireRequest, Lm15Error> {
        generation::image_request(cx, request)
    }
    fn image_generation(
        &self,
        cx: &BuildContext<'_>,
        _request: &crate::types::ImageGenerationRequest,
        _headers: &[(String, String)],
        body: &[u8],
    ) -> Result<crate::types::ImageGenerationResponse, Lm15Error> {
        generation::image_response(cx, body)
    }

    fn file_upload_request(
        &self,
        cx: &BuildContext<'_>,
        request: &crate::types::FileUploadRequest,
    ) -> Result<WireRequest, Lm15Error> {
        files::upload_request(cx, request)
    }
    fn file_info(
        &self,
        cx: &BuildContext<'_>,
        body: &[u8],
    ) -> Result<crate::types::FileInfo, Lm15Error> {
        files::file_info_from_body(cx, body)
    }
    fn file_get_request(
        &self,
        cx: &BuildContext<'_>,
        file_id: &str,
    ) -> Result<WireRequest, Lm15Error> {
        Ok(files::get_request(cx, file_id))
    }
    fn file_list_request(
        &self,
        cx: &BuildContext<'_>,
        limit: u64,
        cursor: Option<&str>,
    ) -> Result<WireRequest, Lm15Error> {
        Ok(files::list_request(cx, limit, cursor))
    }
    fn file_page(
        &self,
        cx: &BuildContext<'_>,
        body: &[u8],
    ) -> Result<crate::types::FilePage, Lm15Error> {
        files::page(cx, body)
    }
    fn file_delete_request(
        &self,
        cx: &BuildContext<'_>,
        file_id: &str,
    ) -> Result<WireRequest, Lm15Error> {
        Ok(files::delete_request(cx, file_id))
    }
    fn file_download_request(
        &self,
        cx: &BuildContext<'_>,
        file_id: &str,
    ) -> Result<WireRequest, Lm15Error> {
        Ok(files::download_request(cx, file_id))
    }
}

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
        let prepared = if cx.policy.provider == "xai" {
            Some(xai_prepare(request, cx.provider)?)
        } else {
            None
        };
        let request = prepared.as_ref().unwrap_or(request);
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

    /// `openai_chat.py:248-264`: `GET /models`, entries under `data`,
    /// `id` verbatim; `_headers()` (content type, static headers).
    fn models_request(&self, cx: &BuildContext<'_>) -> Result<WireRequest, Lm15Error> {
        let mut wire = WireRequest::get("/models");
        wire.headers
            .push(("Content-Type".into(), "application/json".into()));
        apply_static_headers(&mut wire.headers, cx.policy);
        Ok(wire)
    }

    fn parse_models(
        &self,
        cx: &BuildContext<'_>,
        body: &[u8],
    ) -> Result<Vec<ModelInfo>, Lm15Error> {
        let data: Value = serde_json::from_slice(body).map_err(|err| {
            let mut meta = crate::errors::ErrorMeta::new(format!(
                "{}: models body is not JSON: {err}",
                cx.provider
            ));
            meta.provider = Some(cx.provider.to_string());
            Lm15Error::ProviderError(meta)
        })?;
        // Two catalog shapes are in the wild: OpenAI's {"object": "list",
        // "data": [...]} and a bare JSON array (Together, live 2026-09-26: 272
        // entries, no envelope). Anything else is a malformed reply, never an
        // empty catalog: reading it as zero models would lose every entry
        // silently.
        let entries = match &data {
            Value::Array(_) => Some(&data),
            Value::Object(object) => object.get("data").filter(|v| v.is_array()),
            _ => None,
        };
        let Some(entries) = entries else {
            let text = String::from_utf8_lossy(body);
            let excerpt: String = text.chars().take(200).collect();
            let mut meta = crate::errors::ErrorMeta::new(format!(
                "malformed provider reply: a model catalog is {{\"data\": [...]}} or a JSON array of \
                 entries. Body starts: {excerpt:?}"
            ));
            meta.provider = Some(cx.provider.to_string());
            return Err(Lm15Error::ProviderError(meta));
        };
        Ok(model_infos_from_entries(
            Some(entries),
            cx.provider,
            "openai_chat",
            |entry| entry.get("id").and_then(Value::as_str).map(str::to_string),
        ))
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
pub(crate) fn resolve_compat(cx: &BuildContext<'_>, model: &str) -> ResolvedOpenAIChatCompat {
    let partial = cx.compat.openai_chat().unwrap_or(&OpenAIChatCompat::EMPTY);
    if partial.model_overrides.is_empty() {
        partial.resolve()
    } else {
        partial.for_model(model).resolve()
    }
}

/// xAI's receipted silent no-ops become visible adaptations (MAP-13).
/// A forced tool conflicting with structured output still needs a real choice.
fn xai_prepare(request: &Request, provider: &str) -> Result<Request, Lm15Error> {
    use crate::adaptation::{adapt, drop_value, refusal, AdaptationAction::*};
    let mut out = request.clone();
    let config = &mut out.config;
    if let Some(r) = config.reasoning.as_mut().filter(|r| r.is_off()) {
        adapt("config.reasoning.effort",Substituted,Some(serde_json::json!("off")),Some(serde_json::json!("low")),"Grok has no off switch; the lowest reasoning level was sent and spend remains visible in usage")?;
        r.effort = crate::types::ReasoningEffort::Low;
    }
    drop_value(
        "config.logprobs",
        &mut config.logprobs,
        "grok-4.20 and newer ignore logprobs/top_logprobs",
    )?;
    if let Some(choice) = &mut config.tool_choice {
        if choice.mode == ToolChoiceMode::Required && config.response_format.is_some() {
            return Err(refusal(
                provider,
                "config.tool_choice",
                "a real choice is needed: xAI drops a forced tool next to response_format",
            ));
        }
        if !choice.allowed.is_empty()
            && !(choice.allowed.len() == 1 && choice.mode == ToolChoiceMode::Required)
        {
            adapt(
                "config.tool_choice.allowed",
                ClientSide,
                Some(serde_json::json!(choice.allowed)),
                Some(serde_json::json!(choice.allowed)),
                "xAI ignores allowlists; only allowed tools were sent",
            )?;
            out.tools
                .retain(|t| choice.allowed.iter().any(|n| n == t.name()));
            choice.allowed.clear();
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests;

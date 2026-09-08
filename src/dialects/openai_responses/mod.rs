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

pub mod batch;
mod cache;
pub mod files;
pub mod generation;
mod input;
pub mod live;
mod payload;
pub mod response;
mod tools;
pub mod video;

use serde_json::Value;

use crate::errors::{ErrorMeta, Lm15Error};
use crate::registry::DialectId;
use crate::sse::SseEvent;
use crate::types::{ModelInfo, Request, Response, StreamEvent};
use crate::wire::{
    apply_static_headers, model_infos_from_entries, BuildContext, Dialect, Surfaces, WireRequest,
};

pub use cache::model_has_cache_options;
pub use tools::builtin_type;

/// spec/auth.md AUTH-10: the backend value the dialect branches on
/// (`lm15/providers/openai.py:391` `CODEX_BACKEND`).
pub use crate::wire::CODEX_BACKEND;

/// The dialect value.
pub struct OpenAIResponses;

/// The one instance the registry wires.
pub static OPENAI_RESPONSES: OpenAIResponses = OpenAIResponses;

impl Surfaces for OpenAIResponses {
    fn live_url(
        &self,
        cx: &BuildContext<'_>,
        config: &crate::types::LiveConfig,
    ) -> Result<(String, Vec<(String, String)>), Lm15Error> {
        Ok(live::url(cx, config))
    }
    fn live_setup_frames(
        &self,
        cx: &BuildContext<'_>,
        config: &crate::types::LiveConfig,
    ) -> Result<Vec<Value>, Lm15Error> {
        Ok(vec![live::session_update(cx, config)?])
    }
    fn live_encode(
        &self,
        cx: &BuildContext<'_>,
        _config: &crate::types::LiveConfig,
        event: &crate::types::LiveClientEvent,
    ) -> Result<Vec<Value>, Lm15Error> {
        live::encode(cx, event)
    }
    fn live_decode(
        &self,
        _cx: &BuildContext<'_>,
        frame: &[u8],
    ) -> Result<Vec<crate::types::LiveServerEvent>, Lm15Error> {
        Ok(live::decode(frame))
    }

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
        _video_id: Option<&str>,
    ) -> Result<crate::types::VideoJobInfo, Lm15Error> {
        video::job_from_body(cx, body)
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
        cx: &BuildContext<'_>,
        status_body: &serde_json::Map<String, Value>,
    ) -> Result<Option<WireRequest>, Lm15Error> {
        Ok(Some(video::result_fetch(cx, status_body)))
    }
    fn video_part(
        &self,
        cx: &BuildContext<'_>,
        _status_body: &serde_json::Map<String, Value>,
        fetched: crate::wire::Fetched<'_>,
    ) -> Result<crate::types::VideoPart, Lm15Error> {
        video::part(cx, fetched)
    }
    fn video_list_request(
        &self,
        cx: &BuildContext<'_>,
        limit: u64,
        _model: Option<&str>,
    ) -> Result<WireRequest, Lm15Error> {
        Ok(video::list_request(cx, limit))
    }
    fn video_jobs(
        &self,
        cx: &BuildContext<'_>,
        body: &[u8],
    ) -> Result<Vec<crate::types::VideoJobInfo>, Lm15Error> {
        video::jobs(cx, body)
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
    fn speech_generate_request(
        &self,
        cx: &BuildContext<'_>,
        request: &crate::types::SpeechGenerationRequest,
    ) -> Result<WireRequest, Lm15Error> {
        Ok(generation::speech_request(cx, request))
    }
    fn speech_generation(
        &self,
        cx: &BuildContext<'_>,
        _request: &crate::types::SpeechGenerationRequest,
        headers: &[(String, String)],
        body: &[u8],
    ) -> Result<crate::types::SpeechGenerationResponse, Lm15Error> {
        generation::speech_response(cx, headers, body)
    }

    fn batch_upload_request(
        &self,
        cx: &BuildContext<'_>,
        request: &crate::types::BatchRequest,
    ) -> Result<Option<WireRequest>, Lm15Error> {
        batch::upload_request(self, cx, request).map(Some)
    }
    fn batch_submit_request(
        &self,
        cx: &BuildContext<'_>,
        request: &crate::types::BatchRequest,
        upload_body: Option<&serde_json::Map<String, Value>>,
    ) -> Result<WireRequest, Lm15Error> {
        batch::submit_request(cx, request, upload_body)
    }
    fn batch_job(
        &self,
        cx: &BuildContext<'_>,
        body: &[u8],
    ) -> Result<crate::types::BatchJobInfo, Lm15Error> {
        batch::job_from_body(cx, body)
    }
    fn batch_status_request(
        &self,
        cx: &BuildContext<'_>,
        batch_id: &str,
    ) -> Result<WireRequest, Lm15Error> {
        Ok(batch::status_request(cx, batch_id))
    }
    fn batch_cancel_request(
        &self,
        cx: &BuildContext<'_>,
        batch_id: &str,
    ) -> Result<WireRequest, Lm15Error> {
        Ok(batch::cancel_request(cx, batch_id))
    }
    fn batch_result_fetches(
        &self,
        cx: &BuildContext<'_>,
        status_body: &serde_json::Map<String, Value>,
    ) -> Result<Vec<WireRequest>, Lm15Error> {
        Ok(batch::result_fetches(cx, status_body))
    }
    fn batch_entries(
        &self,
        cx: &BuildContext<'_>,
        status_body: &serde_json::Map<String, Value>,
        fetched: &[Vec<u8>],
    ) -> Result<Vec<crate::types::BatchEntry>, Lm15Error> {
        batch::entries(self, cx, status_body, fetched)
    }
    fn batch_list_request(
        &self,
        cx: &BuildContext<'_>,
        limit: u64,
    ) -> Result<WireRequest, Lm15Error> {
        Ok(batch::list_request(cx, limit))
    }
    fn batch_jobs(
        &self,
        cx: &BuildContext<'_>,
        body: &[u8],
    ) -> Result<Vec<crate::types::BatchJobInfo>, Lm15Error> {
        batch::jobs(cx, body)
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

    /// `openai.py:1625-1653`: `GET /models`, entries under `data`, `id`
    /// verbatim. The Codex backend: `?client_version=<policy's>` (the
    /// backend requires it), entries under `models`, the id is `slug`.
    fn models_request(&self, cx: &BuildContext<'_>) -> Result<WireRequest, Lm15Error> {
        let mut wire = WireRequest::get("/models");
        if cx.policy.backend == CODEX_BACKEND {
            let version = cx.policy.backend_option("client_version").unwrap_or("");
            wire.params
                .push(("client_version".into(), version.to_string()));
        }
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
            let mut meta =
                ErrorMeta::new(format!("{}: models body is not JSON: {err}", cx.provider));
            meta.provider = Some(cx.provider.to_string());
            Lm15Error::ProviderError(meta)
        })?;
        let (key, id_key) = if cx.policy.backend == CODEX_BACKEND {
            ("models", "slug")
        } else {
            ("data", "id")
        };
        Ok(model_infos_from_entries(
            data.get(key),
            cx.provider,
            "openai_responses",
            |entry| {
                entry
                    .get(id_key)
                    .and_then(Value::as_str)
                    .map(str::to_string)
            },
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

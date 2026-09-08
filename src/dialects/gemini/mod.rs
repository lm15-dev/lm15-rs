//! The Gemini `generateContent` dialect, request side (module 4 W4).
//!
//! `POST /models/{model}:generateContent`, or
//! `:streamGenerateContent?alt=sse` for a stream. The body is built in
//! the key order the fixtures pin (`serde_json` `preserve_order`): the
//! Gemini doors have no signature, so the order is a family convention
//! here, not a wire fact. The dialect takes no compat value (the
//! reference registry binds none) and reads no policy field: the host
//! places the model (`WireRequest.model`, `endpoint = "generateContent"`)
//! and `emit` places the credential (`x-goog-api-key` on the public door,
//! `?key=` on `vertex-express`, a bearer token on `vertex`).
//!
//! Mapping tables are copied from the reference as data with citations
//! (`lm15/providers/gemini.py`, `lm15/providers/common.py`). Refusals are
//! the MAP-5..8 rules and the no-silent-drop rule (playbooks/port.md
//! rule 4); each is an `Lm15Error` raised before any wire.

pub mod batch;
pub mod cache;
mod config;
mod contents;
pub mod files;
pub mod generation;
pub mod live;
pub mod response;
pub mod video;

use serde_json::{Map, Value};

use crate::errors::{ErrorMeta, Lm15Error};
use crate::registry::DialectId;
use crate::sse::SseEvent;
use crate::types::{BuiltinTool, ModelInfo, Request, Response, StreamEvent, Tool};
use crate::wire::{model_infos_from_entries, BuildContext, Dialect, Surfaces, WireRequest};

use self::config::{cache_plan, generation_config, tool_config};
use self::contents::{contents, system_instruction};

/// The one dialect value.
pub static GEMINI: Gemini = Gemini;

/// The Gemini wire codec (stateless).
#[derive(Debug, Clone, Copy, Default)]
pub struct Gemini;

/// Canonical builtin tool name → Gemini tool key
/// (`lm15/providers/gemini.py:110-113`). An unknown name passes through
/// verbatim as the key (spec/vocabularies.md § Open string namespaces).
pub const GEMINI_BUILTIN_TOOLS: &[(&str, &str)] = &[
    ("web_search", "googleSearch"),
    ("code_execution", "codeExecution"),
];

/// The endpoint name a host path override is keyed by (AUTH-10).
pub const ENDPOINT: &str = "generateContent";

/// True for the Gemini 3.x class: `thinkingLevel`, no full off (MAP-7
/// rule 10; `lm15/providers/gemini.py:120-130`). A model-name table that
/// rots; the server 400s loudly when wrong. Receipted 2026-09-02: 2.5
/// models reject `thinkingLevel`, 3.x models take it; 3.7 Flash accepts
/// `thinkingBudget: 0` and still spends thinking tokens.
pub fn gemini_level_class(model: &str) -> bool {
    let lowered = model.to_ascii_lowercase();
    lowered.starts_with("models/gemini-3") || lowered.starts_with("gemini-3")
}

/// `UnsupportedFeatureError` naming the bound provider.
pub(crate) fn unsupported(cx: &BuildContext<'_>, message: String) -> Lm15Error {
    let mut meta = ErrorMeta::new(format!("{}: {message}", cx.provider));
    meta.provider = Some(cx.provider.to_string());
    Lm15Error::UnsupportedFeatureError(meta)
}

/// `InvalidRequestError` naming the bound provider (a request the wire
/// cannot carry as written; no bytes were sent).
pub(crate) fn invalid(cx: &BuildContext<'_>, message: String) -> Lm15Error {
    let mut meta = ErrorMeta::new(format!("{}: {message}", cx.provider));
    meta.provider = Some(cx.provider.to_string());
    Lm15Error::InvalidRequestError(meta)
}

/// `_model_path` (`lm15/providers/gemini.py:551-553`): `quote(model,
/// safe="/:@")`, prefixed with `models/` unless already so.
pub(crate) fn model_path(model: &str) -> String {
    let quoted = crate::cloud::percent::encode(model, b"/:@");
    if quoted.starts_with("models/") {
        quoted
    } else {
        format!("models/{quoted}")
    }
}

/// `{"googleSearch": {}}` / `{"codeExecution": {}}` / `{<name>: <config>}`
/// (`lm15/providers/gemini.py:145-146`).
fn builtin_tool(tool: &BuiltinTool) -> Value {
    let key = GEMINI_BUILTIN_TOOLS
        .iter()
        .find(|(name, _)| *name == tool.name)
        .map(|(_, key)| *key)
        .unwrap_or(tool.name.as_str());
    let config = tool
        .config
        .clone()
        .map(Value::Object)
        .unwrap_or_else(|| Value::Object(Map::new()));
    let mut out = Map::new();
    out.insert(key.to_string(), config);
    Value::Object(out)
}

/// `tools`: one `functionDeclarations` group for every FunctionTool (in
/// order), then one entry per BuiltinTool (in order) —
/// `lm15/providers/gemini.py:747-759`. `description` is omitted when
/// absent (the reference sends `null`; the field is optional on the wire).
fn tools(request: &Request) -> Option<Value> {
    if request.tools.is_empty() {
        return None;
    }
    let declarations: Vec<Value> = request
        .tools
        .iter()
        .filter_map(|tool| match tool {
            Tool::Function(f) => {
                let mut decl = Map::new();
                decl.insert("name".into(), Value::String(f.name.clone()));
                if let Some(description) = &f.description {
                    decl.insert("description".into(), Value::String(description.clone()));
                }
                decl.insert("parameters".into(), Value::Object(f.parameters.clone()));
                Some(Value::Object(decl))
            }
            Tool::Builtin(_) => None,
        })
        .collect();
    let mut out = Vec::new();
    if !declarations.is_empty() {
        let mut group = Map::new();
        group.insert("functionDeclarations".into(), Value::Array(declarations));
        out.push(Value::Object(group));
    }
    for tool in &request.tools {
        if let Tool::Builtin(b) = tool {
            out.push(builtin_tool(b));
        }
    }
    Some(Value::Array(out))
}

/// The request body (`lm15/providers/gemini.py:646-794` `_payload`), in
/// the fixtures' key order: `contents`, `cachedContent`,
/// `systemInstruction`, `generationConfig`, `toolConfig`, `tools`,
/// `store`, `serviceTier`, then `extensions` verbatim.
fn payload(request: &Request, cx: &BuildContext<'_>) -> Result<Value, Lm15Error> {
    let config = &request.config;

    // MAP-6 on Gemini: the automatic tier needs nothing; prefix intents
    // fall back to it (no cost, visible in usage); `resource` names a
    // stored object that already holds system, tools and the prefix, so
    // the wire carries only the suffix (MAP-6.7); `key` and `retention`
    // name mechanisms this wire does not have.
    let plan = cache_plan(request, cx)?;
    let wire_messages = &request.messages[plan.suffix_from..];
    if plan.resource.is_some() && wire_messages.is_empty() {
        return Err(invalid(
            cx,
            "a request against a stored cache needs at least one message after the prefix".into(),
        ));
    }

    // The tool-choice refusals (MAP-8) are about the caller's intent and
    // fire even when a cached resource makes `toolConfig` unsendable.
    let tool_config = tool_config(request, cx)?;

    let mut body = Map::new();
    body.insert("contents".into(), contents(wire_messages, request, cx)?);
    if let Some(resource) = &plan.resource {
        body.insert("cachedContent".into(), Value::String(resource.clone()));
    }
    if plan.resource.is_none() {
        if let Some(system) = &request.system {
            body.insert("systemInstruction".into(), system_instruction(system, cx)?);
        }
    }
    let mut generation = generation_config(request, cx)?;

    // `extensions.output` selects the response modality
    // (`lm15/providers/gemini.py:765-769`; the image/speech surfaces of
    // module 8 build on it).
    let extensions = config.extensions.as_ref();
    if let Some(output) = extensions.and_then(|e| e.get("output")) {
        let modality = match output.as_str() {
            Some("image") => "IMAGE",
            Some("audio") => "AUDIO",
            _ => {
                return Err(invalid(
                    cx,
                    format!("extensions.output must be \"image\" or \"audio\" (got {output})"),
                ))
            }
        };
        generation.insert(
            "responseModalities".into(),
            Value::Array(vec![Value::String(modality.into())]),
        );
    }
    if !generation.is_empty() {
        body.insert("generationConfig".into(), Value::Object(generation));
    }

    if plan.resource.is_none() {
        if let Some(tool_config) = tool_config {
            body.insert("toolConfig".into(), tool_config);
        }
        if let Some(tools) = tools(request) {
            body.insert("tools".into(), tools);
        }
    }

    // Promoted knobs (changes/2026-09-01-extensions-burn-down.md): `store`
    // verbatim; `service_tier` → top-level `serviceTier` (live 2026-09-01,
    // echoed in `usageMetadata.serviceTier`); `user_id` has no wire field.
    if let Some(store) = config.store {
        body.insert("store".into(), Value::Bool(store));
    }
    if let Some(tier) = &config.service_tier {
        body.insert("serviceTier".into(), Value::String(tier.clone()));
    }
    if config.user_id.is_some() {
        return Err(unsupported(
            cx,
            "config.user_id is not supported — GenerateContent has no end-user attribution field (OpenAI and Anthropic carry it)".into(),
        ));
    }

    // Passthrough (INV-049): every other extension key lands at the top
    // level verbatim, replacing a built key of the same name.
    if let Some(extensions) = extensions {
        for (key, value) in extensions {
            if key == "output" {
                continue;
            }
            body.insert(key.clone(), value.clone());
        }
    }
    Ok(Value::Object(body))
}

impl Surfaces for Gemini {
    fn live_url(
        &self,
        cx: &BuildContext<'_>,
        _config: &crate::types::LiveConfig,
    ) -> Result<(String, Vec<(String, String)>), Lm15Error> {
        Ok((live::url(cx), Vec::new()))
    }
    fn live_setup_frames(
        &self,
        cx: &BuildContext<'_>,
        config: &crate::types::LiveConfig,
    ) -> Result<Vec<Value>, Lm15Error> {
        Ok(vec![live::setup_frame(cx, config)?])
    }
    fn live_encode(
        &self,
        cx: &BuildContext<'_>,
        config: &crate::types::LiveConfig,
        event: &crate::types::LiveClientEvent,
    ) -> Result<Vec<Value>, Lm15Error> {
        live::encode(cx, config, event)
    }
    fn live_decode(
        &self,
        cx: &BuildContext<'_>,
        frame: &[u8],
    ) -> Result<Vec<crate::types::LiveServerEvent>, Lm15Error> {
        Ok(live::decode(cx, frame))
    }
    fn live_setup_complete(&self, cx: &BuildContext<'_>, frame: &[u8]) -> Result<bool, Lm15Error> {
        live::setup_complete(cx, frame)
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
        _cx: &BuildContext<'_>,
        video_id: &str,
    ) -> Result<WireRequest, Lm15Error> {
        Ok(video::status_request(video_id))
    }
    fn video_result_fetch(
        &self,
        cx: &BuildContext<'_>,
        status_body: &Map<String, Value>,
    ) -> Result<Option<WireRequest>, Lm15Error> {
        video::result_fetch(cx, status_body).map(Some)
    }
    fn video_part(
        &self,
        cx: &BuildContext<'_>,
        _status_body: &Map<String, Value>,
        fetched: crate::wire::Fetched<'_>,
    ) -> Result<crate::types::VideoPart, Lm15Error> {
        video::part(cx, fetched)
    }
    fn video_list_request(
        &self,
        cx: &BuildContext<'_>,
        limit: u64,
        model: Option<&str>,
    ) -> Result<WireRequest, Lm15Error> {
        video::list_request(cx, limit, model)
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
        generation::image_request(self, cx, request)
    }
    fn image_generation(
        &self,
        cx: &BuildContext<'_>,
        request: &crate::types::ImageGenerationRequest,
        _headers: &[(String, String)],
        body: &[u8],
    ) -> Result<crate::types::ImageGenerationResponse, Lm15Error> {
        generation::image_response(self, cx, request, body)
    }
    fn speech_generate_request(
        &self,
        cx: &BuildContext<'_>,
        request: &crate::types::SpeechGenerationRequest,
    ) -> Result<WireRequest, Lm15Error> {
        generation::speech_request(self, cx, request)
    }
    fn speech_generation(
        &self,
        cx: &BuildContext<'_>,
        request: &crate::types::SpeechGenerationRequest,
        _headers: &[(String, String)],
        body: &[u8],
    ) -> Result<crate::types::SpeechGenerationResponse, Lm15Error> {
        generation::speech_response(self, cx, request, body)
    }

    fn cache_create_request(
        &self,
        cx: &BuildContext<'_>,
        prefix: &Request,
        ttl_seconds: Option<u64>,
        label: Option<&str>,
    ) -> Result<WireRequest, Lm15Error> {
        cache::create_request(self, cx, prefix, ttl_seconds, label)
    }
    fn cache_info(
        &self,
        cx: &BuildContext<'_>,
        body: &[u8],
    ) -> Result<crate::types::CacheInfo, Lm15Error> {
        cache::info_from_body(cx, body)
    }
    fn cache_get_request(
        &self,
        _cx: &BuildContext<'_>,
        cache_id: &str,
    ) -> Result<WireRequest, Lm15Error> {
        Ok(cache::get_request(cache_id))
    }
    fn cache_list_request(
        &self,
        _cx: &BuildContext<'_>,
        limit: u64,
        cursor: Option<&str>,
    ) -> Result<WireRequest, Lm15Error> {
        Ok(cache::list_request(limit, cursor))
    }
    fn cache_page(
        &self,
        cx: &BuildContext<'_>,
        body: &[u8],
    ) -> Result<crate::types::CachePage, Lm15Error> {
        cache::page(cx, body)
    }
    fn cache_delete_request(
        &self,
        _cx: &BuildContext<'_>,
        cache_id: &str,
    ) -> Result<WireRequest, Lm15Error> {
        Ok(cache::delete_request(cache_id))
    }
    fn cache_update_request(
        &self,
        _cx: &BuildContext<'_>,
        cache_id: &str,
        ttl_seconds: u64,
    ) -> Result<WireRequest, Lm15Error> {
        Ok(cache::update_request(cache_id, ttl_seconds))
    }

    fn batch_upload_request(
        &self,
        _cx: &BuildContext<'_>,
        _request: &crate::types::BatchRequest,
    ) -> Result<Option<WireRequest>, Lm15Error> {
        Ok(None)
    }
    fn batch_submit_request(
        &self,
        cx: &BuildContext<'_>,
        request: &crate::types::BatchRequest,
        _upload_body: Option<&Map<String, Value>>,
    ) -> Result<WireRequest, Lm15Error> {
        batch::submit_request(self, cx, request)
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
        _cx: &BuildContext<'_>,
        batch_id: &str,
    ) -> Result<WireRequest, Lm15Error> {
        Ok(batch::status_request(batch_id))
    }
    fn batch_cancel_request(
        &self,
        _cx: &BuildContext<'_>,
        batch_id: &str,
    ) -> Result<WireRequest, Lm15Error> {
        Ok(batch::cancel_request(batch_id))
    }
    fn batch_result_fetches(
        &self,
        _cx: &BuildContext<'_>,
        _status_body: &Map<String, Value>,
    ) -> Result<Vec<WireRequest>, Lm15Error> {
        Ok(Vec::new()) // inlined in the terminal operation
    }
    fn batch_entries(
        &self,
        cx: &BuildContext<'_>,
        status_body: &Map<String, Value>,
        _fetched: &[Vec<u8>],
    ) -> Result<Vec<crate::types::BatchEntry>, Lm15Error> {
        batch::entries(self, cx, status_body)
    }
    fn batch_list_request(
        &self,
        _cx: &BuildContext<'_>,
        limit: u64,
    ) -> Result<WireRequest, Lm15Error> {
        Ok(batch::list_request(limit))
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
        _cx: &BuildContext<'_>,
        file_id: &str,
    ) -> Result<WireRequest, Lm15Error> {
        Ok(files::get_request(file_id))
    }
    fn file_list_request(
        &self,
        _cx: &BuildContext<'_>,
        limit: u64,
        cursor: Option<&str>,
    ) -> Result<WireRequest, Lm15Error> {
        Ok(files::list_request(limit, cursor))
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
        _cx: &BuildContext<'_>,
        file_id: &str,
    ) -> Result<WireRequest, Lm15Error> {
        Ok(files::delete_request(file_id))
    }
    fn file_download_request(
        &self,
        _cx: &BuildContext<'_>,
        file_id: &str,
    ) -> Result<WireRequest, Lm15Error> {
        Ok(files::download_request(file_id))
    }
}

impl Dialect for Gemini {
    fn dialect(&self) -> DialectId {
        DialectId::Gemini
    }

    fn build(
        &self,
        request: &Request,
        stream: bool,
        cx: &BuildContext<'_>,
    ) -> Result<WireRequest, Lm15Error> {
        let body = payload(request, cx)?;
        let verb = if stream {
            "streamGenerateContent"
        } else {
            "generateContent"
        };
        let mut wire = WireRequest::post(format!("/{}:{verb}", model_path(cx.model)), body);
        if stream {
            wire.params.push(("alt".into(), "sse".into()));
        }
        wire.endpoint = Some(ENDPOINT);
        wire.model = Some(cx.model.to_string());
        Ok(wire)
    }

    fn api_key_header(&self) -> &'static str {
        "x-goog-api-key"
    }

    /// `gemini.py:1419-1446`: `GET /models?pageSize=1000` (53 models, no
    /// `nextPageToken` live 2026-08-31), entries under `models`, the id is
    /// the wire `name` with `models/` stripped (`build` re-prefixes). No
    /// `content-type`: the reference's `_auth_headers()` carries none.
    fn models_request(&self, _cx: &BuildContext<'_>) -> Result<WireRequest, Lm15Error> {
        let mut wire = WireRequest::get("/models");
        wire.params.push(("pageSize".into(), "1000".into()));
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
        Ok(model_infos_from_entries(
            data.get("models"),
            cx.provider,
            "gemini_generate_content",
            |entry| {
                let name = entry.get("name").and_then(Value::as_str)?;
                Some(name.strip_prefix("models/").unwrap_or(name).to_string())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_class_is_the_gemini_3_name_table() {
        assert!(gemini_level_class("gemini-3.7-flash"));
        assert!(gemini_level_class("models/Gemini-3-pro"));
        assert!(!gemini_level_class("gemini-2.5-flash"));
        assert!(!gemini_level_class("gemini-flash-latest"));
    }

    #[test]
    fn model_path_prefixes_and_quotes() {
        assert_eq!(model_path("gemini-2.5-flash"), "models/gemini-2.5-flash");
        assert_eq!(
            model_path("models/gemini-2.5-flash"),
            "models/gemini-2.5-flash"
        );
        assert_eq!(model_path("tunedModels/a b"), "models/tunedModels/a%20b");
    }
}

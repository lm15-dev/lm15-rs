//! grok-imagine video on the chat dialect (`xai.py:173-250`, captured
//! live 2026-09-01): `POST /videos/generations` → `{request_id}`;
//! `GET /videos/{id}` → pending + progress, then done + a PUBLIC MP4
//! URL — URL-addressed, no fetch step; no list endpoint exists.

use serde_json::{Map, Value};

use crate::errors::Lm15Error;
use crate::surfaces::{body_object, provider_error, str_field, unsupported};
use crate::types::{VideoGenerationRequest, VideoJobInfo, VideoPart, VideoStatus};
use crate::wire::{BuildContext, WireRequest};

use super::generation::json_headers;
use crate::cloud::percent::path_id;

pub fn submit_request(
    cx: &BuildContext<'_>,
    request: &VideoGenerationRequest,
) -> Result<WireRequest, Lm15Error> {
    if request.seconds.is_some() {
        return Err(unsupported(cx.provider, "video duration (no wire slot)"));
    }
    if !request.images.is_empty() {
        // The generation wire silently IGNORES unknown fields
        // (pixel-verified 2026-09-01): an unverified image-input mapping
        // could silently produce prompt-only videos.
        let mut err = unsupported(cx.provider, "video input images");
        err.meta_mut().message = format!(
            "{}: video input images are not mapped yet; use extensions until the mapping is \
             live-receipted",
            cx.provider
        );
        return Err(err);
    }
    let mut payload = Map::new();
    payload.insert("model".into(), Value::String(request.model.clone()));
    payload.insert("prompt".into(), Value::String(request.prompt.clone()));
    if let Some(extensions) = &request.extensions {
        payload.extend(extensions.clone());
    }
    let mut wire = WireRequest::post("/videos/generations", Value::Object(payload));
    wire.headers = json_headers(cx);
    Ok(wire)
}

pub fn job_from_body(
    cx: &BuildContext<'_>,
    body: &[u8],
    video_id: Option<&str>,
) -> Result<VideoJobInfo, Lm15Error> {
    let data = body_object(cx.provider, body, "video")?;
    if let Some(request_id) = str_field(&data, "request_id") {
        // The submit acknowledgement: a bare ticket, not yet started.
        return Ok(VideoJobInfo {
            id: request_id,
            status: VideoStatus::Queued,
            progress: None,
            created_at: None,
            model: None,
            provider_data: Some(data),
        });
    }
    let video_id = video_id
        .ok_or_else(|| provider_error(cx.provider, "video body carries no request_id".into()))?;
    let wire_status = data.get("status").and_then(Value::as_str).unwrap_or("");
    let status = match wire_status {
        "pending" => VideoStatus::Running,
        "done" => VideoStatus::Completed,
        "failed" => VideoStatus::Failed,
        other => {
            return Err(provider_error(
                cx.provider,
                format!("unknown video status {other:?}"),
            ))
        }
    };
    Ok(VideoJobInfo {
        id: video_id.to_string(),
        status,
        progress: data
            .get("progress")
            .and_then(|p| p.as_u64().or_else(|| p.as_f64().map(|f| f as u64))),
        created_at: None,
        model: str_field(&data, "model"),
        provider_data: Some(data),
    })
}

pub fn status_request(cx: &BuildContext<'_>, video_id: &str) -> WireRequest {
    let mut wire = WireRequest::get(format!("/videos/{}", path_id(video_id, false)));
    wire.headers = json_headers(cx);
    wire
}

pub fn part(
    cx: &BuildContext<'_>,
    status_body: &Map<String, Value>,
) -> Result<VideoPart, Lm15Error> {
    let url = status_body
        .get("video")
        .and_then(Value::as_object)
        .and_then(|v| str_field(v, "url"))
        .ok_or_else(|| provider_error(cx.provider, "terminal video carries no url".into()))?;
    Ok(VideoPart {
        media_type: "video/mp4".into(),
        url: Some(url),
        ..Default::default()
    })
}

pub fn list_unsupported(cx: &BuildContext<'_>) -> Lm15Error {
    let mut err = unsupported(cx.provider, "video list");
    err.meta_mut().message = format!(
        "{}: the wire has no video list endpoint (probed 2026-09-01: 404) — the ticket you \
         stored is the only copy",
        cx.provider
    );
    err
}

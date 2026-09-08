//! Sora (`openai.py:1870-1905`, captured live 2026-09-01): jobs at
//! `/videos`, `queued` → `in_progress` → `completed`, then
//! `/videos/{id}/content` streams the MP4 (type from `content-type`).

use serde_json::{Map, Value};

use crate::errors::Lm15Error;
use crate::surfaces::{body_object, header, iso_utc, provider_error, str_field, unsupported};
use crate::types::{VideoGenerationRequest, VideoJobInfo, VideoPart, VideoStatus};
use crate::wire::{BuildContext, WireRequest};

use super::files::json_headers;

fn status(wire: &str) -> Option<VideoStatus> {
    Some(match wire {
        "queued" => VideoStatus::Queued,
        "in_progress" => VideoStatus::Running,
        "completed" => VideoStatus::Completed,
        "failed" => VideoStatus::Failed,
        "cancelled" => VideoStatus::Cancelled,
        _ => return None,
    })
}

pub fn submit_request(
    cx: &BuildContext<'_>,
    request: &VideoGenerationRequest,
) -> Result<WireRequest, Lm15Error> {
    if !request.images.is_empty() {
        // Sora's image input (input_reference) is a multipart upload; the
        // mapping is unverified against the live wire. Raising beats a guess.
        let mut err = unsupported(cx.provider, "video input images");
        err.meta_mut().message = format!(
            "{}: video input images (input_reference) are not mapped yet; use the provider \
             door until the mapping is live-receipted",
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
    if let Some(seconds) = request.seconds {
        payload.insert("seconds".into(), Value::String(seconds.to_string())); // a string enum on the wire
    }
    let mut wire = WireRequest::post("/videos", Value::Object(payload));
    wire.headers = json_headers(cx);
    Ok(wire)
}

pub fn job_info(
    cx: &BuildContext<'_>,
    data: &Map<String, Value>,
) -> Result<VideoJobInfo, Lm15Error> {
    let id = str_field(data, "id")
        .ok_or_else(|| provider_error(cx.provider, "video object carries no id".into()))?;
    let wire_status = data.get("status").and_then(Value::as_str).unwrap_or("");
    let status = status(wire_status).ok_or_else(|| {
        provider_error(cx.provider, format!("unknown video status {wire_status:?}"))
    })?;
    Ok(VideoJobInfo {
        id,
        status,
        progress: data
            .get("progress")
            .and_then(|p| p.as_u64().or_else(|| p.as_f64().map(|f| f as u64))),
        created_at: iso_utc(data.get("created_at")),
        model: str_field(data, "model"),
        provider_data: Some(data.clone()),
    })
}

pub fn job_from_body(cx: &BuildContext<'_>, body: &[u8]) -> Result<VideoJobInfo, Lm15Error> {
    job_info(cx, &body_object(cx.provider, body, "video")?)
}

pub fn status_request(cx: &BuildContext<'_>, video_id: &str) -> WireRequest {
    let mut wire = WireRequest::get(format!("/videos/{video_id}"));
    wire.headers = json_headers(cx);
    wire
}

pub fn result_fetch(cx: &BuildContext<'_>, status_body: &Map<String, Value>) -> WireRequest {
    let id = status_body.get("id").and_then(Value::as_str).unwrap_or("");
    let mut wire = WireRequest::get(format!("/videos/{id}/content"));
    wire.headers = json_headers(cx);
    wire
}

pub fn part(
    cx: &BuildContext<'_>,
    fetched: crate::wire::Fetched<'_>,
) -> Result<VideoPart, Lm15Error> {
    let (headers, body) = fetched
        .ok_or_else(|| provider_error(cx.provider, "video content fetch is required".into()))?;
    let content_type = header(headers, "content-type")
        .map(|v| v.split(';').next().unwrap_or("").trim().to_string())
        .filter(|v| !v.is_empty())
        .ok_or_else(|| {
            provider_error(cx.provider, "video content carries no content-type".into())
        })?;
    Ok(VideoPart {
        media_type: content_type,
        data: Some(crate::types::base64_encode(body)),
        ..Default::default()
    })
}

pub fn list_request(cx: &BuildContext<'_>, limit: u64) -> WireRequest {
    let mut wire = WireRequest::get("/videos");
    wire.params.push(("limit".into(), limit.to_string()));
    wire.headers = json_headers(cx);
    wire
}

pub fn jobs(cx: &BuildContext<'_>, body: &[u8]) -> Result<Vec<VideoJobInfo>, Lm15Error> {
    let data = body_object(cx.provider, body, "video list")?;
    data.get("data")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
        .map(|item| job_info(cx, item))
        .collect()
}

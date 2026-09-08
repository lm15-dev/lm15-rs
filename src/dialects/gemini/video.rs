//! Veo (`gemini.py:1720-1800`, captured live 2026-09-01):
//! `:predictLongRunning` returns an operation; poll it by name; the
//! terminal body carries a KEY-BOUND file URI (403 without the header,
//! verified live), so the fetch travels through auth.

use serde_json::{json, Map, Value};

use crate::errors::Lm15Error;
use crate::surfaces::{body_object, header, provider_error, str_field, unsupported};
use crate::types::{VideoGenerationRequest, VideoJobInfo, VideoPart, VideoStatus};
use crate::wire::{wire_model, BuildContext, WireRequest};

use super::model_path;

pub fn submit_request(
    cx: &BuildContext<'_>,
    request: &VideoGenerationRequest,
) -> Result<WireRequest, Lm15Error> {
    if !request.images.is_empty() {
        let mut err = unsupported(cx.provider, "video input images");
        err.meta_mut().message = format!(
            "{}: video input images are not mapped yet; use extensions until the mapping is \
             live-receipted",
            cx.provider
        );
        return Err(err);
    }
    let mut payload = Map::new();
    payload.insert("instances".into(), json!([{"prompt": request.prompt}]));
    if let Some(extensions) = &request.extensions {
        payload.extend(extensions.clone());
    }
    if let Some(seconds) = request.seconds {
        payload
            .entry("parameters")
            .or_insert_with(|| json!({"durationSeconds": seconds}));
    }
    let path = format!(
        "/{}:predictLongRunning",
        model_path(wire_model(cx.provider, &request.model))
    );
    let mut wire = WireRequest::post(path, Value::Object(payload));
    wire.headers
        .push(("Content-Type".into(), "application/json".into()));
    Ok(wire)
}

pub fn job_info(
    cx: &BuildContext<'_>,
    data: &Map<String, Value>,
) -> Result<VideoJobInfo, Lm15Error> {
    let id = str_field(data, "name")
        .ok_or_else(|| provider_error(cx.provider, "video operation carries no name".into()))?;
    let status = if data.get("done").and_then(Value::as_bool) == Some(true) {
        if data.get("error").is_some_and(Value::is_object) {
            VideoStatus::Failed
        } else {
            VideoStatus::Completed
        }
    } else {
        VideoStatus::Running // operations expose no queued/running distinction before done
    };
    Ok(VideoJobInfo {
        id,
        status,
        progress: None,
        created_at: None,
        model: None,
        provider_data: Some(data.clone()),
    })
}

pub fn job_from_body(cx: &BuildContext<'_>, body: &[u8]) -> Result<VideoJobInfo, Lm15Error> {
    job_info(cx, &body_object(cx.provider, body, "video")?)
}

pub fn status_request(video_id: &str) -> WireRequest {
    WireRequest::get(format!("/{video_id}"))
}

fn result_uri(
    cx: &BuildContext<'_>,
    status_body: &Map<String, Value>,
) -> Result<String, Lm15Error> {
    status_body
        .get("response")
        .and_then(Value::as_object)
        .and_then(|r| r.get("generateVideoResponse"))
        .and_then(Value::as_object)
        .and_then(|g| g.get("generatedSamples"))
        .and_then(Value::as_array)
        .and_then(|s| s.first())
        .and_then(Value::as_object)
        .and_then(|s| s.get("video"))
        .and_then(Value::as_object)
        .and_then(|v| str_field(v, "uri"))
        .ok_or_else(|| {
            provider_error(
                cx.provider,
                "terminal video operation carries no video uri".into(),
            )
        })
}

pub fn result_fetch(
    cx: &BuildContext<'_>,
    status_body: &Map<String, Value>,
) -> Result<WireRequest, Lm15Error> {
    let mut wire = WireRequest::get("");
    wire.absolute_url = Some(result_uri(cx, status_body)?);
    Ok(wire)
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
            provider_error(cx.provider, "video download carries no content-type".into())
        })?;
    Ok(VideoPart {
        media_type: content_type,
        data: Some(crate::types::base64_encode(body)),
        ..Default::default()
    })
}

pub fn list_request(
    cx: &BuildContext<'_>,
    limit: u64,
    model: Option<&str>,
) -> Result<WireRequest, Lm15Error> {
    let Some(model) = model.filter(|m| !m.is_empty()) else {
        let mut err = unsupported(cx.provider, "a video list without a model");
        err.meta_mut().message = format!(
            "{}: video jobs list per model — pass model= (operations live under \
             models/<model>/operations)",
            cx.provider
        );
        return Err(err);
    };
    let mut wire = WireRequest::get(format!(
        "/{}/operations",
        model_path(wire_model(cx.provider, model))
    ));
    wire.params.push(("pageSize".into(), limit.to_string()));
    Ok(wire)
}

pub fn jobs(cx: &BuildContext<'_>, body: &[u8]) -> Result<Vec<VideoJobInfo>, Lm15Error> {
    let data = body_object(cx.provider, body, "video list")?;
    data.get("operations")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
        .map(|op| job_info(cx, op))
        .collect()
}

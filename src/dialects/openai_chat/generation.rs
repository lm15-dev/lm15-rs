//! xAI image generation on the chat dialect (`xai.py:127-163`, captured
//! live 2026-09-01): `/images/generations` and `/images/edits` are JSON;
//! `size` has no wire slot; edits take exactly one image as
//! `image:{url|file_id}` (inline data as a data URL).

use serde_json::{json, Map, Value};

use crate::errors::Lm15Error;
use crate::surfaces::{body_object, media_bytes, provider_error, str_field, unsupported};
use crate::types::{ImageGenerationRequest, ImageGenerationResponse, ImagePart, Usage};
use crate::wire::{apply_static_headers, BuildContext, WireRequest};

fn json_headers(cx: &BuildContext<'_>) -> Vec<(String, String)> {
    let mut headers = vec![("content-type".to_string(), "application/json".to_string())];
    apply_static_headers(&mut headers, cx.policy);
    headers
}

fn image_input(cx: &BuildContext<'_>, part: &ImagePart) -> Result<Value, Lm15Error> {
    if let Some(url) = &part.url {
        return Ok(json!({"url": url}));
    }
    if let Some(file_id) = &part.file_id {
        return Ok(json!({"file_id": file_id}));
    }
    if let Some(data) = &part.data {
        return Ok(json!({"url": format!("data:{};base64,{data}", part.media_type)}));
    }
    let bytes = media_bytes(cx.provider, None, part.path.as_deref(), "input image")?;
    Ok(json!({"url": format!("data:{};base64,{}", part.media_type, crate::types::base64_encode(&bytes))}))
}

pub fn image_request(cx: &BuildContext<'_>, request: &ImageGenerationRequest) -> Result<WireRequest, Lm15Error> {
    let mut payload = Map::new();
    payload.insert("model".into(), Value::String(request.model.clone()));
    payload.insert("prompt".into(), Value::String(request.prompt.clone()));
    if let Some(extensions) = &request.extensions {
        payload.extend(extensions.clone());
    }
    if request.size.is_some() {
        // No wire slot: xAI sizes through quality/resolution knobs with
        // their own names (extensions). Raising beats guessing a mapping.
        let mut err = unsupported(cx.provider, "size");
        err.meta_mut().message = format!(
            "{}: size has no wire slot; use extensions for xAI's quality/resolution fields",
            cx.provider
        );
        return Err(err);
    }
    if request.images.is_empty() {
        let mut wire = WireRequest::post("/images/generations", Value::Object(payload));
        wire.headers = json_headers(cx);
        return Ok(wire);
    }
    if request.images.len() > 1 {
        let mut err = unsupported(cx.provider, "more than one input image");
        err.meta_mut().message = format!(
            "{}: image edits take exactly one input image; the wire has no slot for more",
            cx.provider
        );
        return Err(err);
    }
    payload.insert("image".into(), image_input(cx, &request.images[0])?);
    let mut wire = WireRequest::post("/images/edits", Value::Object(payload));
    wire.headers = json_headers(cx);
    Ok(wire)
}

pub fn image_response(cx: &BuildContext<'_>, body: &[u8]) -> Result<ImageGenerationResponse, Lm15Error> {
    let data = body_object(cx.provider, body, "image")?;
    let mut images = Vec::new();
    for item in data.get("data").and_then(Value::as_array).into_iter().flatten() {
        let Some(item) = item.as_object() else { continue };
        let media_type = str_field(item, "mime_type").unwrap_or_else(|| "application/octet-stream".into());
        if let Some(b64) = str_field(item, "b64_json") {
            images.push(ImagePart {
                media_type,
                data: Some(b64),
                ..Default::default()
            });
        } else if let Some(url) = str_field(item, "url") {
            images.push(ImagePart {
                media_type,
                url: Some(url),
                ..Default::default()
            });
        }
    }
    if images.is_empty() {
        return Err(provider_error(cx.provider, "image response carries no images".into()));
    }
    // Captured: usage reports cost_in_usd_ticks only — no token counts
    // exist, so Usage stays empty and the figure lives in provider_data.
    Ok(ImageGenerationResponse {
        images,
        text: None,
        id: None,
        model: None,
        usage: Usage::default(),
        provider_data: Some(data),
    })
}

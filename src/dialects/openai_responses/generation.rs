//! OpenAI image and speech generation (`openai.py:1500-1560`), shared by
//! every OpenAI-shaped door (Azure OpenAI v1, Meta): `/images/generations`
//! (JSON), `/images/edits` (multipart — the wire takes uploaded bytes
//! only; the field name is the compat's `edit_image_field`),
//! `/audio/speech` (raw media back, its type in `content-type`).

use serde_json::{Map, Value};

use crate::compat::OpenAIResponsesEditImageField;
use crate::errors::{ErrorMeta, Lm15Error};
use crate::surfaces::{
    body_object, header, media_bytes, multipart_form_body, provider_error, str_field, unsupported,
    FilePart,
};
use crate::types::{
    AudioPart, ImageGenerationRequest, ImageGenerationResponse, ImagePart, SpeechGenerationRequest,
    SpeechGenerationResponse, Usage,
};
use crate::wire::{apply_static_headers, BuildContext, WireRequest};

use super::files::json_headers;
use super::payload::resolve_compat;

fn extension_string(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

pub fn image_request(
    cx: &BuildContext<'_>,
    request: &ImageGenerationRequest,
) -> Result<WireRequest, Lm15Error> {
    if request.images.is_empty() {
        let mut payload = Map::new();
        payload.insert("model".into(), Value::String(request.model.clone()));
        payload.insert("prompt".into(), Value::String(request.prompt.clone()));
        if let Some(size) = &request.size {
            payload.insert("size".into(), Value::String(size.clone()));
        }
        if let Some(extensions) = &request.extensions {
            for (k, v) in extensions {
                if !v.is_null() {
                    payload.insert(k.clone(), v.clone());
                }
            }
        }
        let mut wire = WireRequest::post("/images/generations", Value::Object(payload));
        wire.headers = json_headers(cx);
        return Ok(wire);
    }
    // Edits are multipart: the wire takes uploaded bytes only.
    for part in &request.images {
        if part.data.is_none() && part.path.is_none() {
            let mut err = unsupported(cx.provider, "url/file_id-addressed input images");
            err.meta_mut().message = format!(
                "{}: image edits take inline data or a local path; url/file_id-addressed \
                 input images have no wire slot",
                cx.provider
            );
            return Err(err);
        }
    }
    let synthetic = crate::wire::batch_entry_request(Some(&request.model));
    let compat = resolve_compat(cx.provider, &synthetic, cx.compat.openai_responses())?;
    let mut fields = vec![
        ("model".to_string(), request.model.clone()),
        ("prompt".to_string(), request.prompt.clone()),
    ];
    if let Some(size) = &request.size {
        fields.push(("size".into(), size.clone()));
    }
    if let Some(extensions) = &request.extensions {
        for (k, v) in extensions {
            fields.push((k.clone(), extension_string(v)));
        }
    }
    let datas = request
        .images
        .iter()
        .map(|part| {
            media_bytes(
                cx.provider,
                part.data.as_deref(),
                part.path.as_deref(),
                "input image",
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let names: Vec<(String, String)> = request
        .images
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let field = match compat.edit_image_field {
                OpenAIResponsesEditImageField::Indexed => format!("image[{i}]"),
                OpenAIResponsesEditImageField::Array => "image[]".to_string(),
            };
            (field, format!("image-{i}"))
        })
        .collect();
    let files: Vec<FilePart<'_>> = request
        .images
        .iter()
        .zip(&datas)
        .zip(&names)
        .map(|((part, data), (field, filename))| FilePart {
            field,
            filename,
            content_type: &part.media_type,
            data,
        })
        .collect();
    let (content_type, body) = multipart_form_body(&fields, &files);
    let mut wire = WireRequest::with_raw("POST", "/images/edits", content_type, body);
    apply_static_headers(&mut wire.headers, cx.policy);
    Ok(wire)
}

pub fn image_response(
    cx: &BuildContext<'_>,
    body: &[u8],
) -> Result<ImageGenerationResponse, Lm15Error> {
    let data = body_object(cx.provider, body, "image")?;
    let media_type = str_field(&data, "output_format").map(|f| format!("image/{f}"));
    let mut images = Vec::new();
    for item in data
        .get("data")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let Some(item) = item.as_object() else {
            continue;
        };
        let media = media_type
            .clone()
            .unwrap_or_else(|| "application/octet-stream".into());
        if let Some(b64) = str_field(item, "b64_json") {
            images.push(ImagePart {
                media_type: media,
                data: Some(b64),
                ..Default::default()
            });
        } else if let Some(url) = str_field(item, "url") {
            images.push(ImagePart {
                media_type: media,
                url: Some(url),
                ..Default::default()
            });
        }
    }
    let usage_obj = data.get("usage").and_then(Value::as_object);
    let usage = Usage {
        input_tokens: usage_obj
            .and_then(|u| u.get("input_tokens"))
            .and_then(Value::as_u64),
        output_tokens: usage_obj
            .and_then(|u| u.get("output_tokens"))
            .and_then(Value::as_u64),
        total_tokens: usage_obj
            .and_then(|u| u.get("total_tokens"))
            .and_then(Value::as_u64),
        ..Default::default()
    };
    // Captured: the images response carries no id and no model echo.
    Ok(ImageGenerationResponse {
        images,
        text: None,
        id: None,
        model: None,
        usage,
        provider_data: Some(data),
    })
}

pub fn speech_request(cx: &BuildContext<'_>, request: &SpeechGenerationRequest) -> WireRequest {
    let mut payload = Map::new();
    payload.insert("model".into(), Value::String(request.model.clone()));
    payload.insert("input".into(), Value::String(request.prompt.clone()));
    if let Some(extensions) = &request.extensions {
        payload.extend(extensions.clone());
    }
    if let Some(voice) = &request.voice {
        payload.insert("voice".into(), Value::String(voice.clone()));
    }
    if let Some(format) = &request.format {
        payload.insert("response_format".into(), Value::String(format.clone()));
    }
    let mut wire = WireRequest::post("/audio/speech", Value::Object(payload));
    wire.headers = json_headers(cx);
    wire
}

pub fn speech_response(
    cx: &BuildContext<'_>,
    headers: &[(String, String)],
    body: &[u8],
) -> Result<SpeechGenerationResponse, Lm15Error> {
    let content_type = header(headers, "content-type")
        .map(|v| v.split(';').next().unwrap_or("").trim().to_string())
        .filter(|v| !v.is_empty())
        .ok_or_else(|| {
            provider_error(
                cx.provider,
                "speech response carries no content-type".into(),
            )
        })?;
    let audio = AudioPart {
        media_type: content_type.clone(),
        data: Some(crate::types::base64_encode(body)),
        ..Default::default()
    };
    let mut provider_data = Map::new();
    provider_data.insert("content_type".into(), Value::String(content_type));
    // The body is raw media: no usage, no id, no model echo exist.
    Ok(SpeechGenerationResponse {
        audio,
        id: None,
        model: None,
        usage: Usage::default(),
        provider_data: Some(provider_data),
    })
}

#[allow(dead_code)]
fn _unused(_: ErrorMeta) {}

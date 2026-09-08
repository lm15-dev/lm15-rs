//! The Anthropic Files API (`anthropic.py:1004-1073`; GA, no beta
//! header): multipart upload, `page` cursors, `downloadable` per file.

use serde_json::{Map, Value};

use crate::errors::Lm15Error;
use crate::surfaces::{
    body_object, iso_utc, multipart_form_body, provider_error, str_field, FilePart,
};
use crate::types::{FileInfo, FilePage, FileReadiness, FileUploadRequest};
use crate::wire::{BuildContext, WireRequest};

use super::surface_headers;

fn json_headers(cx: &BuildContext<'_>) -> Vec<(String, String)> {
    let mut headers = surface_headers(cx.policy);
    headers.push(("content-type".into(), "application/json".into()));
    headers
}

pub fn upload_request(
    cx: &BuildContext<'_>,
    request: &FileUploadRequest,
) -> Result<WireRequest, Lm15Error> {
    let fields: Vec<(String, String)> = request
        .extensions
        .iter()
        .flat_map(|e| e.iter())
        .map(|(k, v)| {
            (
                k.clone(),
                match v {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                },
            )
        })
        .collect();
    let data = request.content()?;
    let (content_type, body) = multipart_form_body(
        &fields,
        &[FilePart {
            field: "file",
            filename: &request.filename,
            content_type: &request.media_type,
            data: &data,
        }],
    );
    let mut wire = WireRequest::with_raw("POST", "/files", content_type.clone(), body);
    // `_headers()` then `content-type` replaced: the static headers first.
    let mut headers = surface_headers(cx.policy);
    headers.push(("content-type".into(), content_type));
    wire.headers = headers;
    Ok(wire)
}

pub fn file_info(cx: &BuildContext<'_>, data: &Map<String, Value>) -> Result<FileInfo, Lm15Error> {
    let id = str_field(data, "id")
        .ok_or_else(|| provider_error(cx.provider, "file object carries no id".into()))?;
    Ok(FileInfo {
        id,
        filename: str_field(data, "filename"),
        media_type: str_field(data, "mime_type"),
        size_bytes: data.get("size_bytes").and_then(Value::as_u64),
        created_at: iso_utc(data.get("created_at")),
        expires_at: iso_utc(data.get("expires_at")),
        readiness: FileReadiness::Ready, // Anthropic files have no processing state
        downloadable: data.get("downloadable").and_then(Value::as_bool),
        provider_data: Some(data.clone()),
    })
}

pub fn file_info_from_body(cx: &BuildContext<'_>, body: &[u8]) -> Result<FileInfo, Lm15Error> {
    file_info(cx, &body_object(cx.provider, body, "file")?)
}

pub fn get_request(cx: &BuildContext<'_>, file_id: &str) -> WireRequest {
    let mut wire = WireRequest::get(format!("/files/{file_id}"));
    wire.headers = json_headers(cx);
    wire
}

pub fn list_request(cx: &BuildContext<'_>, limit: u64, cursor: Option<&str>) -> WireRequest {
    let mut wire = WireRequest::get("/files");
    wire.params.push(("limit".into(), limit.to_string()));
    if let Some(cursor) = cursor {
        wire.params.push(("page".into(), cursor.to_string()));
    }
    wire.headers = json_headers(cx);
    wire
}

pub fn page(cx: &BuildContext<'_>, body: &[u8]) -> Result<FilePage, Lm15Error> {
    let data = body_object(cx.provider, body, "file list")?;
    let mut items = Vec::new();
    if let Some(Value::Array(entries)) = data.get("data") {
        for entry in entries.iter().filter_map(Value::as_object) {
            items.push(file_info(cx, entry)?);
        }
    }
    Ok(FilePage {
        items,
        next_cursor: str_field(&data, "next_page"),
    })
}

pub fn delete_request(cx: &BuildContext<'_>, file_id: &str) -> WireRequest {
    let mut wire = get_request(cx, file_id);
    wire.method = "DELETE".into();
    wire
}

pub fn download_request(cx: &BuildContext<'_>, file_id: &str) -> WireRequest {
    let mut wire = WireRequest::get(format!("/files/{file_id}/content"));
    wire.headers = json_headers(cx);
    wire
}

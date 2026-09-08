//! The OpenAI Files API (`openai.py:1666-1735`), shared with every
//! OpenAI-shaped files door (Azure OpenAI v1, Meta) and the chat dialect.

use serde_json::{Map, Value};

use crate::errors::Lm15Error;
use crate::surfaces::{
    body_object, iso_utc, multipart_form_body, openai_file_readiness, provider_error, str_field,
    u64_field, FilePart,
};
use crate::types::{FileInfo, FilePage, FileUploadRequest};
use crate::wire::{apply_static_headers, BuildContext, WireRequest};

/// `_headers()`: content type, then the policy's static headers.
pub fn json_headers(cx: &BuildContext<'_>) -> Vec<(String, String)> {
    let mut headers = vec![("content-type".to_string(), "application/json".to_string())];
    apply_static_headers(&mut headers, cx.policy);
    headers
}

pub fn upload_request(
    cx: &BuildContext<'_>,
    request: &FileUploadRequest,
) -> Result<WireRequest, Lm15Error> {
    // `purpose` from extensions (default `user_data`), then the rest as
    // form fields in order.
    let mut fields = Vec::new();
    let mut purpose = "user_data".to_string();
    let mut extra = Vec::new();
    if let Some(extensions) = &request.extensions {
        for (key, value) in extensions {
            let text = match value {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            if key == "purpose" {
                purpose = text;
            } else {
                extra.push((key.clone(), text));
            }
        }
    }
    fields.push(("purpose".to_string(), purpose));
    fields.extend(extra);
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
    let mut wire = WireRequest::with_raw("POST", "/files", content_type, body);
    apply_static_headers(&mut wire.headers, cx.policy);
    Ok(wire)
}

pub fn file_info(cx: &BuildContext<'_>, data: &Map<String, Value>) -> Result<FileInfo, Lm15Error> {
    let id = str_field(data, "id")
        .ok_or_else(|| provider_error(cx.provider, "file object carries no id".into()))?;
    Ok(FileInfo {
        id,
        filename: str_field(data, "filename"),
        media_type: None, // OpenAI file metadata reports no MIME type
        size_bytes: data.get("bytes").and_then(Value::as_u64),
        created_at: iso_utc(data.get("created_at")),
        expires_at: iso_utc(data.get("expires_at")),
        readiness: openai_file_readiness(data.get("status")),
        downloadable: None, // purpose-dependent policy, not reported per file
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
        wire.params.push(("after".into(), cursor.to_string()));
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
    let has_more = data
        .get("has_more")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let next_cursor = if has_more && !items.is_empty() {
        str_field(&data, "last_id")
    } else {
        None
    };
    Ok(FilePage { items, next_cursor })
}

pub fn delete_request(cx: &BuildContext<'_>, file_id: &str) -> WireRequest {
    let mut wire = WireRequest::get(format!("/files/{file_id}"));
    wire.method = "DELETE".into();
    wire.headers = json_headers(cx);
    wire
}

pub fn download_request(cx: &BuildContext<'_>, file_id: &str) -> WireRequest {
    let mut wire = WireRequest::get(format!("/files/{file_id}/content"));
    wire.headers = json_headers(cx);
    wire
}

#[allow(dead_code)]
fn _size(data: &Map<String, Value>) -> Option<u64> {
    u64_field(data, "bytes")
}

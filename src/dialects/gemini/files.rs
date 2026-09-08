//! The Gemini Files API (`gemini.py:1464-1568`): a multipart/related
//! upload on the `/upload/v1beta` host, `files/<id>` resources, `state`
//! folded to readiness, `downloadUri` / `source` to `downloadable`.

use serde_json::{json, Map, Value};

use crate::errors::Lm15Error;
use crate::surfaces::{body_object, iso_utc, multipart_related_body, provider_error, str_field, u64_field};
use crate::types::{FileInfo, FilePage, FileReadiness, FileUploadRequest};
use crate::wire::{full_url, BuildContext, WireRequest};

/// `files/<id>` from a canonical id (a URI, a resource name, or a bare id).
pub fn file_resource(file_id: &str) -> String {
    if file_id.contains("://") {
        let trimmed = file_id.trim_end_matches('/');
        if let Some((_, tail)) = trimmed.rsplit_once("/files/") {
            if !tail.is_empty() {
                return format!("files/{tail}");
            }
        }
    }
    if file_id.starts_with("files/") {
        return file_id.to_string();
    }
    format!("files/{file_id}")
}

/// The reference's `upload_base_url` (`gemini.py:400`,
/// `https://generativelanguage.googleapis.com/upload/v1beta`): the base
/// URL with `/upload` in front of its path, so a proxy base URL uploads
/// through the same host.
pub fn upload_base_url(base_url: &str) -> String {
    let base = base_url.trim_end_matches('/');
    match base.find("://").and_then(|i| base[i + 3..].find('/').map(|j| i + 3 + j)) {
        Some(path_start) => format!("{}/upload{}", &base[..path_start], &base[path_start..]),
        None => format!("{base}/upload"),
    }
}

pub fn upload_request(cx: &BuildContext<'_>, request: &FileUploadRequest) -> Result<WireRequest, Lm15Error> {
    let data = request.content()?;
    let (content_type, body) = multipart_related_body(
        &json!({"file": {"display_name": request.filename}}),
        &request.media_type,
        &data,
    );
    let mut wire = WireRequest::with_raw("POST", "/files", content_type.clone(), body);
    // `build_url(url, request.extensions)`: extensions ride as query params.
    let params: Vec<(String, String)> = request
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
    wire.absolute_url = Some(full_url(&format!("{}/files", upload_base_url(cx.base_url)), &params));
    wire.headers = vec![
        ("x-goog-upload-protocol".into(), "multipart".into()),
        ("content-type".into(), content_type),
    ];
    Ok(wire)
}

pub fn file_info(cx: &BuildContext<'_>, data: &Map<String, Value>) -> Result<FileInfo, Lm15Error> {
    let id = str_field(data, "uri")
        .or_else(|| str_field(data, "name"))
        .ok_or_else(|| provider_error(cx.provider, "file object carries no uri or name".into()))?;
    let state = data.get("state").and_then(Value::as_str).unwrap_or("");
    let readiness = if state.ends_with("PROCESSING") {
        FileReadiness::Pending
    } else if state.ends_with("FAILED") {
        FileReadiness::Failed
    } else {
        FileReadiness::Ready // ACTIVE, absent, or unknown
    };
    let downloadable = if str_field(data, "downloadUri").is_some() {
        Some(true)
    } else if data.get("source").and_then(Value::as_str) == Some("UPLOADED") {
        Some(false) // the server's stated rule: only GENERATED files download
    } else {
        None
    };
    Ok(FileInfo {
        id,
        filename: str_field(data, "displayName"),
        media_type: str_field(data, "mimeType"),
        size_bytes: u64_field(data, "sizeBytes"), // int64 as a string on the wire
        created_at: iso_utc(data.get("createTime")),
        expires_at: iso_utc(data.get("expirationTime")),
        readiness,
        downloadable,
        provider_data: Some(data.clone()),
    })
}

pub fn file_info_from_body(cx: &BuildContext<'_>, body: &[u8]) -> Result<FileInfo, Lm15Error> {
    let data = body_object(cx.provider, body, "file")?;
    // The upload wraps the object under `file`; get and list do not.
    match data.get("file").and_then(Value::as_object) {
        Some(inner) => file_info(cx, inner),
        None => file_info(cx, &data),
    }
}

pub fn get_request(file_id: &str) -> WireRequest {
    WireRequest::get(format!("/{}", file_resource(file_id)))
}

pub fn list_request(limit: u64, cursor: Option<&str>) -> WireRequest {
    let mut wire = WireRequest::get("/files");
    wire.params.push(("pageSize".into(), limit.to_string()));
    if let Some(cursor) = cursor {
        wire.params.push(("pageToken".into(), cursor.to_string()));
    }
    wire
}

pub fn page(cx: &BuildContext<'_>, body: &[u8]) -> Result<FilePage, Lm15Error> {
    let data = body_object(cx.provider, body, "file list")?;
    let mut items = Vec::new();
    if let Some(Value::Array(entries)) = data.get("files") {
        for entry in entries.iter().filter_map(Value::as_object) {
            items.push(file_info(cx, entry)?);
        }
    }
    Ok(FilePage {
        items,
        next_cursor: str_field(&data, "nextPageToken"),
    })
}

pub fn delete_request(file_id: &str) -> WireRequest {
    let mut wire = get_request(file_id);
    wire.method = "DELETE".into();
    wire
}

pub fn download_request(file_id: &str) -> WireRequest {
    let mut wire = WireRequest::get(format!("/{}:download", file_resource(file_id)));
    wire.params.push(("alt".into(), "media".into()));
    wire
}

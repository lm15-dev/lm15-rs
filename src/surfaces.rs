//! Shared pieces of the endpoint surfaces (modules 7–8: files, batch,
//! cache, generation, video): the multipart encoders of
//! `lm15/providers/common.py`, byte-for-byte (the harness compares
//! multipart bodies with only the boundary normalized), the timestamp
//! fold, and the readiness tables copied as data.

use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::auth::{format_rfc3339, parse_rfc3339};
use crate::errors::{ErrorMeta, Lm15Error};
use crate::types::FileReadiness;

/// `lm15-<32 hex>`: the reference's `f"lm15-{uuid.uuid4().hex}"`. The
/// hex is a SHA-256 over the clock, the process and a counter — the
/// boundary is the one legitimately random byte sequence in a request,
/// and it only has to be unique, not unguessable.
pub fn boundary() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    let digest = Sha256::new()
        .chain_update(nanos.to_le_bytes())
        .chain_update(std::process::id().to_le_bytes())
        .chain_update(count.to_le_bytes())
        .finalize();
    let hex: String = digest.iter().take(16).map(|b| format!("{b:02x}")).collect();
    format!("lm15-{hex}")
}

/// One file part of a multipart form: `(field, filename, content type, data)`.
pub struct FilePart<'a> {
    pub field: &'a str,
    pub filename: &'a str,
    pub content_type: &'a str,
    pub data: &'a [u8],
}

/// `multipart_form_body`: text fields, then file parts, CRLF framing.
/// Returns the `content-type` header value and the body.
pub fn multipart_form_body(
    fields: &[(String, String)],
    files: &[FilePart<'_>],
) -> (String, Vec<u8>) {
    let boundary = boundary();
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
        );
        body.extend_from_slice(format!("{value}\r\n").as_bytes());
    }
    for file in files {
        let safe_filename = file.filename.replace('"', "%22");
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!(
                "Content-Disposition: form-data; name=\"{}\"; filename=\"{safe_filename}\"\r\n",
                file.field
            )
            .as_bytes(),
        );
        body.extend_from_slice(format!("Content-Type: {}\r\n\r\n", file.content_type).as_bytes());
        body.extend_from_slice(file.data);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

/// `multipart_related_body` (the Gemini media upload): a JSON metadata
/// part, then one media part.
pub fn multipart_related_body(
    metadata: &Value,
    media_type: &str,
    data: &[u8],
) -> (String, Vec<u8>) {
    let boundary = boundary();
    let mut body = Vec::new();
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(b"Content-Type: application/json; charset=UTF-8\r\n\r\n");
    body.extend_from_slice(&serde_json::to_vec(metadata).expect("a JSON value serializes"));
    body.extend_from_slice(b"\r\n");
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(format!("Content-Type: {media_type}\r\n\r\n").as_bytes());
    body.extend_from_slice(data);
    body.extend_from_slice(b"\r\n");
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    (format!("multipart/related; boundary={boundary}"), body)
}

/// `iso_utc`: a provider timestamp — Unix seconds (int or float) or an
/// ISO-8601 string, any offset, up to nanosecond fractions — as canonical
/// `YYYY-MM-DDTHH:MM:SSZ`; `None` when unparseable (the raw value stays
/// in `provider_data`).
pub fn iso_utc(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::Number(n) => {
            let seconds = n.as_i64().or_else(|| n.as_f64().map(|f| f as i64))?;
            Some(format_rfc3339(seconds))
        }
        Value::String(text) if !text.trim().is_empty() => {
            let text = text.trim();
            let trimmed = trim_fraction(text);
            // `datetime.fromisoformat` accepts a date-time without an
            // offset (naive, read as UTC) and a space separator.
            let candidate = if has_offset(&trimmed) {
                trimmed
            } else {
                format!("{trimmed}Z")
            };
            let candidate = candidate.replacen(' ', "T", 1);
            parse_rfc3339(&candidate).map(format_rfc3339)
        }
        _ => None,
    }
}

fn has_offset(text: &str) -> bool {
    text.ends_with('Z')
        || text.ends_with('z')
        || text
            .rfind(['+', '-'])
            .is_some_and(|i| i > 10 && text.len() - i == 6)
}

/// Keep at most six fractional digits (the reference trims nanoseconds
/// to microseconds; the parser then truncates to seconds).
fn trim_fraction(text: &str) -> String {
    let Some(dot) = text.find('.') else {
        return text.to_string();
    };
    let digits_end = text[dot + 1..]
        .find(|c: char| !c.is_ascii_digit())
        .map(|i| dot + 1 + i)
        .unwrap_or(text.len());
    let fraction = &text[dot + 1..digits_end];
    if fraction.len() <= 6 {
        return text.to_string();
    }
    format!("{}.{}{}", &text[..dot], &fraction[..6], &text[digits_end..])
}

/// `_OPENAI_FILE_READINESS` (D6): the fold for every OpenAI-shaped file
/// object (api.openai.com, Azure OpenAI v1, Meta). Absent and unknown
/// words read as ready (the `status` field is deprecated upstream).
pub fn openai_file_readiness(status: Option<&Value>) -> FileReadiness {
    match status.and_then(Value::as_str) {
        Some("uploaded") | Some("pending") => FileReadiness::Pending,
        Some("error") | Some("failed") => FileReadiness::Failed,
        _ => FileReadiness::Ready,
    }
}

/// A provider body that is not a JSON object.
pub fn body_object(
    provider: &str,
    body: &[u8],
    what: &str,
) -> Result<Map<String, Value>, Lm15Error> {
    match serde_json::from_slice::<Value>(body) {
        Ok(Value::Object(map)) => Ok(map),
        Ok(_) => Err(provider_error(
            provider,
            format!("{what} body is not a JSON object"),
        )),
        Err(err) => Err(provider_error(
            provider,
            format!("{what} body is not JSON: {err}"),
        )),
    }
}

pub fn provider_error(provider: &str, message: String) -> Lm15Error {
    let mut meta = ErrorMeta::new(format!("{provider}: {message}"));
    meta.provider = Some(provider.to_string());
    Lm15Error::ProviderError(meta)
}

pub fn unsupported(provider: &str, what: &str) -> Lm15Error {
    let mut meta = ErrorMeta::new(format!("{provider}: {what} not supported"));
    meta.provider = Some(provider.to_string());
    Lm15Error::UnsupportedFeatureError(meta)
}

/// The bytes of a media part: inline base64 data, else the file at
/// `path` read now; a URL or file id is not content.
pub fn media_bytes(
    provider: &str,
    data: Option<&str>,
    path: Option<&std::path::Path>,
    what: &str,
) -> Result<Vec<u8>, Lm15Error> {
    if let Some(data) = data {
        return crate::types::base64_decode(data).map_err(|err| {
            Lm15Error::InvalidRequestError(ErrorMeta::new(format!(
                "{provider}: {what}: {}",
                err.message
            )))
        });
    }
    if let Some(path) = path {
        return std::fs::read(path).map_err(|err| {
            Lm15Error::ConfigurationError(ErrorMeta::new(format!(
                "{provider}: {what} at {}: {err}",
                path.display()
            )))
        });
    }
    let mut meta = ErrorMeta::new(format!("{provider}: {what} carries no content"));
    meta.provider = Some(provider.to_string());
    Err(Lm15Error::UnsupportedFeatureError(meta))
}

/// The first value of a response header, by case-insensitive name.
pub fn header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.as_str())
}

/// A string field, when present and non-empty.
pub fn str_field(map: &Map<String, Value>, key: &str) -> Option<String> {
    map.get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// An integer field (a non-negative count); a JSON string of digits is
/// accepted where the wire carries int64 as text (Gemini `sizeBytes`).
pub fn u64_field(map: &Map<String, Value>, key: &str) -> Option<u64> {
    match map.get(key)? {
        Value::Number(n) => n.as_u64(),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn iso_utc_folds_epochs_and_iso_strings() {
        assert_eq!(
            iso_utc(Some(&json!(1788215944))).as_deref(),
            Some("2026-08-31T22:39:04Z")
        );
        assert_eq!(
            iso_utc(Some(&json!(1788215944.7))).as_deref(),
            Some("2026-08-31T22:39:04Z")
        );
        assert_eq!(
            iso_utc(Some(&json!("2026-08-31T22:39:04.123456789Z"))).as_deref(),
            Some("2026-08-31T22:39:04Z")
        );
        assert_eq!(
            iso_utc(Some(&json!("2026-08-31T23:39:04+01:00"))).as_deref(),
            Some("2026-08-31T22:39:04Z")
        );
        assert_eq!(
            iso_utc(Some(&json!("2026-08-31T22:39:04"))).as_deref(),
            Some("2026-08-31T22:39:04Z")
        );
        assert_eq!(iso_utc(Some(&json!(null))), None);
        assert_eq!(iso_utc(Some(&json!(true))), None);
        assert_eq!(iso_utc(Some(&json!("soon"))), None);
        assert_eq!(iso_utc(None), None);
    }

    #[test]
    fn multipart_form_matches_the_reference_framing() {
        let (content_type, body) = multipart_form_body(
            &[("purpose".into(), "user_data".into())],
            &[FilePart {
                field: "file",
                filename: "a \"b\".txt",
                content_type: "text/plain",
                data: b"hi",
            }],
        );
        let boundary = content_type
            .strip_prefix("multipart/form-data; boundary=")
            .unwrap();
        assert!(boundary.starts_with("lm15-") && boundary.len() == 5 + 32);
        let text = String::from_utf8(body).unwrap().replace(boundary, "B");
        assert_eq!(
            text,
            "--B\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nuser_data\r\n\
             --B\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a %22b%22.txt\"\r\n\
             Content-Type: text/plain\r\n\r\nhi\r\n--B--\r\n"
        );
        assert_ne!(boundary_of(&multipart_form_body(&[], &[]).0), boundary);
    }

    fn boundary_of(content_type: &str) -> &str {
        content_type.split("boundary=").nth(1).unwrap()
    }

    #[test]
    fn multipart_related_matches_the_reference_framing() {
        let (content_type, body) = multipart_related_body(
            &json!({"file": {"display_name": "x"}}),
            "image/png",
            b"\x89PNG",
        );
        let boundary = boundary_of(&content_type).to_string();
        let text = String::from_utf8_lossy(&body).replace(&boundary, "B");
        assert_eq!(
            text,
            "--B\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n{\"file\":{\"display_name\":\"x\"}}\r\n\
             --B\r\nContent-Type: image/png\r\n\r\n\u{fffd}PNG\r\n--B--\r\n"
        );
    }
}

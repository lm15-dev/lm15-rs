//! The OpenAI Batch API over `/v1/responses` (`openai.py:1745-1868`):
//! two-step — the requests travel as an uploaded JSONL file, then
//! `/batches` references it; results come back as output/error files.

use serde_json::{json, Map, Value};

use crate::errors::Lm15Error;
use crate::surfaces::{
    body_object, iso_utc, multipart_form_body, provider_error, str_field, FilePart,
};
use crate::types::{
    BatchEntry, BatchJobInfo, BatchOutcome, BatchRequest, BatchStatus, ErrorDetail,
};
use crate::wire::{batch_entry_request, BuildContext, Dialect, WireRequest};

use super::files::json_headers;
use crate::cloud::percent::path_id;

/// `_openai_batch_status`.
pub fn batch_status(status: &str) -> BatchStatus {
    match status.to_ascii_lowercase().as_str() {
        "completed" => BatchStatus::Completed,
        "failed" => BatchStatus::Failed,
        "cancelled" => BatchStatus::Cancelled,
        "expired" => BatchStatus::Expired,
        "cancelling" | "canceling" => BatchStatus::Cancelling,
        "in_progress" | "finalizing" => BatchStatus::Running,
        _ => BatchStatus::Queued, // validating / queued / anything pre-run
    }
}

pub fn upload_request(
    dialect: &dyn Dialect,
    cx: &BuildContext<'_>,
    request: &BatchRequest,
) -> Result<WireRequest, Lm15Error> {
    let mut lines = String::new();
    for (i, nested) in request.requests.iter().enumerate() {
        let body = dialect
            .build(nested, false, &cx.for_model(&nested.model))?
            .body
            .unwrap_or(Value::Null);
        let line = json!({"custom_id": i.to_string(), "method": "POST", "url": "/v1/responses", "body": body});
        lines.push_str(&serde_json::to_string(&line).expect("a JSON value serializes"));
        lines.push('\n');
    }
    let (content_type, body) = multipart_form_body(
        &[("purpose".into(), "batch".into())],
        &[FilePart {
            field: "file",
            filename: "lm15-batch.jsonl",
            content_type: "application/jsonl",
            data: lines.as_bytes(),
        }],
    );
    let mut wire = WireRequest::with_raw("POST", "/files", content_type, body);
    crate::wire::apply_static_headers(&mut wire.headers, cx.policy);
    Ok(wire)
}

pub fn submit_request(
    cx: &BuildContext<'_>,
    request: &BatchRequest,
    upload_body: Option<&Map<String, Value>>,
) -> Result<WireRequest, Lm15Error> {
    let input_file_id = upload_body
        .and_then(|b| str_field(b, "id"))
        .ok_or_else(|| {
            provider_error(cx.provider, "batch input file upload returned no id".into())
        })?;
    let mut extensions = request.extensions.clone().unwrap_or_default();
    let mut payload = Map::new();
    payload.insert("input_file_id".into(), Value::String(input_file_id));
    payload.insert(
        "endpoint".into(),
        extensions
            .remove("endpoint")
            .unwrap_or_else(|| json!("/v1/responses")),
    );
    payload.insert(
        "completion_window".into(),
        extensions
            .remove("completion_window")
            .unwrap_or_else(|| json!("24h")),
    );
    if let Some(label) = &request.label {
        payload.insert("metadata".into(), json!({"label": label}));
    }
    payload.extend(extensions);
    let mut wire = WireRequest::post("/batches", Value::Object(payload));
    wire.headers = json_headers(cx);
    Ok(wire)
}

pub fn job_info(
    cx: &BuildContext<'_>,
    data: &Map<String, Value>,
) -> Result<BatchJobInfo, Lm15Error> {
    let id = str_field(data, "id")
        .ok_or_else(|| provider_error(cx.provider, "batch object carries no id".into()))?;
    let label = data
        .get("metadata")
        .and_then(Value::as_object)
        .and_then(|m| str_field(m, "label"));
    Ok(BatchJobInfo {
        id,
        status: batch_status(data.get("status").and_then(Value::as_str).unwrap_or("")),
        label,
        created_at: iso_utc(data.get("created_at")),
        provider_data: Some(data.clone()),
    })
}

pub fn job_from_body(cx: &BuildContext<'_>, body: &[u8]) -> Result<BatchJobInfo, Lm15Error> {
    job_info(cx, &body_object(cx.provider, body, "batch")?)
}

pub fn status_request(cx: &BuildContext<'_>, batch_id: &str) -> WireRequest {
    let mut wire = WireRequest::get(format!("/batches/{}", path_id(batch_id, false)));
    wire.headers = json_headers(cx);
    wire
}

pub fn cancel_request(cx: &BuildContext<'_>, batch_id: &str) -> WireRequest {
    let mut wire = WireRequest::get(format!("/batches/{}/cancel", path_id(batch_id, false)));
    wire.method = "POST".into();
    wire.headers = json_headers(cx);
    wire
}

pub fn result_fetches(cx: &BuildContext<'_>, status_body: &Map<String, Value>) -> Vec<WireRequest> {
    ["output_file_id", "error_file_id"]
        .iter()
        .filter_map(|key| str_field(status_body, key))
        .map(|file_id| {
            let mut wire = WireRequest::get(format!("/files/{}/content", path_id(&file_id, false)));
            wire.headers = json_headers(cx);
            wire
        })
        .collect()
}

pub fn entries(
    dialect: &dyn Dialect,
    cx: &BuildContext<'_>,
    status_body: &Map<String, Value>,
    fetched: &[Vec<u8>],
) -> Result<Vec<BatchEntry>, Lm15Error> {
    let job_status = batch_status(
        status_body
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or(""),
    );
    let mut found: std::collections::BTreeMap<u64, BatchEntry> = std::collections::BTreeMap::new();
    for text in fetched {
        for line in String::from_utf8_lossy(text).lines() {
            if line.trim().is_empty() {
                continue;
            }
            let item: Value = serde_json::from_str(line).map_err(|err| {
                provider_error(cx.provider, format!("batch output line is not JSON: {err}"))
            })?;
            let index = item
                .get("custom_id")
                .map(|v| match v {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                })
                .and_then(|s| s.parse::<u64>().ok())
                .ok_or_else(|| {
                    provider_error(cx.provider, "batch output line carries no custom_id".into())
                })?;
            let response_obj = item.get("response").and_then(Value::as_object);
            let status_code = response_obj
                .and_then(|r| r.get("status_code"))
                .and_then(Value::as_u64)
                .unwrap_or(0) as u16;
            let body_obj = response_obj
                .and_then(|r| r.get("body"))
                .and_then(Value::as_object);
            let entry = match body_obj {
                Some(body) if status_code == 200 && !body.is_empty() => {
                    let request = batch_entry_request(body.get("model").and_then(Value::as_str));
                    let bytes = serde_json::to_vec(body).expect("a JSON value serializes");
                    let response =
                        dialect.parse_response(&request, &cx.for_model(&request.model), &bytes)?;
                    BatchEntry {
                        index,
                        outcome: BatchOutcome::Succeeded,
                        response: Some(response),
                        error: None,
                    }
                }
                _ => {
                    let source = body_obj
                        .filter(|b| !b.is_empty())
                        .map(|b| Value::Object(b.clone()))
                        .or_else(|| item.get("error").cloned())
                        .unwrap_or_else(|| json!({}));
                    let err = crate::errors::normalize_error(
                        cx.provider,
                        if status_code == 0 { 400 } else { status_code },
                        &source.to_string(),
                    )
                    .map_err(|e| provider_error(cx.provider, e.message))?;
                    BatchEntry {
                        index,
                        outcome: BatchOutcome::Errored,
                        response: None,
                        error: Some(ErrorDetail {
                            code: err.code(),
                            message: non_empty_or(err.message(), "batch entry errored"),
                            provider_code: err.provider_code().map(str::to_string),
                        }),
                    }
                }
            };
            found.insert(index, entry);
        }
    }
    // Entries the output files never mention (an expired or cancelled
    // batch stops mid-flight): filled from the job's terminal status. A
    // batch cancelled during `validating` reports total=0 — honestly
    // empty, never fabricated from the input side.
    let counts = status_body.get("request_counts").and_then(Value::as_object);
    let total = counts
        .and_then(|c| c.get("total"))
        .and_then(Value::as_u64)
        .filter(|t| *t > 0)
        .unwrap_or_else(|| found.keys().max().map(|m| m + 1).unwrap_or(0));
    let fill = match job_status {
        BatchStatus::Expired => Some(BatchOutcome::Expired),
        BatchStatus::Cancelled => Some(BatchOutcome::Cancelled),
        _ => None,
    };
    Ok((0..total)
        .map(|index| match found.remove(&index) {
            Some(entry) => entry,
            None => match fill {
                Some(outcome) => BatchEntry {
                    index,
                    outcome,
                    response: None,
                    error: None,
                },
                None => BatchEntry {
                    index,
                    outcome: BatchOutcome::Errored,
                    response: None,
                    error: Some(ErrorDetail::new(
                        crate::errors::ErrorCode::Provider,
                        "entry missing from batch output files",
                    )),
                },
            },
        })
        .collect())
}

fn non_empty_or(text: &str, fallback: &str) -> String {
    if text.is_empty() {
        fallback.to_string()
    } else {
        text.to_string()
    }
}

pub fn list_request(cx: &BuildContext<'_>, limit: u64) -> WireRequest {
    let mut wire = WireRequest::get("/batches");
    wire.params.push(("limit".into(), limit.to_string()));
    wire.headers = json_headers(cx);
    wire
}

pub fn jobs(cx: &BuildContext<'_>, body: &[u8]) -> Result<Vec<BatchJobInfo>, Lm15Error> {
    let data = body_object(cx.provider, body, "batch list")?;
    data.get("data")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
        .map(|item| job_info(cx, item))
        .collect()
}

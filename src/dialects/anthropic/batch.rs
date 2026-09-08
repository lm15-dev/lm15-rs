//! The Anthropic Message Batches API (`anthropic.py:1079-1200`):
//! single-step submit, `ended` split on request counts, results fetched
//! from `results_url` and re-sorted to submission order (the live
//! capture of 2026-08-31 returned them out of order).

use serde_json::{json, Map, Value};

use crate::errors::Lm15Error;
use crate::surfaces::{body_object, iso_utc, provider_error, str_field, unsupported};
use crate::types::{BatchEntry, BatchJobInfo, BatchOutcome, BatchRequest, BatchStatus, ErrorDetail};
use crate::wire::{batch_entry_request, BuildContext, Dialect, WireRequest};

use super::surface_headers;

fn json_headers(cx: &BuildContext<'_>) -> Vec<(String, String)> {
    let mut headers = surface_headers(cx.policy);
    headers.push(("content-type".into(), "application/json".into()));
    headers
}

/// `_anthropic_batch_status`: `ended` is the one terminal processing
/// status; the canonical terminal splits on request_counts.
pub fn batch_status(data: &Map<String, Value>) -> BatchStatus {
    let status = data
        .get("processing_status")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    match status.as_str() {
        "in_progress" => BatchStatus::Running,
        "canceling" => BatchStatus::Cancelling,
        "ended" => {
            let counts = data.get("request_counts").and_then(Value::as_object);
            let n = |key: &str| -> u64 {
                counts
                    .and_then(|c| c.get(key))
                    .and_then(|v| v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse().ok())))
                    .unwrap_or(0)
            };
            if n("canceled") > 0 && n("succeeded") == 0 && n("errored") == 0 && n("expired") == 0 {
                BatchStatus::Cancelled
            } else if n("expired") > 0 && n("succeeded") == 0 && n("errored") == 0 && n("canceled") == 0 {
                BatchStatus::Expired
            } else {
                BatchStatus::Completed
            }
        }
        _ => BatchStatus::Queued,
    }
}

pub fn submit_request(dialect: &dyn Dialect, cx: &BuildContext<'_>, request: &BatchRequest) -> Result<WireRequest, Lm15Error> {
    if request.label.is_some() {
        let mut err = unsupported(cx.provider, "batch labels are");
        err.meta_mut().message = format!(
            "{}: batch labels are not supported — the Message Batches create body has no \
             metadata field (verified live 2026-08-31); submit without a label and correlate by id",
            cx.provider
        );
        return Err(err);
    }
    let mut requests = Vec::new();
    for (i, nested) in request.requests.iter().enumerate() {
        let params = dialect.build(nested, false, &cx.for_model(&nested.model))?.body.unwrap_or(Value::Null);
        requests.push(json!({"custom_id": i.to_string(), "params": params}));
    }
    let mut payload = Map::new();
    payload.insert("requests".into(), Value::Array(requests));
    if let Some(extensions) = &request.extensions {
        payload.extend(extensions.clone());
    }
    let mut wire = WireRequest::post("/messages/batches", Value::Object(payload));
    wire.headers = json_headers(cx);
    Ok(wire)
}

pub fn job_info(cx: &BuildContext<'_>, data: &Map<String, Value>) -> Result<BatchJobInfo, Lm15Error> {
    let id = str_field(data, "id")
        .ok_or_else(|| provider_error(cx.provider, "batch object carries no id".into()))?;
    Ok(BatchJobInfo {
        id,
        status: batch_status(data),
        label: None,
        created_at: iso_utc(data.get("created_at")),
        provider_data: Some(data.clone()),
    })
}

pub fn job_from_body(cx: &BuildContext<'_>, body: &[u8]) -> Result<BatchJobInfo, Lm15Error> {
    job_info(cx, &body_object(cx.provider, body, "batch")?)
}

pub fn status_request(cx: &BuildContext<'_>, batch_id: &str) -> WireRequest {
    let mut wire = WireRequest::get(format!("/messages/batches/{batch_id}"));
    wire.headers = json_headers(cx);
    wire
}

pub fn cancel_request(cx: &BuildContext<'_>, batch_id: &str) -> WireRequest {
    let mut wire = WireRequest::get(format!("/messages/batches/{batch_id}/cancel"));
    wire.method = "POST".into();
    wire.headers = json_headers(cx);
    wire
}

pub fn result_fetches(cx: &BuildContext<'_>, status_body: &Map<String, Value>) -> Result<Vec<WireRequest>, Lm15Error> {
    let url = str_field(status_body, "results_url")
        .ok_or_else(|| provider_error(cx.provider, "ended batch carries no results_url".into()))?;
    let mut wire = WireRequest::get("");
    wire.absolute_url = Some(url);
    wire.headers = json_headers(cx);
    Ok(vec![wire])
}

pub fn entries(dialect: &dyn Dialect, cx: &BuildContext<'_>, fetched: &[Vec<u8>]) -> Result<Vec<BatchEntry>, Lm15Error> {
    let Some(text) = fetched.first() else {
        return Ok(Vec::new());
    };
    let mut entries = Vec::new();
    for line in String::from_utf8_lossy(text).lines() {
        if line.trim().is_empty() {
            continue;
        }
        let item: Value = serde_json::from_str(line)
            .map_err(|err| provider_error(cx.provider, format!("batch result line is not JSON: {err}")))?;
        let index = item
            .get("custom_id")
            .map(|v| match v {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            })
            .and_then(|s| s.parse::<u64>().ok())
            .ok_or_else(|| provider_error(cx.provider, "batch result line carries no custom_id".into()))?;
        let result = item.get("result").and_then(Value::as_object);
        let rtype = result.and_then(|r| r.get("type")).and_then(Value::as_str).unwrap_or("");
        let entry = match rtype {
            "succeeded" => {
                let message = result.and_then(|r| r.get("message")).and_then(Value::as_object);
                let request = batch_entry_request(message.and_then(|m| m.get("model")).and_then(Value::as_str));
                let bytes = serde_json::to_vec(&message.cloned().unwrap_or_default()).expect("serializes");
                let response = dialect.parse_response(&request, &cx.for_model(&request.model), &bytes)?;
                BatchEntry {
                    index,
                    outcome: BatchOutcome::Succeeded,
                    response: Some(response),
                    error: None,
                }
            }
            "errored" => {
                let raw = result.and_then(|r| r.get("error")).cloned().unwrap_or_else(|| json!({}));
                let envelope = match &raw {
                    Value::Object(map) if map.contains_key("error") => raw.clone(),
                    _ => json!({"error": raw}),
                };
                let err = crate::errors::normalize_error(cx.provider, 400, &envelope.to_string())
                    .map_err(|e| provider_error(cx.provider, e.message))?;
                BatchEntry {
                    index,
                    outcome: BatchOutcome::Errored,
                    response: None,
                    error: Some(ErrorDetail {
                        code: err.code(),
                        message: if err.message().is_empty() {
                            "batch entry errored".into()
                        } else {
                            err.message().to_string()
                        },
                        provider_code: err.provider_code().map(str::to_string),
                    }),
                }
            }
            "canceled" => BatchEntry {
                index,
                outcome: BatchOutcome::Cancelled,
                response: None,
                error: None,
            },
            "expired" => BatchEntry {
                index,
                outcome: BatchOutcome::Expired,
                response: None,
                error: None,
            },
            other => BatchEntry {
                index,
                outcome: BatchOutcome::Errored,
                response: None,
                error: Some(ErrorDetail::new(
                    crate::errors::ErrorCode::Provider,
                    format!("unrecognized batch result type {other:?}"),
                )),
            },
        };
        entries.push(entry);
    }
    entries.sort_by_key(|e| e.index);
    Ok(entries)
}

pub fn list_request(cx: &BuildContext<'_>, limit: u64) -> WireRequest {
    let mut wire = WireRequest::get("/messages/batches");
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

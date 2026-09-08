//! Gemini batch (`gemini.py:1580-1700`): `:batchGenerateContent` with
//! inlined requests keyed by index, a long-running operation whose
//! terminal body inlines the responses (no result fetch).

use serde_json::{json, Map, Value};

use crate::errors::Lm15Error;
use crate::surfaces::{body_object, iso_utc, provider_error, str_field};
use crate::types::{
    BatchEntry, BatchJobInfo, BatchOutcome, BatchRequest, BatchStatus, ErrorDetail,
};
use crate::wire::{batch_entry_request, wire_model, BuildContext, Dialect, WireRequest};

use super::model_path;
use crate::cloud::percent::path_id;

/// `_gemini_batch_status` (states observed live 2026-08-31:
/// BATCH_STATE_PENDING / RUNNING / SUCCEEDED — the docs' JOB_STATE_*
/// naming is wrong for this endpoint).
pub fn batch_status(data: &Map<String, Value>) -> BatchStatus {
    let state = data
        .get("metadata")
        .and_then(Value::as_object)
        .and_then(|m| m.get("state"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_uppercase();
    match state.as_str() {
        "BATCH_STATE_PENDING" => BatchStatus::Queued,
        "BATCH_STATE_RUNNING" => BatchStatus::Running,
        "BATCH_STATE_CANCELLING" => BatchStatus::Cancelling,
        "BATCH_STATE_SUCCEEDED" => BatchStatus::Completed,
        "BATCH_STATE_FAILED" => BatchStatus::Failed,
        "BATCH_STATE_CANCELLED" => BatchStatus::Cancelled,
        "BATCH_STATE_EXPIRED" => BatchStatus::Expired,
        _ => {
            if data.get("done").and_then(Value::as_bool).unwrap_or(false) {
                BatchStatus::Completed
            } else {
                BatchStatus::Queued
            }
        }
    }
}

pub fn submit_request(
    dialect: &dyn Dialect,
    cx: &BuildContext<'_>,
    request: &BatchRequest,
) -> Result<WireRequest, Lm15Error> {
    let model = request
        .model
        .as_deref()
        .or_else(|| request.requests.first().map(|r| r.model.as_str()))
        .ok_or_else(|| provider_error(cx.provider, "batch carries no model".into()))?;
    let mut requests = Vec::new();
    for (i, nested) in request.requests.iter().enumerate() {
        let body = dialect
            .build(nested, false, &cx.for_model(&nested.model))?
            .body
            .unwrap_or(Value::Null);
        requests.push(json!({"request": body, "metadata": {"key": i.to_string()}}));
    }
    let mut batch = Map::new();
    batch.insert(
        "inputConfig".into(),
        json!({"requests": {"requests": requests}}),
    );
    if let Some(label) = &request.label {
        batch.insert("displayName".into(), Value::String(label.clone()));
    }
    let mut payload = Map::new();
    payload.insert("batch".into(), Value::Object(batch));
    if let Some(extensions) = &request.extensions {
        payload.extend(extensions.clone());
    }
    let path = format!(
        "/{}:batchGenerateContent",
        model_path(wire_model(cx.provider, model))
    );
    let mut wire = WireRequest::post(path, Value::Object(payload));
    wire.headers
        .push(("Content-Type".into(), "application/json".into()));
    Ok(wire)
}

pub fn job_info(
    cx: &BuildContext<'_>,
    data: &Map<String, Value>,
) -> Result<BatchJobInfo, Lm15Error> {
    let id = str_field(data, "name")
        .ok_or_else(|| provider_error(cx.provider, "batch operation carries no name".into()))?;
    let metadata = data.get("metadata").and_then(Value::as_object);
    Ok(BatchJobInfo {
        id,
        status: batch_status(data),
        label: metadata.and_then(|m| str_field(m, "displayName")),
        created_at: iso_utc(metadata.and_then(|m| m.get("createTime"))),
        provider_data: Some(data.clone()),
    })
}

pub fn job_from_body(cx: &BuildContext<'_>, body: &[u8]) -> Result<BatchJobInfo, Lm15Error> {
    job_info(cx, &body_object(cx.provider, body, "batch")?)
}

pub fn status_request(batch_id: &str) -> WireRequest {
    WireRequest::get(format!("/{}", path_id(batch_id, true)))
}

pub fn cancel_request(batch_id: &str) -> WireRequest {
    let mut wire = WireRequest::post(format!("/{}:cancel", path_id(batch_id, true)), json!({}));
    wire.headers
        .push(("Content-Type".into(), "application/json".into()));
    wire
}

pub fn entries(
    dialect: &dyn Dialect,
    cx: &BuildContext<'_>,
    status_body: &Map<String, Value>,
) -> Result<Vec<BatchEntry>, Lm15Error> {
    let response = status_body.get("response").and_then(Value::as_object);
    let mut inlined = response.and_then(|r| r.get("inlinedResponses"));
    if let Some(Value::Object(map)) = inlined {
        inlined = map.get("inlinedResponses");
    }
    let mut entries = Vec::new();
    for (position, item) in inlined
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
    {
        let Some(item) = item.as_object() else {
            continue;
        };
        let index = item
            .get("metadata")
            .and_then(Value::as_object)
            .and_then(|m| m.get("key"))
            .and_then(|k| match k {
                Value::String(s) => s.parse::<u64>().ok(),
                Value::Number(n) => n.as_u64(),
                _ => None,
            })
            .unwrap_or(position as u64);
        let entry = match item.get("response").and_then(Value::as_object) {
            Some(body) => {
                let request = batch_entry_request(body.get("modelVersion").and_then(Value::as_str));
                let bytes = serde_json::to_vec(body).expect("serializes");
                let parsed =
                    dialect.parse_response(&request, &cx.for_model(&request.model), &bytes)?;
                BatchEntry {
                    index,
                    outcome: BatchOutcome::Succeeded,
                    response: Some(parsed),
                    error: None,
                }
            }
            None => {
                // A per-entry failure is a google.rpc.Status, not the HTTP
                // error envelope; mapped directly.
                let err = item.get("error").and_then(Value::as_object);
                let provider_code = err
                    .and_then(|e| e.get("status").or_else(|| e.get("code")))
                    .map(|v| match v {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    });
                BatchEntry {
                    index,
                    outcome: BatchOutcome::Errored,
                    response: None,
                    error: Some(ErrorDetail {
                        code: crate::errors::ErrorCode::Provider,
                        message: err
                            .and_then(|e| str_field(e, "message"))
                            .unwrap_or_else(|| "batch entry errored".into()),
                        provider_code,
                    }),
                }
            }
        };
        entries.push(entry);
    }
    entries.sort_by_key(|e| e.index);
    Ok(entries)
}

pub fn list_request(limit: u64) -> WireRequest {
    let mut wire = WireRequest::get("/batches");
    wire.params.push(("pageSize".into(), limit.to_string()));
    wire
}

pub fn jobs(cx: &BuildContext<'_>, body: &[u8]) -> Result<Vec<BatchJobInfo>, Lm15Error> {
    let data = body_object(cx.provider, body, "batch list")?;
    data.get("operations")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
        .map(|item| job_info(cx, item))
        .collect()
}

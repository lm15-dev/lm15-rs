//! The lm15 vet shim (harness/PROTOCOL.md): newline-delimited JSON on
//! stdin/stdout, one reply per request, same `id`. Thin by rule: it parses
//! the protocol line, calls the public functions users call, and
//! serializes. No network.

use std::io::{self, BufRead, Write};

use serde_json::{json, Map, Value};

use lm15::auth::{parse_rfc3339, Credential};
use lm15::cloud::sigv4::{self, AwsKeys, SigningRequest};
use lm15::errors::{normalize_error, Lm15Error};
use lm15::serde::{roundtrip, validate};
use lm15::stream::materialize_response;
use lm15::types::{BatchRequest, FileUploadRequest, Request, Response, ValidationError};
use lm15::wire::{FixedClock, TransportRequest};
use lm15::{Canonical, HostSettings};

const LANGUAGE: &str = "rust";
const IMPL_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Ops this shim answers (`capabilities.ops`).
const OPS: &[&str] = &[
    "build_request",
    "build_models_request",
    "batch_op_build",
    "cache_op_build",
    "cache_op_parse",
    "batch_op_parse",
    "file_op_build",
    "file_op_parse",
    "capabilities",
    "explain_auth",
    "normalize_error",
    "parse_response",
    "parse_models_response",
    "replay_stream",
    "serde_roundtrip",
    "sigv4_sign",
    "validate",
];

enum Failure {
    Validation(ValidationError),
    Lm15(Box<Lm15Error>),
    /// A typed error plus extra reply fields (`replay_stream`: the event
    /// trace that parsed before assembly refused, MAP-9).
    Lm15Extra(Box<Lm15Error>, Map<String, Value>),
}

impl From<ValidationError> for Failure {
    fn from(err: ValidationError) -> Self {
        Failure::Validation(err)
    }
}

impl From<Lm15Error> for Failure {
    fn from(err: Lm15Error) -> Self {
        Failure::Lm15(Box::new(err))
    }
}

fn field<'a>(msg: &'a Map<String, Value>, key: &str) -> Result<&'a Value, Failure> {
    msg.get(key)
        .ok_or_else(|| ValidationError::type_error(format!("missing field: {key}")).into())
}

fn field_str(msg: &Map<String, Value>, key: &str) -> Result<String, Failure> {
    match field(msg, key)? {
        Value::String(s) => Ok(s.clone()),
        other => Ok(other.to_string()),
    }
}

fn op_capabilities() -> Value {
    json!({
        "language": LANGUAGE,
        "ops": OPS,
        "impl_version": IMPL_VERSION,
    })
}

fn op_serde_roundtrip(msg: &Map<String, Value>) -> Result<Value, Failure> {
    let kind = field_str(msg, "kind")?;
    let value = roundtrip(&kind, field(msg, "value")?)?;
    Ok(json!({ "value": value }))
}

fn op_validate(msg: &Map<String, Value>) -> Result<Value, Failure> {
    let kind = field_str(msg, "kind")?;
    let normalized = validate(&kind, field(msg, "value")?)?;
    Ok(json!({ "ok": true, "normalized": normalized }))
}

fn op_normalize_error(msg: &Map<String, Value>) -> Result<Value, Failure> {
    let provider = field_str(msg, "provider")?;
    let status = field(msg, "status")?
        .as_u64()
        .and_then(|s| u16::try_from(s).ok())
        .ok_or_else(|| ValidationError::type_error("status must be an int"))?;
    let body_text = field_str(msg, "body_text")?;
    let err = normalize_error(&provider, status, &body_text)?;
    Ok(json!({
        "class": err.class_name(),
        "code": err.code().as_str(),
        "provider_code": err.provider_code(),
        "message": err.message(),
    }))
}

fn op_explain_auth(msg: &Map<String, Value>) -> Result<Value, Failure> {
    // PROTOCOL.md explain_auth: the harness owns every input. The env map is
    // the whole environment (never the process env), api_keys_providers get
    // the sentinel planted, credentials_path is the harness-written file.
    let provider = field_str(msg, "provider")?;
    let mut env = std::collections::HashMap::new();
    if let Some(map) = msg.get("env").and_then(Value::as_object) {
        for (k, v) in map {
            if let Some(s) = v.as_str() {
                env.insert(k.clone(), s.to_string());
            }
        }
    }
    let options = lm15::auth::ExplainOptions {
        env: Some(env),
        api_key_providers: msg
            .get("api_keys_providers")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default(),
        credentials_path: msg
            .get("credentials_path")
            .and_then(Value::as_str)
            .map(std::path::PathBuf::from),
    };
    let report = lm15::auth::explain_auth(&provider, &options).map_err(Lm15Error::from)?;
    let steps: Vec<Value> = report
        .steps
        .iter()
        .map(|s| json!({ "kind": s.kind, "state": s.state.as_str() }))
        .collect();
    let report_text = format!("{}\n{}\n{:?}", report.describe(), report, report);
    Ok(json!({ "configured": report.configured, "steps": steps, "report_text": report_text }))
}

/// PROTOCOL.md: `credential` (an AUTH-2 value) wins over the `api_key`
/// shorthand. The shim never reads environment keys.
fn credential_of(msg: &Map<String, Value>) -> Result<Credential, Failure> {
    if let Some(value @ Value::Object(_)) = msg.get("credential") {
        return Ok(Credential::from_json(value)?);
    }
    let api_key = field_str(msg, "api_key")?;
    Ok(Credential::api_key(api_key).map_err(Lm15Error::from)?)
}

/// `now` (RFC 3339, UTC) as Unix seconds: the clock for every
/// time-dependent byte. Absent means the wall clock.
fn clock_of(msg: &Map<String, Value>) -> Result<Option<FixedClock>, Failure> {
    match msg.get("now") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => parse_rfc3339(text)
            .map(|unix| Some(FixedClock(unix)))
            .ok_or_else(|| ValidationError::value(format!("now is not RFC 3339: {text:?}")).into()),
        Some(other) => {
            Err(ValidationError::type_error(format!("now must be a string, got {other}")).into())
        }
    }
}

fn settings_of(msg: &Map<String, Value>) -> Option<HostSettings> {
    let settings = msg.get("settings")?.as_object()?;
    Some(
        settings
            .iter()
            .map(|(k, v)| {
                let value = match v {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                (k.clone(), value)
            })
            .collect(),
    )
}

/// The protocol's `build_request` shape: `url` without its query, decoded
/// `params`, lowercase header names, the JSON body or `null`.
fn transport_request_json(request: &TransportRequest) -> Value {
    let params: Map<String, Value> = request
        .params
        .iter()
        .map(|(k, v)| (k.clone(), Value::String(v.clone())))
        .collect();
    let headers: Map<String, Value> = request
        .headers
        .iter()
        .map(|(k, v)| (k.to_ascii_lowercase(), Value::String(v.clone())))
        .collect();
    let mut out = json!({
        "method": request.method,
        "url": request.url,
        "params": params,
        "headers": headers,
        "body": request.body.clone().unwrap_or(Value::Null),
    });
    if let Some(raw) = &request.raw {
        out["body_b64"] = Value::String(lm15::types::base64_encode(raw));
    }
    out
}

fn op_build_request(msg: &Map<String, Value>) -> Result<Value, Failure> {
    let provider = field_str(msg, "provider")?;
    let credential = credential_of(msg)?;
    let base_url = msg.get("base_url").and_then(Value::as_str);
    let clock = clock_of(msg)?;
    let lm = lm15::registry::adapter_for(
        &provider,
        credential,
        base_url,
        settings_of(msg),
        clock.map(|c| Box::new(c) as Box<dyn lm15::wire::Clock + Send + Sync>),
    )?;
    let request = Request::from_json(field(msg, "canonical_request")?)?;
    let stream = msg.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let transport = lm.build_request(&request, stream)?;
    Ok(transport_request_json(&transport))
}

/// PROTOCOL.md § build_models_request: the shim constructs `openai-codex`
/// with account id `test-account` (a non-JWT key carries none).
fn surface_adapter(
    msg: &Map<String, Value>,
    credential: Credential,
) -> Result<lm15::ProviderLM, Failure> {
    let provider = field_str(msg, "provider")?;
    let definition = lm15::registry::lookup(&provider)
        .ok_or_else(|| Lm15Error::not_configured(format!("unknown provider {provider:?}")))?;
    let mut builder = lm15::LmBuilder::for_entry(definition).api_key(credential);
    if let Some(base_url) = msg.get("base_url").and_then(Value::as_str) {
        builder = builder.base_url(base_url);
    }
    if let Some(settings) = settings_of(msg) {
        builder = builder.settings(settings);
    }
    if let Some(clock) = clock_of(msg)? {
        builder = builder.clock(clock);
    }
    if definition.id == "openai-codex" {
        builder = builder.account_id("test-account");
    }
    Ok(builder.build()?)
}

fn op_file_op_build(msg: &Map<String, Value>) -> Result<Value, Failure> {
    use lm15::adapter::FileOp;
    let lm = surface_adapter(msg, credential_of(msg)?)?;
    let op = field_str(msg, "file_op")?;
    let file_id = msg.get("file_id").and_then(Value::as_str).unwrap_or("");
    let built = match op.as_str() {
        "upload" => {
            let request = FileUploadRequest::from_json(field(msg, "upload_request")?)?;
            lm.file_request(&FileOp::Upload(&request))?
        }
        "get" => lm.file_request(&FileOp::Get(file_id))?,
        "delete" => lm.file_request(&FileOp::Delete(file_id))?,
        "download" => lm.file_request(&FileOp::Download(file_id))?,
        "list" => lm.file_request(&FileOp::List {
            limit: msg.get("limit").and_then(Value::as_u64).unwrap_or(20),
            cursor: msg.get("cursor").and_then(Value::as_str),
        })?,
        other => return Err(ValidationError::value(format!("unknown file_op {other:?}")).into()),
    };
    Ok(transport_request_json(&built))
}

fn op_file_op_parse(msg: &Map<String, Value>) -> Result<Value, Failure> {
    let lm = surface_adapter(msg, Credential::api_key("vet-parse-only").map_err(Lm15Error::from)?)?;
    let status = msg.get("status").and_then(Value::as_u64).unwrap_or(200) as u16;
    let body = body_of(msg)?;
    Ok(match field_str(msg, "kind")?.as_str() {
        "info" => json!({ "file": lm.parse_file_info(status, &body)?.to_json() }),
        "page" => json!({ "page": lm.parse_file_page(status, &body)?.to_json() }),
        other => return Err(ValidationError::value(format!("unknown kind {other:?}")).into()),
    })
}

fn op_batch_op_build(msg: &Map<String, Value>) -> Result<Value, Failure> {
    use lm15::adapter::BatchAction;
    let lm = surface_adapter(msg, credential_of(msg)?)?;
    let action = field_str(msg, "action")?;
    let batch_id = msg.get("batch_id").and_then(Value::as_str).unwrap_or("");
    let request = match msg.get("batch_request") {
        Some(value @ Value::Object(_)) => Some(BatchRequest::from_json(value)?),
        _ => None,
    };
    let need = || -> Result<&BatchRequest, Failure> {
        request
            .as_ref()
            .ok_or_else(|| ValidationError::value("batch_request is required").into())
    };
    let upload_body = msg.get("upload_body").and_then(Value::as_object);
    let status_body = msg.get("status_body").and_then(Value::as_object);
    let built = match action.as_str() {
        "upload" => lm.batch_requests(&BatchAction::Upload(need()?))?,
        "submit" => lm.batch_requests(&BatchAction::Submit {
            request: need()?,
            upload_body,
        })?,
        "status" => lm.batch_requests(&BatchAction::Status(batch_id))?,
        "cancel" => lm.batch_requests(&BatchAction::Cancel(batch_id))?,
        "result_fetches" => lm.batch_requests(&BatchAction::ResultFetches(
            status_body.ok_or_else(|| ValidationError::value("status_body is required"))?,
        ))?,
        "list" => lm.batch_requests(&BatchAction::List(
            msg.get("limit").and_then(Value::as_u64).unwrap_or(20),
        ))?,
        other => return Err(ValidationError::value(format!("unknown action {other:?}")).into()),
    };
    Ok(json!({ "requests": built.iter().map(transport_request_json).collect::<Vec<_>>() }))
}

fn op_batch_op_parse(msg: &Map<String, Value>) -> Result<Value, Failure> {
    let lm = surface_adapter(msg, Credential::api_key("vet-parse-only").map_err(Lm15Error::from)?)?;
    let status = msg.get("status").and_then(Value::as_u64).unwrap_or(200) as u16;
    Ok(match field_str(msg, "kind")?.as_str() {
        "job" => json!({ "job": lm.parse_batch_job(status, &body_of(msg)?)?.to_json() }),
        "list" => json!({ "jobs": lm.parse_batch_jobs(status, &body_of(msg)?)?.iter().map(|j| j.to_json()).collect::<Vec<_>>() }),
        "entries" => {
            let status_body = msg
                .get("status_body")
                .and_then(Value::as_object)
                .ok_or_else(|| ValidationError::value("status_body is required"))?;
            let fetched: Vec<Vec<u8>> = msg
                .get("fetched_b64")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(|b64| lm15::types::base64_decode(b64).map_err(Failure::from))
                .collect::<Result<_, _>>()?;
            let entries = lm.parse_batch_entries(status_body, &fetched)?;
            json!({ "entries": entries.iter().map(|e| e.to_json()).collect::<Vec<_>>() })
        }
        other => return Err(ValidationError::value(format!("unknown kind {other:?}")).into()),
    })
}

fn op_cache_op_build(msg: &Map<String, Value>) -> Result<Value, Failure> {
    use lm15::adapter::CacheOp;
    let lm = surface_adapter(msg, credential_of(msg)?)?;
    let cache_id = msg.get("cache_id").and_then(Value::as_str).unwrap_or("");
    let ttl = msg.get("ttl_seconds").and_then(Value::as_u64);
    let built = match field_str(msg, "cache_op")?.as_str() {
        "create" => {
            let prefix = Request::from_json(field(msg, "prefix_request")?)?;
            lm.cache_request(&CacheOp::Create {
                prefix: &prefix,
                ttl_seconds: ttl,
                label: msg.get("label").and_then(Value::as_str),
            })?
        }
        "get" => lm.cache_request(&CacheOp::Get(cache_id))?,
        "delete" => lm.cache_request(&CacheOp::Delete(cache_id))?,
        "update" => lm.cache_request(&CacheOp::Update {
            cache_id,
            ttl_seconds: ttl.ok_or_else(|| ValidationError::value("ttl_seconds is required"))?,
        })?,
        "list" => lm.cache_request(&CacheOp::List {
            limit: msg.get("limit").and_then(Value::as_u64).unwrap_or(20),
            cursor: msg.get("cursor").and_then(Value::as_str),
        })?,
        other => return Err(ValidationError::value(format!("unknown cache_op {other:?}")).into()),
    };
    Ok(transport_request_json(&built))
}

fn op_cache_op_parse(msg: &Map<String, Value>) -> Result<Value, Failure> {
    let lm = surface_adapter(msg, Credential::api_key("vet-parse-only").map_err(Lm15Error::from)?)?;
    let status = msg.get("status").and_then(Value::as_u64).unwrap_or(200) as u16;
    let body = body_of(msg)?;
    Ok(match field_str(msg, "kind")?.as_str() {
        "info" => json!({ "cache": lm.parse_cache_info(status, &body)?.to_json() }),
        "page" => json!({ "page": lm.parse_cache_page(status, &body)?.to_json() }),
        other => return Err(ValidationError::value(format!("unknown kind {other:?}")).into()),
    })
}

fn op_build_models_request(msg: &Map<String, Value>) -> Result<Value, Failure> {
    let lm = surface_adapter(msg, credential_of(msg)?)?;
    Ok(transport_request_json(&lm.models_request()?))
}

/// PROTOCOL.md § parse_models_response: canonical `model_info` serde
/// INCLUDING `origin.provider_data`; a status of 400 or more is the
/// normalized error as an `ok: false` reply.
fn op_parse_models_response(msg: &Map<String, Value>) -> Result<Value, Failure> {
    let lm = surface_adapter(
        msg,
        Credential::api_key("vet-parse-only").map_err(Lm15Error::from)?,
    )?;
    let status = msg.get("status").and_then(Value::as_u64).unwrap_or(200) as u16;
    let models = lm.parse_models(status, &body_of(msg)?)?;
    Ok(json!({ "models": models.iter().map(|m| m.to_json()).collect::<Vec<_>>() }))
}

/// A parse-side adapter: the harness gives no credential for these ops,
/// so a placeholder stands in (never sent; parsing reads no credential).
fn parse_adapter(msg: &Map<String, Value>) -> Result<lm15::ProviderLM, Failure> {
    let provider = field_str(msg, "provider")?;
    let base_url = msg.get("base_url").and_then(Value::as_str);
    let clock = clock_of(msg)?;
    Ok(lm15::registry::adapter_for(
        &provider,
        "vet-parse-only",
        base_url,
        settings_of(msg),
        clock.map(|c| Box::new(c) as Box<dyn lm15::wire::Clock + Send + Sync>),
    )?)
}

fn body_of(msg: &Map<String, Value>) -> Result<Vec<u8>, Failure> {
    let b64 = field_str(msg, "body_b64")?;
    lm15::types::base64_decode(&b64).map_err(Failure::from)
}

/// PROTOCOL.md: a Response WITHOUT provider_data, plus the `_lm15_unmapped`
/// canary surfaced as the top-level `unmapped` array.
fn response_result(response: &Response) -> Map<String, Value> {
    let mut result = Map::new();
    result.insert("canonical_response".into(), response.to_json());
    if let Some(unmapped) = response
        .provider_data
        .as_ref()
        .and_then(|pd| pd.get("_lm15_unmapped"))
    {
        result.insert("unmapped".into(), unmapped.clone());
    }
    result
}

fn op_parse_response(msg: &Map<String, Value>) -> Result<Value, Failure> {
    let lm = parse_adapter(msg)?;
    let request = Request::from_json(field(msg, "canonical_request")?)?;
    let status = field(msg, "status")?
        .as_u64()
        .and_then(|s| u16::try_from(s).ok())
        .ok_or_else(|| ValidationError::type_error("status must be an int"))?;
    let body = body_of(msg)?;
    let response = lm.parse_response(&request, status, &body)?;
    Ok(Value::Object(response_result(&response)))
}

fn op_replay_stream(msg: &Map<String, Value>) -> Result<Value, Failure> {
    let lm = parse_adapter(msg)?;
    let request = Request::from_json(field(msg, "canonical_request")?)?;
    let body = body_of(msg)?;
    let events = lm.replay_stream(&request, &body)?;
    let event_json: Vec<Value> = events.iter().map(Canonical::to_json).collect();
    match materialize_response(events.iter(), &request) {
        Ok(response) => {
            let mut result = response_result(&response);
            result.insert("events".into(), Value::Array(event_json));
            Ok(Value::Object(result))
        }
        Err(err) if err.partial().is_some() => {
            // The trace parsed; assembly refused (MAP-9): report both.
            let mut extra = Map::new();
            extra.insert("events".into(), Value::Array(event_json));
            Err(Failure::Lm15Extra(Box::new(err), extra))
        }
        Err(err) => Err(err.into()),
    }
}

fn op_sigv4_sign(msg: &Map<String, Value>) -> Result<Value, Failure> {
    let request = field(msg, "request")?
        .as_object()
        .ok_or_else(|| ValidationError::type_error("request must be an object"))?;
    let method = field_str(request, "method")?;
    let url = field_str(request, "url")?;
    // A list value is the same name repeated on the wire, in that order.
    let mut headers: Vec<(String, String)> = Vec::new();
    if let Some(map) = request.get("headers").and_then(Value::as_object) {
        for (name, value) in map {
            match value {
                Value::Array(values) => {
                    for v in values {
                        headers.push((name.clone(), v.as_str().unwrap_or_default().to_string()));
                    }
                }
                Value::String(s) => headers.push((name.clone(), s.clone())),
                other => headers.push((name.clone(), other.to_string())),
            }
        }
    }
    let body = request
        .get("body")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let credential = Credential::from_json(field(msg, "credential")?)?;
    let keys = AwsKeys::from_credential(&credential)?;
    let region = field_str(msg, "region")?;
    let service = field_str(msg, "service")?;
    let now = clock_of(msg)?.ok_or_else(|| ValidationError::type_error("missing field: now"))?;
    let signing = SigningRequest {
        method: &method,
        url: &url,
        headers: &headers,
        payload: body.as_bytes(),
    };
    let signature = sigv4::sign(&signing, &keys, &region, &service, now.0);
    let headers: Map<String, Value> = signature
        .headers
        .iter()
        .map(|(k, v)| (k.clone(), Value::String(v.clone())))
        .collect();
    Ok(json!({
        "canonical_request": signature.canonical_request,
        "string_to_sign": signature.string_to_sign,
        "authorization": signature.authorization,
        "headers": headers,
    }))
}

fn dispatch(op: &str, msg: &Map<String, Value>) -> Result<Value, Failure> {
    match op {
        "capabilities" => Ok(op_capabilities()),
        "serde_roundtrip" => op_serde_roundtrip(msg),
        "validate" => op_validate(msg),
        "normalize_error" => op_normalize_error(msg),
        "explain_auth" => op_explain_auth(msg),
        "build_request" => op_build_request(msg),
        "parse_response" => op_parse_response(msg),
        "build_models_request" => op_build_models_request(msg),
        "file_op_build" => op_file_op_build(msg),
        "cache_op_build" => op_cache_op_build(msg),
        "cache_op_parse" => op_cache_op_parse(msg),
        "batch_op_build" => op_batch_op_build(msg),
        "batch_op_parse" => op_batch_op_parse(msg),
        "file_op_parse" => op_file_op_parse(msg),
        "parse_models_response" => op_parse_models_response(msg),
        "replay_stream" => op_replay_stream(msg),
        "sigv4_sign" => op_sigv4_sign(msg),
        // Module 3b (cloud chains: token exchange, RS256) is not implemented.
        "token_exchange_build" | "token_exchange_parse" => Err(Lm15Error::unsupported_feature(
            format!("op {op:?} needs module 3b (cloud credential chains), which this port does not implement"),
        )
        .into()),
        other => Err(Lm15Error::unsupported_feature(format!(
            "op {other:?} is not implemented by the Rust shim (modules 1-5: {})",
            OPS.join(", ")
        ))
        .into()),
    }
}

fn lm15_error_json(err: &Lm15Error, extra: Map<String, Value>) -> Value {
    let mut error = json!({
        "type": err.class_name(),
        "code": err.code().as_str(),
        "message": err.message(),
    });
    if let Some(partial) = err.partial() {
        error["partial_response"] =
            Value::Object(response_result(partial))["canonical_response"].clone();
    }
    for (key, value) in extra {
        error[key] = value;
    }
    error
}

fn error_reply(id: Value, failure: Failure) -> Value {
    let error = match failure {
        Failure::Validation(err) => json!({
            "type": err.type_name(),
            "message": err.message,
        }),
        Failure::Lm15(err) => lm15_error_json(&err, Map::new()),
        Failure::Lm15Extra(err, extra) => lm15_error_json(&err, extra),
    };
    json!({ "id": id, "ok": false, "error": error })
}

fn handle_line(line: &str) -> Value {
    let msg: Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => {
            return error_reply(
                Value::Null,
                ValidationError::value(format!("invalid JSON request: {e}")).into(),
            )
        }
    };
    let Value::Object(msg) = msg else {
        return error_reply(
            Value::Null,
            ValidationError::value("request must be a JSON object").into(),
        );
    };
    let id = msg.get("id").cloned().unwrap_or(Value::Null);
    let op = msg.get("op").and_then(Value::as_str).unwrap_or_default();
    match dispatch(op, &msg) {
        Ok(result) => json!({ "id": id, "ok": true, "result": result }),
        Err(failure) => error_reply(id, failure),
    }
}

fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let reply = handle_line(&line);
        writeln!(out, "{reply}")?;
        out.flush()?;
    }
    Ok(())
}

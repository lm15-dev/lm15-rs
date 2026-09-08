//! Gemini `cachedContents` (`gemini.py:1583-1680`): the stored tier of
//! MAP-6. The create body reuses the chat codec for `contents`,
//! `systemInstruction` and `tools` (the same mapping, never a second one).

use serde_json::{json, Map, Value};

use crate::errors::Lm15Error;
use crate::surfaces::{body_object, iso_utc, provider_error, str_field, u64_field};
use crate::types::{CacheInfo, CachePage, Request};
use crate::wire::{BuildContext, Dialect, WireRequest};

use super::model_path;

pub fn cache_resource(cache_id: &str) -> String {
    if cache_id.starts_with("cachedContents/") {
        cache_id.to_string()
    } else {
        format!("cachedContents/{cache_id}")
    }
}

pub fn create_request(
    dialect: &dyn Dialect,
    cx: &BuildContext<'_>,
    prefix: &Request,
    ttl_seconds: Option<u64>,
    label: Option<&str>,
) -> Result<WireRequest, Lm15Error> {
    let cx = cx.for_model(&prefix.model);
    let chat = dialect
        .build(prefix, false, &cx)?
        .body
        .unwrap_or(Value::Null);
    let mut body = Map::new();
    body.insert("model".into(), Value::String(model_path(cx.model)));
    body.insert(
        "contents".into(),
        chat.get("contents").cloned().unwrap_or_else(|| json!([])),
    );
    if let Some(system) = chat.get("systemInstruction") {
        body.insert("systemInstruction".into(), system.clone());
    }
    if let Some(tools) = chat.get("tools") {
        body.insert("tools".into(), tools.clone());
    }
    if let Some(ttl) = ttl_seconds {
        body.insert("ttl".into(), Value::String(format!("{ttl}s")));
    }
    if let Some(label) = label {
        body.insert("displayName".into(), Value::String(label.to_string()));
    }
    let mut wire = WireRequest::post("/cachedContents", Value::Object(body));
    wire.headers
        .push(("Content-Type".into(), "application/json".into()));
    Ok(wire)
}

pub fn cache_info(
    cx: &BuildContext<'_>,
    data: &Map<String, Value>,
) -> Result<CacheInfo, Lm15Error> {
    let id = str_field(data, "name")
        .ok_or_else(|| provider_error(cx.provider, "cache object carries no name".into()))?;
    let model = data
        .get("model")
        .and_then(Value::as_str)
        .map(|m| m.strip_prefix("models/").unwrap_or(m).to_string())
        .filter(|m| !m.is_empty())
        .ok_or_else(|| provider_error(cx.provider, "cache object carries no model".into()))?;
    Ok(CacheInfo {
        id,
        model,
        tokens: data
            .get("usageMetadata")
            .and_then(Value::as_object)
            .and_then(|u| u64_field(u, "totalTokenCount")),
        created_at: iso_utc(data.get("createTime")),
        expires_at: iso_utc(data.get("expireTime")),
        label: str_field(data, "displayName"),
        provider_data: Some(data.clone()),
    })
}

pub fn info_from_body(cx: &BuildContext<'_>, body: &[u8]) -> Result<CacheInfo, Lm15Error> {
    cache_info(cx, &body_object(cx.provider, body, "cache")?)
}

pub fn get_request(cache_id: &str) -> WireRequest {
    WireRequest::get(format!("/{}", cache_resource(cache_id)))
}

pub fn list_request(limit: u64, cursor: Option<&str>) -> WireRequest {
    let mut wire = WireRequest::get("/cachedContents");
    wire.params.push(("pageSize".into(), limit.to_string()));
    if let Some(cursor) = cursor {
        wire.params.push(("pageToken".into(), cursor.to_string()));
    }
    wire
}

pub fn page(cx: &BuildContext<'_>, body: &[u8]) -> Result<CachePage, Lm15Error> {
    let data = body_object(cx.provider, body, "cache list")?;
    let items = data
        .get("cachedContents")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
        .map(|e| cache_info(cx, e))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(CachePage {
        items,
        next_cursor: str_field(&data, "nextPageToken"),
    })
}

pub fn delete_request(cache_id: &str) -> WireRequest {
    let mut wire = get_request(cache_id);
    wire.method = "DELETE".into();
    wire
}

pub fn update_request(cache_id: &str, ttl_seconds: u64) -> WireRequest {
    let mut wire = WireRequest::json(
        "PATCH",
        format!("/{}", cache_resource(cache_id)),
        json!({"ttl": format!("{ttl_seconds}s")}),
    );
    wire.headers
        .push(("Content-Type".into(), "application/json".into()));
    wire
}

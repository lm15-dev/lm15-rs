//! Anthropic adapter (Messages API).

use crate::errors::{map_http_error, LM15Error};
use crate::transport::{HttpRequest, HttpResponse, SSEEvent, UreqTransport};
use crate::types::*;
use super::{Adapter, ProviderManifest};
use super::common::parts_to_text;
use serde_json::Value;
use std::collections::HashMap;

pub struct AnthropicAdapter {
    pub api_key: String,
    pub base_url: String,
    pub api_version: String,
    pub transport: UreqTransport,
}

impl AnthropicAdapter {
    pub fn new(api_key: &str, transport: UreqTransport) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com/v1".into(),
            api_version: "2023-06-01".into(),
            transport,
        }
    }

    fn headers(&self) -> HashMap<String, String> {
        let mut h = HashMap::new();
        h.insert("x-api-key".into(), self.api_key.clone());
        h.insert("anthropic-version".into(), self.api_version.clone());
        h.insert("content-type".into(), "application/json".into());
        h
    }

    fn part_payload(&self, p: &Part) -> Value {
        match &p.part_type {
            PartType::Text => serde_json::json!({"type": "text", "text": p.text.as_deref().unwrap_or("")}),
            PartType::Image => {
                if let Some(s) = &p.source {
                    if s.source_type == "url" {
                        return serde_json::json!({"type": "image", "source": {"type": "url", "url": s.url}});
                    }
                    return serde_json::json!({"type": "image", "source": {"type": "base64", "media_type": s.media_type, "data": s.data}});
                }
                serde_json::json!({"type": "text", "text": ""})
            }
            PartType::Document => {
                if let Some(s) = &p.source {
                    if s.source_type == "url" {
                        return serde_json::json!({"type": "document", "source": {"type": "url", "url": s.url}});
                    }
                    return serde_json::json!({"type": "document", "source": {"type": "base64", "media_type": s.media_type, "data": s.data}});
                }
                serde_json::json!({"type": "text", "text": ""})
            }
            PartType::ToolCall => {
                serde_json::json!({
                    "type": "tool_use",
                    "id": p.id,
                    "name": p.name,
                    "input": p.input.as_ref().unwrap_or(&HashMap::new()),
                })
            }
            PartType::ToolResult => {
                let text = p.content.as_ref().map(|c| parts_to_text(c)).unwrap_or_default();
                let mut out = serde_json::json!({"type": "tool_result", "tool_use_id": p.id});
                if !text.is_empty() {
                    out["content"] = Value::String(text);
                }
                if p.is_error == Some(true) {
                    out["is_error"] = Value::Bool(true);
                }
                out
            }
            _ => serde_json::json!({"type": "text", "text": p.text.as_deref().unwrap_or("")}),
        }
    }

    fn build_payload(&self, request: &LMRequest, stream: bool) -> Value {
        let messages: Vec<Value> = request.messages.iter().map(|m| {
            let content: Vec<Value> = m.parts.iter().map(|p| self.part_payload(p)).collect();
            let role = match m.role { Role::Tool => "user", Role::User => "user", Role::Assistant => "assistant" };
            serde_json::json!({"role": role, "content": content})
        }).collect();

        let max_tokens = request.config.max_tokens.unwrap_or(1024);
        let mut payload = serde_json::json!({
            "model": request.model, "messages": messages,
            "stream": stream, "max_tokens": max_tokens,
        });

        if let Some(system) = &request.system {
            payload["system"] = Value::String(system.clone());
        }
        if let Some(temp) = request.config.temperature {
            payload["temperature"] = serde_json::json!(temp);
        }

        let tools: Vec<Value> = request.tools.iter()
            .filter(|t| t.tool_type == "function")
            .map(|t| serde_json::json!({
                "name": t.name, "description": t.description,
                "input_schema": t.parameters.as_ref().unwrap_or(&HashMap::new()),
            }))
            .collect();
        if !tools.is_empty() {
            payload["tools"] = Value::Array(tools);
        }

        if let Some(reasoning) = &request.config.reasoning {
            if reasoning.get("enabled").and_then(|v| v.as_bool()).unwrap_or(false) {
                let budget = reasoning.get("budget").and_then(|v| v.as_i64()).unwrap_or(1024);
                payload["thinking"] = serde_json::json!({"type": "enabled", "budget_tokens": budget});
            }
        }

        if let Some(provider) = &request.config.provider {
            for (k, v) in provider {
                if k != "prompt_caching" { payload[k] = v.clone(); }
            }
        }

        payload
    }
}

fn is_context_msg(msg: &str) -> bool {
    let m = msg.to_lowercase();
    m.contains("prompt is too long") || m.contains("too many tokens")
        || m.contains("context window") || m.contains("context length")
}

impl Adapter for AnthropicAdapter {
    fn provider_name(&self) -> &str { "anthropic" }

    fn manifest(&self) -> ProviderManifest {
        ProviderManifest { provider: "anthropic".into(), env_keys: vec!["ANTHROPIC_API_KEY".into()] }
    }

    fn transport(&self) -> &UreqTransport { &self.transport }

    fn build_request(&self, request: &LMRequest, stream: bool) -> HttpRequest {
        let body = serde_json::to_vec(&self.build_payload(request, stream)).unwrap_or_default();
        HttpRequest {
            method: "POST".into(), url: format!("{}/messages", self.base_url),
            headers: self.headers(), params: HashMap::new(),
            body: Some(body), timeout: Some(std::time::Duration::from_secs(if stream { 120 } else { 60 })),
        }
    }

    fn normalize_error(&self, status: u16, body: &str) -> LM15Error {
        if let Ok(data) = serde_json::from_str::<Value>(body) {
            if let Some(err) = data.get("error").and_then(|e| e.as_object()) {
                let msg = err.get("message").and_then(|m| m.as_str()).unwrap_or("");
                let err_type = err.get("type").and_then(|t| t.as_str()).unwrap_or("");

                if err_type == "invalid_request_error" && is_context_msg(msg) {
                    return LM15Error::ContextLength(msg.into());
                }

                return match err_type {
                    "authentication_error" | "permission_error" => LM15Error::Auth(msg.into()),
                    "billing_error" => LM15Error::Billing(msg.into()),
                    "rate_limit_error" => LM15Error::RateLimit(msg.into()),
                    "api_error" | "overloaded_error" => LM15Error::Server(msg.into()),
                    "timeout_error" => LM15Error::Timeout(msg.into()),
                    _ => map_http_error(status, msg),
                };
            }
        }
        map_http_error(status, &body[..body.len().min(200)])
    }

    fn parse_response(&self, request: &LMRequest, response: &HttpResponse) -> Result<LMResponse, LM15Error> {
        let data: Value = serde_json::from_slice(&response.body).map_err(|e| LM15Error::Provider(e.to_string()))?;

        let mut parts = Vec::new();
        if let Some(content) = data.get("content").and_then(|c| c.as_array()) {
            for block in content {
                let bt = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
                match bt {
                    "text" => parts.push(Part::text(block.get("text").and_then(|t| t.as_str()).unwrap_or(""))),
                    "tool_use" => {
                        let id = block.get("id").and_then(|i| i.as_str()).unwrap_or("");
                        let name = block.get("name").and_then(|n| n.as_str()).unwrap_or("");
                        let input: JsonObject = block.get("input").and_then(|i| serde_json::from_value(i.clone()).ok()).unwrap_or_default();
                        parts.push(Part::tool_call(id, name, input));
                    }
                    "thinking" => parts.push(Part::thinking(block.get("thinking").and_then(|t| t.as_str()).unwrap_or(""))),
                    "redacted_thinking" => {
                        let mut p = Part::thinking("[redacted]");
                        p.redacted = Some(true);
                        parts.push(p);
                    }
                    _ => {}
                }
            }
        }

        if parts.is_empty() { parts.push(Part::text("")); }
        let has_tc = parts.iter().any(|p| p.part_type == PartType::ToolCall);
        let finish = if has_tc { FinishReason::ToolCall } else { FinishReason::Stop };

        let u = data.get("usage");
        let mut usage = Usage {
            input_tokens: u.and_then(|u| u.get("input_tokens")).and_then(|t| t.as_i64()).unwrap_or(0),
            output_tokens: u.and_then(|u| u.get("output_tokens")).and_then(|t| t.as_i64()).unwrap_or(0),
            ..Usage::default()
        };
        usage.total_tokens = usage.input_tokens + usage.output_tokens;
        usage.cache_read_tokens = u.and_then(|u| u.get("cache_read_input_tokens")).and_then(|t| t.as_i64());
        usage.cache_write_tokens = u.and_then(|u| u.get("cache_creation_input_tokens")).and_then(|t| t.as_i64());

        Ok(LMResponse {
            id: data.get("id").and_then(|i| i.as_str()).unwrap_or("").into(),
            model: data.get("model").and_then(|m| m.as_str()).unwrap_or(&request.model).into(),
            message: Message { role: Role::Assistant, parts, name: None },
            finish_reason: finish, usage,
            provider: serde_json::from_value(data).ok(),
        })
    }

    fn parse_stream_event(&self, _request: &LMRequest, raw: &SSEEvent) -> Result<Option<StreamEvent>, LM15Error> {
        if raw.data.is_empty() { return Ok(None); }
        let p: Value = serde_json::from_str(&raw.data).map_err(|_| LM15Error::Provider("invalid JSON".into()))?;
        let et = p.get("type").and_then(|t| t.as_str()).unwrap_or("");

        match et {
            "message_start" => {
                let msg = p.get("message");
                Ok(Some(StreamEvent::start(
                    msg.and_then(|m| m.get("id")).and_then(|i| i.as_str()).unwrap_or(""),
                    msg.and_then(|m| m.get("model")).and_then(|m| m.as_str()).unwrap_or(""),
                )))
            }
            "content_block_start" => {
                let block = p.get("content_block");
                if block.and_then(|b| b.get("type")).and_then(|t| t.as_str()) == Some("tool_use") {
                    let idx = p.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                    let mut raw_delta = JsonObject::new();
                    raw_delta.insert("type".into(), "tool_call".into());
                    raw_delta.insert("id".into(), block.and_then(|b| b.get("id")).cloned().unwrap_or(Value::Null));
                    raw_delta.insert("name".into(), block.and_then(|b| b.get("name")).cloned().unwrap_or(Value::Null));
                    raw_delta.insert("input".into(), "".into());
                    return Ok(Some(StreamEvent { event_type: "delta".into(), part_index: Some(idx), delta_raw: Some(raw_delta), ..StreamEvent::default() }));
                }
                Ok(None)
            }
            "content_block_delta" => {
                let delta = p.get("delta");
                let idx = p.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                let dt = delta.and_then(|d| d.get("type")).and_then(|t| t.as_str()).unwrap_or("");
                match dt {
                    "text_delta" => {
                        let text = delta.and_then(|d| d.get("text")).and_then(|t| t.as_str()).unwrap_or("");
                        Ok(Some(StreamEvent::text_delta(idx, text)))
                    }
                    "input_json_delta" => {
                        let input = delta.and_then(|d| d.get("partial_json")).and_then(|t| t.as_str()).unwrap_or("");
                        let mut raw_delta = JsonObject::new();
                        raw_delta.insert("type".into(), "tool_call".into());
                        raw_delta.insert("input".into(), input.into());
                        Ok(Some(StreamEvent { event_type: "delta".into(), part_index: Some(idx), delta_raw: Some(raw_delta), ..StreamEvent::default() }))
                    }
                    "thinking_delta" => {
                        let text = delta.and_then(|d| d.get("thinking")).and_then(|t| t.as_str()).unwrap_or("");
                        Ok(Some(StreamEvent { event_type: "delta".into(), part_index: Some(idx),
                            delta: Some(PartDelta { delta_type: "thinking".into(), text: Some(text.into()), data: None, input: None }),
                            ..StreamEvent::default() }))
                    }
                    _ => Ok(None),
                }
            }
            "message_stop" => Ok(Some(StreamEvent::end(FinishReason::Stop, Usage::default()))),
            "error" => {
                let e = p.get("error");
                let code = e.and_then(|e| e.get("type")).and_then(|t| t.as_str()).unwrap_or("provider");
                let msg = e.and_then(|e| e.get("message")).and_then(|m| m.as_str()).unwrap_or("");
                Ok(Some(StreamEvent::error(ErrorInfo { code: code.into(), message: msg.into(), provider_code: Some(code.into()) })))
            }
            _ => Ok(None),
        }
    }

    fn file_upload(&self, request: &FileUploadRequest) -> Result<FileUploadResponse, LM15Error> {
        let mut headers = HashMap::new();
        headers.insert("x-api-key".into(), self.api_key.clone());
        headers.insert("anthropic-version".into(), self.api_version.clone());
        headers.insert("content-type".into(), request.media_type.clone());
        headers.insert("x-filename".into(), request.filename.clone());

        let req = HttpRequest {
            method: "POST".into(), url: format!("{}/files", self.base_url),
            headers, params: HashMap::new(),
            body: Some(request.bytes_data.clone()),
            timeout: Some(std::time::Duration::from_secs(120)),
        };
        let resp = self.transport.request(&req)?;
        if resp.status >= 400 {
            return Err(self.normalize_error(resp.status, &resp.text()));
        }
        let data: Value = serde_json::from_slice(&resp.body).map_err(|e| LM15Error::Provider(e.to_string()))?;
        let id = data.get("id").and_then(|i| i.as_str())
            .or_else(|| data.pointer("/file/id").and_then(|i| i.as_str()))
            .unwrap_or("");
        Ok(FileUploadResponse { id: id.into(), provider: serde_json::from_value(data).ok() })
    }
}

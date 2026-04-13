//! OpenAI adapter (Responses API).

use crate::errors::{map_http_error, LM15Error};
use crate::transport::{HttpRequest, HttpResponse, SSEEvent, UreqTransport};
use crate::types::*;
use super::{Adapter, ProviderManifest};
use super::common::{part_to_openai_input, parts_to_text};
use serde_json::Value;
use std::collections::HashMap;

pub struct OpenAIAdapter {
    pub api_key: String,
    pub base_url: String,
    pub transport: UreqTransport,
}

impl OpenAIAdapter {
    pub fn new(api_key: &str, transport: UreqTransport) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".into(),
            transport,
        }
    }

    fn headers(&self) -> HashMap<String, String> {
        let mut h = HashMap::new();
        h.insert("Authorization".into(), format!("Bearer {}", self.api_key));
        h.insert("Content-Type".into(), "application/json".into());
        h
    }

    fn build_input(messages: &[Message]) -> Vec<Value> {
        let mut items: Vec<Value> = Vec::new();
        for msg in messages {
            if msg.role == Role::Tool {
                for part in &msg.parts {
                    if part.part_type == PartType::ToolResult {
                        if let Some(id) = &part.id {
                            let text = part.content.as_ref().map(|c| parts_to_text(c)).unwrap_or_default();
                            items.push(serde_json::json!({
                                "type": "function_call_output",
                                "call_id": id,
                                "output": text,
                            }));
                        }
                    }
                }
                continue;
            }
            let content_parts: Vec<Value> = msg.parts.iter()
                .filter(|p| p.part_type != PartType::ToolCall && p.part_type != PartType::ToolResult)
                .map(part_to_openai_input)
                .collect();
            if !content_parts.is_empty() {
                items.push(serde_json::json!({"role": msg.role, "content": content_parts}));
            }
            for part in &msg.parts {
                if part.part_type == PartType::ToolCall {
                    if let (Some(id), Some(name)) = (&part.id, &part.name) {
                        let args = part.input.as_ref()
                            .map(|i| serde_json::to_string(i).unwrap_or_else(|_| "{}".into()))
                            .unwrap_or_else(|| "{}".into());
                        items.push(serde_json::json!({
                            "type": "function_call",
                            "call_id": id,
                            "name": name,
                            "arguments": args,
                        }));
                    }
                }
            }
        }
        items
    }

    fn payload(&self, request: &LMRequest, stream: bool) -> Value {
        let mut p = serde_json::json!({
            "model": request.model,
            "input": Self::build_input(&request.messages),
            "stream": stream,
        });

        if let Some(system) = &request.system {
            p["instructions"] = Value::String(system.clone());
        }
        if let Some(max_tokens) = request.config.max_tokens {
            p["max_output_tokens"] = Value::Number(max_tokens.into());
        }
        if let Some(temp) = request.config.temperature {
            p["temperature"] = serde_json::json!(temp);
        }

        let tools: Vec<Value> = request.tools.iter()
            .filter(|t| t.tool_type == "function")
            .map(|t| serde_json::json!({
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters.as_ref().unwrap_or(&HashMap::new()),
            }))
            .collect();
        if !tools.is_empty() {
            p["tools"] = Value::Array(tools);
        }

        if let Some(provider) = &request.config.provider {
            for (k, v) in provider {
                if k != "prompt_caching" {
                    p[k] = v.clone();
                }
            }
        }

        p
    }
}

impl Adapter for OpenAIAdapter {
    fn provider_name(&self) -> &str { "openai" }

    fn manifest(&self) -> ProviderManifest {
        ProviderManifest {
            provider: "openai".into(),
            env_keys: vec!["OPENAI_API_KEY".into()],
        }
    }

    fn transport(&self) -> &UreqTransport { &self.transport }

    fn build_request(&self, request: &LMRequest, stream: bool) -> HttpRequest {
        let body = serde_json::to_vec(&self.payload(request, stream)).unwrap_or_default();
        let timeout = if stream { 120 } else { 60 };
        HttpRequest {
            method: "POST".into(),
            url: format!("{}/responses", self.base_url),
            headers: self.headers(),
            params: HashMap::new(),
            body: Some(body),
            timeout: Some(std::time::Duration::from_secs(timeout)),
        }
    }

    fn normalize_error(&self, status: u16, body: &str) -> LM15Error {
        if let Ok(data) = serde_json::from_str::<Value>(body) {
            if let Some(err) = data.get("error").and_then(|e| e.as_object()) {
                let msg = err.get("message").and_then(|m| m.as_str()).unwrap_or("");
                let code = err.get("code").and_then(|c| c.as_str()).unwrap_or("");
                let err_type = err.get("type").and_then(|t| t.as_str()).unwrap_or("");

                if code == "context_length_exceeded" { return LM15Error::ContextLength(msg.into()); }
                if code == "insufficient_quota" || err_type == "insufficient_quota" { return LM15Error::Billing(msg.into()); }
                if code == "invalid_api_key" || err_type == "authentication_error" { return LM15Error::Auth(msg.into()); }
                if code == "rate_limit_exceeded" || err_type == "rate_limit_error" { return LM15Error::RateLimit(msg.into()); }

                let final_msg = if !code.is_empty() && !msg.contains(code) {
                    format!("{msg} ({code})")
                } else {
                    msg.into()
                };
                return map_http_error(status, &final_msg);
            }
        }
        map_http_error(status, &body[..body.len().min(200)])
    }

    fn parse_response(&self, request: &LMRequest, response: &HttpResponse) -> Result<LMResponse, LM15Error> {
        let data: Value = serde_json::from_slice(&response.body).map_err(|e| LM15Error::Provider(e.to_string()))?;

        // In-band error
        if let Some(err) = data.get("error").and_then(|e| e.as_object()) {
            let msg = err.get("message").and_then(|m| m.as_str()).unwrap_or("server error");
            return Err(LM15Error::Server(msg.into()));
        }

        let mut parts = Vec::new();
        if let Some(output) = data.get("output").and_then(|o| o.as_array()) {
            for item in output {
                let item_type = item.get("type").and_then(|t| t.as_str()).unwrap_or("");
                if item_type == "message" {
                    if let Some(content) = item.get("content").and_then(|c| c.as_array()) {
                        for c in content {
                            let ct = c.get("type").and_then(|t| t.as_str()).unwrap_or("");
                            match ct {
                                "output_text" | "text" => {
                                    parts.push(Part::text(c.get("text").and_then(|t| t.as_str()).unwrap_or("")));
                                }
                                "refusal" => {
                                    parts.push(Part::refusal(c.get("refusal").and_then(|t| t.as_str()).unwrap_or("")));
                                }
                                _ => {}
                            }
                        }
                    }
                } else if item_type == "function_call" {
                    let call_id = item.get("call_id").and_then(|c| c.as_str()).unwrap_or("");
                    let name = item.get("name").and_then(|n| n.as_str()).unwrap_or("");
                    let args_str = item.get("arguments").and_then(|a| a.as_str()).unwrap_or("{}");
                    let args: JsonObject = serde_json::from_str(args_str).unwrap_or_default();
                    parts.push(Part::tool_call(call_id, name, args));
                }
            }
        }

        if parts.is_empty() {
            let text = data.get("output_text").and_then(|t| t.as_str()).unwrap_or("");
            parts.push(Part::text(text));
        }

        let has_tool_call = parts.iter().any(|p| p.part_type == PartType::ToolCall);
        let finish = if has_tool_call { FinishReason::ToolCall } else { FinishReason::Stop };

        let usage = parse_openai_usage(&data);

        Ok(LMResponse {
            id: data.get("id").and_then(|i| i.as_str()).unwrap_or("").into(),
            model: data.get("model").and_then(|m| m.as_str()).unwrap_or(&request.model).into(),
            message: Message { role: Role::Assistant, parts, name: None },
            finish_reason: finish,
            usage,
            provider: serde_json::from_value(data).ok(),
        })
    }

    fn parse_stream_event(&self, request: &LMRequest, raw: &SSEEvent) -> Result<Option<StreamEvent>, LM15Error> {
        if raw.data.is_empty() { return Ok(None); }
        if raw.data == "[DONE]" {
            return Ok(Some(StreamEvent::end(FinishReason::Stop, Usage::default())));
        }

        let p: Value = serde_json::from_str(&raw.data).map_err(|_| LM15Error::Provider("invalid JSON".into()))?;
        let et = p.get("type").and_then(|t| t.as_str()).unwrap_or("");

        match et {
            "response.created" => {
                let id = p.pointer("/response/id").and_then(|i| i.as_str()).unwrap_or("");
                Ok(Some(StreamEvent::start(id, &request.model)))
            }
            "response.output_text.delta" | "response.refusal.delta" => {
                let delta = p.get("delta").and_then(|d| d.as_str()).unwrap_or("");
                Ok(Some(StreamEvent::text_delta(0, delta)))
            }
            "response.output_item.added" => {
                let item = p.get("item").and_then(|i| i.as_object());
                if let Some(item) = item {
                    if item.get("type").and_then(|t| t.as_str()) == Some("function_call") {
                        let idx = p.get("output_index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                        let mut raw_delta = JsonObject::new();
                        raw_delta.insert("type".into(), "tool_call".into());
                        raw_delta.insert("id".into(), item.get("call_id").cloned().unwrap_or(Value::Null));
                        raw_delta.insert("name".into(), item.get("name").cloned().unwrap_or(Value::Null));
                        raw_delta.insert("input".into(), item.get("arguments").cloned().unwrap_or("".into()));
                        return Ok(Some(StreamEvent { event_type: "delta".into(), part_index: Some(idx), delta_raw: Some(raw_delta), ..StreamEvent::default() }));
                    }
                }
                Ok(None)
            }
            "response.function_call_arguments.delta" => {
                let idx = p.get("output_index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                let mut raw_delta = JsonObject::new();
                raw_delta.insert("type".into(), "tool_call".into());
                raw_delta.insert("id".into(), p.get("call_id").cloned().unwrap_or(Value::Null));
                raw_delta.insert("name".into(), p.get("name").cloned().unwrap_or(Value::Null));
                raw_delta.insert("input".into(), p.get("delta").cloned().unwrap_or("".into()));
                Ok(Some(StreamEvent { event_type: "delta".into(), part_index: Some(idx), delta_raw: Some(raw_delta), ..StreamEvent::default() }))
            }
            "response.completed" => {
                let resp = p.get("response");
                let usage = resp.map(|r| parse_openai_usage(r)).unwrap_or_default();
                let has_fc = resp
                    .and_then(|r| r.get("output"))
                    .and_then(|o| o.as_array())
                    .map(|arr| arr.iter().any(|i| i.get("type").and_then(|t| t.as_str()) == Some("function_call")))
                    .unwrap_or(false);
                let finish = if has_fc { FinishReason::ToolCall } else { FinishReason::Stop };
                Ok(Some(StreamEvent::end(finish, usage)))
            }
            "response.error" | "error" => {
                let err = p.get("error");
                let code = err.and_then(|e| e.get("code")).and_then(|c| c.as_str())
                    .or_else(|| err.and_then(|e| e.get("type")).and_then(|t| t.as_str()))
                    .unwrap_or("provider");
                let msg = err.and_then(|e| e.get("message")).and_then(|m| m.as_str()).unwrap_or("");
                Ok(Some(StreamEvent::error(ErrorInfo { code: code.into(), message: msg.into(), provider_code: Some(code.into()) })))
            }
            _ => Ok(None),
        }
    }

    fn embeddings(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse, LM15Error> {
        let payload = serde_json::json!({"model": request.model, "input": request.inputs});
        let body = serde_json::to_vec(&payload).unwrap_or_default();
        let req = HttpRequest {
            method: "POST".into(),
            url: format!("{}/embeddings", self.base_url),
            headers: self.headers(),
            params: HashMap::new(),
            body: Some(body),
            timeout: Some(std::time::Duration::from_secs(60)),
        };
        let resp = self.transport.request(&req)?;
        if resp.status >= 400 {
            return Err(self.normalize_error(resp.status, &resp.text()));
        }
        let data: Value = serde_json::from_slice(&resp.body).map_err(|e| LM15Error::Provider(e.to_string()))?;
        let vectors = data.get("data").and_then(|d| d.as_array())
            .map(|arr| arr.iter().map(|item| {
                item.get("embedding").and_then(|e| e.as_array())
                    .map(|v| v.iter().filter_map(|x| x.as_f64()).collect())
                    .unwrap_or_default()
            }).collect())
            .unwrap_or_default();

        Ok(EmbeddingResponse { model: request.model.clone(), vectors, usage: Usage::default(), provider: None })
    }
}

fn parse_openai_usage(data: &Value) -> Usage {
    let u = data.get("usage");
    let u_in = u.and_then(|u| u.get("input_tokens_details"));
    let u_out = u.and_then(|u| u.get("output_tokens_details"));
    Usage {
        input_tokens: u.and_then(|u| u.get("input_tokens")).and_then(|t| t.as_i64()).unwrap_or(0),
        output_tokens: u.and_then(|u| u.get("output_tokens")).and_then(|t| t.as_i64()).unwrap_or(0),
        total_tokens: u.and_then(|u| u.get("total_tokens")).and_then(|t| t.as_i64()).unwrap_or(0),
        reasoning_tokens: u_out.and_then(|d| d.get("reasoning_tokens")).and_then(|t| t.as_i64()),
        cache_read_tokens: u_in.and_then(|d| d.get("cached_tokens")).and_then(|t| t.as_i64()),
        input_audio_tokens: u_in.and_then(|d| d.get("audio_tokens")).and_then(|t| t.as_i64()),
        output_audio_tokens: u_out.and_then(|d| d.get("audio_tokens")).and_then(|t| t.as_i64()),
        ..Usage::default()
    }
}

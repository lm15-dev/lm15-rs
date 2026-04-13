//! Gemini adapter (GenerativeLanguage API).

use crate::errors::{map_http_error, LM15Error};
use crate::transport::{HttpRequest, HttpResponse, SSEEvent, UreqTransport};
use crate::types::*;
use super::{Adapter, ProviderManifest};
use serde_json::Value;
use std::collections::HashMap;

fn builtin_to_gemini(tool: &Tool) -> Value {
    let wire_key = match tool.name.as_str() {
        "web_search" => "googleSearch",
        "code_execution" => "codeExecution",
        other => other,
    };
    let cfg = tool.builtin_config.as_ref()
        .map(|c| serde_json::json!(c))
        .unwrap_or(serde_json::json!({}));
    serde_json::json!({ wire_key: cfg })
}

pub struct GeminiAdapter {
    pub api_key: String,
    pub base_url: String,
    pub transport: UreqTransport,
}

impl GeminiAdapter {
    pub fn new(api_key: &str, transport: UreqTransport) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://generativelanguage.googleapis.com/v1beta".into(),
            transport,
        }
    }

    fn model_path(&self, model: &str) -> String {
        if model.starts_with("models/") { model.into() } else { format!("models/{model}") }
    }

    fn auth_headers(&self) -> HashMap<String, String> {
        let mut h = HashMap::new();
        h.insert("x-goog-api-key".into(), self.api_key.clone());
        h.insert("Content-Type".into(), "application/json".into());
        h
    }

    fn part_payload(&self, p: &Part) -> Value {
        match &p.part_type {
            PartType::Text => serde_json::json!({"text": p.text.as_deref().unwrap_or("")}),
            PartType::Image | PartType::Audio | PartType::Video | PartType::Document => {
                if let Some(s) = &p.source {
                    let mime = s.media_type.as_deref().unwrap_or("application/octet-stream");
                    if s.source_type == "url" {
                        return serde_json::json!({"fileData": {"mimeType": mime, "fileUri": s.url}});
                    }
                    if s.source_type == "base64" {
                        return serde_json::json!({"inlineData": {"mimeType": mime, "data": s.data}});
                    }
                    if s.source_type == "file" {
                        return serde_json::json!({"fileData": {"mimeType": mime, "fileUri": s.file_id}});
                    }
                }
                serde_json::json!({"text": ""})
            }
            PartType::ToolCall => {
                let mut fc = serde_json::json!({"name": p.name.as_deref().unwrap_or(""), "args": p.input.as_ref().unwrap_or(&HashMap::new())});
                if let Some(id) = &p.id {
                    fc["id"] = Value::String(id.clone());
                }
                serde_json::json!({"functionCall": fc})
            }
            PartType::ToolResult => {
                let text = p.content.as_ref().map(|c| {
                    c.iter().filter(|x| x.part_type == PartType::Text)
                        .filter_map(|x| x.text.as_deref()).collect::<Vec<_>>().join("")
                }).unwrap_or_default();
                let mut fr = serde_json::json!({"name": p.name.as_deref().unwrap_or("tool"), "response": {"result": text}});
                if let Some(id) = &p.id {
                    fr["id"] = Value::String(id.clone());
                }
                serde_json::json!({"functionResponse": fr})
            }
            _ => serde_json::json!({"text": p.text.as_deref().unwrap_or("")}),
        }
    }

    fn build_payload(&self, request: &LMRequest) -> Value {
        let contents: Vec<Value> = request.messages.iter().map(|m| {
            let role = if m.role == Role::Assistant { "model" } else { "user" };
            let parts: Vec<Value> = m.parts.iter().map(|p| self.part_payload(p)).collect();
            serde_json::json!({"role": role, "parts": parts})
        }).collect();

        let mut payload = serde_json::json!({"contents": contents});

        if let Some(system) = &request.system {
            payload["systemInstruction"] = serde_json::json!({"parts": [{"text": system}]});
        }

        let mut cfg = serde_json::Map::new();
        if let Some(temp) = request.config.temperature {
            cfg.insert("temperature".into(), serde_json::json!(temp));
        }
        if let Some(max_tokens) = request.config.max_tokens {
            cfg.insert("maxOutputTokens".into(), serde_json::json!(max_tokens));
        }
        if let Some(stop) = &request.config.stop {
            cfg.insert("stopSequences".into(), serde_json::json!(stop));
        }
        if !cfg.is_empty() {
            payload["generationConfig"] = Value::Object(cfg);
        }

        let func_decls: Vec<Value> = request.tools.iter()
            .filter(|t| t.tool_type == "function")
            .map(|t| serde_json::json!({
                "name": t.name, "description": t.description,
                "parameters": t.parameters.as_ref().unwrap_or(&HashMap::new()),
            }))
            .collect();
        let mut tools_wire: Vec<Value> = Vec::new();
        if !func_decls.is_empty() {
            tools_wire.push(serde_json::json!({"functionDeclarations": func_decls}));
        }
        for t in &request.tools {
            if t.tool_type == "builtin" {
                tools_wire.push(builtin_to_gemini(t));
            }
        }
        if !tools_wire.is_empty() {
            payload["tools"] = Value::Array(tools_wire);
        }

        if let Some(provider) = &request.config.provider {
            for (k, v) in provider {
                if k != "prompt_caching" && k != "output" { payload[k] = v.clone(); }
            }
            if let Some(output) = provider.get("output").and_then(|o| o.as_str()) {
                let gen_cfg = payload.get_mut("generationConfig")
                    .and_then(|g| g.as_object_mut());
                match output {
                    "image" => { if let Some(g) = gen_cfg { g.insert("responseModalities".into(), serde_json::json!(["IMAGE"])); } }
                    "audio" => { if let Some(g) = gen_cfg { g.insert("responseModalities".into(), serde_json::json!(["AUDIO"])); } }
                    _ => {}
                }
            }
        }

        payload
    }
}

fn is_context_msg(msg: &str) -> bool {
    let m = msg.to_lowercase();
    (m.contains("token") && (m.contains("limit") || m.contains("exceed")))
        || m.contains("too long") || m.contains("context length")
}

impl Adapter for GeminiAdapter {
    fn provider_name(&self) -> &str { "gemini" }

    fn manifest(&self) -> ProviderManifest {
        ProviderManifest { provider: "gemini".into(), env_keys: vec!["GEMINI_API_KEY".into(), "GOOGLE_API_KEY".into()] }
    }

    fn transport(&self) -> &UreqTransport { &self.transport }

    fn build_request(&self, request: &LMRequest, stream: bool) -> HttpRequest {
        let endpoint = if stream { "streamGenerateContent" } else { "generateContent" };
        let mut params = HashMap::new();
        if stream { params.insert("alt".into(), "sse".into()); }

        let body = serde_json::to_vec(&self.build_payload(request)).unwrap_or_default();
        HttpRequest {
            method: "POST".into(),
            url: format!("{}/{}:{}", self.base_url, self.model_path(&request.model), endpoint),
            headers: self.auth_headers(), params,
            body: Some(body), timeout: Some(std::time::Duration::from_secs(if stream { 120 } else { 60 })),
        }
    }

    fn normalize_error(&self, status: u16, body: &str) -> LM15Error {
        if let Ok(data) = serde_json::from_str::<Value>(body) {
            if let Some(err) = data.get("error").and_then(|e| e.as_object()) {
                let msg = err.get("message").and_then(|m| m.as_str()).unwrap_or("");
                let err_status = err.get("status").and_then(|s| s.as_str()).unwrap_or("");
                if is_context_msg(msg) { return LM15Error::ContextLength(msg.into()); }
                return match err_status {
                    "INVALID_ARGUMENT" | "NOT_FOUND" => LM15Error::InvalidRequest(msg.into()),
                    "FAILED_PRECONDITION" => LM15Error::Billing(msg.into()),
                    "PERMISSION_DENIED" => LM15Error::Auth(msg.into()),
                    "RESOURCE_EXHAUSTED" => LM15Error::RateLimit(msg.into()),
                    "INTERNAL" | "UNAVAILABLE" => LM15Error::Server(msg.into()),
                    "DEADLINE_EXCEEDED" => LM15Error::Timeout(msg.into()),
                    _ => map_http_error(status, msg),
                };
            }
        }
        map_http_error(status, &body[..body.len().min(200)])
    }

    fn parse_response(&self, request: &LMRequest, response: &HttpResponse) -> Result<LMResponse, LM15Error> {
        let data: Value = serde_json::from_slice(&response.body).map_err(|e| LM15Error::Provider(e.to_string()))?;

        // In-band error
        if let Some(reason) = data.pointer("/promptFeedback/blockReason").and_then(|r| r.as_str()) {
            if reason != "BLOCK_REASON_UNSPECIFIED" {
                return Err(LM15Error::InvalidRequest(format!("Prompt blocked: {reason}")));
            }
        }

        let candidate = data.get("candidates").and_then(|c| c.get(0));
        let content = candidate.and_then(|c| c.get("content"));
        let raw_parts = content.and_then(|c| c.get("parts")).and_then(|p| p.as_array());

        let mut parts = Vec::new();
        if let Some(raw_parts) = raw_parts {
            for p in raw_parts {
                if let Some(text) = p.get("text").and_then(|t| t.as_str()) {
                    parts.push(Part::text(text));
                } else if let Some(fc) = p.get("functionCall").and_then(|f| f.as_object()) {
                    let id = fc.get("id").and_then(|i| i.as_str()).unwrap_or("fc_0");
                    let name = fc.get("name").and_then(|n| n.as_str()).unwrap_or("");
                    let args: JsonObject = fc.get("args").and_then(|a| serde_json::from_value(a.clone()).ok()).unwrap_or_default();
                    parts.push(Part::tool_call(id, name, args));
                } else if let Some(inline) = p.get("inlineData").and_then(|i| i.as_object()) {
                    let mime = inline.get("mimeType").and_then(|m| m.as_str()).unwrap_or("application/octet-stream");
                    let data_str = inline.get("data").and_then(|d| d.as_str()).unwrap_or("");
                    if mime.starts_with("image/") {
                        parts.push(Part::image_base64(data_str, mime));
                    } else if mime.starts_with("audio/") {
                        parts.push(Part::audio_base64(data_str, mime));
                    }
                }
            }
        }

        if parts.is_empty() { parts.push(Part::text("")); }
        let has_tc = parts.iter().any(|p| p.part_type == PartType::ToolCall);

        let um = data.get("usageMetadata");
        let mut usage = Usage {
            input_tokens: um.and_then(|u| u.get("promptTokenCount")).and_then(|t| t.as_i64()).unwrap_or(0),
            output_tokens: um.and_then(|u| u.get("candidatesTokenCount")).and_then(|t| t.as_i64()).unwrap_or(0),
            total_tokens: um.and_then(|u| u.get("totalTokenCount")).and_then(|t| t.as_i64()).unwrap_or(0),
            ..Usage::default()
        };
        usage.cache_read_tokens = um.and_then(|u| u.get("cachedContentTokenCount")).and_then(|t| t.as_i64());
        usage.reasoning_tokens = um.and_then(|u| u.get("thoughtsTokenCount")).and_then(|t| t.as_i64());

        Ok(LMResponse {
            id: data.get("responseId").and_then(|i| i.as_str()).unwrap_or("").into(),
            model: request.model.clone(),
            message: Message { role: Role::Assistant, parts, name: None },
            finish_reason: if has_tc { FinishReason::ToolCall } else { FinishReason::Stop },
            usage,
            provider: serde_json::from_value(data).ok(),
        })
    }

    fn parse_stream_event(&self, _request: &LMRequest, raw: &SSEEvent) -> Result<Option<StreamEvent>, LM15Error> {
        if raw.data.is_empty() { return Ok(None); }
        let payload: Value = serde_json::from_str(&raw.data).map_err(|_| LM15Error::Provider("invalid JSON".into()))?;

        if let Some(err) = payload.get("error").and_then(|e| e.as_object()) {
            let code = err.get("status").or(err.get("code")).and_then(|c| c.as_str()).unwrap_or("provider");
            let msg = err.get("message").and_then(|m| m.as_str()).unwrap_or("");
            return Ok(Some(StreamEvent::error(ErrorInfo { code: code.into(), message: msg.into(), provider_code: Some(code.into()) })));
        }

        let cands = payload.get("candidates").and_then(|c| c.as_array());
        let part = cands.and_then(|c| c.first())
            .and_then(|c| c.pointer("/content/parts"))
            .and_then(|p| p.as_array())
            .and_then(|p| p.first());

        if let Some(part) = part {
            if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                return Ok(Some(StreamEvent::text_delta(0, text)));
            }
            if let Some(fc) = part.get("functionCall").and_then(|f| f.as_object()) {
                let args = serde_json::to_string(fc.get("args").unwrap_or(&Value::Object(Default::default()))).unwrap_or_default();
                let mut raw_delta = JsonObject::new();
                raw_delta.insert("type".into(), "tool_call".into());
                raw_delta.insert("id".into(), fc.get("id").cloned().unwrap_or("fc_0".into()));
                raw_delta.insert("name".into(), fc.get("name").cloned().unwrap_or(Value::Null));
                raw_delta.insert("input".into(), args.into());
                return Ok(Some(StreamEvent { event_type: "delta".into(), part_index: Some(0), delta_raw: Some(raw_delta), ..StreamEvent::default() }));
            }
            if let Some(inline) = part.get("inlineData").and_then(|i| i.as_object()) {
                let mime = inline.get("mimeType").and_then(|m| m.as_str()).unwrap_or("");
                if mime.starts_with("audio/") {
                    let data = inline.get("data").and_then(|d| d.as_str()).unwrap_or("");
                    return Ok(Some(StreamEvent { event_type: "delta".into(), part_index: Some(0),
                        delta: Some(PartDelta { delta_type: "audio".into(), text: None, data: Some(data.into()), input: None }),
                        ..StreamEvent::default() }));
                }
            }
        }

        Ok(None)
    }

    fn embeddings(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse, LM15Error> {
        let model_path = self.model_path(&request.model);
        if request.inputs.len() <= 1 {
            let input = request.inputs.first().map(|s| s.as_str()).unwrap_or("");
            let payload = serde_json::json!({"model": model_path, "content": {"parts": [{"text": input}]}});
            let body = serde_json::to_vec(&payload).unwrap_or_default();
            let req = HttpRequest {
                method: "POST".into(), url: format!("{}/{}:embedContent", self.base_url, model_path),
                headers: self.auth_headers(), params: HashMap::new(),
                body: Some(body), timeout: Some(std::time::Duration::from_secs(60)),
            };
            let resp = self.transport.request(&req)?;
            if resp.status >= 400 { return Err(self.normalize_error(resp.status, &resp.text())); }
            let data: Value = serde_json::from_slice(&resp.body).map_err(|e| LM15Error::Provider(e.to_string()))?;
            let values = data.pointer("/embedding/values").and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|x| x.as_f64()).collect()).unwrap_or_default();
            return Ok(EmbeddingResponse { model: request.model.clone(), vectors: vec![values], usage: Usage::default(), provider: None });
        }

        let requests: Vec<Value> = request.inputs.iter()
            .map(|input| serde_json::json!({"model": model_path, "content": {"parts": [{"text": input}]}}))
            .collect();
        let payload = serde_json::json!({"requests": requests});
        let body = serde_json::to_vec(&payload).unwrap_or_default();
        let req = HttpRequest {
            method: "POST".into(), url: format!("{}/{}:batchEmbedContents", self.base_url, model_path),
            headers: self.auth_headers(), params: HashMap::new(),
            body: Some(body), timeout: Some(std::time::Duration::from_secs(60)),
        };
        let resp = self.transport.request(&req)?;
        if resp.status >= 400 { return Err(self.normalize_error(resp.status, &resp.text())); }
        let data: Value = serde_json::from_slice(&resp.body).map_err(|e| LM15Error::Provider(e.to_string()))?;
        let vectors = data.get("embeddings").and_then(|e| e.as_array())
            .map(|arr| arr.iter().map(|e| e.get("values").and_then(|v| v.as_array())
                .map(|v| v.iter().filter_map(|x| x.as_f64()).collect()).unwrap_or_default()
            ).collect()).unwrap_or_default();
        Ok(EmbeddingResponse { model: request.model.clone(), vectors, usage: Usage::default(), provider: None })
    }

    fn file_upload(&self, request: &FileUploadRequest) -> Result<FileUploadResponse, LM15Error> {
        let upload_base = self.base_url.replace("/v1beta", "/upload/v1beta");
        let mut headers = HashMap::new();
        headers.insert("x-goog-api-key".into(), self.api_key.clone());
        headers.insert("X-Goog-Upload-Protocol".into(), "raw".into());
        headers.insert("X-Goog-Upload-File-Name".into(), request.filename.clone());
        headers.insert("Content-Type".into(), request.media_type.clone());

        let req = HttpRequest {
            method: "POST".into(), url: format!("{upload_base}/files"),
            headers, params: HashMap::new(),
            body: Some(request.bytes_data.clone()),
            timeout: Some(std::time::Duration::from_secs(120)),
        };
        let resp = self.transport.request(&req)?;
        if resp.status >= 400 { return Err(self.normalize_error(resp.status, &resp.text())); }
        let data: Value = serde_json::from_slice(&resp.body).map_err(|e| LM15Error::Provider(e.to_string()))?;
        let id = data.pointer("/file/name").or(data.get("name")).and_then(|n| n.as_str()).unwrap_or("");
        Ok(FileUploadResponse { id: id.into(), provider: serde_json::from_value(data).ok() })
    }
}

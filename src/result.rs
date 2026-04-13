//! Result — lazy stream-backed response with auto tool execution.

use crate::cost::{lookup_cost, CostBreakdown};
use crate::errors::{error_for_code, is_transient, LM15Error};
use crate::types::*;
use std::collections::HashMap;

/// Function that opens a stream for a request.
pub type StartStreamFn = Box<dyn Fn(&LMRequest) -> Result<Vec<StreamEvent>, LM15Error> + Send + Sync>;

/// Callback when a result is finalized.
pub type OnFinishedFn = Box<dyn FnMut(&LMRequest, &LMResponse) + Send>;

/// Tool execution function.
pub type ToolFn = Box<dyn Fn(&JsonObject) -> Result<serde_json::Value, String> + Send + Sync>;

/// Options for creating a Result.
pub struct ResultOpts {
    pub request: LMRequest,
    pub start_stream: StartStreamFn,
    pub on_finished: Option<OnFinishedFn>,
    pub callable_registry: HashMap<String, ToolFn>,
    pub on_tool_call: Option<Box<dyn Fn(&ToolCallInfo) -> Option<String> + Send + Sync>>,
    pub max_tool_rounds: usize,
    pub retries: usize,
}

/// Lazy stream-backed response.
///
/// Call `.text()` to block and get text, `.stream()` to iterate chunks,
/// or `.response()` to get the full `LMResponse`.
pub struct LMResult {
    opts: ResultOpts,
    response: Option<LMResponse>,
    consumed: bool,
}

impl LMResult {
    /// Create a new Result.
    pub fn new(opts: ResultOpts) -> Self {
        Self { opts, response: None, consumed: false }
    }

    /// Block and return the full response.
    pub fn response(&mut self) -> Result<&LMResponse, LM15Error> {
        if self.response.is_none() {
            self.consume()?;
        }
        self.response.as_ref().ok_or_else(|| LM15Error::Provider("no response".into()))
    }

    /// Block and return the response text.
    pub fn text(&mut self) -> Result<String, LM15Error> {
        let resp = self.response()?;
        Ok(resp.text().unwrap_or_default())
    }

    /// Block and return the thinking text.
    pub fn thinking(&mut self) -> Result<Option<String>, LM15Error> {
        let resp = self.response()?;
        Ok(resp.thinking())
    }

    /// Block and return tool calls.
    pub fn tool_calls(&mut self) -> Result<Vec<Part>, LM15Error> {
        let resp = self.response()?;
        Ok(resp.tool_calls().into_iter().cloned().collect())
    }

    /// Block and return the finish reason.
    pub fn finish_reason(&mut self) -> Result<FinishReason, LM15Error> {
        let resp = self.response()?;
        Ok(resp.finish_reason.clone())
    }

    /// Block and return usage.
    pub fn usage(&mut self) -> Result<Usage, LM15Error> {
        let resp = self.response()?;
        Ok(resp.usage.clone())
    }

    /// Block and return estimated cost, or None if cost tracking is disabled.
    pub fn cost(&mut self) -> Result<Option<CostBreakdown>, LM15Error> {
        let resp = self.response()?;
        Ok(lookup_cost(&resp.model, &resp.usage))
    }

    /// Block and return first image part.
    pub fn image(&mut self) -> Result<Option<Part>, LM15Error> {
        let resp = self.response()?;
        Ok(resp.image().cloned())
    }

    /// Block and return first audio part.
    pub fn audio(&mut self) -> Result<Option<Part>, LM15Error> {
        let resp = self.response()?;
        Ok(resp.audio().cloned())
    }

    /// Block and parse response text as JSON.
    pub fn json<T: serde::de::DeserializeOwned>(&mut self) -> Result<T, LM15Error> {
        let resp = self.response()?;
        resp.json::<T>().map_err(|e| LM15Error::Provider(e))
    }

    /// Stream chunks for incremental consumption.
    pub fn stream(&mut self) -> Result<Vec<StreamChunk>, LM15Error> {
        self.run_stream()
    }

    // ── Internal ───────────────────────────────────────────────────

    fn consume(&mut self) -> Result<(), LM15Error> {
        let _chunks = self.run_stream()?;
        // response is set by run_stream
        Ok(())
    }

    fn run_stream(&mut self) -> Result<Vec<StreamChunk>, LM15Error> {
        if self.consumed {
            return Ok(Vec::new());
        }
        self.consumed = true;

        let mut current_request = self.opts.request.clone();
        let mut all_chunks = Vec::new();
        let mut rounds = 0;

        loop {
            let mut state = RoundState::new(&current_request);

            // Retry loop
            let events = self.stream_with_retries(&current_request)?;

            for event in &events {
                if event.event_type == "error" {
                    if let Some(err) = &event.error {
                        return Err(error_for_code(&err.code, &err.message));
                    }
                    return Err(LM15Error::Provider("stream error".into()));
                }
                let chunks = state.apply(event);
                all_chunks.extend(chunks);
            }

            let resp = state.materialize();
            self.response = Some(resp.clone());

            // Yield tool_call chunks
            let tool_calls: Vec<Part> = resp.tool_calls().into_iter().cloned().collect();
            for tc in &tool_calls {
                all_chunks.push(StreamChunk {
                    chunk_type: "tool_call".into(),
                    text: None,
                    name: tc.name.clone(),
                    input: tc.input.clone(),
                    response: None,
                });
            }

            // Auto-execute tools
            if resp.finish_reason == FinishReason::ToolCall
                && !tool_calls.is_empty()
                && rounds < self.opts.max_tool_rounds
            {
                let executed = self.execute_tools(&tool_calls);
                if executed.len() == tool_calls.len() {
                    for outcome in &executed {
                        all_chunks.push(StreamChunk {
                            chunk_type: "tool_result".into(),
                            text: Some(outcome.preview.clone()),
                            name: Some(outcome.name.clone()),
                            input: None,
                            response: None,
                        });
                    }

                    let tool_parts: Vec<Part> = executed.into_iter().map(|e| e.part).collect();
                    let tool_msg = Message {
                        role: Role::Tool,
                        parts: tool_parts,
                        name: None,
                    };

                    let mut new_messages = current_request.messages.clone();
                    new_messages.push(resp.message.clone());
                    new_messages.push(tool_msg);

                    current_request = LMRequest {
                        model: current_request.model.clone(),
                        messages: new_messages,
                        system: current_request.system.clone(),
                        tools: current_request.tools.clone(),
                        config: current_request.config.clone(),
                    };
                    rounds += 1;
                    continue;
                }
            }

            // Finalize
            if let Some(on_finished) = &mut self.opts.on_finished {
                on_finished(&current_request, &resp);
            }

            all_chunks.push(StreamChunk {
                chunk_type: "finished".into(),
                text: None,
                name: None,
                input: None,
                response: Some(resp),
            });
            return Ok(all_chunks);
        }
    }

    fn stream_with_retries(&self, request: &LMRequest) -> Result<Vec<StreamEvent>, LM15Error> {
        let mut last_err = None;
        for attempt in 0..=self.opts.retries {
            match (self.opts.start_stream)(request) {
                Ok(events) => return Ok(events),
                Err(e) => {
                    if attempt == self.opts.retries || !is_transient(&e) {
                        return Err(e);
                    }
                    last_err = Some(e);
                    std::thread::sleep(std::time::Duration::from_millis(200 * (1 << attempt)));
                }
            }
        }
        Err(last_err.unwrap_or_else(|| LM15Error::Provider("unreachable".into())))
    }

    fn execute_tools(&self, tool_calls: &[Part]) -> Vec<ExecutedTool> {
        let mut results = Vec::new();
        for tc in tool_calls {
            let id = tc.id.as_deref().unwrap_or("");
            let name = tc.name.as_deref().unwrap_or("tool");
            let input = tc.input.as_ref().cloned().unwrap_or_default();

            let info = ToolCallInfo {
                id: id.into(),
                name: name.into(),
                input: input.clone(),
            };

            // Check on_tool_call callback
            if let Some(callback) = &self.opts.on_tool_call {
                if let Some(override_result) = callback(&info) {
                    results.push(ExecutedTool {
                        name: name.into(),
                        part: Part::tool_result(id, vec![Part::text(&override_result)], Some(name)),
                        preview: override_result,
                    });
                    continue;
                }
            }

            // Check callable registry
            if let Some(func) = self.opts.callable_registry.get(name) {
                let output = match func(&input) {
                    Ok(val) => val.to_string(),
                    Err(e) => format!("error: {e}"),
                };
                results.push(ExecutedTool {
                    name: name.into(),
                    part: Part::tool_result(id, vec![Part::text(&output)], Some(name)),
                    preview: output,
                });
                continue;
            }

            // Can't execute — return partial
            return results;
        }
        results
    }
}

struct ExecutedTool {
    name: String,
    part: Part,
    preview: String,
}

// ── RoundState ─────────────────────────────────────────────────────

struct RoundState {
    request: LMRequest,
    started_id: String,
    started_model: String,
    finish_reason: Option<FinishReason>,
    usage: Option<Usage>,
    text_parts: Vec<String>,
    thinking_parts: Vec<String>,
    audio_chunks: Vec<String>,
    tool_call_raw: HashMap<usize, String>,
    tool_call_meta: HashMap<usize, HashMap<String, serde_json::Value>>,
}

impl RoundState {
    fn new(request: &LMRequest) -> Self {
        Self {
            request: request.clone(),
            started_id: String::new(),
            started_model: String::new(),
            finish_reason: None,
            usage: None,
            text_parts: Vec::new(),
            thinking_parts: Vec::new(),
            audio_chunks: Vec::new(),
            tool_call_raw: HashMap::new(),
            tool_call_meta: HashMap::new(),
        }
    }

    fn apply(&mut self, event: &StreamEvent) -> Vec<StreamChunk> {
        let mut chunks = Vec::new();

        match event.event_type.as_str() {
            "start" => {
                if let Some(id) = &event.id {
                    self.started_id = id.clone();
                }
                if let Some(model) = &event.model {
                    self.started_model = model.clone();
                }
            }
            "end" => {
                self.finish_reason = event.finish_reason.clone();
                self.usage = event.usage.clone();
            }
            "delta" => {
                if let Some(delta) = &event.delta {
                    match delta.delta_type.as_str() {
                        "text" => {
                            let text = delta.text.as_deref().unwrap_or("");
                            self.text_parts.push(text.to_string());
                            chunks.push(StreamChunk {
                                chunk_type: "text".into(),
                                text: Some(text.to_string()),
                                name: None, input: None, response: None,
                            });
                        }
                        "thinking" => {
                            let text = delta.text.as_deref().unwrap_or("");
                            self.thinking_parts.push(text.to_string());
                            chunks.push(StreamChunk {
                                chunk_type: "thinking".into(),
                                text: Some(text.to_string()),
                                name: None, input: None, response: None,
                            });
                        }
                        "audio" => {
                            let data = delta.data.as_deref().unwrap_or("");
                            self.audio_chunks.push(data.to_string());
                            chunks.push(StreamChunk {
                                chunk_type: "audio".into(),
                                text: Some(data.to_string()),
                                name: None, input: None, response: None,
                            });
                        }
                        "tool_call" => {
                            let idx = event.part_index.unwrap_or(0);
                            let raw = delta.input.as_deref().unwrap_or("");
                            self.push_tool_call(idx, raw);
                        }
                        _ => {}
                    }
                }

                // Handle raw delta (tool calls with structured fields)
                if let Some(raw_delta) = &event.delta_raw {
                    if raw_delta.get("type").and_then(|v| v.as_str()) == Some("tool_call") {
                        let idx = event.part_index.unwrap_or(0);
                        let meta = self.tool_call_meta.entry(idx).or_default();
                        if let Some(id) = raw_delta.get("id") {
                            meta.insert("id".into(), id.clone());
                        }
                        if let Some(name) = raw_delta.get("name") {
                            meta.insert("name".into(), name.clone());
                        }
                        if let Some(input) = raw_delta.get("input") {
                            let input_str = input.as_str().unwrap_or("");
                            self.push_tool_call(idx, input_str);
                        }
                    }
                }
            }
            _ => {}
        }

        chunks
    }

    fn push_tool_call(&mut self, idx: usize, raw_input: &str) {
        let agg = self.tool_call_raw.entry(idx).or_default();
        agg.push_str(raw_input);
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(agg) {
            if let Some(obj) = parsed.as_object() {
                let meta = self.tool_call_meta.entry(idx).or_default();
                meta.insert("input".into(), serde_json::Value::Object(obj.clone()));
            }
        }
    }

    fn materialize(self) -> LMResponse {
        let mut parts = Vec::new();

        if !self.thinking_parts.is_empty() {
            parts.push(Part::thinking(self.thinking_parts.join("")));
        }
        if !self.text_parts.is_empty() {
            parts.push(Part::text(self.text_parts.join("")));
        }
        if !self.audio_chunks.is_empty() {
            parts.push(Part::audio_base64(&self.audio_chunks.join(""), "audio/wav"));
        }

        // Tool calls
        let tool_names: Vec<&str> = self.request.tools.iter()
            .filter(|t| t.tool_type == "function")
            .map(|t| t.name.as_str())
            .collect();

        let mut indices: Vec<usize> = self.tool_call_meta.keys().copied().collect();
        indices.sort();

        for (pos, idx) in indices.iter().enumerate() {
            let meta = &self.tool_call_meta[idx];

            let payload: JsonObject = meta.get("input")
                .and_then(|v| v.as_object())
                .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                .unwrap_or_else(|| {
                    serde_json::from_str(self.tool_call_raw.get(idx).map(|s| s.as_str()).unwrap_or("{}"))
                        .unwrap_or_default()
                });

            let name = meta.get("name").and_then(|v| v.as_str())
                .unwrap_or_else(|| {
                    if tool_names.len() == 1 { tool_names[0] }
                    else if pos < tool_names.len() { tool_names[pos] }
                    else { "tool" }
                });

            let id = meta.get("id").and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("tool_call_{idx}"));

            parts.push(Part::tool_call(&id, name, payload));
        }

        if parts.is_empty() {
            parts.push(Part::text(""));
        }

        let has_tc = parts.iter().any(|p| p.part_type == PartType::ToolCall);
        let finish = match &self.finish_reason {
            Some(f) if *f == FinishReason::Stop && has_tc => FinishReason::ToolCall,
            Some(f) => f.clone(),
            None if has_tc => FinishReason::ToolCall,
            None => FinishReason::Stop,
        };

        let model = if self.started_model.is_empty() {
            self.request.model.clone()
        } else {
            self.started_model
        };

        LMResponse {
            id: self.started_id,
            model,
            message: Message { role: Role::Assistant, parts, name: None },
            finish_reason: finish,
            usage: self.usage.unwrap_or_default(),
            provider: None,
        }
    }
}

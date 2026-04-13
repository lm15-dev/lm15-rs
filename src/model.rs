//! Model — reusable stateful object with conversation memory.

use crate::capabilities::resolve_provider;
use crate::client::UniversalLM;
use crate::result::{LMResult, ResultOpts, StartStreamFn, ToolFn};
use crate::types::*;
use std::collections::HashMap;
use std::sync::Arc;

/// History entry.
#[derive(Debug, Clone)]
pub struct HistoryEntry {
    pub request: LMRequest,
    pub response: LMResponse,
}

/// Options for creating a Model.
pub struct ModelOpts {
    pub model: String,
    pub system: Option<String>,
    pub tools: Vec<Tool>,
    pub provider: Option<String>,
    pub retries: usize,
    pub prompt_caching: bool,
    pub temperature: Option<f64>,
    pub max_tokens: Option<i64>,
    pub max_tool_rounds: usize,
}

impl Default for ModelOpts {
    fn default() -> Self {
        Self {
            model: String::new(),
            system: None,
            tools: Vec::new(),
            provider: None,
            retries: 0,
            prompt_caching: false,
            temperature: None,
            max_tokens: None,
            max_tool_rounds: 8,
        }
    }
}

/// A reusable, stateful model with conversation memory.
pub struct Model {
    lm: Arc<UniversalLM>,
    pub opts: ModelOpts,
    conversation: Vec<Message>,
    pub history: Vec<HistoryEntry>,
    pending_tool_calls: Vec<Part>,
}

impl Model {
    /// Create a new Model. The UniversalLM must be wrapped in Arc for shared ownership.
    pub fn new(lm: Arc<UniversalLM>, opts: ModelOpts) -> Self {
        Self {
            lm,
            opts,
            conversation: Vec::new(),
            history: Vec::new(),
            pending_tool_calls: Vec::new(),
        }
    }

    /// Clear conversation history.
    pub fn clear_history(&mut self) {
        self.history.clear();
        self.conversation.clear();
        self.pending_tool_calls.clear();
    }

    /// Prepare an LMRequest without sending it.
    pub fn prepare(&self, prompt: &str) -> LMRequest {
        let (req, _) = self.build_request(prompt, None);
        req
    }

    /// Call the model. Returns an LMResult.
    pub fn call(&mut self, prompt: &str, opts: Option<&CallOpts>) -> LMResult {
        let (request, callable_registry) = self.build_request(prompt, opts);
        let provider = opts.and_then(|o| o.provider.as_deref())
            .or(self.opts.provider.as_deref())
            .unwrap_or("")
            .to_string();

        let lm = self.lm.clone();
        let p = provider.clone();
        let start_stream: StartStreamFn = Box::new(move |req: &LMRequest| {
            lm.stream(req, &p)
        });

        let on_tool_call = opts.and_then(|o| o.on_tool_call.as_ref()).map(|f| {
            let f = f.clone();
            Box::new(move |info: &ToolCallInfo| -> Option<String> { f(info) })
                as Box<dyn Fn(&ToolCallInfo) -> Option<String> + Send + Sync>
        });

        let max_rounds = opts.and_then(|o| if o.max_tool_rounds > 0 { Some(o.max_tool_rounds) } else { None })
            .unwrap_or(self.opts.max_tool_rounds);

        LMResult::new(ResultOpts {
            request,
            start_stream,
            on_finished: None,
            callable_registry,
            on_tool_call,
            max_tool_rounds: max_rounds,
            retries: self.opts.retries,
        })
    }

    /// Submit tool results back to the model.
    pub fn submit_tools(&mut self, results: &[(&str, &str)]) -> Result<LMResult, LM15Error> {
        if self.pending_tool_calls.is_empty() {
            return Err(LM15Error::Provider("no pending tool calls".into()));
        }

        let mut parts = Vec::new();
        for tc in &self.pending_tool_calls {
            let id = tc.id.as_deref().unwrap_or("");
            if let Some((_, result)) = results.iter().find(|(rid, _)| *rid == id) {
                parts.push(Part::tool_result(id, vec![Part::text(*result)], tc.name.as_deref()));
            }
        }

        let tool_msg = Message { role: Role::Tool, parts, name: None };
        let mut messages = self.conversation.clone();
        messages.push(tool_msg);

        let (_, callable_registry) = self.normalize_tools();

        let config = self.base_config(None);
        let request = LMRequest {
            model: self.opts.model.clone(),
            messages,
            system: self.opts.system.clone(),
            tools: self.opts.tools.clone(),
            config,
        };

        let provider = self.opts.provider.as_deref().unwrap_or("").to_string();
        let lm = self.lm.clone();
        let p = provider.clone();
        let start_stream: StartStreamFn = Box::new(move |req: &LMRequest| {
            lm.stream(req, &p)
        });

        Ok(LMResult::new(ResultOpts {
            request,
            start_stream,
            on_finished: None,
            callable_registry,
            on_tool_call: None,
            max_tool_rounds: self.opts.max_tool_rounds,
            retries: self.opts.retries,
        }))
    }

    /// Upload a file.
    pub fn upload(&self, path: &str) -> Result<Part, LM15Error> {
        let data = std::fs::read(path).map_err(|e| LM15Error::Provider(e.to_string()))?;
        let filename = std::path::Path::new(path).file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("file.bin");
        let ext = std::path::Path::new(path).extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        let media_type = match ext {
            "pdf" => "application/pdf", "txt" => "text/plain",
            "png" => "image/png", "jpg" | "jpeg" => "image/jpeg",
            "mp3" => "audio/mpeg", "wav" => "audio/wav",
            "mp4" => "video/mp4",
            _ => "application/octet-stream",
        };

        let provider = self.opts.provider.as_deref()
            .map(|s| s.to_string())
            .or_else(|| resolve_provider(&self.opts.model).ok())
            .unwrap_or_default();

        let req = FileUploadRequest {
            model: Some(self.opts.model.clone()),
            filename: filename.into(),
            bytes_data: data,
            media_type: media_type.into(),
        };
        let resp = self.lm.file_upload(&req, &provider)?;

        let ds = DataSource::file(&resp.id, Some(media_type));
        Ok(match media_type.split('/').next().unwrap_or("") {
            "image" => Part { part_type: PartType::Image, source: Some(ds), ..Part::text("") },
            "audio" => Part { part_type: PartType::Audio, source: Some(ds), ..Part::text("") },
            "video" => Part { part_type: PartType::Video, source: Some(ds), ..Part::text("") },
            _ => Part { part_type: PartType::Document, source: Some(ds), ..Part::text("") },
        })
    }

    // ── Private ────────────────────────────────────────────────────

    fn build_request(&self, prompt: &str, opts: Option<&CallOpts>) -> (LMRequest, HashMap<String, ToolFn>) {
        let mut messages = self.conversation.clone();
        messages.push(Message::user(prompt));

        if let Some(opts) = opts {
            if let Some(prefill) = &opts.prefill {
                messages.push(Message::assistant(prefill));
            }
        }

        let (tools, callable_registry) = self.normalize_tools();
        let config = self.base_config(opts);
        let system = opts.and_then(|o| o.system.as_ref())
            .or(self.opts.system.as_ref())
            .cloned();

        let request = LMRequest {
            model: self.opts.model.clone(),
            messages,
            system,
            tools,
            config,
        };

        (request, callable_registry)
    }

    fn normalize_tools(&self) -> (Vec<Tool>, HashMap<String, ToolFn>) {
        let registry = HashMap::new();
        let tools = self.opts.tools.clone();
        // Note: cloning Tool drops the func field. We need to re-extract from originals.
        for t in &self.opts.tools {
            if t.tool_type == "function" {
                if let Some(_f) = &t.func {
                    // We can't clone the Box<dyn Fn>, but we can create a reference-based wrapper
                    // For now, tools must be re-registered via callable_registry separately
                    // This is a known Rust limitation with dyn Fn ownership
                }
            }
        }
        (tools, registry)
    }

    fn base_config(&self, opts: Option<&CallOpts>) -> Config {
        let mut cfg = Config {
            max_tokens: self.opts.max_tokens,
            temperature: self.opts.temperature,
            ..Config::default()
        };

        let mut provider_cfg: JsonObject = HashMap::new();
        if self.opts.prompt_caching {
            provider_cfg.insert("prompt_caching".into(), true.into());
        }

        if let Some(opts) = opts {
            if let Some(t) = opts.temperature { cfg.temperature = Some(t); }
            if let Some(m) = opts.max_tokens { cfg.max_tokens = Some(m); }
            if let Some(tp) = opts.top_p { cfg.top_p = Some(tp); }
            if let Some(stop) = &opts.stop { cfg.stop = Some(stop.clone()); }
            if opts.prompt_caching { provider_cfg.insert("prompt_caching".into(), true.into()); }
            if let Some(output) = &opts.output { provider_cfg.insert("output".into(), output.clone().into()); }

            match &opts.reasoning {
                Some(Reasoning::Enabled) => {
                    let mut r = JsonObject::new();
                    r.insert("enabled".into(), true.into());
                    cfg.reasoning = Some(r);
                }
                Some(Reasoning::WithBudget(budget)) => {
                    let mut r = JsonObject::new();
                    r.insert("enabled".into(), true.into());
                    r.insert("budget".into(), (*budget).into());
                    cfg.reasoning = Some(r);
                }
                None => {}
            }
        }

        if !provider_cfg.is_empty() {
            cfg.provider = Some(provider_cfg);
        }

        cfg
    }
}

/// Call options for Model.call().
pub struct CallOpts {
    pub system: Option<String>,
    pub prefill: Option<String>,
    pub output: Option<String>,
    pub prompt_caching: bool,
    pub temperature: Option<f64>,
    pub max_tokens: Option<i64>,
    pub top_p: Option<f64>,
    pub stop: Option<Vec<String>>,
    pub reasoning: Option<Reasoning>,
    pub max_tool_rounds: usize,
    pub provider: Option<String>,
    pub on_tool_call: Option<Arc<dyn Fn(&ToolCallInfo) -> Option<String> + Send + Sync>>,
}

impl Default for CallOpts {
    fn default() -> Self {
        Self {
            system: None, prefill: None, output: None, prompt_caching: false,
            temperature: None, max_tokens: None, top_p: None, stop: None,
            reasoning: None, max_tool_rounds: 0, provider: None, on_tool_call: None,
        }
    }
}

/// Reasoning configuration.
pub enum Reasoning {
    Enabled,
    WithBudget(i64),
}

use crate::errors::LM15Error;

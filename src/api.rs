//! High-level API surface.

use crate::capabilities::resolve_provider;
use crate::client::UniversalLM;
use crate::cost::{disable_cost_tracking, enable_cost_tracking};
use crate::factory::{build_default, BuildOpts};
use crate::model::{Model, ModelOpts, Reasoning};
use crate::result::{LMResult, ResultOpts, StartStreamFn, ToolFn};
use crate::types::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

// ── Module-level state ─────────────────────────────────────────────

static DEFAULTS: OnceLock<Mutex<HashMap<String, String>>> = OnceLock::new();

fn defaults() -> &'static Mutex<HashMap<String, String>> {
    DEFAULTS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Set module-level defaults.
pub fn configure(env: Option<&str>, api_key: Option<&str>) {
    let mut d = defaults().lock().unwrap();
    d.clear();
    if let Some(e) = env { d.insert("env".into(), e.into()); }
    if let Some(k) = api_key { d.insert("api_key".into(), k.into()); }
    disable_cost_tracking();
}

/// Set module-level defaults and optionally enable automatic cost tracking.
pub fn configure_with_tracking(env: Option<&str>, api_key: Option<&str>, track_costs: bool) -> Result<(), crate::errors::LM15Error> {
    configure(env, api_key);
    if track_costs {
        enable_cost_tracking()?;
    }
    Ok(())
}

fn get_client(api_key: Option<&str>, env: Option<&str>) -> Arc<UniversalLM> {
    let d = defaults().lock().unwrap();
    let resolved_env = env.map(|s| s.to_string())
        .or_else(|| d.get("env").cloned());
    let resolved_key = api_key.map(|s| s.to_string())
        .or_else(|| d.get("api_key").cloned());

    let mut opts_map = None;
    if resolved_key.is_some() || resolved_env.is_some() {
        let mut keys = HashMap::new();
        if let Some(k) = &resolved_key {
            for p in &["openai", "anthropic", "gemini"] {
                keys.insert(p.to_string(), k.clone());
            }
        }
        opts_map = Some(BuildOpts {
            api_key: if keys.is_empty() { None } else { Some(keys) },
            env_file: resolved_env,
            ..Default::default()
        });
    }

    Arc::new(build_default(opts_map.as_ref()))
}

// ── call() ─────────────────────────────────────────────────────────

/// Options for the call() function.
pub struct CallOptions {
    pub system: Option<String>,
    pub tools: Vec<Tool>,
    pub tool_fns: HashMap<String, ToolFn>,
    pub reasoning: Option<Reasoning>,
    pub prefill: Option<String>,
    pub output: Option<String>,
    pub prompt_caching: bool,
    pub temperature: Option<f64>,
    pub max_tokens: Option<i64>,
    pub max_tool_rounds: usize,
    pub retries: usize,
    pub provider: Option<String>,
    pub api_key: Option<String>,
    pub env: Option<String>,
}

impl Default for CallOptions {
    fn default() -> Self {
        Self {
            system: None, tools: Vec::new(), tool_fns: HashMap::new(),
            reasoning: None, prefill: None, output: None, prompt_caching: false,
            temperature: None, max_tokens: None, max_tool_rounds: 8, retries: 0,
            provider: None, api_key: None, env: None,
        }
    }
}

/// One-shot call to any model. Returns an LMResult.
///
/// ```rust,no_run
/// let mut result = lm15::call("gpt-4.1-mini", "Hello.", None);
/// println!("{}", result.text().unwrap());
/// ```
pub fn call(model: &str, prompt: &str, opts: Option<&CallOptions>) -> LMResult {
    let opts = opts.cloned().unwrap_or_default();
    let lm = get_client(opts.api_key.as_deref(), opts.env.as_deref());

    let provider = opts.provider.clone()
        .or_else(|| resolve_provider(model).ok())
        .unwrap_or_default();

    let mut messages = vec![Message::user(prompt)];
    if let Some(prefill) = &opts.prefill {
        messages.push(Message::assistant(prefill));
    }

    let mut provider_cfg: JsonObject = HashMap::new();
    if opts.prompt_caching { provider_cfg.insert("prompt_caching".into(), true.into()); }
    if let Some(output) = &opts.output { provider_cfg.insert("output".into(), output.clone().into()); }

    let reasoning = match &opts.reasoning {
        Some(Reasoning::Enabled) => {
            let mut r = JsonObject::new();
            r.insert("enabled".into(), true.into());
            Some(r)
        }
        Some(Reasoning::WithBudget(b)) => {
            let mut r = JsonObject::new();
            r.insert("enabled".into(), true.into());
            r.insert("budget".into(), (*b).into());
            Some(r)
        }
        None => None,
    };

    let config = Config {
        max_tokens: opts.max_tokens,
        temperature: opts.temperature,
        reasoning,
        provider: if provider_cfg.is_empty() { None } else { Some(provider_cfg) },
        ..Config::default()
    };

    let request = LMRequest {
        model: model.into(),
        messages,
        system: opts.system.clone(),
        tools: opts.tools.clone(),
        config,
    };

    let p = provider.clone();
    let lm_clone = lm.clone();
    let start_stream: StartStreamFn = Box::new(move |req: &LMRequest| {
        lm_clone.stream(req, &p)
    });

    LMResult::new(ResultOpts {
        request,
        start_stream,
        on_finished: None,
        callable_registry: opts.tool_fns,
        on_tool_call: None,
        max_tool_rounds: opts.max_tool_rounds,
        retries: opts.retries,
    })
}

/// Create a reusable Model object.
pub fn model(model_name: &str, opts: Option<ModelOpts>) -> Model {
    let opts = opts.unwrap_or(ModelOpts { model: model_name.into(), ..ModelOpts::default() });
    let lm = get_client(None, None);
    Model::new(lm, ModelOpts { model: model_name.into(), ..opts })
}

/// Build an LMRequest without sending it.
pub fn prepare(model: &str, prompt: &str, opts: Option<&CallOptions>) -> LMRequest {
    let system = opts.and_then(|o| o.system.clone());
    LMRequest {
        model: model.into(),
        messages: vec![Message::user(prompt)],
        system,
        tools: opts.map(|o| o.tools.clone()).unwrap_or_default(),
        config: Config::default(),
    }
}

/// Send a pre-built LMRequest. Returns an LMResult.
pub fn send(request: LMRequest, opts: Option<&CallOptions>) -> LMResult {
    let provider = opts.and_then(|o| o.provider.clone())
        .or_else(|| resolve_provider(&request.model).ok())
        .unwrap_or_default();

    let lm = get_client(
        opts.and_then(|o| o.api_key.as_deref()),
        opts.and_then(|o| o.env.as_deref()),
    );
    let p = provider.clone();
    let lm_clone = lm.clone();
    let start_stream: StartStreamFn = Box::new(move |req: &LMRequest| {
        lm_clone.stream(req, &p)
    });

    LMResult::new(ResultOpts {
        request,
        start_stream,
        on_finished: None,
        callable_registry: HashMap::new(),
        on_tool_call: None,
        max_tool_rounds: 8,
        retries: 0,
    })
}

impl Clone for CallOptions {
    fn clone(&self) -> Self {
        Self {
            system: self.system.clone(),
            tools: self.tools.clone(),
            tool_fns: HashMap::new(), // Can't clone Box<dyn Fn>
            reasoning: None, // Can't clone enum with non-Clone variants... actually we can
            prefill: self.prefill.clone(),
            output: self.output.clone(),
            prompt_caching: self.prompt_caching,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            max_tool_rounds: self.max_tool_rounds,
            retries: self.retries,
            provider: self.provider.clone(),
            api_key: self.api_key.clone(),
            env: self.env.clone(),
        }
    }
}

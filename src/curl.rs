//! Dump lm15 requests as curl commands or structured HTTP for comparison.

use crate::capabilities::resolve_provider;
use crate::errors::LM15Error;
use crate::factory::{build_default, BuildOpts};
use crate::transport::HttpRequest;
use crate::types::*;
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;

const AUTH_HEADERS: &[&str] = &["authorization", "x-api-key", "x-goog-api-key"];

/// JSON-serializable representation of an HTTP request.
#[derive(Debug, Clone, Serialize)]
pub struct HttpRequestDump {
    pub method: String,
    pub url: String,
    pub headers: HashMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<HashMap<String, String>>,
    pub body: Value,
}

/// Options for building curl / HTTP dumps.
#[derive(Debug, Clone, Default)]
pub struct CurlOptions {
    pub stream: bool,
    pub provider: Option<String>,
    pub api_key: Option<String>,
    pub env: Option<String>,
    pub messages: Option<Vec<Message>>,
    pub system: Option<String>,
    pub tools: Vec<Tool>,
    pub reasoning: Option<JsonObject>,
    pub prefill: Option<String>,
    pub output: Option<String>,
    pub prompt_caching: bool,
    pub temperature: Option<f64>,
    pub max_tokens: Option<i64>,
    pub top_p: Option<f64>,
    pub stop: Option<Vec<String>>,
    /// Provider-specific passthrough fields merged into config.provider.
    pub provider_config: Option<JsonObject>,
}

fn build_lm_request(model: &str, prompt: Option<&str>, opts: &CurlOptions) -> Result<LMRequest, LM15Error> {
    let mut messages = if let Some(messages) = &opts.messages {
        messages.clone()
    } else if let Some(prompt) = prompt {
        vec![Message::user(prompt)]
    } else {
        return Err(LM15Error::Provider("either prompt or messages is required".into()));
    };

    if let Some(prefill) = &opts.prefill {
        messages.push(Message::assistant(prefill));
    }

    let mut provider_cfg: JsonObject = HashMap::new();
    if opts.prompt_caching {
        provider_cfg.insert("prompt_caching".into(), true.into());
    }
    if let Some(output) = &opts.output {
        provider_cfg.insert("output".into(), output.clone().into());
    }

    // Merge provider_config passthrough
    if let Some(pc) = &opts.provider_config {
        for (k, v) in pc {
            provider_cfg.insert(k.clone(), v.clone());
        }
    }

    let config = Config {
        max_tokens: opts.max_tokens,
        temperature: opts.temperature,
        top_p: opts.top_p,
        stop: opts.stop.clone(),
        reasoning: opts.reasoning.clone(),
        provider: if provider_cfg.is_empty() { None } else { Some(provider_cfg) },
        ..Config::default()
    };

    Ok(LMRequest {
        model: model.into(),
        messages,
        system: opts.system.clone(),
        tools: opts.tools.clone(),
        config,
    })
}

/// Build the provider-level HTTP request without sending it.
pub fn build_http_request(
    model: &str,
    prompt: Option<&str>,
    opts: Option<&CurlOptions>,
) -> Result<HttpRequest, LM15Error> {
    let opts = opts.cloned().unwrap_or_default();
    let request = build_lm_request(model, prompt, &opts)?;
    let provider = opts.provider.unwrap_or(resolve_provider(model)?);

    let build_opts = if opts.api_key.is_some() || opts.env.is_some() {
        let mut keys = HashMap::new();
        if let Some(key) = opts.api_key {
            for p in ["openai", "anthropic", "gemini"] {
                keys.insert(p.to_string(), key.clone());
            }
        }
        Some(BuildOpts {
            api_key: if keys.is_empty() { None } else { Some(keys) },
            env_file: opts.env,
            ..Default::default()
        })
    } else {
        None
    };

    let client = build_default(build_opts.as_ref());
    client.build_http_request(&request, &provider, opts.stream)
}

/// Convert an HTTP request to a JSON-serializable structure.
pub fn http_request_to_dict(req: &HttpRequest) -> HttpRequestDump {
    let body = req.body.as_ref().map(|b| serde_json::from_slice::<Value>(b).unwrap_or(Value::String("<binary>".into()))).unwrap_or(Value::Null);

    let mut headers = HashMap::new();
    for (k, v) in &req.headers {
        if AUTH_HEADERS.contains(&k.to_ascii_lowercase().as_str()) {
            headers.insert(k.clone(), "REDACTED".into());
        } else {
            headers.insert(k.clone(), v.clone());
        }
    }

    HttpRequestDump {
        method: req.method.clone(),
        url: req.url.clone(),
        headers,
        params: if req.params.is_empty() { None } else { Some(req.params.clone()) },
        body,
    }
}

fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// Convert an HTTP request to a curl command string.
pub fn http_request_to_curl(req: &HttpRequest, redact_auth: bool) -> String {
    let mut parts = vec!["curl".to_string()];

    if req.method != "GET" {
        parts.push(format!("-X {}", req.method));
    }

    let mut url = req.url.clone();
    if !req.params.is_empty() {
        let qs = req.params.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join("&");
        url = format!("{url}?{qs}");
    }
    parts.push(shell_quote(&url));

    for (k, v) in &req.headers {
        let value = if redact_auth && AUTH_HEADERS.contains(&k.to_ascii_lowercase().as_str()) {
            "REDACTED".to_string()
        } else {
            v.clone()
        };
        parts.push(format!("-H {}", shell_quote(&format!("{k}: {value}"))));
    }

    if let Some(body) = &req.body {
        if let Ok(json) = serde_json::from_slice::<Value>(body) {
            let pretty = serde_json::to_string_pretty(&json).unwrap_or_else(|_| String::from_utf8_lossy(body).into());
            parts.push(format!("-d {}", shell_quote(&pretty)));
        } else {
            parts.push("--data-binary @-".into());
        }
    }

    parts.join(" \\\n  ")
}

/// Build a curl command for the given call parameters.
pub fn dump_curl(model: &str, prompt: Option<&str>, opts: Option<&CurlOptions>) -> Result<String, LM15Error> {
    let req = build_http_request(model, prompt, opts)?;
    Ok(http_request_to_curl(&req, true))
}

/// Build the structured HTTP request dump for comparison.
pub fn dump_http(model: &str, prompt: Option<&str>, opts: Option<&CurlOptions>) -> Result<HttpRequestDump, LM15Error> {
    let req = build_http_request(model, prompt, opts)?;
    Ok(http_request_to_dict(&req))
}

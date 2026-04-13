//! Factory — build a configured UniversalLM.

use crate::client::UniversalLM;
use crate::provider::Adapter;
use crate::provider::openai::OpenAIAdapter;
use crate::provider::anthropic::AnthropicAdapter;
use crate::provider::gemini::GeminiAdapter;
use crate::transport::UreqTransport;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::time::Duration;

struct AdapterDef {
    provider: &'static str,
    env_keys: &'static [&'static str],
    create: fn(&str, UreqTransport) -> Box<dyn Adapter>,
}

const ADAPTERS: &[AdapterDef] = &[
    AdapterDef {
        provider: "openai",
        env_keys: &["OPENAI_API_KEY"],
        create: |key, t| Box::new(OpenAIAdapter::new(key, t)),
    },
    AdapterDef {
        provider: "anthropic",
        env_keys: &["ANTHROPIC_API_KEY"],
        create: |key, t| Box::new(AnthropicAdapter::new(key, t)),
    },
    AdapterDef {
        provider: "gemini",
        env_keys: &["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        create: |key, t| Box::new(GeminiAdapter::new(key, t)),
    },
];

/// Options for building a default client.
#[derive(Default)]
pub struct BuildOpts {
    pub api_key: Option<HashMap<String, String>>,
    pub env_file: Option<String>,
    pub timeout: Option<Duration>,
}

/// Build a default UniversalLM from environment variables and optional .env file.
pub fn build_default(opts: Option<&BuildOpts>) -> UniversalLM {
    let timeout = opts.and_then(|o| o.timeout).unwrap_or(Duration::from_secs(30));
    let _transport = UreqTransport::new(timeout);

    let env_key_map = build_env_key_map();

    // Parse explicit keys
    let explicit = opts.and_then(|o| o.api_key.as_ref()).cloned().unwrap_or_default();

    // Parse .env file
    let mut file_keys = HashMap::new();
    if let Some(env_file) = opts.and_then(|o| o.env_file.as_ref()) {
        file_keys = parse_env_file(env_file, &env_key_map);
    }

    let mut client = UniversalLM::new();

    for def in ADAPTERS {
        let key = explicit.get(def.provider)
            .cloned()
            .or_else(|| file_keys.get(def.provider).cloned())
            .or_else(|| {
                def.env_keys.iter().find_map(|var| env::var(var).ok())
            });

        if let Some(key) = key {
            let t = UreqTransport::new(timeout);
            client.register((def.create)(&key, t));
        }
    }

    client
}

/// Return {provider: [env_keys]} for all core adapters.
pub fn providers() -> HashMap<String, Vec<String>> {
    let mut out = HashMap::new();
    for def in ADAPTERS {
        out.insert(def.provider.to_string(), def.env_keys.iter().map(|s| s.to_string()).collect());
    }
    out
}

fn build_env_key_map() -> HashMap<String, String> {
    let mut map = HashMap::new();
    for def in ADAPTERS {
        for key in def.env_keys {
            map.insert(key.to_string(), def.provider.to_string());
        }
    }
    map
}

fn parse_env_file(path: &str, env_key_map: &HashMap<String, String>) -> HashMap<String, String> {
    let mut result = HashMap::new();
    let expanded = if path.starts_with("~/") {
        dirs_or_home().map(|h| format!("{}/{}", h, &path[2..])).unwrap_or_else(|| path.to_string())
    } else {
        path.to_string()
    };

    let content = match fs::read_to_string(&expanded) {
        Ok(c) => c,
        Err(_) => return result,
    };

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        let line = line.strip_prefix("export ").unwrap_or(line);
        if let Some((key, value)) = line.split_once('=') {
            let key = key.trim();
            let mut value = value.trim();
            if (value.starts_with('"') && value.ends_with('"')) ||
               (value.starts_with('\'') && value.ends_with('\'')) {
                value = &value[1..value.len()-1];
            }
            if let Some(provider) = env_key_map.get(key) {
                if !value.is_empty() {
                    result.insert(provider.clone(), value.to_string());
                    if env::var(key).is_err() {
                        // SAFETY: single-threaded init, before any threads are spawned
                        unsafe { env::set_var(key, value); }
                    }
                }
            }
        }
    }
    result
}

fn dirs_or_home() -> Option<String> {
    env::var("HOME").ok()
}

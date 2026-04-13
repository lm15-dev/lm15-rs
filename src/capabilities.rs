//! Model → provider resolution.

use crate::errors::LM15Error;

struct Pattern {
    prefix: &'static str,
    provider: &'static str,
}

const PATTERNS: &[Pattern] = &[
    Pattern { prefix: "claude", provider: "anthropic" },
    Pattern { prefix: "gemini", provider: "gemini" },
    Pattern { prefix: "gpt", provider: "openai" },
    Pattern { prefix: "o1", provider: "openai" },
    Pattern { prefix: "o3", provider: "openai" },
    Pattern { prefix: "o4", provider: "openai" },
    Pattern { prefix: "chatgpt", provider: "openai" },
    Pattern { prefix: "dall-e", provider: "openai" },
    Pattern { prefix: "tts", provider: "openai" },
    Pattern { prefix: "whisper", provider: "openai" },
];

/// Resolve a provider name from a model name.
pub fn resolve_provider(model: &str) -> Result<String, LM15Error> {
    let lower = model.to_lowercase();
    for p in PATTERNS {
        if lower.starts_with(p.prefix) {
            return Ok(p.provider.to_string());
        }
    }
    Err(LM15Error::UnsupportedModel(format!(
        "unable to resolve provider for model '{model}'\n\n\
         To fix: use provider parameter, or check the model name"
    )))
}

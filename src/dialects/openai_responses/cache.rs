//! Prompt caching (MAP-6) on the Responses wire
//! (`lm15/providers/openai.py:159-283`): the model-class detector, the
//! breakpoint placement rules and the shared body fields.

use serde_json::{json, Map, Value};

use crate::compat::OpenAICacheControl;
use crate::types::{CacheConfig, CacheMode, CachePrefix, CacheRetention, Request};

use super::unsupported;

/// `openai.py:159-174` `openai_model_has_cache_options`: the GPT-5.6+
/// class, the only one that takes `prompt_cache_options` (older models
/// answer 400, and their writes are free). A name table that rots; a
/// family not spelled `gpt-<major>.<minor>` keeps implicit writes on
/// `mode="off"`, and `extensions.prompt_cache_options` overrides.
pub fn model_has_cache_options(model: &str) -> bool {
    gpt_version(&model.to_ascii_lowercase()).is_some_and(|version| version >= (5, 6))
}

/// `^gpt-(\d+)\.(\d+)` (`openai.py:159` `_GPT_VERSION_RE`).
fn gpt_version(lower: &str) -> Option<(u32, u32)> {
    fn leading_number(text: &str) -> Option<(u32, &str)> {
        let len = text.chars().take_while(char::is_ascii_digit).count();
        let number = text[..len].parse().ok()?;
        Some((number, &text[len..]))
    }
    let (major, rest) = leading_number(lower.strip_prefix("gpt-")?)?;
    let (minor, _) = leading_number(rest.strip_prefix('.')?)?;
    Some((major, minor))
}

fn active(cache: Option<&CacheConfig>) -> Option<&CacheConfig> {
    cache.filter(|cache| cache.mode != CacheMode::Off)
}

/// `openai.py:177-196` `_cache_breakpoint_index`: the message that
/// carries `prompt_cache_breakpoint`, clamped to the last message; only
/// under the `openai` cache control.
pub fn breakpoint_index(request: &Request, cache_control: OpenAICacheControl) -> Option<usize> {
    if cache_control != OpenAICacheControl::OpenAI {
        return None;
    }
    let index = active(request.config.cache.as_ref())?.prefix_until_index?;
    let last = request.messages.len().saturating_sub(1) as u64;
    Some(index.min(last) as usize)
}

/// `openai.py:209-215` `_cache_stable_prefix`: `prefix="stable"` marks
/// the end of system + tools.
pub fn stable_prefix(request: &Request, cache_control: OpenAICacheControl) -> bool {
    cache_control == OpenAICacheControl::OpenAI
        && active(request.config.cache.as_ref())
            .is_some_and(|cache| cache.prefix == Some(CachePrefix::Stable))
}

/// `openai.py:199-206` `_has_explicit_breakpoint`: a mark is placed
/// (`prefix_until_index`, or `prefix="stable"` with a system prompt).
fn has_explicit_breakpoint(request: &Request, cache_control: OpenAICacheControl) -> bool {
    breakpoint_index(request, cache_control).is_some()
        || (stable_prefix(request, cache_control) && request.system.is_some())
}

/// `openai.py:218-271` `_cache_common_payload`: the off switch, key,
/// retention and explicit mode; `resource` refuses (no stored-cache tier).
pub fn cache_payload(
    provider: &str,
    model: &str,
    request: &Request,
    payload: &mut Map<String, Value>,
    cache_control: OpenAICacheControl,
) -> Result<(), crate::errors::Lm15Error> {
    let Some(cache) = request.config.cache.as_ref() else {
        return Ok(());
    };
    match cache_control {
        OpenAICacheControl::OpenAIImplicit => {
            // The two documented fields only; no off switch, no mark
            // (changes/2026-09-03-meta-live.md §2).
            if cache.mode != CacheMode::Off {
                if let Some(key) = cache.key.as_deref().filter(|k| !k.is_empty()) {
                    payload.insert("prompt_cache_key".into(), Value::String(key.into()));
                }
                if cache.retention == Some(CacheRetention::Long) {
                    payload.insert("prompt_cache_retention".into(), json!("24h"));
                }
            }
            if cache.resource.is_some() {
                return Err(unsupported(
                    provider,
                    "cache.resource is not supported — this provider has no stored-cache tier; \
                     it caches every prompt prefix automatically",
                ));
            }
            Ok(())
        }
        OpenAICacheControl::OpenAI => {
            if cache.mode == CacheMode::Off {
                // Option 2 (ratified 2026-09-01): the real off switch where
                // the class has one; nothing where writes are free anyway.
                if model_has_cache_options(model) {
                    payload.insert("prompt_cache_options".into(), json!({"mode": "explicit"}));
                }
                return Ok(());
            }
            if let Some(key) = cache.key.as_deref().filter(|k| !k.is_empty()) {
                payload.insert("prompt_cache_key".into(), Value::String(key.into()));
            }
            if cache.retention == Some(CacheRetention::Long) {
                // Every class takes "24h" (review probe 2, 2026-09-02).
                payload.insert("prompt_cache_retention".into(), json!("24h"));
            }
            if model_has_cache_options(model) && has_explicit_breakpoint(request, cache_control) {
                // The mark and the mode go together (review probe 3).
                payload.insert("prompt_cache_options".into(), json!({"mode": "explicit"}));
            }
            if cache.resource.is_some() {
                return Err(unsupported(
                    provider,
                    "cache.resource is not supported — this provider has no stored-cache tier; \
                     it caches by marks on blocks (prefix / prefix_until_index) and automatically",
                ));
            }
            Ok(())
        }
        // `none` / `anthropic`: nothing on this wire (MAP-6 rule 4: the
        // automatic tier, observable in usage).
        OpenAICacheControl::None | OpenAICacheControl::Anthropic => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cache_options_model_class() {
        assert!(model_has_cache_options("gpt-5.6-sol"));
        assert!(model_has_cache_options("GPT-6.0"));
        assert!(model_has_cache_options("gpt-5.10"));
        assert!(!model_has_cache_options("gpt-5.4-mini"));
        assert!(!model_has_cache_options("gpt-5-mini"));
        assert!(!model_has_cache_options("gpt-4.1-mini"));
        assert!(!model_has_cache_options("o3"));
        assert!(!model_has_cache_options("gpt-"));
        assert!(!model_has_cache_options("gpt-5."));
    }
}

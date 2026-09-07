//! Prompt caching on the Chat Completions wire (MAP-6), gated by the
//! compat's `cache_control`. The helpers mirror the ones both OpenAI
//! dialects share in the reference (`lm15/providers/openai.py:159-283`);
//! the model-class detector is copied as data.

use serde_json::{json, Map, Value};

use super::text::unsupported;
use crate::compat::OpenAICacheControl;
use crate::errors::Lm15Error;
use crate::types::{CacheMode, CachePrefix, CacheRetention, Request, Role};

/// The GPT-5.6-and-later class takes `prompt_cache_options`
/// (`lm15/providers/openai.py:159-175` `openai_model_has_cache_options`):
/// `gpt-<major>.<minor>` with `(major, minor) >= (5, 6)`. A stated,
/// rotting table: a future family whose name does not match keeps the
/// provider's implicit writes on `mode="off"`; `extensions` overrides.
pub(super) fn model_has_cache_options(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    let Some(rest) = lower.strip_prefix("gpt-") else {
        return false;
    };
    let digits = |s: &str| -> (Option<u32>, usize) {
        let end = s.bytes().take_while(u8::is_ascii_digit).count();
        (s[..end].parse().ok(), end)
    };
    let (major, end) = digits(rest);
    let Some(major) = major else {
        return false;
    };
    let Some(after_dot) = rest[end..].strip_prefix('.') else {
        return false;
    };
    let (minor, _) = digits(after_dot);
    let Some(minor) = minor else {
        return false;
    };
    (major, minor) >= (5, 6)
}

/// The message index that carries `prompt_cache_breakpoint`
/// (`openai.py:177-196`): `prefix_until_index`, clamped to the last
/// message, only when the compat names OpenAI's cache control.
pub(super) fn breakpoint_index(
    request: &Request,
    cache_control: OpenAICacheControl,
) -> Option<usize> {
    let cache = request.config.cache.as_ref()?;
    if cache.mode == CacheMode::Off || cache_control != OpenAICacheControl::OpenAI {
        return None;
    }
    let index = cache.prefix_until_index?;
    let last = request.messages.len().saturating_sub(1);
    Some(usize::try_from(index).map_or(last, |i| i.min(last)))
}

/// `prefix="stable"`: the mark rides on the system message
/// (`openai.py:209-216`).
pub(super) fn stable_prefix(request: &Request, cache_control: OpenAICacheControl) -> bool {
    request.config.cache.as_ref().is_some_and(|cache| {
        cache.mode != CacheMode::Off
            && cache.prefix == Some(CachePrefix::Stable)
            && cache_control == OpenAICacheControl::OpenAI
    })
}

/// True when this request places a mark (`openai.py:199-206`): a
/// `prefix="stable"` with no system prompt places none.
fn has_explicit_breakpoint(request: &Request, cache_control: OpenAICacheControl) -> bool {
    breakpoint_index(request, cache_control).is_some()
        || (stable_prefix(request, cache_control) && request.system.is_some())
}

/// The top-level MAP-6 fields (`openai.py:218-271`): off switch, key,
/// retention, explicit mode next to a mark; `resource` refuses on both
/// OpenAI cache controls (no stored-cache tier). `none` and `anthropic`
/// send nothing.
pub(super) fn cache_payload(
    request: &Request,
    payload: &mut Map<String, Value>,
    cache_control: OpenAICacheControl,
    provider: &str,
) -> Result<(), Lm15Error> {
    let Some(cache) = request.config.cache.as_ref() else {
        return Ok(());
    };
    match cache_control {
        OpenAICacheControl::None | OpenAICacheControl::Anthropic => return Ok(()),
        OpenAICacheControl::OpenAIImplicit => {
            // The key and the retention hint only: no off switch (the
            // server has none) and no mark (an undocumented field the
            // server swallows — Meta, live 2026-09-03).
            if cache.mode != CacheMode::Off {
                if let Some(key) = &cache.key {
                    payload.insert("prompt_cache_key".into(), Value::String(key.clone()));
                }
                if cache.retention == Some(CacheRetention::Long) {
                    payload.insert("prompt_cache_retention".into(), Value::String("24h".into()));
                }
            }
            if cache.resource.is_some() {
                return Err(unsupported(
                    provider,
                    "cache.resource is not supported — this provider has no stored-cache tier; \
                     it caches every prompt prefix automatically",
                ));
            }
            return Ok(());
        }
        OpenAICacheControl::OpenAI => {}
    }
    if cache.mode == CacheMode::Off {
        // Option 2 (ratified 2026-09-01): the real off switch where the
        // model class has one; nothing where writes are free anyway.
        if model_has_cache_options(&request.model) {
            payload.insert("prompt_cache_options".into(), json!({"mode": "explicit"}));
        }
        return Ok(());
    }
    if let Some(key) = &cache.key {
        payload.insert("prompt_cache_key".into(), Value::String(key.clone()));
    }
    if cache.retention == Some(CacheRetention::Long) {
        payload.insert("prompt_cache_retention".into(), Value::String("24h".into()));
    }
    if model_has_cache_options(&request.model) && has_explicit_breakpoint(request, cache_control) {
        // The mark and the mode go together (MAP-6 rule 4, review probe 3).
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

/// The refusal for a breakpoint that cannot ride on text
/// (`openai.py:274-283`).
pub(super) fn breakpoint_unsupported(provider: &str, index: usize, role: Role) -> Lm15Error {
    unsupported(
        provider,
        format!(
            "cache.prefix_until_index={index} points at a {role} message whose last block is \
             not text — the wire carries prompt_cache_breakpoint on text input blocks only. \
             Point the prefix at a user/developer message that ends with text, or omit \
             prefix_until_index (implicit caching still applies)."
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_class_detector_matches_the_reference_regex() {
        assert!(model_has_cache_options("gpt-5.6-sol"));
        assert!(model_has_cache_options("GPT-5.6"));
        assert!(model_has_cache_options("gpt-6.0-mini"));
        assert!(model_has_cache_options("gpt-5.10"));
        assert!(!model_has_cache_options("gpt-5.4-mini"));
        assert!(!model_has_cache_options("gpt-4.1-mini"));
        assert!(!model_has_cache_options("gpt-5-mini"));
        assert!(!model_has_cache_options("o3"));
        assert!(!model_has_cache_options("gpt-"));
    }
}

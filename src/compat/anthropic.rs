//! Anthropic Messages compat (`lm15/compat.py:831-1038`). Data only: the
//! Anthropic dialect (module 4 W1) consults the resolved value at the
//! named points.

use super::{JsonObject, Knob, ReasoningEfforts, SendReject, ToolResultMedia};

/// `lm15/compat.py:839-853` `AnthropicThinkingFormat`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnthropicThinkingFormat {
    /// The model-class rule (MAP-7 rule 10).
    Anthropic,
    /// `thinking.type=enabled|disabled` + `output_config.effort`; off MUST be sent.
    Deepseek,
    /// Every model is the adaptive class; off is `thinking.type=disabled`.
    Adaptive,
    /// `output_config.effort` alone; off is `thinking.type=disabled`.
    Effort,
}

/// `lm15/compat.py:854-861` `AnthropicThinkingReplay`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnthropicThinkingReplay {
    /// Only a signed block goes back as `thinking` (decision G).
    Signed,
    /// An unsigned block goes back as `thinking` without a signature.
    Unsigned,
}

/// `lm15/compat.py:862-867` `AnthropicSamplingParams`.
pub type AnthropicSamplingParams = SendReject;
/// `lm15/compat.py:872-875` `AnthropicStructuredOutput`.
pub type AnthropicStructuredOutput = SendReject;
/// `lm15/compat.py:876-878` `AnthropicParallelToolCalls`.
pub type AnthropicParallelToolCalls = SendReject;

/// `lm15/compat.py:868-871` `AnthropicCacheControl`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnthropicCacheControl {
    Anthropic,
    /// The server ignores marks and caches implicitly; nothing is placed.
    None,
}

/// Partial Anthropic Messages compat (`lm15/compat.py:881-931`).
///
/// `model_prefixes` refuses, before the wire, any model id that does not
/// start with one of the prefixes (DeepSeek silently serves `claude-*`
/// as its own models); `None` means any model id goes out as typed.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AnthropicCompat {
    pub thinking_format: Option<Knob<AnthropicThinkingFormat>>,
    pub thinking_replay: Option<Knob<AnthropicThinkingReplay>>,
    pub cache_control: Option<Knob<AnthropicCacheControl>>,
    pub structured_output: Option<Knob<AnthropicStructuredOutput>>,
    pub parallel_tool_calls: Option<Knob<AnthropicParallelToolCalls>>,
    pub sampling_params: Option<Knob<AnthropicSamplingParams>>,
    /// MAP-10: media inside `tool_result.content` (`lm15/compat.py` `tool_result_media`).
    pub tool_result_media: Option<Knob<ToolResultMedia>>,
    pub reasoning_efforts: Option<ReasoningEfforts>,
    pub model_prefixes: Option<&'static [&'static str]>,
    pub extensions: Option<JsonObject>,
}

impl AnthropicCompat {
    pub const EMPTY: AnthropicCompat = AnthropicCompat {
        thinking_format: None,
        thinking_replay: None,
        cache_control: None,
        structured_output: None,
        parallel_tool_calls: None,
        sampling_params: None,
        tool_result_media: None,
        reasoning_efforts: None,
        model_prefixes: None,
        extensions: None,
    };

    /// The named preset (`lm15/compat.py:925-931`; aliases accepted).
    pub fn preset(name: &str) -> Option<&'static AnthropicCompat> {
        super::find(ANTHROPIC_PRESETS, name)
    }

    /// `lm15/compat.py:1017-1038` `resolve_anthropic_compat`.
    pub fn resolve(&self) -> ResolvedAnthropicCompat {
        ResolvedAnthropicCompat {
            thinking_format: Knob::resolve(
                self.thinking_format,
                AnthropicThinkingFormat::Anthropic,
            ),
            thinking_replay: Knob::resolve(self.thinking_replay, AnthropicThinkingReplay::Signed),
            cache_control: Knob::resolve(self.cache_control, AnthropicCacheControl::Anthropic),
            structured_output: Knob::resolve(self.structured_output, SendReject::Send),
            parallel_tool_calls: Knob::resolve(self.parallel_tool_calls, SendReject::Send),
            sampling_params: Knob::resolve(self.sampling_params, SendReject::Send),
            // tool_result.content takes image and document blocks (anthropic exact, 2026-09-07).
            tool_result_media: Knob::resolve(self.tool_result_media, ToolResultMedia::Native),
            reasoning_efforts: self.reasoning_efforts,
            model_prefixes: self.model_prefixes,
            extensions: self.extensions.clone(),
        }
    }
}

/// Fully resolved Anthropic compat (`lm15/compat.py:1002-1014`).
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedAnthropicCompat {
    pub thinking_format: AnthropicThinkingFormat,
    pub thinking_replay: AnthropicThinkingReplay,
    pub cache_control: AnthropicCacheControl,
    pub structured_output: AnthropicStructuredOutput,
    pub parallel_tool_calls: AnthropicParallelToolCalls,
    pub sampling_params: AnthropicSamplingParams,
    pub tool_result_media: ToolResultMedia,
    pub reasoning_efforts: Option<ReasoningEfforts>,
    pub model_prefixes: Option<&'static [&'static str]>,
    pub extensions: Option<JsonObject>,
}

impl Default for ResolvedAnthropicCompat {
    fn default() -> Self {
        AnthropicCompat::EMPTY.resolve()
    }
}

/// The server presets (`lm15/compat.py`), generated from lm15-contract
/// tables/providers.json into `src/generated/tables.rs`; the receipt behind
/// each knob is cited at the reference table.
pub const ANTHROPIC_PRESETS: &[(&str, AnthropicCompat)] =
    crate::generated::tables::ANTHROPIC_PRESETS;

/// The roots the presets supply (`lm15/compat.py`), generated likewise.
pub const ANTHROPIC_PRESET_BASE_URLS: &[(&str, &str)] =
    crate::generated::tables::ANTHROPIC_PRESET_BASE_URLS;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_resolves_to_the_dialect_defaults() {
        let resolved = AnthropicCompat::EMPTY.resolve();
        assert_eq!(resolved.thinking_format, AnthropicThinkingFormat::Anthropic);
        assert_eq!(resolved.thinking_replay, AnthropicThinkingReplay::Signed);
        assert_eq!(resolved.cache_control, AnthropicCacheControl::Anthropic);
        assert_eq!(resolved.structured_output, SendReject::Send);
        assert_eq!(resolved.sampling_params, SendReject::Send);
        assert!(resolved.model_prefixes.is_none());
    }

    #[test]
    fn presets_resolve_as_the_reference_table_says() {
        let deepseek = AnthropicCompat::preset("deepseek").unwrap().resolve();
        assert_eq!(deepseek.thinking_format, AnthropicThinkingFormat::Deepseek);
        assert_eq!(deepseek.structured_output, SendReject::Reject);
        assert_eq!(deepseek.model_prefixes, Some(&["deepseek-"][..]));
        let moonshot = AnthropicCompat::preset("moonshotai").unwrap().resolve();
        assert_eq!(moonshot.thinking_replay, AnthropicThinkingReplay::Unsigned);
        assert_eq!(moonshot.reasoning_efforts.unwrap().len(), 3);
        assert!(AnthropicCompat::preset("groq").is_none());
    }
}

//! Text renderings the Chat Completions wire needs: the lossy text form
//! of a part sequence (`lm15/providers/common.py:53-66` `parts_to_text`)
//! and the inline-media data URI (`common.py:73-76` `media_data_uri`).

use crate::errors::{ErrorMeta, Lm15Error};
use crate::types::Part;

/// An `UnsupportedFeatureError` naming the provider (the refusal class
/// every MAP-5..8 cell on this dialect pins).
pub(super) fn unsupported(provider: &str, message: impl Into<String>) -> Lm15Error {
    let mut meta = ErrorMeta::new(format!("{provider}: {}", message.into()));
    meta.provider = Some(provider.to_string());
    Lm15Error::UnsupportedFeatureError(meta)
}

/// Lossy text rendering for wire fields that accept text only
/// (`common.py:53-66`): text verbatim, non-empty thinking text, citations
/// as `title — url — text`, everything else contributes nothing.
pub(super) fn parts_to_text(parts: &[Part]) -> String {
    let mut out: Vec<String> = Vec::new();
    for part in parts {
        match part {
            Part::Text(text) => out.push(text.text.clone()),
            Part::Thinking(thinking) if !thinking.text.is_empty() => {
                out.push(thinking.text.clone())
            }
            Part::Citation(citation) => {
                let bits: Vec<&str> = [
                    citation.title.as_deref(),
                    citation.url.as_deref(),
                    citation.text.as_deref(),
                ]
                .into_iter()
                .flatten()
                .filter(|s| !s.is_empty())
                .collect();
                if !bits.is_empty() {
                    out.push(bits.join(" — "));
                }
            }
            _ => {}
        }
    }
    out.join("\n")
}

/// `data:<media_type>;base64,<data>` (`common.py:73-76`).
pub(super) fn data_uri(media_type: &str, data: &str) -> String {
    format!("data:{media_type};base64,{data}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CitationPart, ToolCallPart};

    #[test]
    fn text_rendering_joins_text_thinking_and_citations() {
        let parts = vec![
            Part::text("a"),
            Part::thinking(""),
            Part::thinking("t"),
            Part::Citation(CitationPart {
                url: Some("https://x".into()),
                title: Some("X".into()),
                text: None,
                continuation: Vec::new(),
            }),
            Part::ToolCall(ToolCallPart::default()),
        ];
        assert_eq!(parts_to_text(&parts), "a\nt\nX — https://x");
        assert_eq!(parts_to_text(&[]), "");
        assert_eq!(data_uri("image/png", "AA=="), "data:image/png;base64,AA==");
    }

    #[test]
    fn refusal_names_the_provider() {
        let err = unsupported("groq", "no");
        assert_eq!(err.class_name(), "UnsupportedFeatureError");
        assert_eq!(err.provider(), Some("groq"));
        assert!(err.message().starts_with("groq: no"));
    }
}

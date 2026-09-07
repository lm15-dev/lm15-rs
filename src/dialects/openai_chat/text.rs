//! The Chat Completions dialect's refusal constructor and the inline-media
//! data URI (`common.py:73-76` `media_data_uri`). Text rendering lives in
//! `dialects::content` (MAP-10: a media part raises, never a caption).

use crate::errors::{ErrorMeta, Lm15Error};

/// An `UnsupportedFeatureError` naming the provider (the refusal class
/// every MAP-5..8 cell on this dialect pins).
pub(super) fn unsupported(provider: &str, message: impl Into<String>) -> Lm15Error {
    let mut meta = ErrorMeta::new(format!("{provider}: {}", message.into()));
    meta.provider = Some(provider.to_string());
    Lm15Error::UnsupportedFeatureError(meta)
}

/// `data:<media_type>;base64,<data>` (`common.py:73-76`).
pub(super) fn data_uri(media_type: &str, data: &str) -> String {
    format!("data:{media_type};base64,{data}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialects::content::parts_to_text;
    use crate::types::{CitationPart, Part, ToolCallPart};

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
        assert_eq!(parts_to_text(&parts, "p", "x").unwrap(), "a\nt\nX — https://x");
        assert_eq!(parts_to_text(&[], "p", "x").unwrap(), "");
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

//! The four wire dialects (module 4). Each lives in its own directory and
//! is selected by `dialect_for`; the registry maps a provider string to a
//! `DialectId`, the access policy binds the door.

use crate::registry::DialectId;
use crate::wire::Dialect;

pub mod anthropic;
pub mod content;
pub mod gemini;
pub mod openai_chat;
pub mod openai_responses;

/// The codec for a dialect id.
pub fn dialect_for(id: DialectId) -> &'static dyn Dialect {
    match id {
        DialectId::Anthropic => &anthropic::ANTHROPIC,
        DialectId::OpenaiResponses => &openai_responses::OPENAI_RESPONSES,
        DialectId::OpenaiChat => &openai_chat::OPENAI_CHAT,
        DialectId::Gemini => &gemini::GEMINI,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_dialect_names_itself() {
        for id in [
            DialectId::Anthropic,
            DialectId::OpenaiResponses,
            DialectId::OpenaiChat,
            DialectId::Gemini,
        ] {
            assert_eq!(dialect_for(id).dialect(), id);
        }
        assert_eq!(
            dialect_for(DialectId::Gemini).api_key_header(),
            "x-goog-api-key"
        );
    }
}

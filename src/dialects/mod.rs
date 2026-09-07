//! The four wire dialects (module 4 W1–W4). Each lives in its own
//! directory and is wired in through exactly one line below; until it
//! lands, a stub answers `UnsupportedFeatureError` for every request.
//!
//! A dialect worker touches only its own directory plus its wiring line.

use crate::errors::{ErrorMeta, Lm15Error};
use crate::registry::DialectId;
use crate::types::Request;
use crate::wire::{BuildContext, Dialect, WireRequest};

// W1: `pub mod anthropic;`
// W2: `pub mod openai_responses;`
// W3: `pub mod openai_chat;`
pub mod gemini;

/// A dialect not yet implemented: every build is an honest refusal.
struct Stub(DialectId);

impl Dialect for Stub {
    fn dialect(&self) -> DialectId {
        self.0
    }

    fn build(
        &self,
        _request: &Request,
        _stream: bool,
        cx: &BuildContext<'_>,
    ) -> Result<WireRequest, Lm15Error> {
        let mut meta = ErrorMeta::new(format!(
            "module 4 dialect {} not yet implemented",
            self.0.as_str()
        ));
        meta.provider = Some(cx.provider.to_string());
        Err(Lm15Error::UnsupportedFeatureError(meta))
    }

    fn api_key_header(&self) -> &'static str {
        match self.0 {
            DialectId::Gemini => "x-goog-api-key",
            _ => "x-api-key",
        }
    }
}

static ANTHROPIC_STUB: Stub = Stub(DialectId::Anthropic);
static OPENAI_RESPONSES_STUB: Stub = Stub(DialectId::OpenaiResponses);
static OPENAI_CHAT_STUB: Stub = Stub(DialectId::OpenaiChat);

/// The codec for a dialect id.
pub fn dialect_for(id: DialectId) -> &'static dyn Dialect {
    match id {
        DialectId::Anthropic => &ANTHROPIC_STUB, // W1 wiring point: `&anthropic::ANTHROPIC`
        DialectId::OpenaiResponses => &OPENAI_RESPONSES_STUB, // W2 wiring point: `&openai_responses::OPENAI_RESPONSES`
        DialectId::OpenaiChat => &OPENAI_CHAT_STUB, // W3 wiring point: `&openai_chat::OPENAI_CHAT`
        DialectId::Gemini => &gemini::GEMINI,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_stub_names_its_dialect_and_refuses() {
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

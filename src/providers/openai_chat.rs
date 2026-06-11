//! OpenAI Chat Completions dialect adapter. Stage B: error normalization.
//!
//! OpenAI-compatible servers reuse the same error envelope family; the
//! mapping is shared verbatim with the Responses adapter (reference:
//! OpenAIChatLM.normalize_error = OpenAILM.normalize_error).

use crate::errors::Lm15Error;

pub fn normalize_error(status: u16, body: &str) -> Lm15Error {
    super::openai::normalize_error_as(status, body, "openai_chat")
}

//! Conversation — stateless message accumulator.

use crate::types::{LMResponse, Message, Part};

/// Accumulates messages for multi-turn interactions.
pub struct Conversation {
    pub system: Option<String>,
    messages: Vec<Message>,
}

impl Conversation {
    pub fn new(system: Option<&str>) -> Self {
        Self { system: system.map(Into::into), messages: Vec::new() }
    }

    /// Add a user message.
    pub fn user(&mut self, text: &str) {
        self.messages.push(Message::user(text));
    }

    /// Add a user message with mixed parts.
    pub fn user_parts(&mut self, parts: Vec<Part>) {
        self.messages.push(Message {
            role: crate::types::Role::User,
            parts,
            name: None,
        });
    }

    /// Add an assistant response.
    pub fn assistant(&mut self, response: &LMResponse) {
        self.messages.push(response.message.clone());
    }

    /// Add a prefill message.
    pub fn prefill(&mut self, text: &str) {
        self.messages.push(Message::assistant(text));
    }

    /// Get all messages.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Clear all messages.
    pub fn clear(&mut self) {
        self.messages.clear();
    }
}

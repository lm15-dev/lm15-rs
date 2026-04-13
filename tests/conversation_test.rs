use lm15::*;

#[test]
fn test_conversation_basic() {
    let mut conv = Conversation::new(Some("test system"));
    conv.user("hello");
    conv.user("world");
    assert_eq!(conv.messages().len(), 2);
    assert_eq!(conv.system.as_deref(), Some("test system"));
}

#[test]
fn test_conversation_assistant() {
    let mut conv = Conversation::new(None);
    conv.user("hi");
    conv.assistant(&LMResponse {
        id: "r1".into(),
        model: "test".into(),
        message: Message { role: Role::Assistant, parts: vec![Part::text("hello")], name: None },
        finish_reason: FinishReason::Stop,
        usage: Usage::default(),
        provider: None,
    });
    assert_eq!(conv.messages().len(), 2);
    assert_eq!(conv.messages()[1].role, Role::Assistant);
}

#[test]
fn test_conversation_clear() {
    let mut conv = Conversation::new(None);
    conv.user("hi");
    conv.clear();
    assert_eq!(conv.messages().len(), 0);
}

#[test]
fn test_conversation_prefill() {
    let mut conv = Conversation::new(None);
    conv.user("Output JSON.");
    conv.prefill("{");
    assert_eq!(conv.messages().len(), 2);
    assert_eq!(conv.messages()[1].role, Role::Assistant);
}

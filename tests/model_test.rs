use lm15::*;
use std::sync::Arc;

#[test]
fn test_model_prepare() {
    let client = Arc::new(UniversalLM::new());
    let m = Model::new(client, ModelOpts {
        model: "test".into(),
        system: Some("test system".into()),
        ..ModelOpts::default()
    });
    let req = m.prepare("hello");
    assert_eq!(req.model, "test");
    assert_eq!(req.system.as_deref(), Some("test system"));
    assert_eq!(req.messages.len(), 1);
}

#[test]
fn test_model_history() {
    let client = Arc::new(UniversalLM::new());
    let mut m = Model::new(client, ModelOpts { model: "test".into(), ..ModelOpts::default() });
    m.history.push(HistoryEntry {
        request: LMRequest { model: "test".into(), messages: vec![Message::user("hi")], system: None, tools: vec![], config: Config::default() },
        response: LMResponse {
            id: "r1".into(), model: "test".into(),
            message: Message { role: Role::Assistant, parts: vec![Part::text("ok")], name: None },
            finish_reason: FinishReason::Stop, usage: Usage::default(), provider: None,
        },
    });
    assert_eq!(m.history.len(), 1);
}

#[test]
fn test_model_clear_history() {
    let client = Arc::new(UniversalLM::new());
    let mut m = Model::new(client, ModelOpts { model: "test".into(), ..ModelOpts::default() });
    m.history.push(HistoryEntry {
        request: LMRequest { model: "test".into(), messages: vec![Message::user("hi")], system: None, tools: vec![], config: Config::default() },
        response: LMResponse {
            id: "r1".into(), model: "test".into(),
            message: Message { role: Role::Assistant, parts: vec![Part::text("ok")], name: None },
            finish_reason: FinishReason::Stop, usage: Usage::default(), provider: None,
        },
    });
    m.clear_history();
    assert_eq!(m.history.len(), 0);
}

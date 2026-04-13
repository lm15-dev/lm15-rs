use lm15::*;

#[test]
fn test_client_rejects_unknown_provider() {
    let client = UniversalLM::new();
    let request = LMRequest {
        model: "test".into(),
        messages: vec![Message::user("hi")],
        system: None,
        tools: vec![],
        config: Config::default(),
    };
    let result = client.complete(&request, "nope");
    assert!(result.is_err());
}

#[test]
fn test_providers() {
    let p = providers();
    assert!(p.contains_key("openai"));
    assert!(p.contains_key("anthropic"));
    assert!(p.contains_key("gemini"));
    assert!(p["openai"].contains(&"OPENAI_API_KEY".to_string()));
}

use lm15::resolve_provider;

#[test]
fn test_resolve_provider() {
    assert_eq!(resolve_provider("claude-sonnet-4-5").unwrap(), "anthropic");
    assert_eq!(resolve_provider("gpt-4.1-mini").unwrap(), "openai");
    assert_eq!(resolve_provider("gemini-2.5-flash").unwrap(), "gemini");
    assert_eq!(resolve_provider("o1-preview").unwrap(), "openai");
    assert_eq!(resolve_provider("o3-mini").unwrap(), "openai");
}

#[test]
fn test_resolve_provider_unknown() {
    assert!(resolve_provider("llama-3").is_err());
}

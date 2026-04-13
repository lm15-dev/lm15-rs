use lm15::*;
use std::collections::HashMap;

#[test]
fn test_text_part() {
    let p = Part::text("hello");
    assert_eq!(p.part_type, PartType::Text);
    assert_eq!(p.text.as_deref(), Some("hello"));
}

#[test]
fn test_thinking_part() {
    let p = Part::thinking("hmm");
    assert_eq!(p.part_type, PartType::Thinking);
    assert_eq!(p.text.as_deref(), Some("hmm"));
}

#[test]
fn test_tool_call_part() {
    let mut input = JsonObject::new();
    input.insert("city".into(), "Paris".into());
    let p = Part::tool_call("c1", "weather", input);
    assert_eq!(p.part_type, PartType::ToolCall);
    assert_eq!(p.id.as_deref(), Some("c1"));
    assert_eq!(p.name.as_deref(), Some("weather"));
}

#[test]
fn test_image_url() {
    let p = Part::image_url("https://example.com/img.png");
    assert_eq!(p.part_type, PartType::Image);
    assert!(p.source.is_some());
    assert_eq!(p.source.as_ref().unwrap().url.as_deref(), Some("https://example.com/img.png"));
}

#[test]
fn test_data_source_bytes() {
    let ds = DataSource::base64("AQID", "application/octet-stream");
    let bytes = ds.bytes().unwrap();
    assert_eq!(bytes, vec![1, 2, 3]);
}

#[test]
fn test_data_source_bytes_error() {
    let ds = DataSource::url("https://example.com", None);
    assert!(ds.bytes().is_err());
}

#[test]
fn test_user_message() {
    let m = Message::user("hello");
    assert_eq!(m.role, Role::User);
    assert_eq!(m.parts.len(), 1);
    assert_eq!(m.parts[0].text.as_deref(), Some("hello"));
}

#[test]
fn test_response_text() {
    let resp = LMResponse {
        id: "r1".into(),
        model: "test".into(),
        message: Message {
            role: Role::Assistant,
            parts: vec![Part::text("Hello"), Part::text("World")],
            name: None,
        },
        finish_reason: FinishReason::Stop,
        usage: Usage::default(),
        provider: None,
    };
    assert_eq!(resp.text(), Some("Hello\nWorld".into()));
}

#[test]
fn test_response_tool_calls() {
    let resp = LMResponse {
        id: "r1".into(),
        model: "test".into(),
        message: Message {
            role: Role::Assistant,
            parts: vec![
                Part::text("thinking..."),
                Part::tool_call("c1", "weather", HashMap::new()),
            ],
            name: None,
        },
        finish_reason: FinishReason::ToolCall,
        usage: Usage::default(),
        provider: None,
    };
    let calls = resp.tool_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name.as_deref(), Some("weather"));
}

#[test]
fn test_response_json() {
    let resp = LMResponse {
        id: "r1".into(),
        model: "test".into(),
        message: Message {
            role: Role::Assistant,
            parts: vec![Part::text(r#"{"name": "Alice", "age": 30}"#)],
            name: None,
        },
        finish_reason: FinishReason::Stop,
        usage: Usage::default(),
        provider: None,
    };
    let data: serde_json::Value = resp.json().unwrap();
    assert_eq!(data["name"], "Alice");
    assert_eq!(data["age"], 30);
}

#[test]
fn test_tool_function() {
    let t = Tool::function("test", "A test tool", HashMap::new());
    assert_eq!(t.tool_type, "function");
    assert_eq!(t.name, "test");
}

#[test]
fn test_tool_with_fn() {
    let t = Tool::function_with_fn("greet", "Say hello", HashMap::new(), |_args| {
        Ok(serde_json::json!("Hello!"))
    });
    assert!(t.func.is_some());
    let result = (t.func.as_ref().unwrap())(&HashMap::new()).unwrap();
    assert_eq!(result, serde_json::json!("Hello!"));
}

use lm15::*;
use lm15::result::{LMResult, ResultOpts, StartStreamFn, ToolFn};
use std::collections::HashMap;

fn make_text_stream(text: &str) -> StartStreamFn {
    let text = text.to_string();
    Box::new(move |_req: &LMRequest| {
        Ok(vec![
            StreamEvent::start("r1", "test"),
            StreamEvent::text_delta(0, &text),
            StreamEvent::end(FinishReason::Stop, Usage::default()),
        ])
    })
}

#[test]
fn test_result_text() {
    let mut r = LMResult::new(ResultOpts {
        request: LMRequest { model: "test".into(), messages: vec![Message::user("hi")], system: None, tools: vec![], config: Config::default() },
        start_stream: make_text_stream("Hello!"),
        on_finished: None,
        callable_registry: HashMap::new(),
        on_tool_call: None,
        max_tool_rounds: 8,
        retries: 0,
    });
    assert_eq!(r.text().unwrap(), "Hello!");
}

#[test]
fn test_result_response() {
    let mut r = LMResult::new(ResultOpts {
        request: LMRequest { model: "test".into(), messages: vec![Message::user("hi")], system: None, tools: vec![], config: Config::default() },
        start_stream: make_text_stream("ok"),
        on_finished: None,
        callable_registry: HashMap::new(),
        on_tool_call: None,
        max_tool_rounds: 8,
        retries: 0,
    });
    let resp = r.response().unwrap();
    assert_eq!(resp.finish_reason, FinishReason::Stop);
}

#[test]
fn test_result_stream_chunks() {
    let mut r = LMResult::new(ResultOpts {
        request: LMRequest { model: "test".into(), messages: vec![Message::user("hi")], system: None, tools: vec![], config: Config::default() },
        start_stream: make_text_stream("Hello"),
        on_finished: None,
        callable_registry: HashMap::new(),
        on_tool_call: None,
        max_tool_rounds: 8,
        retries: 0,
    });
    let chunks = r.stream().unwrap();
    assert!(chunks.iter().any(|c| c.chunk_type == "text"));
    assert!(chunks.iter().any(|c| c.chunk_type == "finished"));
}

#[test]
fn test_result_tool_auto_exec() {
    let call_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let cc = call_count.clone();

    let start_stream: StartStreamFn = Box::new(move |_req: &LMRequest| {
        let n = cc.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if n == 0 {
            // First call: return tool call
            let mut raw = JsonObject::new();
            raw.insert("type".into(), "tool_call".into());
            raw.insert("id".into(), "c1".into());
            raw.insert("name".into(), "greet".into());
            raw.insert("input".into(), "{}".into());
            Ok(vec![
                StreamEvent::start("r1", "test"),
                StreamEvent { event_type: "delta".into(), part_index: Some(0), delta_raw: Some(raw), ..StreamEvent::default() },
                StreamEvent::end(FinishReason::ToolCall, Usage::default()),
            ])
        } else {
            // Second call: return text
            Ok(vec![
                StreamEvent::start("r2", "test"),
                StreamEvent::text_delta(0, "done"),
                StreamEvent::end(FinishReason::Stop, Usage::default()),
            ])
        }
    });

    let mut registry: HashMap<String, ToolFn> = HashMap::new();
    registry.insert("greet".into(), Box::new(|_args| Ok(serde_json::json!("Hello!"))));

    let mut r = LMResult::new(ResultOpts {
        request: LMRequest {
            model: "test".into(),
            messages: vec![Message::user("hi")],
            system: None,
            tools: vec![Tool::function("greet", "Say hi", HashMap::new())],
            config: Config::default(),
        },
        start_stream,
        on_finished: None,
        callable_registry: registry,
        on_tool_call: None,
        max_tool_rounds: 2,
        retries: 0,
    });

    assert_eq!(r.text().unwrap(), "done");
    assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 2);
}

#[test]
fn test_result_json() {
    let mut r = LMResult::new(ResultOpts {
        request: LMRequest { model: "test".into(), messages: vec![Message::user("hi")], system: None, tools: vec![], config: Config::default() },
        start_stream: make_text_stream(r#"{"name":"Alice","age":30}"#),
        on_finished: None,
        callable_registry: HashMap::new(),
        on_tool_call: None,
        max_tool_rounds: 8,
        retries: 0,
    });
    let data: serde_json::Value = r.json().unwrap();
    assert_eq!(data["name"], "Alice");
    assert_eq!(data["age"], 30);
}

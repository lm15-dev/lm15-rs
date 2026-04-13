use lm15::*;
use std::time::Duration;

fn make_response() -> LMResponse {
    LMResponse {
        id: "r1".into(), model: "test".into(),
        message: Message { role: Role::Assistant, parts: vec![Part::text("ok")], name: None },
        finish_reason: FinishReason::Stop,
        usage: Usage { input_tokens: 5, output_tokens: 2, total_tokens: 7, ..Usage::default() },
        provider: None,
    }
}

#[test]
fn test_with_retries() {
    let mw = with_retries(2, Duration::from_millis(1));
    let attempts = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let a = attempts.clone();

    let result = mw(
        &LMRequest { model: "test".into(), messages: vec![Message::user("hi")], system: None, tools: vec![], config: Config::default() },
        &|_req| {
            let n = a.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if n < 2 {
                Err(LM15Error::RateLimit("429".into()))
            } else {
                Ok(make_response())
            }
        },
    );

    assert!(result.is_ok());
    assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 3);
}

#[test]
fn test_with_cache() {
    let mut cache = std::collections::HashMap::new();
    let mut cached_fn = with_cache(&mut cache);
    let call_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let cc = call_count.clone();

    let req = LMRequest { model: "test".into(), messages: vec![Message::user("hi")], system: None, tools: vec![], config: Config::default() };

    cached_fn(&req, &|_req| {
        cc.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(make_response())
    }).unwrap();
    cached_fn(&req, &|_req| {
        cc.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(make_response())
    }).unwrap();

    assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 1);
}

#[test]
fn test_with_history() {
    let mut history = Vec::new();
    {
        let mut history_fn = with_history(&mut history);
        let req = LMRequest { model: "test".into(), messages: vec![Message::user("hi")], system: None, tools: vec![], config: Config::default() };
        history_fn(&req, &|_req| Ok(make_response())).unwrap();
    }

    assert_eq!(history.len(), 1);
    assert_eq!(history[0].model, "test");
}

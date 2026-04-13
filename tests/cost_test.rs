use lm15::*;
use std::collections::HashMap;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

#[test]
fn test_estimate_cost_basic() {
    let usage = Usage { input_tokens: 1000, output_tokens: 500, total_tokens: 1500, ..Usage::default() };
    let mut rates = HashMap::new();
    rates.insert("input".into(), 3.0);
    rates.insert("output".into(), 15.0);
    let cost = estimate_cost(&usage, &rates, "openai");
    assert!(cost.total > 0.0);
    assert!(close(cost.input, 1000.0 * 3.0 / 1_000_000.0));
    assert!(close(cost.output, 500.0 * 15.0 / 1_000_000.0));
}

#[test]
fn test_estimate_cost_anthropic_cache() {
    let usage = Usage {
        input_tokens: 500, output_tokens: 200, total_tokens: 700,
        cache_read_tokens: Some(300), cache_write_tokens: Some(100),
        ..Usage::default()
    };
    let mut rates = HashMap::new();
    rates.insert("input".into(), 3.0);
    rates.insert("output".into(), 15.0);
    rates.insert("cache_read".into(), 1.5);
    rates.insert("cache_write".into(), 3.75);
    let cost = estimate_cost(&usage, &rates, "anthropic");
    assert!(close(cost.input, 500.0 * 3.0 / 1_000_000.0));
    assert!(close(cost.cache_read, 300.0 * 1.5 / 1_000_000.0));
}

#[test]
fn test_estimate_cost_openai_reasoning() {
    let usage = Usage {
        input_tokens: 100, output_tokens: 500, total_tokens: 600,
        reasoning_tokens: Some(200), ..Usage::default()
    };
    let mut rates = HashMap::new();
    rates.insert("input".into(), 3.0);
    rates.insert("output".into(), 15.0);
    rates.insert("reasoning".into(), 15.0);
    let cost = estimate_cost(&usage, &rates, "openai");
    // text_output = 500 - 200 = 300
    assert!(close(cost.output, 300.0 * 15.0 / 1_000_000.0));
    assert!(close(cost.reasoning, 200.0 * 15.0 / 1_000_000.0));
}

#[test]
fn test_estimate_cost_gemini_reasoning() {
    let usage = Usage {
        input_tokens: 100, output_tokens: 300, total_tokens: 600,
        reasoning_tokens: Some(200), ..Usage::default()
    };
    let mut rates = HashMap::new();
    rates.insert("output".into(), 15.0);
    rates.insert("reasoning".into(), 15.0);
    let cost = estimate_cost(&usage, &rates, "gemini");
    // text_output = 300 (not subtracted for gemini)
    assert!(close(cost.output, 300.0 * 15.0 / 1_000_000.0));
}

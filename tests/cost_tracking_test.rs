use lm15::*;
use std::collections::HashMap;
use std::sync::Arc;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

fn fake_spec(model: &str, provider: &str, input: f64, output: f64) -> ModelSpec {
    let mut raw = HashMap::new();
    raw.insert("cost".into(), serde_json::json!({"input": input, "output": output}));
    ModelSpec {
        id: model.into(),
        provider: provider.into(),
        context_window: None,
        max_output: None,
        input_modalities: vec![],
        output_modalities: vec![],
        tool_call: false,
        structured_output: false,
        reasoning: false,
        raw,
    }
}

#[test]
fn test_lookup_cost_none_when_disabled() {
    disable_cost_tracking();
    let usage = Usage { input_tokens: 100, output_tokens: 50, total_tokens: 150, ..Usage::default() };
    assert!(lookup_cost("gpt-4.1-mini", &usage).is_none());
}

#[test]
fn test_result_cost_works_with_index() {
    let mut index = HashMap::new();
    index.insert("gpt-4.1-mini".into(), fake_spec("gpt-4.1-mini", "openai", 3.0, 15.0));
    set_cost_index(Some(index));

    let mut result = LMResult::new(lm15::result::ResultOpts {
        request: LMRequest { model: "gpt-4.1-mini".into(), messages: vec![Message::user("hi")], system: None, tools: vec![], config: Config::default() },
        start_stream: Box::new(|_req| Ok(vec![
            StreamEvent::start("r1", "gpt-4.1-mini"),
            StreamEvent::text_delta(0, "hello"),
            StreamEvent::end(FinishReason::Stop, Usage { input_tokens: 1000, output_tokens: 500, total_tokens: 1500, ..Usage::default() }),
        ])),
        on_finished: None,
        callable_registry: HashMap::new(),
        on_tool_call: None,
        max_tool_rounds: 8,
        retries: 0,
    });

    let cost = result.cost().unwrap().unwrap();
    assert!(close(cost.input, 1000.0 * 3.0 / 1_000_000.0));
    assert!(close(cost.output, 500.0 * 15.0 / 1_000_000.0));
}

#[test]
fn test_model_total_cost_sums_history() {
    let mut index = HashMap::new();
    index.insert("gpt-4.1-mini".into(), fake_spec("gpt-4.1-mini", "openai", 3.0, 15.0));
    set_cost_index(Some(index));

    let client = Arc::new(UniversalLM::new());
    let mut model = Model::new(client, ModelOpts { model: "gpt-4.1-mini".into(), ..ModelOpts::default() });
    model.history.push(HistoryEntry {
        request: LMRequest { model: "gpt-4.1-mini".into(), messages: vec![Message::user("one")], system: None, tools: vec![], config: Config::default() },
        response: LMResponse {
            id: "r1".into(),
            model: "gpt-4.1-mini".into(),
            message: Message { role: Role::Assistant, parts: vec![Part::text("ok")], name: None },
            finish_reason: FinishReason::Stop,
            usage: Usage { input_tokens: 1000, output_tokens: 500, total_tokens: 1500, ..Usage::default() },
            provider: None,
        },
    });
    model.history.push(HistoryEntry {
        request: LMRequest { model: "gpt-4.1-mini".into(), messages: vec![Message::user("two")], system: None, tools: vec![], config: Config::default() },
        response: LMResponse {
            id: "r2".into(),
            model: "gpt-4.1-mini".into(),
            message: Message { role: Role::Assistant, parts: vec![Part::text("ok")], name: None },
            finish_reason: FinishReason::Stop,
            usage: Usage { input_tokens: 1000, output_tokens: 500, total_tokens: 1500, ..Usage::default() },
            provider: None,
        },
    });

    let total = model.total_cost().unwrap();
    assert!(close(total.total, 2.0 * ((1000.0 * 3.0 / 1_000_000.0) + (500.0 * 15.0 / 1_000_000.0))));
}

#[test]
fn test_model_total_cost_zero_for_empty_history_when_enabled() {
    let mut index = HashMap::new();
    index.insert("gpt-4.1-mini".into(), fake_spec("gpt-4.1-mini", "openai", 3.0, 15.0));
    set_cost_index(Some(index));

    let client = Arc::new(UniversalLM::new());
    let model = Model::new(client, ModelOpts { model: "gpt-4.1-mini".into(), ..ModelOpts::default() });
    let total = model.total_cost().unwrap();
    assert!(close(total.total, 0.0));
}

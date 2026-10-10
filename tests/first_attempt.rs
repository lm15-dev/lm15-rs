//! 2026-10-10: a budget alone fills effort (MAP-7 rule 3 read the other way);
//! a DataPart answer reads through text() (types.md §Response convenience).
use lm15::types::{effort_for_budget, Reasoning, ReasoningEffort};

#[test]
fn a_budget_alone_fills_effort_from_the_grading_table() {
    use ReasoningEffort::*;
    let got: Vec<_> = [512, 1024, 2047, 2048, 8192, 16384, 24576, 32768, 1_000_000]
        .iter()
        .map(|b| effort_for_budget(*b))
        .collect();
    assert_eq!(
        got,
        vec![Minimal, Minimal, Minimal, Low, Medium, High, Xhigh, Max, Max]
    );
    let r = Reasoning::with_budget(1024);
    assert_eq!((r.effort, r.thinking_budget), (Minimal, Some(1024)));
}

#[test]
fn a_data_part_answer_reads_through_text() {
    use lm15::types::{DataPart, FinishReason, Message, Part, Response, Role, TextPart, Usage};
    let data = |v: serde_json::Value| {
        Part::Data(DataPart {
            value: v,
            probabilities: None,
            method: None,
            continuation: vec![],
        })
    };
    let response = |parts: Vec<Part>| Response {
        id: None,
        model: "m".into(),
        message: Message {
            role: Role::Assistant,
            parts,
            continuation: vec![],
        },
        finish_reason: FinishReason::Stop,
        usage: Usage::default(),
        logprobs: None,
        logprobs_complete: true,
        adaptations: vec![],
        provider_data: None,
    };
    let only = response(vec![data(serde_json::json!({"ok": true}))]);
    assert_eq!(only.text().as_deref(), Some(r#"{"ok":true}"#));
    let mixed = response(vec![
        Part::Text(TextPart::new("hi")),
        data(serde_json::json!({"ok": true})),
    ]);
    assert_eq!(mixed.text(), None);
}

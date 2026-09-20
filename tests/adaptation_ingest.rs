//! MAP-13 promotions and deprecated request spelling translations.
use lm15::dialects::openai_chat::ingest::request_from_openai_chat;
use lm15::types::{Tool, ToolChoiceMode};
use serde_json::json;

#[test]
fn legacy_declarations_and_forcing_translate_without_records() {
    let request = request_from_openai_chat(
        &json!({
            "model":"m", "messages":[{"role":"user","content":"hi"}],
            "functions":[{"name":"lookup","parameters":{"type":"object","properties":{}}}],
            "function_call":{"name":"lookup"},
            "seed":42, "frequency_penalty":0.0, "presence_penalty":0.5, "top_k":20,
            "prediction":{"type":"content","content":"predicted"}
        }),
        None,
    )
    .unwrap();
    assert!(matches!(&request.tools[0],Tool::Function(f) if f.name=="lookup"));
    let choice = request.config.tool_choice.unwrap();
    assert_eq!(choice.mode, ToolChoiceMode::Required);
    assert_eq!(choice.allowed, vec!["lookup".to_string()]);
    assert_eq!(request.config.seed, Some(42));
    assert_eq!(request.config.top_k, Some(20));
    assert_eq!(request.config.frequency_penalty, Some(0.0));
    assert_eq!(request.config.presence_penalty, Some(0.5));
    let extensions = request.config.extensions.unwrap();
    assert!(extensions.contains_key("prediction"));
    assert!(!extensions.contains_key("seed"));
}

#[test]
fn dual_spellings_and_invalid_legacy_choice_refuse_locally() {
    for extras in [
        json!({"functions":[],"tools":[]}),
        json!({"function_call":"auto","tool_choice":"auto"}),
        json!({"function_call":"required"}),
    ] {
        let mut body = json!({"model":"m","messages":[{"role":"user","content":"hi"}]});
        body.as_object_mut()
            .unwrap()
            .extend(extras.as_object().unwrap().clone());
        assert!(request_from_openai_chat(&body, None).is_err());
    }
}

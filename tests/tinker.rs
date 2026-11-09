//! Tinker (Thinking Machines): the two doors, the reasoning switch per base
//! family, a checkpoint that follows its declared base model, URI model
//! names, FastAPI error bodies (MAP-21), the stop sequence returned in the
//! text (MAP-22) and the catalog's fine-tune origin (MAP-23). lm15-contract
//! `changes/2026-11-09-tinker-live.md`; the corpus pins the same rules
//! through the harness, these tests keep them visible here.

use lm15::registry::lookup;
use lm15::router::{is_uri, openai_chat_model_string, LMRouter, RouterConfig};
use lm15::types::*;
use lm15::LmBuilder;
use serde_json::{json, Value};

const CK: &str =
    "tinker://e68146d2-1a2e-56e8-b42f-1f5957705501:train:0/sampler_weights/lm15-untrained";

fn chat(base: &[(&str, &str)]) -> lm15::ProviderLM {
    LmBuilder::for_entry(lookup("tinker").unwrap())
        .api_key("k")
        .base_models(base.iter().map(|(a, b)| (a.to_string(), b.to_string())))
        .build()
        .unwrap()
}

fn ask(model: &str, effort: Option<ReasoningEffort>) -> Request {
    let mut request = Request::new(model, vec![Message::user("hi").unwrap()]).unwrap();
    request.config.reasoning = effort.map(Reasoning::new);
    request
}

fn body(lm: &lm15::ProviderLM, request: &Request) -> Value {
    let built = lm.build_request(request, false).unwrap();
    built.body.clone().unwrap()
}

#[test]
fn two_doors_one_key_and_the_alias() {
    let c = lookup("tinker").unwrap();
    let m = lookup("tinker-anthropic").unwrap();
    assert_eq!(c.access().env_keys, ["TINKER_API_KEY"]);
    assert_eq!(m.access().env_keys, ["TINKER_API_KEY"]);
    assert_eq!(lookup("thinkingmachines").unwrap().id, "tinker");
}

#[test]
fn a_uri_is_one_name_and_aliases_route() {
    let router = LMRouter::with_config(RouterConfig::new().env([("X", "y")])).unwrap();
    for (model, provider, wire) in [
        (CK.to_string(), "tinker", CK),
        (format!("tinker:{CK}"), "tinker", CK),
        (format!("tinker-anthropic:{CK}"), "tinker-anthropic", CK),
        (format!("thinkingmachines:{CK}"), "tinker", CK),
        (
            "thinkingmachines:Qwen/Qwen3-8B".to_string(),
            "tinker",
            "Qwen/Qwen3-8B",
        ),
    ] {
        let r = router.resolve(&model).unwrap();
        assert_eq!(
            (r.provider.as_str(), r.model.as_str()),
            (provider, wire),
            "{model}"
        );
    }
    assert!(router.resolve("openai://x").is_err());
    assert!(is_uri(CK) && !is_uri("tinker:Qwen/Qwen3-8B"));
    assert_eq!(
        openai_chat_model_string("thinkingmachines/Qwen/Qwen3-8B").unwrap(),
        "tinker:Qwen/Qwen3-8B"
    );
}

#[test]
fn reasoning_off_per_family_and_a_checkpoints_default() {
    let lm = chat(&[]);
    for (model, off) in [
        ("thinkingmachines/Inkling-Small", json!("none")),
        ("Qwen/Qwen3.5-4B", json!(false)),
        ("moonshotai/Kimi-K2.6", json!(false)),
        (
            "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
            json!(false),
        ),
        ("Qwen/Qwen3.8-27B", json!(false)),
        ("deepseek-ai/DeepSeek-V4.1-Flash:peft:262144", json!(false)),
        (CK, json!(false)),
    ] {
        let request = ask(model, Some(ReasoningEffort::Off));
        assert_eq!(body(&lm, &request)["reasoning_effort"], off, "{model}");
        assert!(lm.plan(&request).unwrap().is_empty(), "{model}");
    }
}

#[test]
fn an_effort_word_on_a_switch_is_on_and_recorded() {
    let lm = chat(&[]);
    let request = ask("Qwen/Qwen3.5-4B", Some(ReasoningEffort::High));
    assert_eq!(body(&lm, &request)["reasoning_effort"], json!(true));
    let records = lm.plan(&request).unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].field, "config.reasoning.effort");
    assert_eq!(records[0].asked, Some(json!("high")));
    assert_eq!(records[0].applied, Some(json!("on")));
    for model in ["thinkingmachines/Inkling", "openai/gpt-oss-20b", CK] {
        assert_eq!(
            body(&lm, &ask(model, Some(ReasoningEffort::Low)))["reasoning_effort"],
            json!("low"),
            "{model}"
        );
    }
}

#[test]
fn a_checkpoint_follows_its_declared_base_model() {
    let inkling = chat(&[(CK, "thinkingmachines/Inkling-Small")]);
    assert_eq!(
        body(&inkling, &ask(CK, Some(ReasoningEffort::Off)))["reasoning_effort"],
        json!("none")
    );
    let qwen = chat(&[(CK, "Qwen/Qwen3.5-4B")]);
    assert_eq!(
        body(&qwen, &ask(CK, Some(ReasoningEffort::Medium)))["reasoning_effort"],
        json!(true)
    );
    let router = LMRouter::with_config(
        RouterConfig::new()
            .env([("TINKER_API_KEY", "k")])
            .base_models([(CK, "thinkingmachines/Inkling")]),
    )
    .unwrap();
    let lm = router.lm(CK).unwrap();
    assert_eq!(
        lm.base_models().get(CK).map(String::as_str),
        Some("thinkingmachines/Inkling")
    );
    assert_eq!(
        body(&lm, &ask(CK, Some(ReasoningEffort::Off)))["reasoning_effort"],
        json!("none")
    );
    assert!(LMRouter::with_config(RouterConfig::new().base_models([(CK, "")])).is_err());
}

#[test]
fn response_format_is_dropped_and_recorded() {
    let lm = chat(&[]);
    for format in [
        json!({"type": "json_object"}),
        json!({"type": "json_schema", "name": "x", "schema": {"type": "object"}}),
    ] {
        let mut request = ask("thinkingmachines/Inkling", None);
        request.config.response_format = Some(format.as_object().unwrap().clone());
        assert!(body(&lm, &request).get("response_format").is_none());
        let records = lm.plan(&request).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].field, "config.response_format");
    }
}

#[test]
fn map22_the_returned_stop_sequence_is_cut() {
    let lm = chat(&[]);
    let mut request = ask("thinkingmachines/Inkling-Small", None);
    request.config.stop = vec![" 7".into()];
    let reply = json!({"id": "x", "object": "chat.completion", "model": "thinkingmachines/Inkling-Small",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "1 2 3 4 5 6 7"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12, "prompt_tokens_details": {"cached_tokens": 0}}});
    let response = lm
        .parse_response(&request, 200, reply.to_string().as_bytes())
        .unwrap();
    assert_eq!(response.text().unwrap(), "1 2 3 4 5 6");
    assert_eq!(response.usage.input_tokens, Some(5)); // the server read the whole reply: usage stays
    let frames = [
        json!({"choices": [{"index": 0, "delta": {"role": "assistant", "content": "1 2 3 4 5 6 7"}, "finish_reason": null}]}),
        json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}}),
    ];
    let sse: String = frames
        .iter()
        .map(|f| format!("data: {f}\n\n"))
        .collect::<String>()
        + "data: [DONE]\n\n";
    let streamed = lm.parse_stream(&request, sse.as_bytes()).unwrap();
    assert_eq!(streamed.text().unwrap(), "1 2 3 4 5 6");
    let plain = LmBuilder::for_entry(lookup("openai-chat").unwrap())
        .api_key("k")
        .build()
        .unwrap();
    let mut other = ask("gpt-4.1-mini", None);
    other.config.stop = vec!["END".into()];
    let reply = json!({"id": "x", "object": "chat.completion", "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "a END b"}, "finish_reason": "stop"}]});
    assert_eq!(
        plain
            .parse_response(&other, 200, reply.to_string().as_bytes())
            .unwrap()
            .text()
            .unwrap(),
        "a END b"
    );
}

#[test]
fn map21_and_map15_detail_bodies_keep_message_and_class() {
    for (provider, status, body, class) in [
        (
            "tinker",
            401,
            json!({"detail": "Unable to validate api-key"}),
            "AuthError",
        ),
        (
            "tinker",
            404,
            json!({"detail": format!("Weights not found for model_path {CK}")}),
            "UnsupportedModelError",
        ),
        (
            "tinker",
            400,
            json!({"detail": "Sampling is not supported for x. (org)"}),
            "UnsupportedModelError",
        ),
        (
            "tinker",
            400,
            json!({"detail": "model_path 'tinker://x' is not a valid tinker path. Expected"}),
            "UnsupportedModelError",
        ),
        (
            "tinker",
            422,
            json!({"detail": "model returned no tool call when tool_choice required one"}),
            "InvalidRequestError",
        ),
        (
            "tinker-anthropic",
            404,
            json!({"detail": format!("Weights not found for model_path {CK}")}),
            "UnsupportedModelError",
        ),
    ] {
        let err = lm15::errors::normalize_error(provider, status, &body.to_string()).unwrap();
        assert_eq!(err.class_name(), class, "{body}");
        assert!(
            err.message().starts_with(body["detail"].as_str().unwrap()),
            "{body}"
        );
    }
    let validation = json!({"detail": "Validation Error", "errors": [{"loc": "body -> messages -> 0 -> role", "msg": "Input should be 'system'", "type": "literal_error"}]});
    assert_eq!(
        lm15::errors::detail_message(&validation).unwrap(),
        "Validation Error: body -> messages -> 0 -> role: Input should be 'system'"
    );
}

#[test]
fn map23_the_catalog_names_each_checkpoints_base() {
    let lm = chat(&[]);
    let body = json!({"object": "list", "data": [{"id": CK, "object": "model", "created": 1, "owned_by": "x", "base_model": "Qwen/Qwen3.5-4B"}]});
    let models = lm.parse_models(200, body.to_string().as_bytes()).unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].origin.r#type, "fine-tune");
    assert_eq!(
        models[0].origin.base_model.as_deref(),
        Some("Qwen/Qwen3.5-4B")
    );
}

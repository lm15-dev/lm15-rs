//! The build-time refusals of the Anthropic dialect (MAP-5..8 and port.md
//! rule 4): one test per refusal, each asserting the class and ErrorCode
//! the family pins and that nothing was built.

use serde_json::{json, Value};

use lm15::compat::{AnthropicCompat, Compat, Knob};
use lm15::registry::adapter_for;
use lm15::types::{
    BuiltinTool, CacheConfig, CachePrefix, CacheRetention, Config, DocumentPart, FunctionTool,
    ImagePart, JsonObject, Message, Part, Reasoning, ReasoningEffort, ReasoningSummary, Request,
    SystemContent, Tool, ToolChoice, ToolChoiceMode,
};
use lm15::{AnthropicLM, Lm15Error, ProviderLM};

fn object(value: Value) -> JsonObject {
    match value {
        Value::Object(map) => map,
        _ => unreachable!(),
    }
}

fn function(name: &str) -> Tool {
    Tool::Function(FunctionTool::new(name, None, FunctionTool::default_parameters()).unwrap())
}

fn cfg(f: impl FnOnce(&mut Config)) -> Config {
    let mut config = Config::default();
    f(&mut config);
    config
}

fn request(model: &str, config: Config) -> Request {
    Request {
        model: model.into(),
        messages: vec![Message::user("hi").unwrap()],
        system: None,
        tools: Vec::new(),
        config,
    }
}

fn anthropic() -> ProviderLM {
    AnthropicLM::builder().api_key("k").build().unwrap()
}

fn assert_refusal(result: Result<lm15::TransportRequest, Lm15Error>, class: &str, code: &str) {
    match result {
        Ok(out) => panic!("built {} instead of refusing", out.url),
        Err(err) => {
            assert_eq!(err.class_name(), class, "{err}");
            assert_eq!(err.code().as_str(), code, "{err}");
            assert_eq!(err.provider(), Some("anthropic"), "{err}");
        }
    }
}

fn assert_unsupported(result: Result<lm15::TransportRequest, Lm15Error>) {
    assert_refusal(result, "UnsupportedFeatureError", "unsupported_feature");
}

// ─── MAP-5 / MAP-7 ──────────────────────────────────────────────────

#[test]
fn map5_reasoning_off_omits_thinking_on_the_public_api() {
    let config = cfg(|c| {
        c.reasoning = Some(Reasoning::new(ReasoningEffort::Off));
    });
    let out = anthropic()
        .build_request(&request("claude-sonnet-4-5", config), false)
        .unwrap();
    let body = out.body.unwrap();
    assert!(body.get("thinking").is_none());
    assert_eq!(body["max_tokens"], json!(1024));
}

#[test]
fn map7_thinking_budget_raises_on_the_adaptive_class() {
    let config = cfg(|c| {
        c.reasoning = Some(Reasoning {
            effort: ReasoningEffort::High,
            thinking_budget: Some(4096),
            summary: None,
        });
    });
    assert_unsupported(
        anthropic().build_request(&request("claude-sonnet-5", config.clone()), false),
    );
    // The manual class takes it as budget_tokens.
    let out = anthropic()
        .build_request(&request("claude-sonnet-4-5", config), false)
        .unwrap();
    let body = out.body.unwrap();
    assert_eq!(
        body["thinking"],
        json!({"type": "enabled", "budget_tokens": 4096})
    );
    assert_eq!(body["max_tokens"], json!(4096 + 1024));
}

#[test]
fn map7_minimal_raises_on_the_adaptive_class_only() {
    let config = cfg(|c| {
        c.reasoning = Some(Reasoning::new(ReasoningEffort::Minimal));
    });
    assert_unsupported(
        anthropic().build_request(&request("claude-opus-4-6", config.clone()), false),
    );
    let out = anthropic()
        .build_request(&request("claude-haiku-4-5", config), false)
        .unwrap();
    assert_eq!(
        out.body.unwrap()["thinking"],
        json!({"type": "enabled", "budget_tokens": 1024})
    );
}

#[test]
fn map7_summary_detail_levels_raise_and_auto_is_satisfied() {
    for summary in [ReasoningSummary::Concise, ReasoningSummary::Detailed] {
        let config = cfg(|c| {
            c.reasoning = Some(Reasoning {
                effort: ReasoningEffort::Medium,
                thinking_budget: None,
                summary: Some(summary),
            });
        });
        assert_unsupported(anthropic().build_request(&request("claude-sonnet-4-5", config), false));
    }
    let config = cfg(|c| {
        c.reasoning = Some(Reasoning {
            effort: ReasoningEffort::Medium,
            thinking_budget: None,
            summary: Some(ReasoningSummary::Auto),
        });
    });
    let out = anthropic()
        .build_request(&request("claude-sonnet-5", config), false)
        .unwrap();
    let body = out.body.unwrap();
    assert_eq!(body["thinking"], json!({"type": "adaptive"}));
    assert_eq!(body["output_config"], json!({"effort": "medium"}));
}

// ─── MAP-6 ──────────────────────────────────────────────────────────

#[test]
fn map6_cache_key_raises() {
    let config = cfg(|c| {
        c.cache = Some(CacheConfig {
            key: Some("affinity".into()),
            ..CacheConfig::default()
        });
    });
    assert_unsupported(anthropic().build_request(&request("claude-sonnet-4-5", config), false));
}

#[test]
fn map6_cache_resource_raises_on_every_door_of_the_wire() {
    let config = cfg(|c| {
        c.cache = Some(CacheConfig {
            resource: Some("cache_1".into()),
            ..CacheConfig::default()
        });
    });
    assert_unsupported(
        anthropic().build_request(&request("claude-sonnet-4-5", config.clone()), false),
    );
    // A server that ignores marks has no stored tier either.
    let deepseek = adapter_for("deepseek-anthropic", "k", None, None, None).unwrap();
    let err = deepseek
        .build_request(&request("deepseek-v4-flash", config), false)
        .unwrap_err();
    assert_eq!(err.class_name(), "UnsupportedFeatureError");
    assert_eq!(err.provider(), Some("deepseek-anthropic"));
}

#[test]
fn map6_long_retention_is_a_ttl_mark_or_a_refusal() {
    let config = cfg(|c| {
        c.cache = Some(CacheConfig {
            retention: Some(CacheRetention::Long),
            prefix: Some(CachePrefix::History),
            ..CacheConfig::default()
        });
    });
    let mut req = request("claude-sonnet-4-5", config.clone());
    req.system = Some(SystemContent::Text("sys".into()));
    let body = anthropic()
        .build_request(&req, false)
        .unwrap()
        .body
        .unwrap();
    let marker = json!({"type": "ephemeral", "ttl": "1h"});
    assert_eq!(body["system"][0]["cache_control"], marker);
    assert_eq!(body["messages"][0]["content"][0]["cache_control"], marker);

    let deepseek = adapter_for("deepseek-anthropic", "k", None, None, None).unwrap();
    let err = deepseek
        .build_request(&request("deepseek-v4-flash", config), false)
        .unwrap_err();
    assert_eq!(err.class_name(), "UnsupportedFeatureError");
}

#[test]
fn map6_prefix_marks_and_the_off_switch() {
    // prefix_until_index clamps to the last message; mode=off places nothing.
    let config = cfg(|c| {
        c.cache = Some(CacheConfig {
            prefix_until_index: Some(99),
            ..CacheConfig::default()
        });
    });
    let mut req = request("claude-sonnet-4-5", config);
    req.system = Some(SystemContent::Text("sys".into()));
    let body = anthropic()
        .build_request(&req, false)
        .unwrap()
        .body
        .unwrap();
    assert_eq!(
        body["messages"][0]["content"][0]["cache_control"],
        json!({"type": "ephemeral"})
    );
    assert_eq!(
        body["system"],
        json!([{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}])
    );

    let config = cfg(|c| {
        c.cache = Some(CacheConfig {
            mode: lm15::types::CacheMode::Off,
            ..CacheConfig::default()
        });
    });
    let mut req = request("claude-sonnet-4-5", config);
    req.system = Some(SystemContent::Text("sys".into()));
    let body = anthropic()
        .build_request(&req, false)
        .unwrap()
        .body
        .unwrap();
    assert_eq!(body["system"], json!("sys"));
    assert!(body["messages"][0]["content"][0]
        .get("cache_control")
        .is_none());
}

// ─── MAP-8 ──────────────────────────────────────────────────────────

#[test]
fn map8_allowed_subset_raises_and_full_allowlists_relax() {
    let config = cfg(|c| {
        c.tool_choice = Some(ToolChoice {
            mode: ToolChoiceMode::Auto,
            allowed: vec!["a".into()],
            parallel: None,
        });
    });
    let mut req = request("claude-sonnet-4-5", config);
    req.tools = vec![function("a"), function("b")];
    assert_unsupported(anthropic().build_request(&req, false));

    req.config.tool_choice = Some(ToolChoice {
        mode: ToolChoiceMode::Required,
        allowed: vec!["b".into(), "a".into()],
        parallel: Some(false),
    });
    let body = anthropic()
        .build_request(&req, false)
        .unwrap()
        .body
        .unwrap();
    assert_eq!(
        body["tool_choice"],
        json!({"type": "any", "disable_parallel_tool_use": true})
    );
    req.config.tool_choice = Some(ToolChoice {
        mode: ToolChoiceMode::Auto,
        allowed: vec!["a".into(), "b".into()],
        parallel: Some(true),
    });
    let body = anthropic()
        .build_request(&req, false)
        .unwrap()
        .body
        .unwrap();
    assert_eq!(body["tool_choice"], json!({"type": "auto"}));
}

#[test]
fn map8_json_object_raises_and_json_schema_drops_name_and_strict() {
    let config = cfg(|c| {
        c.response_format = Some(object(json!({"type": "json_object"})));
    });
    assert_unsupported(anthropic().build_request(&request("claude-sonnet-5", config), false));

    let config = cfg(|c| {
        c.response_format = Some(object(json!({
            "type": "json_schema", "name": "place", "strict": true,
            "schema": {"type": "object", "properties": {}, "additionalProperties": false}
        })));
    });
    let body = anthropic()
        .build_request(&request("claude-sonnet-5", config), false)
        .unwrap()
        .body
        .unwrap();
    assert_eq!(
        body["output_config"],
        json!({"format": {"type": "json_schema", "schema": {"type": "object", "properties": {}, "additionalProperties": false}}})
    );
}

#[test]
fn store_and_logprobs_raise() {
    let config = cfg(|c| {
        c.store = Some(false);
    });
    assert_unsupported(anthropic().build_request(&request("claude-sonnet-4-5", config), false));
    let config = cfg(|c| {
        c.logprobs = Some(0);
    });
    assert_unsupported(anthropic().build_request(&request("claude-sonnet-4-5", config), false));
}

// ─── Parts with no block (port.md rule 4) ───────────────────────────

#[test]
fn audio_video_binary_parts_raise_in_messages_and_tool_results() {
    let audio = Part::Audio(lm15::types::AudioPart::from_url("https://x/a.wav").unwrap());
    let req = Request {
        model: "claude-sonnet-4-5".into(),
        messages: vec![Message::user(vec![Part::text("listen"), audio.clone()]).unwrap()],
        system: None,
        tools: Vec::new(),
        config: Config::default(),
    };
    assert_unsupported(anthropic().build_request(&req, false));

    let req = Request {
        model: "claude-sonnet-4-5".into(),
        messages: vec![Message::tool("call_1", vec![audio]).unwrap()],
        system: None,
        tools: Vec::new(),
        config: Config::default(),
    };
    assert_unsupported(anthropic().build_request(&req, false));
}

#[test]
fn system_parts_are_text_blocks_and_media_raises() {
    let mut req = request("claude-sonnet-4-5", Config::default());
    req.system = Some(SystemContent::Parts(vec![Part::text("a"), Part::text("b")]));
    let body = anthropic()
        .build_request(&req, false)
        .unwrap()
        .body
        .unwrap();
    assert_eq!(
        body["system"],
        json!([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
    );
    req.system = Some(SystemContent::Parts(vec![
        Part::text("a"),
        Part::Image(ImagePart::from_url("https://x/i.png").unwrap()),
    ]));
    assert_unsupported(anthropic().build_request(&req, false));
}

#[test]
fn media_blocks_and_an_unreadable_path() {
    let mut req = request("claude-sonnet-4-5", Config::default());
    req.messages = vec![Message::user(vec![
        Part::Image(ImagePart::from_data("image/png", "data:image/png;base64,aGk=").unwrap()),
        Part::Document(DocumentPart::from_file_id("file_1").unwrap()),
    ])
    .unwrap()];
    let body = anthropic()
        .build_request(&req, false)
        .unwrap()
        .body
        .unwrap();
    assert_eq!(
        body["messages"][0]["content"],
        json!([
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGk="}},
            {"type": "document", "source": {"type": "file", "file_id": "file_1"}},
        ])
    );
    req.messages = vec![Message::user(vec![Part::Document(
        DocumentPart::from_path("/nonexistent/lm15/none.pdf").unwrap(),
    )])
    .unwrap()];
    assert_refusal(
        anthropic().build_request(&req, false),
        "InvalidRequestError",
        "invalid_request",
    );
}

#[test]
fn builtin_forcing_and_the_code_execution_beta() {
    let config = cfg(|c| {
        c.tool_choice = Some(ToolChoice {
            mode: ToolChoiceMode::Required,
            allowed: vec!["code_execution".into()],
            parallel: None,
        });
    });
    let mut req = request("claude-sonnet-4-5", config);
    req.tools = vec![Tool::Builtin(
        BuiltinTool::new("code_execution", None).unwrap(),
    )];
    let out = anthropic().build_request(&req, false).unwrap();
    assert_eq!(
        out.header("anthropic-beta"),
        Some("code-execution-2025-05-22")
    );
    let body = out.body.unwrap();
    assert_eq!(
        body["tools"],
        json!([{"type": "code_execution_20250522", "name": "code_execution"}])
    );
    assert_eq!(
        body["tool_choice"],
        json!({"type": "tool", "name": "code_execution"})
    );
}

#[test]
fn a_custom_compat_value_binds_to_the_named_constructor() {
    let compat = AnthropicCompat {
        model_prefixes: Some(&["claude-"]),
        structured_output: Some(Knob::Set(lm15::compat::SendReject::Reject)),
        ..AnthropicCompat::EMPTY
    };
    let lm = AnthropicLM::builder()
        .api_key("k")
        .compat(Compat::Anthropic(compat))
        .build()
        .unwrap();
    assert_refusal(
        lm.build_request(&request("gpt-x", Config::default()), false),
        "UnsupportedModelError",
        "unsupported_model",
    );
    let config = cfg(|c| {
        c.response_format = Some(object(json!({"type": "json_schema", "schema": {}})));
    });
    assert_unsupported(lm.build_request(&request("claude-sonnet-4-5", config), false));
}

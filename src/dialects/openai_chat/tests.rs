//! Unit tests: one per MAP-5..8 refusal, one per compat knob value, one
//! per `changes/`-documented corner. The fixture corpus is exercised by
//! `tests/dialect_openai_chat_*.rs` and the harness.

use serde_json::{json, Value};

use super::*;
use crate::auth::{ACCESS_POLICIES, OPENAI_CHAT_API, XAI};
use crate::cloud::hosts::HostSettings;
use crate::compat::{
    Compat, IncludeOmit, Knob, OpenAICacheControl, OpenAIChatAssistantAfterToolResult,
    OpenAIChatAssistantReasoningContent, OpenAIChatInstructionRole, OpenAIChatMaxTokensField,
    OpenAIChatThinkingFormat, OpenAIChatThinkingReplay, OpenAIChatUserField, SendReject,
};
use crate::types::{
    BuiltinTool, CacheConfig, CacheMode, CachePrefix, CacheRetention, CitationPart, Config,
    FunctionTool, ImagePart, Message, Part, Reasoning, ReasoningEffort, ReasoningSummary, Tool,
    ToolChoice, ToolChoiceMode,
};

fn tool(name: &str) -> Tool {
    Tool::Function(FunctionTool {
        name: name.into(),
        description: Some("d".into()),
        parameters: FunctionTool::default_parameters(),
    })
}

fn request(config: Config) -> Request {
    Request {
        model: "m".into(),
        messages: vec![Message::user("hi").unwrap()],
        system: None,
        tools: vec![tool("get_weather")],
        config,
    }
}

fn build_with(
    compat: Compat,
    policy: &crate::auth::AccessPolicy,
    request: &Request,
    stream: bool,
) -> Result<Value, Lm15Error> {
    let settings = HostSettings::new();
    let cx = BuildContext {
        provider: policy.provider,
        policy,
        settings: &settings,
        compat: &compat,
        base_url: "https://x/v1",
        model: &request.model,
    };
    let wire = OPENAI_CHAT.build(request, stream, &cx)?;
    assert_eq!(wire.path, "/chat/completions");
    assert_eq!(wire.endpoint, Some("chat/completions"));
    assert_eq!(wire.model.as_deref(), Some(request.model.as_str()));
    Ok(wire.body.unwrap())
}

fn build(compat: OpenAIChatCompat, request: &Request) -> Result<Value, Lm15Error> {
    build_with(Compat::OpenAIChat(compat), &OPENAI_CHAT_API, request, false)
}

fn preset(name: &str) -> OpenAIChatCompat {
    OpenAIChatCompat::preset(name).unwrap().clone()
}

fn refuses(result: Result<Value, Lm15Error>, needle: &str) {
    let err = result.expect_err("expected a refusal");
    assert_eq!(err.class_name(), "UnsupportedFeatureError");
    assert_eq!(err.code().as_str(), "unsupported_feature");
    assert!(err.message().contains(needle), "{err}");
}

fn keys(body: &Value) -> Vec<&str> {
    body.as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect()
}

// ─── Key order and defaults ───────────────────────────────────────────

#[test]
fn body_keys_follow_the_reference_order() {
    let mut config = Config {
        max_tokens: Some(5),
        temperature: Some(0.2),
        top_p: Some(0.9),
        stop: vec!["x".into()],
        logprobs: Some(2),
        tool_choice: Some(ToolChoice {
            mode: ToolChoiceMode::Auto,
            allowed: Vec::new(),
            parallel: Some(false),
        }),
        response_format: Some(json!({"type": "json_object"}).as_object().unwrap().clone()),
        reasoning: Some(Reasoning::new(ReasoningEffort::Low)),
        cache: Some(CacheConfig {
            key: Some("k".into()),
            retention: Some(CacheRetention::Long),
            ..Default::default()
        }),
        service_tier: Some("flex".into()),
        user_id: Some("u".into()),
        store: Some(false),
        ..Default::default()
    };
    config.extensions = Some(json!({"n": 2, "cache": 1}).as_object().unwrap().clone());
    let body = build_with(
        Compat::OpenAIChat(OpenAIChatCompat::EMPTY),
        &OPENAI_CHAT_API,
        &request(config),
        true,
    )
    .unwrap();
    assert_eq!(
        keys(&body),
        vec![
            "model",
            "messages",
            "stream",
            "stream_options",
            "max_completion_tokens",
            "temperature",
            "top_p",
            "stop",
            "logprobs",
            "top_logprobs",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "response_format",
            "reasoning_effort",
            "prompt_cache_key",
            "prompt_cache_retention",
            "service_tier",
            "user",
            "store",
            "n",
        ]
    );
    assert_eq!(body["stream_options"], json!({"include_usage": true}));
    assert_eq!(body["logprobs"], json!(true));
    assert_eq!(body["top_logprobs"], json!(2));
    assert_eq!(body["store"], json!(false));
    assert_eq!(body["temperature"], json!(0.2));
    assert!(
        body.get("cache").is_none(),
        "reserved extension key is not forwarded"
    );
}

#[test]
fn a_binding_without_a_chat_compat_reads_the_empty_partial() {
    let body = build_with(
        Compat::None,
        &OPENAI_CHAT_API,
        &request(Config::default()),
        false,
    )
    .unwrap();
    assert_eq!(keys(&body), vec!["model", "messages", "tools"]);
    assert_eq!(body["tools"][0]["function"]["description"], json!("d"));
}

#[test]
fn a_tool_without_description_omits_the_key() {
    let mut req = request(Config::default());
    req.tools = vec![Tool::Function(FunctionTool {
        name: "t".into(),
        description: None,
        parameters: FunctionTool::default_parameters(),
    })];
    let body = build(OpenAIChatCompat::EMPTY, &req).unwrap();
    assert_eq!(
        body["tools"][0]["function"],
        json!({"name": "t", "parameters": {"type": "object", "properties": {}}})
    );
}

#[test]
fn headers_carry_content_type_and_the_policy_statics() {
    let settings = HostSettings::new();
    let compat = Compat::OpenAIChat(OpenAIChatCompat::EMPTY);
    let cx = BuildContext {
        provider: "xai",
        policy: &XAI,
        settings: &settings,
        compat: &compat,
        base_url: "https://api.x.ai/v1",
        model: "grok-4.6",
    };
    let wire = OPENAI_CHAT
        .build(&request(Config::default()), false, &cx)
        .unwrap();
    assert_eq!(
        wire.headers,
        vec![("Content-Type".to_string(), "application/json".to_string())]
    );
    assert_eq!(wire.body.unwrap()["model"], json!("grok-4.6"));
}

// ─── Compat knobs, one per value ──────────────────────────────────────

#[test]
fn instruction_role_system_and_developer() {
    let mut req = request(Config::default());
    req.system = Some("sys".into());
    req.messages.insert(0, Message::developer("dev").unwrap());
    let body = build(preset("meta"), &req).unwrap();
    assert_eq!(
        body["messages"][0],
        json!({"role": "developer", "content": "sys"})
    );
    assert_eq!(
        body["messages"][1],
        json!({"role": "developer", "content": "dev"})
    );
    let body = build(preset("deepseek"), &req).unwrap();
    assert_eq!(
        body["messages"][0],
        json!({"role": "system", "content": "sys"})
    );
    assert_eq!(
        body["messages"][1],
        json!({"role": "system", "content": "dev"})
    );
    let mut compat = OpenAIChatCompat::EMPTY;
    compat.instruction_role = Some(Knob::Set(OpenAIChatInstructionRole::Developer));
    assert_eq!(
        build(compat, &req).unwrap()["messages"][0]["role"],
        json!("developer")
    );
}

#[test]
fn max_tokens_field_both_spellings() {
    let req = request(Config {
        max_tokens: Some(7),
        ..Default::default()
    });
    let body = build(preset("xai"), &req).unwrap();
    assert_eq!(body["max_tokens"], json!(7));
    assert!(body.get("max_completion_tokens").is_none());
    let body = build(preset("bedrock"), &req).unwrap();
    assert_eq!(body["max_completion_tokens"], json!(7));
    let mut compat = OpenAIChatCompat::EMPTY;
    compat.max_tokens_field = Some(Knob::Set(OpenAIChatMaxTokensField::MaxTokens));
    assert_eq!(build(compat, &req).unwrap()["max_tokens"], json!(7));
}

#[test]
fn stream_usage_include_and_omit() {
    let req = request(Config::default());
    let body = build_with(
        Compat::OpenAIChat(preset("groq")),
        &OPENAI_CHAT_API,
        &req,
        true,
    )
    .unwrap();
    assert_eq!(body["stream"], json!(true));
    assert_eq!(body["stream_options"], json!({"include_usage": true}));
    let mut compat = OpenAIChatCompat::EMPTY;
    compat.stream_usage = Some(Knob::Set(IncludeOmit::Omit));
    let body = build_with(Compat::OpenAIChat(compat), &OPENAI_CHAT_API, &req, true).unwrap();
    assert_eq!(body["stream"], json!(true));
    assert!(body.get("stream_options").is_none());
    let body = build_with(
        Compat::OpenAIChat(preset("groq")),
        &OPENAI_CHAT_API,
        &req,
        false,
    )
    .unwrap();
    assert!(body.get("stream").is_none());
}

fn tool_loop() -> Request {
    Request {
        model: "m".into(),
        messages: vec![
            Message::user("weather?").unwrap(),
            Message::assistant(vec![
                Part::thinking("think"),
                Part::tool_call(
                    "c1",
                    "get_weather",
                    json!({"city": "Paris"}).as_object().unwrap().clone(),
                )
                .unwrap(),
            ])
            .unwrap(),
            Message::new(
                crate::types::Role::Tool,
                vec![Part::ToolResult(crate::types::ToolResultPart {
                    id: "c1".into(),
                    content: vec![Part::text("Sunny")],
                    name: Some("get_weather".into()),
                    is_error: false,
                    continuation: Vec::new(),
                })],
            )
            .unwrap(),
            Message::user("and Rome?").unwrap(),
        ],
        system: None,
        tools: vec![tool("get_weather")],
        config: Config::default(),
    }
}

#[test]
fn tool_result_name_include_and_omit() {
    let req = tool_loop();
    let body = build(OpenAIChatCompat::EMPTY, &req).unwrap();
    assert_eq!(
        body["messages"][2],
        json!({"role": "tool", "tool_call_id": "c1", "content": "Sunny"})
    );
    let mut compat = OpenAIChatCompat::EMPTY;
    compat.tool_result_name = Some(Knob::Set(IncludeOmit::Include));
    let body = build(compat, &req).unwrap();
    assert_eq!(
        body["messages"][2],
        json!({"role": "tool", "tool_call_id": "c1", "content": "Sunny", "name": "get_weather"})
    );
}

#[test]
fn assistant_after_tool_result_insert_and_omit() {
    let req = tool_loop();
    let body = build(OpenAIChatCompat::EMPTY, &req).unwrap();
    assert_eq!(body["messages"].as_array().unwrap().len(), 4);
    let mut compat = OpenAIChatCompat::EMPTY;
    compat.assistant_after_tool_result =
        Some(Knob::Set(OpenAIChatAssistantAfterToolResult::Insert));
    let body = build(compat.clone(), &req).unwrap();
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 5);
    assert_eq!(messages[3], json!({"role": "assistant", "content": ""}));
    assert_eq!(messages[4]["role"], json!("user"));
    // Never after the last tool row: the server produces that turn.
    let mut trailing = req.clone();
    trailing.messages.pop();
    let body = build(compat, &trailing).unwrap();
    assert_eq!(body["messages"].as_array().unwrap().len(), 3);
}

#[test]
fn thinking_replay_as_text_native_and_omit() {
    let req = tool_loop();
    let body = build(OpenAIChatCompat::EMPTY, &req).unwrap();
    assert_eq!(body["messages"][1]["content"], json!("think"));
    assert!(body["messages"][1].get("reasoning_content").is_none());
    assert_eq!(
        body["messages"][1]["tool_calls"][0],
        json!({"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}})
    );
    let body = build(preset("zai"), &req).unwrap();
    assert_eq!(body["messages"][1]["content"], Value::Null);
    assert_eq!(body["messages"][1]["reasoning_content"], json!("think"));
    assert_eq!(
        keys(&body["messages"][1]),
        vec!["role", "content", "reasoning_content", "tool_calls"]
    );
    let mut compat = OpenAIChatCompat::EMPTY;
    compat.thinking_replay = Some(Knob::Set(OpenAIChatThinkingReplay::Omit));
    let body = build(compat, &req).unwrap();
    assert_eq!(body["messages"][1]["content"], Value::Null);
    assert!(body["messages"][1].get("reasoning_content").is_none());
}

#[test]
fn assistant_reasoning_content_include_empty_and_omit() {
    let mut req = tool_loop();
    req.messages[1] = Message::assistant(vec![Part::text("calling")]).unwrap();
    // deepseek: every assistant turn carries reasoning_content, even empty.
    let body = build(preset("deepseek"), &req).unwrap();
    assert_eq!(
        body["messages"][1],
        json!({"role": "assistant", "content": "calling", "reasoning_content": ""})
    );
    // zai: native replay, but no key when there is no thinking.
    let body = build(preset("zai"), &req).unwrap();
    assert_eq!(
        body["messages"][1],
        json!({"role": "assistant", "content": "calling"})
    );
    let mut compat = preset("deepseek");
    compat.assistant_reasoning_content = Some(Knob::Set(OpenAIChatAssistantReasoningContent::Omit));
    assert!(build(compat, &req).unwrap()["messages"][1]
        .get("reasoning_content")
        .is_none());
}

#[test]
fn thinking_format_every_shape_on_and_off() {
    let on = |format| {
        let mut compat = OpenAIChatCompat::EMPTY;
        compat.thinking_format = Some(Knob::Set(format));
        let req = request(Config {
            reasoning: Some(Reasoning::new(ReasoningEffort::Low)),
            ..Default::default()
        });
        build(compat, &req)
    };
    let off = |format| {
        let mut compat = OpenAIChatCompat::EMPTY;
        compat.thinking_format = Some(Knob::Set(format));
        let req = request(Config {
            reasoning: Some(Reasoning::new(ReasoningEffort::Off)),
            ..Default::default()
        });
        build(compat, &req)
    };
    use OpenAIChatThinkingFormat as F;
    let body = on(F::ReasoningEffort).unwrap();
    assert_eq!(body["reasoning_effort"], json!("low"));
    assert_eq!(
        off(F::ReasoningEffort).unwrap()["reasoning_effort"],
        json!("none")
    );

    assert_eq!(
        on(F::Openrouter).unwrap()["reasoning"],
        json!({"effort": "low"})
    );
    assert_eq!(
        off(F::Openrouter).unwrap()["reasoning"],
        json!({"enabled": false})
    );

    let body = on(F::Deepseek).unwrap();
    assert_eq!(body["thinking"], json!({"type": "enabled"}));
    assert_eq!(body["reasoning_effort"], json!("low"));
    assert_eq!(keys(&body)[3..], ["thinking", "reasoning_effort"]);
    let body = off(F::Deepseek).unwrap();
    assert_eq!(body["thinking"], json!({"type": "disabled"}));
    assert!(body.get("reasoning_effort").is_none());

    let body = on(F::Kimi).unwrap();
    assert_eq!(body["reasoning_effort"], json!("low"));
    assert!(body.get("thinking").is_none());
    let body = off(F::Kimi).unwrap();
    assert_eq!(body["thinking"], json!({"type": "disabled"}));
    assert!(body.get("reasoning_effort").is_none());

    assert_eq!(on(F::Qwen).unwrap()["enable_thinking"], json!(true));
    assert_eq!(off(F::Qwen).unwrap()["enable_thinking"], json!(false));

    assert_eq!(
        on(F::QwenChatTemplate).unwrap()["chat_template_kwargs"],
        json!({"enable_thinking": true, "preserve_thinking": true})
    );
    assert_eq!(
        off(F::QwenChatTemplate).unwrap()["chat_template_kwargs"],
        json!({"enable_thinking": false})
    );

    // MAP-5 / MAP-7: no dial on the wire is a raise, not an omission.
    refuses(on(F::None), "thinking_format='none'");
    refuses(off(F::None), "thinking_format='none'");
}

#[test]
fn strict_tools_include_and_omit() {
    let req = request(Config::default());
    assert!(
        build(preset("openai"), &req).unwrap()["tools"][0]["function"]
            .get("strict")
            .is_none()
    );
    let mut compat = OpenAIChatCompat::EMPTY;
    compat.strict_tools = Some(Knob::Set(IncludeOmit::Include));
    assert_eq!(
        build(compat, &req).unwrap()["tools"][0]["function"]["strict"],
        json!(false)
    );
}

#[test]
fn builtin_tools_groq_and_reject() {
    let mut req = request(Config::default());
    req.tools = vec![
        Tool::Builtin(BuiltinTool::new("web_search", None).unwrap()),
        Tool::Builtin(
            BuiltinTool::new(
                "code_execution",
                Some(json!({"x": 1}).as_object().unwrap().clone()),
            )
            .unwrap(),
        ),
    ];
    let body = build(preset("groq"), &req).unwrap();
    assert_eq!(
        body["tools"],
        json!([{"type": "browser_search"}, {"type": "code_interpreter", "x": 1}])
    );
    refuses(build(preset("openai"), &req), "function tools only");
    req.tools = vec![Tool::Builtin(
        BuiltinTool::new("file_search", None).unwrap(),
    )];
    refuses(build(preset("groq"), &req), "no Groq wire mapping");
    // Named builtin forcing raises on the dialect; plain required flows.
    req.tools = vec![Tool::Builtin(BuiltinTool::new("web_search", None).unwrap())];
    req.config.tool_choice = Some(ToolChoice {
        mode: ToolChoiceMode::Required,
        allowed: vec!["web_search".into()],
        parallel: None,
    });
    refuses(build(preset("groq"), &req), "cannot force builtin tools");
    req.config.tool_choice = Some(ToolChoice {
        mode: ToolChoiceMode::Required,
        ..Default::default()
    });
    assert_eq!(
        build(preset("groq"), &req).unwrap()["tool_choice"],
        json!("required")
    );
}

fn cache_request(cache: CacheConfig, model: &str) -> Request {
    let mut req = request(Config {
        cache: Some(cache),
        ..Default::default()
    });
    req.model = model.into();
    req.system = Some("sys".into());
    req
}

#[test]
fn cache_control_openai_off_switch_and_marks() {
    // mode=off on the 5.6 class: explicit mode, no marks (openai_chat.cache_off).
    let req = cache_request(
        CacheConfig {
            mode: CacheMode::Off,
            ..Default::default()
        },
        "gpt-5.6-sol",
    );
    let body = build(preset("openai"), &req).unwrap();
    assert_eq!(body["prompt_cache_options"], json!({"mode": "explicit"}));
    assert_eq!(body["messages"][0]["content"], json!("sys"));
    // Older classes: nothing (writes are free).
    let mut older = req.clone();
    older.model = "gpt-4.1-mini".into();
    assert!(build(preset("openai"), &older)
        .unwrap()
        .get("prompt_cache_options")
        .is_none());

    // prefix=stable marks the system block and travels with explicit mode.
    let req = cache_request(
        CacheConfig {
            prefix: Some(CachePrefix::Stable),
            ..Default::default()
        },
        "gpt-5.6-sol",
    );
    let body = build(preset("openai"), &req).unwrap();
    assert_eq!(
        body["messages"][0]["content"],
        json!([{"type": "text", "text": "sys", "prompt_cache_breakpoint": {"mode": "explicit"}}])
    );
    assert_eq!(body["prompt_cache_options"], json!({"mode": "explicit"}));
    // Stable with no system prompt: no mark, no mode.
    let mut no_system = req.clone();
    no_system.system = None;
    let body = build(preset("openai"), &no_system).unwrap();
    assert!(body.get("prompt_cache_options").is_none());
    assert_eq!(body["messages"][0]["content"], json!("hi"));

    // prefix=history: nothing on the wire (implicit mode already marks it).
    let req = cache_request(
        CacheConfig {
            prefix: Some(CachePrefix::History),
            ..Default::default()
        },
        "gpt-5.6-sol",
    );
    let body = build(preset("openai"), &req).unwrap();
    assert!(body.get("prompt_cache_options").is_none());
    assert_eq!(body["messages"][0]["content"], json!("sys"));

    // prefix_until_index on a user message ending in text: the mark rides
    // on the array form; clamped to the last message.
    let req = cache_request(
        CacheConfig {
            prefix_until_index: Some(9),
            ..Default::default()
        },
        "gpt-5.6-sol",
    );
    let body = build(preset("openai"), &req).unwrap();
    assert_eq!(
        body["messages"][1]["content"],
        json!([{"type": "text", "text": "hi", "prompt_cache_breakpoint": {"mode": "explicit"}}])
    );
    assert_eq!(body["prompt_cache_options"], json!({"mode": "explicit"}));
    // Same mark on a pre-5.6 model: the mark, not the mode (the 400 is the contract).
    let mut older = req.clone();
    older.model = "gpt-5.4-mini".into();
    let body = build(preset("openai"), &older).unwrap();
    assert!(body.get("prompt_cache_options").is_none());
    assert_eq!(
        body["messages"][1]["content"][0]["prompt_cache_breakpoint"],
        json!({"mode": "explicit"})
    );
    // key and retention.
    let req = cache_request(
        CacheConfig {
            key: Some("k".into()),
            retention: Some(CacheRetention::Long),
            ..Default::default()
        },
        "gpt-4.1",
    );
    let body = build(preset("openai"), &req).unwrap();
    assert_eq!(body["prompt_cache_key"], json!("k"));
    assert_eq!(body["prompt_cache_retention"], json!("24h"));
    assert!(body.get("prompt_cache_options").is_none());
}

#[test]
fn cache_breakpoint_refuses_assistant_tool_and_non_text_prefixes() {
    let mut req = tool_loop();
    req.config.cache = Some(CacheConfig {
        prefix_until_index: Some(1),
        ..Default::default()
    });
    refuses(build(preset("openai"), &req), "assistant message");
    req.config.cache = Some(CacheConfig {
        prefix_until_index: Some(2),
        ..Default::default()
    });
    refuses(build(preset("openai"), &req), "tool message");
    // A user message whose last block is an image.
    let mut req = request(Config {
        cache: Some(CacheConfig {
            prefix_until_index: Some(0),
            ..Default::default()
        }),
        ..Default::default()
    });
    req.messages = vec![Message::user(vec![
        Part::text("see"),
        Part::Image(ImagePart::from_url("https://img").unwrap()),
    ])
    .unwrap()];
    refuses(build(preset("openai"), &req), "not text");
}

#[test]
fn cache_control_none_and_anthropic_send_nothing() {
    let req = cache_request(
        CacheConfig {
            key: Some("k".into()),
            prefix: Some(CachePrefix::Stable),
            retention: Some(CacheRetention::Long),
            ..Default::default()
        },
        "gpt-5.6",
    );
    for name in ["groq", "xai", "deepseek", "bedrock"] {
        let body = build(preset(name), &req).unwrap();
        assert_eq!(keys(&body), vec!["model", "messages", "tools"], "{name}");
        assert_eq!(body["messages"][0]["content"], json!("sys"));
    }
    let mut compat = OpenAIChatCompat::EMPTY;
    compat.cache_control = Some(Knob::Set(OpenAICacheControl::Anthropic));
    assert_eq!(
        keys(&build(compat, &req).unwrap()),
        vec!["model", "messages", "tools"]
    );
}

#[test]
fn cache_control_openai_implicit_forwards_the_key_and_no_mark() {
    let req = cache_request(
        CacheConfig {
            key: Some("k".into()),
            prefix: Some(CachePrefix::Stable),
            retention: Some(CacheRetention::Long),
            ..Default::default()
        },
        "gpt-5.6",
    );
    let body = build(preset("meta"), &req).unwrap();
    assert_eq!(body["messages"][0]["content"], json!("sys"));
    assert_eq!(body["prompt_cache_key"], json!("k"));
    assert_eq!(body["prompt_cache_retention"], json!("24h"));
    assert!(body.get("prompt_cache_options").is_none());
    let off = cache_request(
        CacheConfig {
            mode: CacheMode::Off,
            ..Default::default()
        },
        "gpt-5.6",
    );
    assert_eq!(
        keys(&build(preset("moonshotai"), &off).unwrap()),
        vec!["model", "messages", "tools"]
    );
}

#[test]
fn cache_resource_refuses_on_both_openai_controls() {
    let req = cache_request(
        CacheConfig {
            resource: Some("cache-1".into()),
            ..Default::default()
        },
        "gpt-5.6",
    );
    refuses(build(preset("openai"), &req), "cache.resource");
    refuses(build(preset("meta"), &req), "cache.resource");
    assert!(build(preset("groq"), &req).is_ok());
}

#[test]
fn user_field_user_user_id_and_safety_identifier() {
    let req = request(Config {
        user_id: Some("u".into()),
        ..Default::default()
    });
    assert_eq!(build(preset("bedrock"), &req).unwrap()["user"], json!("u"));
    assert_eq!(
        build(preset("deepseek"), &req).unwrap()["user_id"],
        json!("u")
    );
    assert_eq!(
        build(preset("meta"), &req).unwrap()["safety_identifier"],
        json!("u")
    );
    let mut compat = OpenAIChatCompat::EMPTY;
    compat.user_field = Some(Knob::Set(OpenAIChatUserField::UserId));
    assert_eq!(build(compat, &req).unwrap()["user_id"], json!("u"));
}

#[test]
fn forced_tool_choice_send_and_reject() {
    let required = request(Config {
        tool_choice: Some(ToolChoice {
            mode: ToolChoiceMode::Required,
            ..Default::default()
        }),
        ..Default::default()
    });
    assert_eq!(
        build(preset("bedrock"), &required).unwrap()["tool_choice"],
        json!("required")
    );
    refuses(build(preset("zai"), &required), "silently ignored");
    let none = request(Config {
        tool_choice: Some(ToolChoice {
            mode: ToolChoiceMode::None,
            ..Default::default()
        }),
        ..Default::default()
    });
    refuses(build(preset("zai"), &none), "mode=\"none\"");
    let auto = request(Config {
        tool_choice: Some(ToolChoice::default()),
        ..Default::default()
    });
    assert_eq!(
        build(preset("zai"), &auto).unwrap()["tool_choice"],
        json!("auto")
    );
    let allowed = request(Config {
        tool_choice: Some(ToolChoice {
            mode: ToolChoiceMode::Auto,
            allowed: vec!["get_weather".into()],
            parallel: None,
        }),
        ..Default::default()
    });
    refuses(build(preset("zai"), &allowed), "allowed=");
    let mut compat = OpenAIChatCompat::EMPTY;
    compat.forced_tool_choice = Some(Knob::Set(SendReject::Reject));
    refuses(build(compat, &required), "silently ignored");
}

#[test]
fn json_schema_send_and_reject() {
    let schema = request(Config {
        response_format: Some(
            json!({"type": "json_schema", "schema": {"type": "object"}})
                .as_object()
                .unwrap()
                .clone(),
        ),
        ..Default::default()
    });
    let body = build(preset("bedrock"), &schema).unwrap();
    assert_eq!(
        body["response_format"],
        json!({"type": "json_schema", "json_schema": {"name": "response", "schema": {"type": "object"}}})
    );
    refuses(build(preset("zai"), &schema), "json_schema");
    let object = request(Config {
        response_format: Some(json!({"type": "json_object"}).as_object().unwrap().clone()),
        ..Default::default()
    });
    assert_eq!(
        build(preset("zai"), &object).unwrap()["response_format"],
        json!({"type": "json_object"})
    );
    let mut compat = OpenAIChatCompat::EMPTY;
    compat.json_schema = Some(Knob::Set(SendReject::Reject));
    refuses(build(compat, &schema), "json_schema");
}

#[test]
fn reasoning_efforts_allowlist_refuses_before_the_wire() {
    let medium = request(Config {
        reasoning: Some(Reasoning::new(ReasoningEffort::Medium)),
        ..Default::default()
    });
    refuses(
        build(preset("moonshotai"), &medium),
        "accepts low, high, max",
    );
    let max = request(Config {
        reasoning: Some(Reasoning::new(ReasoningEffort::Max)),
        ..Default::default()
    });
    assert_eq!(
        build(preset("moonshotai"), &max).unwrap()["reasoning_effort"],
        json!("max")
    );
    // No allowlist: the word goes verbatim and the server judges.
    assert_eq!(
        build(preset("openai"), &medium).unwrap()["reasoning_effort"],
        json!("medium")
    );
}

#[test]
fn model_overrides_resolve_per_request_model() {
    let mut req = request(Config {
        tool_choice: Some(ToolChoice {
            mode: ToolChoiceMode::Required,
            ..Default::default()
        }),
        ..Default::default()
    });
    req.model = "openai.gpt-oss-20b-1:0".into();
    refuses(build(preset("bedrock"), &req), "silently ignored");
    req.model = "deepseek.v3.2".into();
    assert_eq!(
        build(preset("bedrock"), &req).unwrap()["tool_choice"],
        json!("required")
    );
    req.model = "google.gemma-3".into();
    refuses(build(preset("bedrock"), &req), "silently ignored");
    assert_eq!(
        build(preset("bedrock-mantle"), &req).unwrap()["tool_choice"],
        json!("required")
    );
}

#[test]
fn routing_and_compat_extensions_ride_the_body() {
    let mut compat = OpenAIChatCompat::EMPTY;
    compat.routing = Some(json!({"order": ["a"]}).as_object().unwrap().clone());
    compat.extensions = Some(
        json!({"chat_template_kwargs": {"x": 1}})
            .as_object()
            .unwrap()
            .clone(),
    );
    let body = build(compat, &request(Config::default())).unwrap();
    assert_eq!(body["provider"], json!({"order": ["a"]}));
    assert_eq!(body["chat_template_kwargs"], json!({"x": 1}));
}

// ─── MAP-5..8 refusals ────────────────────────────────────────────────

#[test]
fn map7_budget_and_summary_levels_refuse_and_auto_is_groqs_knob() {
    let budget = request(Config {
        reasoning: Some(Reasoning {
            effort: ReasoningEffort::Low,
            thinking_budget: Some(1024),
            summary: None,
        }),
        ..Default::default()
    });
    refuses(build(preset("openai"), &budget), "thinking_budget");
    for level in [ReasoningSummary::Concise, ReasoningSummary::Detailed] {
        let req = request(Config {
            reasoning: Some(Reasoning {
                effort: ReasoningEffort::Low,
                thinking_budget: None,
                summary: Some(level),
            }),
            ..Default::default()
        });
        refuses(build(preset("openai"), &req), "detail level");
    }
    let auto = request(Config {
        reasoning: Some(Reasoning {
            effort: ReasoningEffort::Low,
            thinking_budget: None,
            summary: Some(ReasoningSummary::Auto),
        }),
        ..Default::default()
    });
    let body = build(preset("groq"), &auto).unwrap();
    assert_eq!(body["reasoning_format"], json!("parsed"));
    assert_eq!(keys(&body)[3..], ["reasoning_format", "reasoning_effort"]);
    assert!(build(preset("openai"), &auto)
        .unwrap()
        .get("reasoning_format")
        .is_none());
}

#[test]
fn map8_tool_choice_forms() {
    let mut req = request(Config::default());
    req.tools.push(tool("lookup"));
    let choice = |mode, allowed: &[&str]| {
        let mut r = req.clone();
        r.config.tool_choice = Some(ToolChoice {
            mode,
            allowed: allowed.iter().map(|s| s.to_string()).collect(),
            parallel: None,
        });
        build(preset("openai"), &r).unwrap()["tool_choice"].clone()
    };
    assert_eq!(choice(ToolChoiceMode::Auto, &[]), json!("auto"));
    assert_eq!(choice(ToolChoiceMode::Required, &[]), json!("required"));
    assert_eq!(choice(ToolChoiceMode::None, &[]), json!("none"));
    assert_eq!(
        choice(ToolChoiceMode::Required, &["lookup"]),
        json!({"type": "function", "function": {"name": "lookup"}})
    );
    assert_eq!(
        choice(ToolChoiceMode::Auto, &["lookup"]),
        json!({"type": "allowed_tools", "allowed_tools": {"mode": "auto", "tools": [{"type": "function", "function": {"name": "lookup"}}]}})
    );
    assert_eq!(
        choice(ToolChoiceMode::Required, &["lookup", "get_weather"]),
        json!({"type": "allowed_tools", "allowed_tools": {"mode": "required", "tools": [
            {"type": "function", "function": {"name": "lookup"}},
            {"type": "function", "function": {"name": "get_weather"}}
        ]}})
    );
}

#[test]
fn xai_refusal_table() {
    let xai = |config: Config| {
        let mut req = request(config);
        req.tools.push(tool("lookup"));
        build_with(Compat::OpenAIChat(preset("xai")), &XAI, &req, false)
    };
    refuses(
        xai(Config {
            reasoning: Some(Reasoning::new(ReasoningEffort::Off)),
            ..Default::default()
        }),
        "cannot be disabled",
    );
    refuses(
        xai(Config {
            logprobs: Some(0),
            ..Default::default()
        }),
        "logprobs",
    );
    refuses(
        xai(Config {
            tool_choice: Some(ToolChoice {
                mode: ToolChoiceMode::Auto,
                allowed: vec!["lookup".into()],
                parallel: None,
            }),
            ..Default::default()
        }),
        "allowed subsets",
    );
    refuses(
        xai(Config {
            tool_choice: Some(ToolChoice {
                mode: ToolChoiceMode::Required,
                allowed: vec!["lookup".into(), "get_weather".into()],
                parallel: None,
            }),
            ..Default::default()
        }),
        "allowed subsets",
    );
    refuses(
        xai(Config {
            tool_choice: Some(ToolChoice {
                mode: ToolChoiceMode::Required,
                ..Default::default()
            }),
            response_format: Some(json!({"type": "json_object"}).as_object().unwrap().clone()),
            ..Default::default()
        }),
        "response_format",
    );
    // The forced single-function form held live; reasoning on is the deepseek shape.
    let body = xai(Config {
        tool_choice: Some(ToolChoice {
            mode: ToolChoiceMode::Required,
            allowed: vec!["lookup".into()],
            parallel: None,
        }),
        reasoning: Some(Reasoning::new(ReasoningEffort::High)),
        ..Default::default()
    })
    .unwrap();
    assert_eq!(
        body["tool_choice"],
        json!({"type": "function", "function": {"name": "lookup"}})
    );
    assert_eq!(body["thinking"], json!({"type": "enabled"}));
    assert_eq!(body["reasoning_effort"], json!("high"));
    // The same table is not applied to other bindings of the xai preset.
    let off = request(Config {
        reasoning: Some(Reasoning::new(ReasoningEffort::Off)),
        ..Default::default()
    });
    assert_eq!(
        build(preset("xai"), &off).unwrap()["thinking"],
        json!({"type": "disabled"})
    );
    assert!(ACCESS_POLICIES.iter().any(|p| p.provider == "xai"));
}

// ─── Content and no-silent-drop cells ─────────────────────────────────

#[test]
fn user_content_text_and_image_forms() {
    let mut req = request(Config::default());
    req.messages = vec![Message::user(vec![
        Part::text("look"),
        Part::Image(ImagePart::from_url("https://img/x.png").unwrap()),
        Part::Image(ImagePart {
            detail: Some(crate::types::ImageDetail::Low),
            ..ImagePart::from_data("image/jpeg", "AAAA").unwrap()
        }),
    ])
    .unwrap()];
    let body = build(OpenAIChatCompat::EMPTY, &req).unwrap();
    assert_eq!(
        body["messages"][0]["content"],
        json!([
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "https://img/x.png"}},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA", "detail": "low"}}
        ])
    );
    req.messages =
        vec![Message::user(vec![Part::Image(ImagePart::from_file_id("f1").unwrap())]).unwrap()];
    refuses(build(OpenAIChatCompat::EMPTY, &req), "addressed by file_id");
    req.messages = vec![Message::user(vec![
        Part::text("hear"),
        Part::Audio(crate::types::AudioPart::from_data("audio/wav", "AAAA").unwrap()),
    ])
    .unwrap()];
    refuses(build(OpenAIChatCompat::EMPTY, &req), "audio part");
}

#[test]
fn system_parts_render_as_text_and_refusals_replay_as_text() {
    let mut req = request(Config::default());
    req.system = Some(vec![Part::text("a"), Part::text("b")].into());
    req.messages
        .push(Message::assistant(vec![Part::refusal("no").unwrap(), Part::text("t")]).unwrap());
    let body = build(OpenAIChatCompat::EMPTY, &req).unwrap();
    assert_eq!(
        body["messages"][0],
        json!({"role": "system", "content": "a\nb"})
    );
    assert_eq!(
        body["messages"][2],
        json!({"role": "assistant", "content": "no\nt"})
    );
}

#[test]
fn assistant_media_refuses_and_citations_are_dropped() {
    let mut req = request(Config::default());
    req.messages.push(
        Message::assistant(vec![
            Part::text("see"),
            Part::Citation(CitationPart {
                url: Some("https://src".into()),
                title: None,
                text: None,
                continuation: Vec::new(),
            }),
        ])
        .unwrap(),
    );
    let body = build(OpenAIChatCompat::EMPTY, &req).unwrap();
    assert_eq!(
        body["messages"][1],
        json!({"role": "assistant", "content": "see"})
    );
    req.messages.push(
        Message::assistant(vec![Part::Image(
            ImagePart::from_url("https://img").unwrap(),
        )])
        .unwrap(),
    );
    refuses(build(OpenAIChatCompat::EMPTY, &req), "assistant image part");
}

#[test]
fn tool_result_media_follows_the_preset_and_never_a_placeholder() {
    // MAP-10: the base wire's tool row is text-only → a raise naming the
    // door; a preset that proved the array form live (xai) sends
    // text + image_url blocks; a document raises on an `images` preset.
    let image = Part::Image(ImagePart::from_url("https://img").unwrap());
    let mut req = tool_loop();
    req.messages[2] = Message::new(
        crate::types::Role::Tool,
        vec![Part::tool_result("c1", vec![Part::text("panel"), image.clone()]).unwrap()],
    )
    .unwrap();
    refuses(build(OpenAIChatCompat::EMPTY, &req), "text-only tool results");
    let body = build(preset("xai"), &req).unwrap();
    assert_eq!(
        body["messages"][2]["content"],
        json!([{"type": "text", "text": "panel"}, {"type": "image_url", "image_url": {"url": "https://img"}}])
    );
    assert!(!body.to_string().contains("[{\"type\": \"image\"}]"));
    // text-only stays a string; is_error rides as the prefix (rule 5)
    let mut err = crate::types::ToolResultPart::new("c1", "boom").unwrap();
    err.is_error = true;
    req.messages[2] = Message::new(crate::types::Role::Tool, vec![Part::ToolResult(err)]).unwrap();
    assert_eq!(build(preset("xai"), &req).unwrap()["messages"][2]["content"], json!("[error] boom"));
    let document = Part::Document(crate::types::DocumentPart {
        media_type: "application/pdf".into(),
        data: Some("UERG".into()),
        ..Default::default()
    });
    req.messages[2] = Message::new(
        crate::types::Role::Tool,
        vec![Part::tool_result("c1", vec![document]).unwrap()],
    )
    .unwrap();
    refuses(build(preset("xai"), &req), "carries images but not document");
}

#[test]
fn top_k_has_no_slot_and_refuses() {
    let req = request(Config {
        top_k: Some(40),
        ..Default::default()
    });
    refuses(build(preset("vllm"), &req), "top_k");
}

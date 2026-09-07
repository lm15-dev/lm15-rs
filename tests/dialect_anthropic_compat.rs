//! One test per `AnthropicCompat` knob value: the wire each value produces
//! on the same request, and the `changes/`-documented corners the presets
//! carry (`changes/2026-09-03-deepseek-anthropic-live.md`,
//! `changes/2026-09-03-meta-live.md`, `changes/2026-09-03-moonshotai-wires.md`).

use serde_json::{json, Value};

use lm15::compat::{
    AnthropicCacheControl, AnthropicCompat, AnthropicThinkingFormat, AnthropicThinkingReplay,
    Compat, Knob, SendReject,
};
use lm15::types::{
    CacheConfig, Config, ContinuationState, FunctionTool, JsonObject, Message, Part, Reasoning,
    ReasoningEffort, Request, ThinkingPart, Tool, ToolChoice, ToolChoiceMode,
};
use lm15::{AnthropicLM, ClaudeCodeLM, Lm15Error, ProviderLM};

fn bound(compat: AnthropicCompat) -> ProviderLM {
    AnthropicLM::builder()
        .api_key("k")
        .compat(Compat::Anthropic(compat))
        .build()
        .unwrap()
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

fn reasoning(effort: ReasoningEffort) -> Config {
    cfg(|c| c.reasoning = Some(Reasoning::new(effort)))
}

fn body(lm: &ProviderLM, request: &Request) -> Value {
    lm.build_request(request, false).unwrap().body.unwrap()
}

fn unsupported(result: Result<lm15::TransportRequest, Lm15Error>) {
    let err = result.expect_err("a refusal");
    assert_eq!(err.class_name(), "UnsupportedFeatureError", "{err}");
    assert_eq!(err.code().as_str(), "unsupported_feature");
}

fn set<T: Copy>(value: T) -> Option<Knob<T>> {
    Some(Knob::Set(value))
}

// ─── thinking_format ────────────────────────────────────────────────

#[test]
fn thinking_format_anthropic_is_the_model_class_table() {
    let lm = bound(AnthropicCompat::EMPTY);
    let manual = body(
        &lm,
        &request("claude-sonnet-4-5", reasoning(ReasoningEffort::Low)),
    );
    assert_eq!(
        manual["thinking"],
        json!({"type": "enabled", "budget_tokens": 2048})
    );
    assert_eq!(manual["max_tokens"], json!(2048 + 1024));
    assert!(manual.get("output_config").is_none());
    let adaptive = body(
        &lm,
        &request("claude-sonnet-5", reasoning(ReasoningEffort::Low)),
    );
    assert_eq!(adaptive["thinking"], json!({"type": "adaptive"}));
    assert_eq!(adaptive["output_config"], json!({"effort": "low"}));
    assert_eq!(adaptive["max_tokens"], json!(1024));
    let off = body(
        &lm,
        &request("claude-sonnet-5", reasoning(ReasoningEffort::Off)),
    );
    assert!(off.get("thinking").is_none());
    // `Knob::Auto` is the same dialect default.
    let auto = bound(AnthropicCompat {
        thinking_format: Some(Knob::Auto),
        ..AnthropicCompat::EMPTY
    });
    assert_eq!(
        body(
            &auto,
            &request("claude-sonnet-4-5", reasoning(ReasoningEffort::Low))
        ),
        manual
    );
}

#[test]
fn thinking_format_deepseek_sends_off_and_enabled_with_effort() {
    let lm = bound(AnthropicCompat {
        thinking_format: set(AnthropicThinkingFormat::Deepseek),
        ..AnthropicCompat::EMPTY
    });
    let on = body(
        &lm,
        &request("deepseek-v4-flash", reasoning(ReasoningEffort::Xhigh)),
    );
    assert_eq!(on["thinking"], json!({"type": "enabled"}));
    assert_eq!(on["output_config"], json!({"effort": "xhigh"}));
    assert_eq!(on["max_tokens"], json!(1024));
    let off = body(
        &lm,
        &request("deepseek-v4-flash", reasoning(ReasoningEffort::Off)),
    );
    assert_eq!(off["thinking"], json!({"type": "disabled"}));
    let absent = body(&lm, &request("deepseek-v4-flash", Config::default()));
    assert!(absent.get("thinking").is_none());
    // `minimal` goes out verbatim: the server judges (400 there, live).
    let minimal = body(
        &lm,
        &request("deepseek-v4-flash", reasoning(ReasoningEffort::Minimal)),
    );
    assert_eq!(minimal["output_config"], json!({"effort": "minimal"}));
    // A budget is a silent no-op there: refused.
    let mut config = reasoning(ReasoningEffort::Low);
    config.reasoning.as_mut().unwrap().thinking_budget = Some(4096);
    unsupported(lm.build_request(&request("deepseek-v4-flash", config), false));
}

#[test]
fn thinking_format_adaptive_has_no_model_table() {
    let lm = bound(AnthropicCompat {
        thinking_format: set(AnthropicThinkingFormat::Adaptive),
        ..AnthropicCompat::EMPTY
    });
    let on = body(
        &lm,
        &request("muse-spark-1.3", reasoning(ReasoningEffort::Low)),
    );
    assert_eq!(on["thinking"], json!({"type": "adaptive"}));
    assert_eq!(on["output_config"], json!({"effort": "low"}));
    // Off reaches the wire so the server refuses it loudly (Meta always reasons).
    let off = body(
        &lm,
        &request("muse-spark-1.3", reasoning(ReasoningEffort::Off)),
    );
    assert_eq!(off["thinking"], json!({"type": "disabled"}));
    let mut config = reasoning(ReasoningEffort::Low);
    config.reasoning.as_mut().unwrap().thinking_budget = Some(4096);
    unsupported(lm.build_request(&request("muse-spark-1.3", config), false));
}

#[test]
fn thinking_format_effort_sends_the_dial_alone() {
    let lm = bound(AnthropicCompat {
        thinking_format: set(AnthropicThinkingFormat::Effort),
        ..AnthropicCompat::EMPTY
    });
    let on = body(&lm, &request("kimi-k3", reasoning(ReasoningEffort::High)));
    assert!(on.get("thinking").is_none());
    assert_eq!(on["output_config"], json!({"effort": "high"}));
    let off = body(&lm, &request("kimi-k3", reasoning(ReasoningEffort::Off)));
    assert_eq!(off["thinking"], json!({"type": "disabled"}));
    assert!(off.get("output_config").is_none());
}

// ─── thinking_replay ────────────────────────────────────────────────

fn unsigned_thinking_turn() -> Vec<Message> {
    vec![
        Message::user("q").unwrap(),
        Message::assistant(vec![
            Part::Thinking(ThinkingPart::new("why")),
            Part::text("a"),
        ])
        .unwrap(),
        Message::user("next").unwrap(),
    ]
}

#[test]
fn thinking_replay_signed_replays_unsigned_blocks_as_text() {
    let lm = bound(AnthropicCompat {
        thinking_replay: set(AnthropicThinkingReplay::Signed),
        ..AnthropicCompat::EMPTY
    });
    let mut req = request("claude-sonnet-4-5", Config::default());
    req.messages = unsigned_thinking_turn();
    let out = body(&lm, &req);
    assert_eq!(
        out["messages"][1]["content"],
        json!([{"type": "text", "text": "why"}, {"type": "text", "text": "a"}])
    );
    // A signed block replays natively under either value.
    let mut data = JsonObject::new();
    data.insert("signature".into(), json!("sig"));
    let signed = ThinkingPart {
        text: "why".into(),
        continuation: vec![
            ContinuationState::new("anthropic", "thinking_signature", data).unwrap(),
        ],
    };
    req.messages[1] = Message::assistant(vec![Part::Thinking(signed)]).unwrap();
    assert_eq!(
        body(&lm, &req)["messages"][1]["content"],
        json!([{"type": "thinking", "thinking": "why", "signature": "sig"}])
    );
}

#[test]
fn thinking_replay_unsigned_keeps_the_thinking_block() {
    let lm = bound(AnthropicCompat {
        thinking_replay: set(AnthropicThinkingReplay::Unsigned),
        ..AnthropicCompat::EMPTY
    });
    let mut req = request("claude-sonnet-4-5", Config::default());
    req.messages = unsigned_thinking_turn();
    assert_eq!(
        body(&lm, &req)["messages"][1]["content"],
        json!([{"type": "thinking", "thinking": "why"}, {"type": "text", "text": "a"}])
    );
}

// ─── cache_control ──────────────────────────────────────────────────

#[test]
fn cache_control_anthropic_marks_and_none_places_nothing() {
    let config = cfg(|c| {
        c.cache = Some(CacheConfig::default());
    });
    let mut req = request("claude-sonnet-4-5", config);
    req.system = Some("sys".into());
    let marks = bound(AnthropicCompat {
        cache_control: set(AnthropicCacheControl::Anthropic),
        ..AnthropicCompat::EMPTY
    });
    assert_eq!(
        body(&marks, &req)["system"],
        json!([{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}])
    );
    let none = bound(AnthropicCompat {
        cache_control: set(AnthropicCacheControl::None),
        ..AnthropicCompat::EMPTY
    });
    assert_eq!(body(&none, &req)["system"], json!("sys"));
    // An explicit CacheConfig is not an error on a "none" server, and a
    // key has nothing to attach to there.
    req.config.cache = Some(CacheConfig {
        key: Some("k".into()),
        ..CacheConfig::default()
    });
    assert_eq!(body(&none, &req)["system"], json!("sys"));
}

// ─── structured_output ──────────────────────────────────────────────

#[test]
fn structured_output_send_and_reject() {
    let config = cfg(|c| {
        c.response_format = Some(
            json!({"type": "json_schema", "schema": {"type": "object"}})
                .as_object()
                .cloned()
                .unwrap(),
        );
    });
    let req = request("claude-sonnet-4-5", config);
    let send = bound(AnthropicCompat {
        structured_output: set(SendReject::Send),
        ..AnthropicCompat::EMPTY
    });
    assert_eq!(
        body(&send, &req)["output_config"],
        json!({"format": {"type": "json_schema", "schema": {"type": "object"}}})
    );
    let reject = bound(AnthropicCompat {
        structured_output: set(SendReject::Reject),
        ..AnthropicCompat::EMPTY
    });
    unsupported(reject.build_request(&req, false));
}

// ─── parallel_tool_calls ────────────────────────────────────────────

#[test]
fn parallel_tool_calls_send_and_reject() {
    let config = cfg(|c| {
        c.tool_choice = Some(ToolChoice {
            mode: ToolChoiceMode::Auto,
            allowed: Vec::new(),
            parallel: Some(false),
        });
    });
    let mut req = request("claude-sonnet-4-5", config);
    req.tools = vec![Tool::Function(
        FunctionTool::new("t", None, FunctionTool::default_parameters()).unwrap(),
    )];
    let send = bound(AnthropicCompat {
        parallel_tool_calls: set(SendReject::Send),
        ..AnthropicCompat::EMPTY
    });
    assert_eq!(
        body(&send, &req)["tool_choice"],
        json!({"type": "auto", "disable_parallel_tool_use": true})
    );
    let reject = bound(AnthropicCompat {
        parallel_tool_calls: set(SendReject::Reject),
        ..AnthropicCompat::EMPTY
    });
    unsupported(reject.build_request(&req, false));
    // No preference: nothing to refuse.
    req.config.tool_choice.as_mut().unwrap().parallel = None;
    assert_eq!(body(&reject, &req)["tool_choice"], json!({"type": "auto"}));
}

// ─── sampling_params ────────────────────────────────────────────────

#[test]
fn sampling_params_send_and_reject() {
    let config = cfg(|c| {
        c.top_k = Some(5);
    });
    let req = request("claude-sonnet-4-5", config);
    let send = bound(AnthropicCompat {
        sampling_params: set(SendReject::Send),
        ..AnthropicCompat::EMPTY
    });
    assert_eq!(body(&send, &req)["top_k"], json!(5));
    let reject = bound(AnthropicCompat {
        sampling_params: set(SendReject::Reject),
        ..AnthropicCompat::EMPTY
    });
    unsupported(reject.build_request(&req, false));
    let config = cfg(|c| {
        c.top_p = Some(0.5);
    });
    unsupported(reject.build_request(&request("claude-sonnet-4-5", config), false));
}

// ─── reasoning_efforts, model_prefixes ──────────────────────────────

#[test]
fn reasoning_efforts_allowlist_refuses_words_the_server_would_swallow() {
    let lm = bound(AnthropicCompat {
        thinking_format: set(AnthropicThinkingFormat::Effort),
        reasoning_efforts: Some(&[ReasoningEffort::Low, ReasoningEffort::High]),
        ..AnthropicCompat::EMPTY
    });
    assert_eq!(
        body(&lm, &request("kimi-k3", reasoning(ReasoningEffort::High)))["output_config"],
        json!({"effort": "high"})
    );
    unsupported(lm.build_request(
        &request("kimi-k3", reasoning(ReasoningEffort::Medium)),
        false,
    ));
    // Off is never on the list and never refused by it.
    assert_eq!(
        body(&lm, &request("kimi-k3", reasoning(ReasoningEffort::Off)))["thinking"],
        json!({"type": "disabled"})
    );
}

#[test]
fn model_prefixes_refuse_a_substituted_model() {
    let lm = bound(AnthropicCompat {
        model_prefixes: Some(&["deepseek-", "kimi-"]),
        ..AnthropicCompat::EMPTY
    });
    assert!(lm
        .build_request(&request("kimi-k3", Config::default()), false)
        .is_ok());
    let err = lm
        .build_request(&request("claude-opus-4-1", Config::default()), false)
        .unwrap_err();
    assert_eq!(err.class_name(), "UnsupportedModelError");
    assert_eq!(err.code().as_str(), "unsupported_model");
    // The prefix check reads the wire model: `provider:` is stripped first.
    assert!(lm
        .build_request(&request("anthropic:kimi-k3", Config::default()), false)
        .is_ok());
}

// ─── the claude-code policy binding (AUTH-10) ───────────────────────

#[test]
fn claude_code_binding_puts_the_prefix_first_and_joins_betas() {
    let lm = ClaudeCodeLM::builder().api_key("tok").build().unwrap();
    let mut req = request("claude-sonnet-4-5", Config::default());
    req.system = Some("You are terse.".into());
    let out = lm.build_request(&req, false).unwrap();
    assert_eq!(out.url, "https://api.anthropic.com/v1/messages");
    assert_eq!(out.header("authorization"), Some("Bearer tok"));
    assert_eq!(out.header("anthropic-version"), Some("2023-06-01"));
    assert_eq!(
        out.header("anthropic-beta"),
        Some("claude-code-20250219,oauth-2025-04-20")
    );
    assert_eq!(out.header("x-app"), Some("cli"));
    assert_eq!(out.header("user-agent"), Some("claude-cli/2.1.170"));
    assert_eq!(
        out.header("anthropic-dangerous-direct-browser-access"),
        Some("true")
    );
    assert!(out.header("x-api-key").is_none());
    let body = out.body.unwrap();
    assert_eq!(
        body["system"],
        json!([
            {"type": "text", "text": "You are Claude Code, Anthropic's official CLI for Claude."},
            {"type": "text", "text": "You are terse."},
        ])
    );
    // No caller system: the prefix alone; with cache marks the caller's
    // block keeps its mark after the prefix.
    req.system = None;
    assert_eq!(
        body_of(&lm, &req)["system"],
        json!([{"type": "text", "text": "You are Claude Code, Anthropic's official CLI for Claude."}])
    );
    req.system = Some("sys".into());
    req.config.cache = Some(CacheConfig::default());
    assert_eq!(
        body_of(&lm, &req)["system"],
        json!([
            {"type": "text", "text": "You are Claude Code, Anthropic's official CLI for Claude."},
            {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}},
        ])
    );
}

fn body_of(lm: &ProviderLM, req: &Request) -> Value {
    body(lm, req)
}

// ─── extensions (INV-049) ───────────────────────────────────────────

#[test]
fn extensions_pass_through_verbatim_and_override_in_place() {
    let lm = bound(AnthropicCompat::EMPTY);
    let mut config = reasoning(ReasoningEffort::Low);
    config.extensions = Some(
        json!({"thinking": {"type": "enabled", "budget_tokens": 1, "display": "summarized"}, "inference_geo": "global", "prompt_caching": true})
            .as_object()
            .cloned()
            .unwrap(),
    );
    let out = body(&lm, &request("claude-sonnet-4-5", config));
    assert_eq!(
        out["thinking"],
        json!({"type": "enabled", "budget_tokens": 1, "display": "summarized"})
    );
    assert_eq!(out["inference_geo"], json!("global"));
    assert!(out.get("prompt_caching").is_none());
    let keys: Vec<&str> = out
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        keys,
        vec![
            "model",
            "messages",
            "stream",
            "max_tokens",
            "thinking",
            "inference_geo"
        ]
    );
}

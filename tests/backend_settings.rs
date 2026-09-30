//! AUTH-10 backend settings (amended 2026-09-30) and MAP-7 rule 6's default
//! max_tokens: lm15-contract changes/2026-09-30-claude-code-client-version.md.

use std::collections::BTreeMap;
use std::path::PathBuf;

use lm15::auth::{
    claude_code_version_guidance, CLAUDE_CODE, DEFAULT_CLAUDE_CODE_VERSION, OPENAI_CODEX,
};
use lm15::types::{Config, Message, Reasoning, ReasoningEffort, Request};
use lm15::{
    AnthropicLM, ClaudeCodeLM, LMRouter, Lm15Error, OpenAICodexLM, ProviderLM, RouterConfig,
};
use serde_json::Value;

const REFUSAL: &str = "Claude Code 2.1.170 does not support this model; version 2.1.280 or newer is required. Run 'claude update', or update the Claude desktop app, then try again.";

fn say_hi(model: &str) -> Request {
    Request::new(model, vec![Message::user("hi").unwrap()]).unwrap()
}

fn user_agent(lm: &ProviderLM) -> String {
    lm.build_request(&say_hi("claude-opus-5-5"), false)
        .unwrap()
        .header("user-agent")
        .unwrap()
        .to_string()
}

fn settings(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// A routed subscription door reads its CLI's login file: a scratch HOME.
fn scratch_home(tag: &str) -> PathBuf {
    let home = std::env::temp_dir().join(format!(
        "lm15-backend-settings-{tag}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(home.join(".claude")).unwrap();
    std::fs::create_dir_all(home.join(".codex")).unwrap();
    let expires = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis()
        + 3_600_000;
    std::fs::write(
        home.join(".claude/.credentials.json"),
        format!(r#"{{"claudeAiOauth": {{"accessToken": "tok", "refreshToken": "r", "expiresAt": {expires}}}}}"#),
    )
    .unwrap();
    std::fs::write(
        home.join(".codex/auth.json"),
        r#"{"tokens": {"access_token": "tok", "refresh_token": "r", "account_id": "acct"}}"#,
    )
    .unwrap();
    home
}

#[test]
fn the_table_default_is_the_header_and_the_option() {
    assert_eq!(DEFAULT_CLAUDE_CODE_VERSION, "2.1.285");
    assert_eq!(
        CLAUDE_CODE.backend_option("client_version"),
        Some(DEFAULT_CLAUDE_CODE_VERSION)
    );
    assert_eq!(
        CLAUDE_CODE
            .headers
            .iter()
            .find(|(k, _)| *k == "user-agent")
            .map(|(_, v)| *v),
        Some("claude-cli/2.1.285")
    );
    assert_eq!(
        CLAUDE_CODE.backend_settings[0].env,
        &["LM15_CLAUDE_CODE_VERSION"]
    );
    assert_eq!(
        OPENAI_CODEX.backend_settings[0].env,
        &["LM15_CODEX_CLIENT_VERSION"]
    );
    let lm = ClaudeCodeLM::builder().api_key("k").build().unwrap();
    assert_eq!(user_agent(&lm), "claude-cli/2.1.285");
}

#[test]
fn the_setting_moves_the_header_and_an_adapter_reads_no_environment() {
    let lm = ClaudeCodeLM::builder()
        .api_key("k")
        .setting("client_version", "2.1.280")
        .build()
        .unwrap();
    assert_eq!(user_agent(&lm), "claude-cli/2.1.280");
    let lm = AnthropicLM::builder()
        .access_policy(&CLAUDE_CODE)
        .api_key("k")
        .settings(settings(&[("client_version", "2.1.281")]))
        .build()
        .unwrap();
    assert_eq!(user_agent(&lm), "claude-cli/2.1.281");
    let lm = ClaudeCodeLM::builder()
        .api_key("k")
        .env(settings(&[("LM15_CLAUDE_CODE_VERSION", "9.9.9")]))
        .build()
        .unwrap();
    assert_eq!(user_agent(&lm), "claude-cli/2.1.285");
}

#[test]
fn the_router_reads_the_setting_then_the_environment_then_the_table() {
    let home = scratch_home("router");
    let home = home.to_string_lossy().to_string();
    let explicit = LMRouter::with_config(
        RouterConfig::new()
            .env([
                ("HOME", home.as_str()),
                ("LM15_CLAUDE_CODE_VERSION", "2.1.282"),
            ])
            .setting("claude_code", "client_version", "2.1.281"),
    )
    .unwrap();
    assert_eq!(
        user_agent(&explicit.lm("claude-code:claude-opus-5-5").unwrap()),
        "claude-cli/2.1.281"
    );
    let from_env = LMRouter::with_config(RouterConfig::new().env([
        ("HOME", home.as_str()),
        ("LM15_CLAUDE_CODE_VERSION", "2.1.282"),
    ]))
    .unwrap();
    assert_eq!(
        user_agent(&from_env.lm("claude-code:claude-opus-5-5").unwrap()),
        "claude-cli/2.1.282"
    );
    let by_default =
        LMRouter::with_config(RouterConfig::new().env([("HOME", home.as_str())])).unwrap();
    assert_eq!(
        user_agent(&by_default.lm("claude-code:claude-opus-5-5").unwrap()),
        "claude-cli/2.1.285"
    );
    let codex = LMRouter::with_config(RouterConfig::new().env([
        ("HOME", home.as_str()),
        ("LM15_CODEX_CLIENT_VERSION", "0.151.0"),
    ]))
    .unwrap();
    let lm = codex.lm("openai-codex:gpt-5.4-mini").unwrap();
    assert_eq!(
        lm.settings().get("client_version").map(String::as_str),
        Some("0.151.0")
    );
}

#[test]
fn codex_client_version_is_the_same_setting() {
    let lm = OpenAICodexLM::builder()
        .api_key("k")
        .account_id("a")
        .setting("client_version", "0.150.0")
        .build()
        .unwrap();
    let models = lm.models_request().unwrap();
    assert!(
        models
            .params
            .iter()
            .any(|(k, v)| k == "client_version" && v == "0.150.0"),
        "{:?}",
        models.params
    );
}

#[test]
fn a_setting_nothing_reads_is_refused_not_dropped() {
    let err = ClaudeCodeLM::builder()
        .api_key("k")
        .setting("version", "2.1.280")
        .build()
        .unwrap_err();
    assert!(
        matches!(err, Lm15Error::NotConfiguredError(_))
            && err.to_string().contains("known: client_version"),
        "{err}"
    );
    let router = LMRouter::with_config(RouterConfig::new().api_key("anthropic", "k").setting(
        "anthropic",
        "client_version",
        "1",
    ))
    .unwrap();
    let err = router.lm("anthropic:claude-opus-5-5").unwrap_err();
    assert!(
        err.to_string().contains("this door takes no settings"),
        "{err}"
    );
}

#[test]
fn the_minimum_version_refusal_names_the_setting() {
    let body = serde_json::json!({"type": "error", "error": {"type": "invalid_request_error", "message": REFUSAL}, "request_id": "req_1"}).to_string();
    let err = lm15::errors::normalize_error("claude-code", 400, &body).unwrap();
    assert_eq!(err.class_name(), "InvalidRequestError");
    assert_eq!(
        err.message(),
        format!("{REFUSAL}\n\n  To fix:\n    - lm15 sends this version itself; updating Claude Code does not change it\n    - Set the claude-code setting client_version to 2.1.280 or newer (or LM15_CLAUDE_CODE_VERSION=2.1.280)\n")
    );
    let err = lm15::errors::normalize_error("anthropic", 400, &body).unwrap();
    assert_eq!(err.message(), REFUSAL);
    assert_eq!(claude_code_version_guidance("model: x"), "model: x");
}

#[test]
fn an_unset_max_tokens_is_the_models_ceiling() {
    let cases: &[(&str, Option<u64>, u64, u64)] = &[
        ("claude-opus-5-5", None, 128000, 128000),
        ("claude-haiku-4-5", None, 64000, 64000),
        ("claude-sonnet-4-5", Some(32768), 64000, 64000 - 32768),
        ("claude-haiku-4-5", Some(64000), 64000 + 16384, 16384),
        (
            "anthropic.claude-haiku-4-5-20251001-v1:0",
            None,
            64000,
            64000,
        ),
        ("claude-3-5-haiku-20241022", None, 8192, 8192),
        ("deepseek-v4-flash", None, 16384, 16384),
    ];
    let lm = AnthropicLM::builder().api_key("k").build().unwrap();
    for (model, budget, wire, applied) in cases {
        let mut request = say_hi(model);
        if let Some(budget) = budget {
            let mut config = Config::default();
            let mut reasoning = Reasoning::new(ReasoningEffort::High);
            reasoning.thinking_budget = Some(*budget);
            config.reasoning = Some(reasoning);
            request.config = config;
        }
        let built = lm.build_request(&request, false).unwrap();
        let body: Value = built.body.clone().expect("a JSON body");
        assert_eq!(body["max_tokens"], Value::from(*wire), "{model}");
        let plan = lm.plan(&request).unwrap();
        let record = plan
            .iter()
            .find(|a| a.field == "config.max_tokens")
            .expect(model);
        assert_eq!(record.applied, Some(Value::from(*applied)), "{model}");
    }
}

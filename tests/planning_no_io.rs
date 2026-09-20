use lm15::serde::Canonical;
use lm15::transport::NoTransport;
use lm15::{LmBuilder, Request};
use serde_json::json;

#[test]
fn planning_does_not_read_media_but_building_a_sendable_request_does() {
    let missing = std::env::temp_dir()
        .join(format!("lm15-missing-preview-{}", std::process::id()))
        .join("image.png");
    assert!(!missing.exists(), "the fixture path must not exist");
    let request = Request::from_json(&json!({
        "model": "test-model",
        "messages": [{"role": "user", "parts": [{"type": "image", "media_type": "image/png", "path": missing.to_str().unwrap()}]}]
    })).unwrap();
    for provider in ["openai", "openai-chat", "anthropic", "gemini"] {
        let definition = lm15::registry::lookup(provider).unwrap();
        let builder = LmBuilder::for_entry(definition)
            .api_key("synthetic-key")
            .transport(NoTransport);
        builder.plan(&request).unwrap();
        let lm = builder.build().unwrap();
        lm.plan(&request).unwrap();
        assert!(
            lm.build_request(&request, false).is_err(),
            "{provider}: actual request must not use preview bytes"
        );
    }
}

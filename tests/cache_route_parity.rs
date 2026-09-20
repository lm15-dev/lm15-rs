//! Source-only cached destination regressions. Scripted transport; no live I/O.
use lm15::testing::{FakeResponse, FakeTransport};
use lm15::{
    CacheInfo, CachedPrefix, Canonical, Config, DeclaredProvider, LMRouter, Message,
    OpenAIChatCompat, OpenAIChatLM, Request, RouterConfig,
};
use serde_json::json;

fn req(model: &str) -> Request {
    Request::new(model, vec![Message::user("prefix").unwrap()]).unwrap()
}
fn config() -> RouterConfig {
    RouterConfig::new().env([] as [(&str, &str); 0])
}

#[tokio::test]
async fn automatic_cache_roundtrip_keeps_azure_and_declared_destinations() {
    for destination in ["azure", "private-chat"] {
        let body = if destination == "azure" {
            json!({"id":"r", "status":"completed", "output":[{"type":"message", "role":"assistant", "content":[{"type":"output_text", "text":"ok"}]}]})
        } else {
            json!({"id":"r", "choices":[{"message":{"role":"assistant", "content":"ok"}, "finish_reason":"stop"}]})
        };
        let transport = FakeTransport::new([FakeResponse::json(&body), FakeResponse::json(&body)]);
        let declared = DeclaredProvider::chat(
            "private-chat",
            "https://private.invalid/v1",
            OpenAIChatCompat::EMPTY,
        )
        .aliases(["my-gateway"]);
        let router = LMRouter::with_config(
            config()
                .providers([declared])
                .api_key(destination, "key")
                .base_url("azure", "https://account.services.ai.azure.com")
                .transport(transport.clone()),
        )
        .unwrap();
        let input_provider = if destination == "private-chat" {
            "my_gateway"
        } else {
            destination
        };
        let cached = router
            .cache(&req(&format!("{input_provider}:gpt-4.1-mini")), None, None)
            .await
            .unwrap();
        assert!(transport.requests().is_empty());
        assert_eq!(cached.provider.as_deref(), Some(destination));
        assert_eq!(cached.prefix.model, "gpt-4.1-mini");
        let restored = CachedPrefix::from_json(&cached.to_json()).unwrap();
        let suffix = restored
            .request_text("question", Config::default())
            .unwrap();
        assert_eq!(suffix.model, format!("{destination}:gpt-4.1-mini"));
        assert_eq!(router.resolve(&suffix.model).unwrap().provider, destination);
        router.complete(&suffix).await.unwrap();
        let direct = router.lm(&suffix.model).unwrap();
        direct.plan(&suffix).unwrap();
        direct.complete(&suffix).await.unwrap();
        assert_eq!(transport.requests().len(), 2);
        for wire in transport.requests() {
            assert_eq!(wire.body.as_ref().unwrap()["model"], "gpt-4.1-mini");
            assert!(wire.url.contains(if destination == "azure" {
                "account.services.ai.azure.com"
            } else {
                "private.invalid"
            }));
        }
    }
}

#[tokio::test]
async fn resource_model_remains_provider_fact_and_route_survives_roundtrip() {
    let transport = FakeTransport::new([FakeResponse::json(
        &json!({"name":"cachedContents/c", "model":"models/gemini-2.5-flash"}),
    )]);
    let router =
        LMRouter::with_config(config().api_key("gemini", "key").transport(transport)).unwrap();
    let cached = router
        .cache(&req("gemini:gemini-2.5-flash"), Some(60), None)
        .await
        .unwrap();
    assert_eq!(cached.resource.as_ref().unwrap().model, "gemini-2.5-flash");
    assert_eq!(cached.prefix.model, cached.resource.as_ref().unwrap().model);
    assert_eq!(cached.provider.as_deref(), Some("gemini"));
    let restored = CachedPrefix::from_json(&cached.to_json()).unwrap();
    assert_eq!(restored.resource, cached.resource);
    let suffix = restored
        .request_text("question", Config::default())
        .unwrap();
    assert_eq!(suffix.model, "gemini:gemini-2.5-flash");
    assert_eq!(
        suffix.config.cache.as_ref().unwrap().resource.as_deref(),
        Some("cachedContents/c")
    );
    assert_eq!(router.resolve(&suffix.model).unwrap().provider, "gemini");
}

#[test]
fn suffix_destination_validation_and_legacy_canonical_shape() {
    let legacy = CachedPrefix::new(req("m"), None).unwrap();
    assert_eq!(legacy.to_json(), json!({"prefix":req("m").to_json()}));
    assert_eq!(
        legacy.request_text("q", Config::default()).unwrap().model,
        "m"
    );
    let cached = CachedPrefix::new(
        req("m"),
        Some(CacheInfo {
            id: "c".into(),
            model: "m".into(),
            ..Default::default()
        }),
    )
    .unwrap()
    .with_provider("private_chat")
    .unwrap();
    assert_eq!(cached.provider.as_deref(), Some("private-chat"));
    for model in ["m", "private-chat:m", "private_chat:m"] {
        assert_eq!(
            cached.request_suffix(&req(model), None).unwrap().model,
            "private-chat:m"
        );
    }
    for model in ["azure:m", "openai:m", "private-chat:other"] {
        assert!(cached.request_suffix(&req(model), None).is_err());
    }
    for provider in ["", "a:b", "a/b", "a b", "\t"] {
        assert!(legacy.clone().with_provider(provider).is_err());
    }
    assert!(CachedPrefix::from_json(&json!({"prefix":req("m").to_json(), "provider":42})).is_err());
    assert_eq!(
        CachedPrefix::from_json(&json!({"prefix":req("m").to_json(), "provider":"private_chat"}))
            .unwrap()
            .provider
            .as_deref(),
        Some("private-chat")
    );
    assert!(CachedPrefix::new(
        req("m"),
        Some(CacheInfo {
            id: "c".into(),
            model: "azure:m".into(),
            ..Default::default()
        })
    )
    .is_err());
}

#[tokio::test]
async fn direct_cache_keeps_explicit_own_route_only_and_strips_once() {
    let lm = OpenAIChatLM::builder()
        .api_key("k")
        .base_url("https://custom.invalid/v1")
        .build()
        .unwrap();
    assert!(lm
        .cache(&req("m"), None, None)
        .await
        .unwrap()
        .provider
        .is_none());
    let cached = lm.cache(&req("openai_chat:m"), None, None).await.unwrap();
    assert_eq!(cached.provider.as_deref(), Some("openai-chat"));
    assert_eq!(cached.prefix.model, "m");
    assert_eq!(
        cached.request_text("q", Config::default()).unwrap().model,
        "openai-chat:m"
    );
    for (model, expected) in [
        ("openai_chat:m", "m"),
        ("openai-chat:openai-chat:m", "openai-chat:m"),
        ("other:m", "other:m"),
        ("arn:aws:bedrock:model", "arn:aws:bedrock:model"),
    ] {
        let wire = lm.build_request(&req(model), false).unwrap();
        assert_eq!(wire.body.as_ref().unwrap()["model"], expected);
        lm.plan(&req(model)).unwrap();
    }
}

#[cfg(feature = "blocking")]
#[test]
fn blocking_router_cache_keeps_nondefault_destination() {
    let router = lm15::blocking::LMRouter::with_config(
        config()
            .api_key("azure", "k")
            .base_url("azure", "https://account.services.ai.azure.com")
            .transport(FakeTransport::default()),
    )
    .unwrap();
    let cached = router
        .cache(&req("azure:gpt-4.1-mini"), None, None)
        .unwrap();
    assert_eq!(cached.provider.as_deref(), Some("azure"));
    let suffix = CachedPrefix::from_json(&cached.to_json())
        .unwrap()
        .request_text("q", Config::default())
        .unwrap();
    assert_eq!(router.resolve(&suffix.model).unwrap().provider, "azure");
    let wire = router
        .lm(&suffix.model)
        .unwrap()
        .build_request(&suffix, false)
        .unwrap();
    assert_eq!(wire.body.as_ref().unwrap()["model"], "gpt-4.1-mini");
}

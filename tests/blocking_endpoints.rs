//! Blocking wrappers must return endpoint values/handles, never leaked futures.
#![cfg(feature = "blocking")]

use lm15::blocking::{LMRouter, ProviderLM};
use lm15::transport::{BoxFuture, Transport, TransportResponse};
use lm15::types::*;
use lm15::wire::TransportRequest;
use lm15::{Lm15Error, OpenAILM, RouterConfig, WaitOptions};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Clone, Default)]
struct Replies {
    replies: Arc<Mutex<VecDeque<(String, Vec<u8>)>>>,
    requests: Arc<Mutex<Vec<TransportRequest>>>,
}
impl Replies {
    fn json(&self, value: Value) {
        self.replies.lock().unwrap().push_back((
            "application/json".into(),
            serde_json::to_vec(&value).unwrap(),
        ));
    }
    fn bytes(&self, content_type: &str, bytes: &[u8]) {
        self.replies
            .lock()
            .unwrap()
            .push_back((content_type.into(), bytes.into()));
    }
    fn count(&self) -> usize {
        self.requests.lock().unwrap().len()
    }
}
impl Transport for Replies {
    fn send(
        &self,
        request: TransportRequest,
    ) -> BoxFuture<'_, Result<TransportResponse, Lm15Error>> {
        self.requests.lock().unwrap().push(request);
        let (content_type, bytes) = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected I/O");
        Box::pin(async move {
            Ok(TransportResponse::buffered(
                200,
                vec![("content-type".into(), content_type)],
                bytes,
            ))
        })
    }
}
fn provider(replies: &Replies) -> ProviderLM {
    OpenAILM::builder()
        .api_key("test-key")
        .transport(replies.clone())
        .build_blocking()
        .unwrap()
}
fn file() -> Value {
    json!({"id":"f", "filename":"x.txt", "bytes":3, "status":"processed"})
}
fn video(status: &str) -> Value {
    json!({"id":"v", "status":status, "model":"sora-2", "progress":100})
}
fn batch(status: &str) -> Value {
    json!({"id":"b", "status":status})
}

#[test]
fn file_lifecycle_and_generation_are_blocking_values() {
    let replies = Replies::default();
    let lm = provider(&replies);
    replies.json(file());
    assert_eq!(
        lm.file_upload(&FileUploadRequest {
            filename: "x.txt".into(),
            bytes_data: Some(b"abc".to_vec()),
            ..Default::default()
        })
        .unwrap()
        .id,
        "f"
    );
    replies.json(file());
    assert_eq!(lm.file_get("f").unwrap().id, "f");
    replies.json(json!({"data":[file()]}));
    assert_eq!(lm.file_list(10, None).unwrap().items.len(), 1);
    replies.bytes("application/octet-stream", b"abc");
    assert_eq!(lm.file_download("f").unwrap(), b"abc");
    replies.json(file());
    assert!(lm
        .file_wait_ready("f", Duration::from_millis(1), Some(Duration::from_secs(1)))
        .unwrap()
        .ready());
    replies.json(json!({"id":"f", "deleted":true}));
    lm.file_delete("f").unwrap();

    replies.json(json!({"data":[{"b64_json":"YQ=="}], "output_format":"png"}));
    let image = lm
        .image_generate(&ImageGenerationRequest {
            model: "gpt-image-1".into(),
            prompt: "a fox".into(),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(image.images.len(), 1);
    replies.bytes("audio/mpeg", b"speech");
    let speech = lm
        .speech_generate(&SpeechGenerationRequest {
            model: "tts-1".into(),
            prompt: "hi".into(),
            voice: Some("alloy".into()),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(speech.audio.media_type, "audio/mpeg");
    assert_eq!(replies.count(), 8);
}

#[test]
fn blocking_handles_mutate_only_on_explicit_verbs() {
    let replies = Replies::default();
    let lm = provider(&replies);
    replies.json(video("queued"));
    let mut job = lm
        .video_generate(&VideoGenerationRequest {
            model: "sora-2".into(),
            prompt: "a fox".into(),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(job.id(), "v");
    assert!(!job.done());
    assert_eq!(replies.count(), 1);
    replies.json(video("completed"));
    job.wait(WaitOptions::default().poll_every(Duration::from_millis(1)))
        .unwrap();
    assert!(job.done());
    replies.json(video("completed"));
    replies.bytes("video/mp4", b"video");
    assert_eq!(job.result().unwrap().media_type, "video/mp4");
    replies.json(video("completed"));
    assert!(lm.video_job("v").unwrap().done());
    replies.json(json!({"data":[video("completed")]}));
    assert_eq!(lm.video_jobs(10, None).unwrap().len(), 1);

    replies.json(batch("in_progress"));
    let mut batch = lm.batch_job("b").unwrap();
    assert_eq!(batch.status(), BatchStatus::Running);
    replies.json(json!({"id":"b", "status":"cancelled"}));
    batch.cancel().unwrap();
    assert!(batch.done());
    replies.json(json!({"data":[{"id":"b", "status":"completed"}]}));
    assert!(lm.batches(10).unwrap()[0].done());
}

#[test]
fn router_routes_auxiliary_models_and_keeps_cache_automatic_tier_pure() {
    let replies = Replies::default();
    let router = LMRouter::with_config(
        RouterConfig::new()
            .env([("OPENAI_API_KEY", "test-key")])
            .transport(replies.clone()),
    )
    .unwrap();
    replies.json(file());
    assert_eq!(router.file_get("openai:gpt-4.1-mini", "f").unwrap().id, "f");
    replies.json(json!({"data":[{"b64_json":"YQ=="}]}));
    router
        .image_generate(&ImageGenerationRequest {
            model: "openai:gpt-image-1".into(),
            prompt: "fox".into(),
            ..Default::default()
        })
        .unwrap();
    let request = replies.requests.lock().unwrap()[1].clone();
    assert_eq!(request.body.unwrap()["model"], "gpt-image-1");
    let prefix = Request::new(
        "openai:gpt-4.1-mini",
        vec![Message::user("prefix").unwrap()],
    )
    .unwrap();
    let cached = router.cache(&prefix, None, None).unwrap();
    assert!(cached.resource.is_none());
    assert_eq!(cached.prefix.model, "gpt-4.1-mini");
    assert_eq!(replies.count(), 2, "automatic caching is pure");
}

#[test]
fn resource_cache_create_get_list_update_delete_are_blocking() {
    let replies = Replies::default();
    let router = LMRouter::with_config(
        RouterConfig::new()
            .env([("GEMINI_API_KEY", "test-key")])
            .transport(replies.clone()),
    )
    .unwrap();
    let info = json!({"name":"cachedContents/c", "model":"models/gemini-2.5-flash", "usageMetadata":{"totalTokenCount":3}});
    let prefix = Request::new(
        "gemini:gemini-2.5-flash",
        vec![Message::user("prefix").unwrap()],
    )
    .unwrap();
    replies.json(info.clone());
    let cached = router.cache(&prefix, Some(60), Some("prefix")).unwrap();
    assert_eq!(cached.resource.unwrap().id, "cachedContents/c");
    replies.json(info.clone());
    assert_eq!(
        router
            .cache_get("gemini:gemini-2.5-flash", "c")
            .unwrap()
            .tokens,
        Some(3)
    );
    replies.json(json!({"cachedContents":[info.clone()]}));
    assert_eq!(
        router
            .cache_list("gemini:gemini-2.5-flash", 10, None)
            .unwrap()
            .items
            .len(),
        1
    );
    replies.json(info);
    router
        .cache_update("gemini:gemini-2.5-flash", "c", 120)
        .unwrap();
    replies.json(json!({}));
    router.cache_delete("gemini:gemini-2.5-flash", "c").unwrap();
    assert_eq!(replies.count(), 5);
}

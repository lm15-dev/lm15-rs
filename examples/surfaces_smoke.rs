//! The endpoint surfaces live: files (upload, get, list, delete) on
//! OpenAI, Anthropic and Gemini; a Gemini cache object (create, get,
//! update, delete); one OpenAI speech generation. Env-gated; not a gate.
#![allow(clippy::result_large_err)]

use lm15::{FileUploadRequest, LMRouter, Request, SpeechGenerationRequest};

async fn files(router: &LMRouter, model: &str) -> Result<String, lm15::Lm15Error> {
    let lm = router.lm(model)?;
    let upload = FileUploadRequest {
        filename: "lm15-smoke.txt".into(),
        media_type: "text/plain".into(),
        bytes_data: Some(b"The quick brown fox jumps over the lazy dog.\n".to_vec()),
        ..Default::default()
    };
    let info = lm.file_upload(&upload).await?;
    let ready = lm
        .file_wait_ready(
            &info.id,
            std::time::Duration::from_secs(1),
            Some(std::time::Duration::from_secs(30)),
        )
        .await?;
    let page = lm.file_list(20, None).await?;
    let listed = page.items.iter().any(|f| f.id == info.id);
    lm.file_delete(&info.id).await?;
    Ok(format!(
        "uploaded {} ({} bytes, {}) listed={listed} deleted",
        info.id,
        info.size_bytes.unwrap_or(0),
        ready.readiness.as_str()
    ))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let router = LMRouter::new();
    for (env_key, model) in [
        ("OPENAI_API_KEY", "openai:gpt-4.1-mini"),
        ("ANTHROPIC_API_KEY", "anthropic:claude-haiku-4-5"),
        ("GEMINI_API_KEY", "gemini:gemini-2.5-flash"),
    ] {
        if std::env::var(env_key).is_err() {
            println!("files {model:<32} skipped");
            continue;
        }
        match files(&router, model).await {
            Ok(text) => println!("files {model:<32} OK   {text}"),
            Err(err) => println!("files {model:<32} FAIL {err}"),
        }
    }
    if std::env::var("GEMINI_API_KEY").is_ok() {
        let lm = router.lm("gemini:gemini-2.5-flash")?;
        let prefix = Request {
            model: "gemini-2.5-flash".into(),
            system: Some(lm15::SystemContent::Text(
                "You answer with one number.".into(),
            )),
            messages: vec![lm15::Message::user(
                "Reference notes. ".repeat(600).as_str(),
            )?],
            ..Default::default()
        };
        match lm
            .cache_create(&prefix, Some(300), Some("lm15-rs-smoke"))
            .await
        {
            Ok(info) => {
                let got = lm.cache_get(&info.id).await?;
                let updated = lm.cache_update(&info.id, 600).await?;
                lm.cache_delete(&info.id).await?;
                println!(
                    "cache gemini                             OK   {} tokens={:?} label={:?} expires {:?} -> {:?}",
                    info.id, got.tokens, got.label, got.expires_at, updated.expires_at
                );
            }
            Err(err) => println!("cache gemini                             FAIL {err}"),
        }
    }
    if std::env::var("OPENAI_API_KEY").is_ok() {
        let lm = router.lm("openai:gpt-4o-mini-tts")?;
        let request = SpeechGenerationRequest {
            model: "gpt-4o-mini-tts".into(),
            prompt: "Hi.".into(),
            voice: Some("alloy".into()),
            ..Default::default()
        };
        match lm.speech_generate(&request).await {
            Ok(response) => println!(
                "speech openai                            OK   {} ({} base64 chars)",
                response.audio.media_type,
                response.audio.data.as_ref().map(String::len).unwrap_or(0)
            ),
            Err(err) => println!("speech openai                            FAIL {err}"),
        }
    }
    Ok(())
}

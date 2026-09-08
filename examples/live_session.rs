//! One live text turn over a real websocket, per provider with a key in
//! the environment. Env-gated; not a gate.
#![allow(clippy::result_large_err)]

use lm15::{LMRouter, LiveConfig, LiveServerEvent};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let router = LMRouter::new();
    for (env_key, model) in [
        ("OPENAI_API_KEY", "openai:gpt-realtime-mini"),
        ("GEMINI_API_KEY", "gemini:gemini-3.1-flash-live-preview"),
    ] {
        if std::env::var(env_key).is_err() {
            println!("{model:<44} skipped ({env_key} unset)");
            continue;
        }
        let resolution = router.resolve(model)?;
        let lm = router.lm(model)?;
        let config = LiveConfig {
            model: resolution.model.clone(),
            system: Some(lm15::SystemContent::Text("Be terse.".into())),
            ..Default::default()
        };
        let started = std::time::Instant::now();
        let mut session = lm.live(&config).await?;
        session.send_text("Reply with exactly: live hello").await?;
        let mut text = String::new();
        let mut usage = None;
        let mut error = None;
        while let Some(event) = session.recv().await? {
            match event {
                LiveServerEvent::Text(t) => text.push_str(&t.text),
                LiveServerEvent::TurnEnd(end) => {
                    usage = Some(end.usage);
                    break;
                }
                LiveServerEvent::Error(e) => {
                    error = Some(e.error);
                    break;
                }
                _ => {}
            }
        }
        session.close().await?;
        println!(
            "{model:<44} {} {}ms text={:?} usage(out)={:?}{}",
            if error.is_none() { "OK  " } else { "FAIL" },
            started.elapsed().as_millis(),
            text.trim(),
            usage.as_ref().and_then(|u| u.output_tokens),
            error
                .map(|e| format!(" error={}: {}", e.code.as_str(), e.message))
                .unwrap_or_default(),
        );
    }
    Ok(())
}

//! Gemini image and speech generation (`gemini.py:1902-1960`): the same
//! `generateContent` call as chat, through the chat codec — a synthetic
//! `Request` in, the chat `Response` out, its image / audio parts lifted.

use serde_json::{json, Map, Value};

use crate::errors::Lm15Error;
use crate::surfaces::{provider_error, unsupported};
use crate::types::{
    Config, ImageGenerationRequest, ImageGenerationResponse, Message, Part, Request, Role,
    SpeechGenerationRequest, SpeechGenerationResponse, TextPart,
};
use crate::wire::{BuildContext, Dialect, WireRequest};

pub fn image_lm_request(request: &ImageGenerationRequest) -> Result<Request, Lm15Error> {
    let mut extensions = request.extensions.clone().unwrap_or_default();
    if let Some(size) = &request.size {
        let mut generation = extensions
            .get("generationConfig")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let mut image_config = generation
            .get("imageConfig")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        image_config
            .entry("aspectRatio")
            .or_insert_with(|| Value::String(size.clone()));
        generation.insert("imageConfig".into(), Value::Object(image_config));
        extensions.insert("generationConfig".into(), Value::Object(generation));
    }
    let mut parts = vec![Part::Text(TextPart {
        text: request.prompt.clone(),
        ..Default::default()
    })];
    parts.extend(request.images.iter().cloned().map(Part::Image));
    Ok(Request {
        model: request.model.clone(),
        messages: vec![
            Message::new(Role::User, parts).map_err(|e| provider_error("gemini", e.message))?
        ],
        config: Config {
            extensions: if extensions.is_empty() {
                None
            } else {
                Some(extensions)
            },
            ..Default::default()
        },
        ..Default::default()
    })
}

pub fn image_request(
    dialect: &dyn Dialect,
    cx: &BuildContext<'_>,
    request: &ImageGenerationRequest,
) -> Result<WireRequest, Lm15Error> {
    let lm_request = image_lm_request(request)?;
    dialect.build(&lm_request, false, &cx.for_model(&lm_request.model))
}

pub fn image_response(
    dialect: &dyn Dialect,
    cx: &BuildContext<'_>,
    request: &ImageGenerationRequest,
    body: &[u8],
) -> Result<ImageGenerationResponse, Lm15Error> {
    let lm_request = image_lm_request(request)?;
    let chat = dialect.parse_response(&lm_request, &cx.for_model(&lm_request.model), body)?;
    let images: Vec<_> = chat
        .message
        .parts
        .iter()
        .filter_map(|p| match p {
            Part::Image(image) => Some(image.clone()),
            _ => None,
        })
        .collect();
    if images.is_empty() {
        return Err(provider_error(
            cx.provider,
            "model returned no image parts".into(),
        ));
    }
    let text: String = chat
        .message
        .parts
        .iter()
        .filter_map(|p| match p {
            Part::Text(t) if !t.text.is_empty() => Some(t.text.as_str()),
            _ => None,
        })
        .collect();
    Ok(ImageGenerationResponse {
        images,
        text: if text.is_empty() { None } else { Some(text) },
        id: chat.id,
        model: Some(chat.model),
        usage: chat.usage,
        provider_data: chat.provider_data,
    })
}

pub fn speech_lm_request(
    cx: &BuildContext<'_>,
    request: &SpeechGenerationRequest,
) -> Result<Request, Lm15Error> {
    if request.format.is_some() {
        // No wire slot: Gemini TTS always answers PCM (captured
        // audio/L16;codec=pcm;rate=24000). Raising beats dropping.
        let mut err = unsupported(cx.provider, "speech format");
        err.meta_mut().message = format!(
            "{}: speech format cannot be chosen; the wire always returns PCM",
            cx.provider
        );
        return Err(err);
    }
    let mut generation = Map::new();
    generation.insert("responseModalities".into(), json!(["AUDIO"]));
    if let Some(voice) = &request.voice {
        generation.insert(
            "speechConfig".into(),
            json!({"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}),
        );
    }
    let mut extensions = Map::new();
    extensions.insert("generationConfig".into(), Value::Object(generation));
    if let Some(extra) = &request.extensions {
        extensions.extend(extra.clone());
    }
    Ok(Request {
        model: request.model.clone(),
        messages: vec![Message::user(request.prompt.as_str())
            .map_err(|e| provider_error("gemini", e.message))?],
        config: Config {
            extensions: Some(extensions),
            ..Default::default()
        },
        ..Default::default()
    })
}

pub fn speech_request(
    dialect: &dyn Dialect,
    cx: &BuildContext<'_>,
    request: &SpeechGenerationRequest,
) -> Result<WireRequest, Lm15Error> {
    let lm_request = speech_lm_request(cx, request)?;
    dialect.build(&lm_request, false, &cx.for_model(&lm_request.model))
}

pub fn speech_response(
    dialect: &dyn Dialect,
    cx: &BuildContext<'_>,
    request: &SpeechGenerationRequest,
    body: &[u8],
) -> Result<SpeechGenerationResponse, Lm15Error> {
    let lm_request = speech_lm_request(cx, request)?;
    let chat = dialect.parse_response(&lm_request, &cx.for_model(&lm_request.model), body)?;
    let audio = chat
        .message
        .parts
        .iter()
        .find_map(|p| match p {
            Part::Audio(audio) => Some(audio.clone()),
            _ => None,
        })
        .ok_or_else(|| provider_error(cx.provider, "model returned no audio part".into()))?;
    Ok(SpeechGenerationResponse {
        audio,
        id: chat.id,
        model: Some(chat.model),
        usage: chat.usage,
        provider_data: chat.provider_data,
    })
}

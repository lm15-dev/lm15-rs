//! TypeSafe System One: one user part is the state, verbatim (2026-09-19);
//! an image part is a reference in the state and its bytes an asset
//! (changes/2026-10-14-jev-images.md); a bounding box is Jev's Box question,
//! answered verbatim with no distribution (changes/2026-10-15-bounding-box.md).
use crate::errors::{ErrorMeta, Lm15Error};
use crate::judgments::{self, JudgmentKind};
use crate::registry::DialectId;
use crate::serde::Canonical;
use crate::sse::SseEvent;
use crate::types::*;
use crate::wire::{
    apply_static_headers, model_infos_from_entries, BuildContext, Dialect, Surfaces, WireRequest,
};
use serde_json::{json, Value};

#[derive(Debug, Clone, Copy, Default)]
pub struct TypeSafe;
pub static TYPESAFE: TypeSafe = TypeSafe;
impl Surfaces for TypeSafe {}

fn refuse(cx: &BuildContext<'_>, feature: &str, reason: impl Into<String>) -> Lm15Error {
    judgments::refusal(cx.provider, feature, reason)
}

/// Images and Box questions are off unless each request that carries one
/// sends this header (live 400 without it: 2026-10-14 for an image,
/// 2026-10-15 for a Box question with no image).
pub const IMAGES_HEADER: &str = "TypeSafe-Danger-Zone-Images";

/// The asset formats systemone reads: PNG and JPEG, nothing else.
fn image_format(part: &ImagePart) -> Option<&'static str> {
    match part.media_type.to_ascii_lowercase().as_str() {
        "image/png" => Some("png"),
        "image/jpeg" | "image/jpg" => Some("jpg"),
        _ => None,
    }
}

/// The n-th image's reference (1-based, in part order); `>` closes it so
/// `<lm15-image-1>` never matches inside `<lm15-image-10>`.
fn image_reference(n: usize) -> String {
    format!("<lm15-image-{n}>")
}

/// An image reaches systemone as inline PNG or JPEG bytes, or not at all
/// (MAP-10 rule 1); lm15 never fetches or converts media.
fn check_image(part: &ImagePart, cx: &BuildContext<'_>, at: &str) -> Result<(), Lm15Error> {
    if part.url.is_some() || part.file_id.is_some() {
        let how = if part.url.is_some() {
            "a URL"
        } else {
            "a file id"
        };
        return Err(refuse(cx, at, format!("systemone takes image bytes, not {how}; fetch the image and pass its data or path (changes/2026-10-14-jev-images.md)")));
    }
    if image_format(part).is_none() {
        return Err(refuse(cx, at, format!("systemone reads PNG or JPEG images, got {:?}; convert the image first (lm15 never converts media)", part.media_type)));
    }
    Ok(())
}

fn image_base64(part: &ImagePart, cx: &BuildContext<'_>) -> Result<String, Lm15Error> {
    if let Some(data) = &part.data {
        return Ok(base64_payload(data));
    }
    let path = part
        .path
        .as_ref()
        .expect("INV-011: a checked image has data or a path");
    let bytes = crate::adaptation::read_media(path).map_err(|err| {
        crate::dialects::openai_responses::invalid_request(
            cx.provider,
            format!("image part path {} cannot be read: {err}", path.display()),
        )
    })?;
    Ok(base64_encode(&bytes))
}

/// The state and its assets (2026-09-19 D1/D2; 2026-10-14 D1/D2): one
/// text or data part verbatim, one image its reference, a message holding
/// an image beside other parts the array of its parts in order.
pub fn state(request: &Request, cx: &BuildContext<'_>) -> Result<(Value, JsonObject), Lm15Error> {
    for (m, message) in request.messages.iter().enumerate() {
        for (p, part) in message.parts.iter().enumerate() {
            let at = format!("messages[{m}].parts[{p}]");
            match part {
                Part::Text(_) | Part::Data(_) => {}
                Part::Image(image) => check_image(image, cx, &at)?,
                other => {
                    return Err(refuse(
                        cx,
                        &at,
                        format!("a {} part has no slot on the systemone wire (MAP-10); Jev reads text, data, or PNG/JPEG images", other.type_name()),
                    ))
                }
            }
        }
    }
    if request.system.is_some() {
        return Err(refuse(cx, "system", "Jev has no system prompt; put context in named state keys, or framing in each question's description"));
    }
    if request.messages.len() != 1 {
        return Err(refuse(
            cx,
            "messages",
            "Jev judges one state; put a transcript in a data array or object",
        ));
    }
    let message = &request.messages[0];
    if message.role != Role::User {
        return Err(refuse(
            cx,
            "messages[0].role",
            "Jev's state must be a user message",
        ));
    }
    let has_image = message.parts.iter().any(|p| matches!(p, Part::Image(_)));
    if message.parts.len() != 1 && !has_image {
        return Err(refuse(
            cx,
            "messages[0].parts",
            "Jev's state is one text or data part; put several pieces in named data keys",
        ));
    }
    let mut assets = JsonObject::new();
    let mut element = |part: &Part| -> Result<Value, Lm15Error> {
        Ok(match part {
            Part::Text(text) => Value::String(text.text.clone()),
            Part::Data(data) => data.value.clone(),
            Part::Image(image) => {
                let reference = image_reference(assets.len() + 1);
                let format = image_format(image).expect("checked above");
                assets.insert(
                    reference.clone(),
                    json!({"type": "image", "format": format, "data_base64": image_base64(image, cx)?}),
                );
                Value::String(reference)
            }
            _ => unreachable!("validated above"),
        })
    };
    let state = if message.parts.len() == 1 {
        element(&message.parts[0])?
    } else {
        Value::Array(
            message
                .parts
                .iter()
                .map(&mut element)
                .collect::<Result<_, _>>()?,
        )
    };
    Ok((state, assets))
}

pub fn payload(request: &Request, cx: &BuildContext<'_>) -> Result<Value, Lm15Error> {
    if !request.tools.is_empty() {
        return Err(refuse(
            cx,
            "tools",
            "tools have no slot on systemone; the program depends on them",
        ));
    }
    if request.config.tool_choice.is_some() {
        return Err(refuse(
            cx,
            "config.tool_choice",
            "tool_choice has no slot on systemone",
        ));
    }
    let found = judgments::request_judgments(request);
    let schema = request
        .config
        .response_format
        .as_ref()
        .and_then(|f| f.get("schema"));
    if found.is_empty()
        || schema.is_some_and(|s| !judgments::non_judgment_properties(s, &found).is_empty())
    {
        return Err(refuse(cx, "config.response_format", "Jev answers declared judgments only; provide json_schema properties that are booleans, choices, or ordered levels"));
    }
    let config = request.config.to_json();
    for name in [
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "stop",
        "seed",
        "frequency_penalty",
        "presence_penalty",
        "reasoning",
        "logprobs",
        "store",
        "user_id",
        "service_tier",
        "cache",
    ] {
        if let Some(value) = config.get(name) {
            crate::adaptation::adapt(
                &format!("config.{name}"),
                AdaptationAction::Dropped,
                Some(value.clone()),
                None,
                "no such control on systemone; Jev returns decisions, not generated samples",
            )?;
        }
    }
    let mut questions = JsonObject::new();
    for j in &found {
        let instruction = match &j.instruction {
            Some(text) => text.clone(),
            None => {
                crate::adaptation::adapt(&format!("config.response_format.schema.properties.{}.description", j.name), AdaptationAction::Defaulted,
                    None, Some(j.name.clone().into()), "Jev does not see property names; use the property name as the missing instruction")?;
                j.name.clone()
            }
        };
        let question = match j.kind {
            JudgmentKind::Boolean => json!({"type":"noul", "instructions":instruction}),
            // Jev's Box takes no criteria (docs.typesafe.ai/primitives/box).
            JudgmentKind::Box => json!({"type":"box", "instructions":instruction}),
            JudgmentKind::Choice => {
                if j.keys.len() > judgments::MAX_CHOICE_KEYS {
                    return Err(refuse(
                        cx,
                        &format!("config.response_format.schema.properties.{}", j.name),
                        "a Jev choice accepts at most 255 keys",
                    ));
                }
                let criteria: JsonObject = j
                    .keys
                    .iter()
                    .map(|k| {
                        (
                            k.clone(),
                            j.descriptions
                                .get(k)
                                .map(|d| Value::from(d.clone()))
                                .unwrap_or(Value::Null),
                        )
                    })
                    .collect();
                json!({"type":"choice", "instructions":instruction, "criteria":criteria})
            }
            JudgmentKind::Ordered => {
                if j.keys.len() > judgments::MAX_ORDERED_LEVELS {
                    return Err(refuse(
                        cx,
                        &format!("config.response_format.schema.properties.{}", j.name),
                        "a Jev score accepts at most 10 levels",
                    ));
                }
                // Titles are metadata, never part of Jev's criterion wording (MAP-14 §2).
                let criteria: Vec<_> = j
                    .keys
                    .iter()
                    .map(|k| {
                        j.descriptions
                            .get(k)
                            .filter(|d| !d.is_empty())
                            .unwrap_or(k)
                            .clone()
                    })
                    .collect();
                json!({"type":"score", "instructions":instruction, "criteria":criteria})
            }
        };
        questions.insert(j.name.clone(), question);
    }
    let (state, assets) = state(request, cx)?;
    let mut body = json!({"model": cx.model, "state": state});
    if !assets.is_empty() {
        body["assets"] = Value::Object(assets);
    }
    body["questions"] = Value::Object(questions);
    if let Some(extensions) = &request.config.extensions {
        for (key, value) in extensions {
            if key == "n" && value.as_f64().is_some_and(|n| n > 1.0) {
                return Err(refuse(
                    cx,
                    "config.extensions.n",
                    "n > 1 has no canonical multiple-response representation",
                ));
            }
            body[key] = value.clone();
        }
    }
    Ok(body)
}

fn malformed(cx: &BuildContext<'_>, path: &str, why: &str) -> Lm15Error {
    let mut meta = ErrorMeta::new(format!("malformed systemone reply at {path}: {why}"));
    meta.provider = Some(cx.provider.to_string());
    Lm15Error::ProviderError(meta)
}
fn probability(value: Option<&Value>, cx: &BuildContext<'_>, path: &str) -> Result<f64, Lm15Error> {
    value
        .and_then(Value::as_f64)
        .filter(|p| p.is_finite() && (0.0..=1.0).contains(p))
        .ok_or_else(|| malformed(cx, path, "expected a finite number in [0, 1]"))
}

pub fn parse(
    request: &Request,
    cx: &BuildContext<'_>,
    data: &Value,
) -> Result<Response, Lm15Error> {
    if !data.is_object() {
        return Err(malformed(cx, "$", "expected an object"));
    }
    let found = judgments::request_judgments(request);
    if found.is_empty() {
        return Err(refuse(
            cx,
            "config.response_format",
            "a systemone answer requires declared judgments",
        ));
    }
    let answers = data
        .get("answers")
        .and_then(Value::as_object)
        .ok_or_else(|| malformed(cx, "answers", "expected every declared judgment"))?;
    if answers.len() != found.len() || found.iter().any(|j| !answers.contains_key(&j.name)) {
        return Err(malformed(
            cx,
            "answers",
            "keys must match the declared judgments exactly",
        ));
    }
    let mut value = JsonObject::new();
    let mut probabilities = JsonObject::new();
    for j in found {
        let path = format!("answers.{}", j.name);
        let answer = answers[&j.name]
            .as_object()
            .ok_or_else(|| malformed(cx, &path, "expected an answer object"))?;
        let expected = match j.kind {
            JudgmentKind::Boolean => "noul",
            JudgmentKind::Choice => "choice",
            JudgmentKind::Ordered => "score",
            JudgmentKind::Box => "box",
        };
        if answer.get("type").and_then(Value::as_str) != Some(expected) {
            return Err(malformed(
                cx,
                &format!("{path}.type"),
                "answer kind does not match the declaration",
            ));
        }
        if j.kind == JudgmentKind::Box {
            // The four edges verbatim, in Jev's order; a box has no
            // distribution, and its `confidence` stays in provider_data.
            let edges = answer
                .get("box")
                .and_then(Value::as_object)
                .filter(|b| {
                    b.len() == judgments::BOX_EDGES.len()
                        && judgments::BOX_EDGES.iter().all(|e| b.contains_key(*e))
                })
                .ok_or_else(|| {
                    malformed(
                        cx,
                        &format!("{path}.box"),
                        "expected exactly top, bottom, left and right",
                    )
                })?;
            let mut box_value = JsonObject::new();
            for edge in judgments::BOX_EDGES {
                probability(edges.get(edge), cx, &format!("{path}.box.{edge}"))?; // a fraction: finite, in [0, 1]
                                                                                  // Numbers as written (an opaque value, serde number rule 4).
                box_value.insert(edge.into(), edges[edge].clone());
            }
            value.insert(j.name.clone(), Value::Object(box_value));
            continue;
        }
        if j.kind == JudgmentKind::Boolean {
            let p = probability(answer.get("noul"), cx, &format!("{path}.noul"))?;
            value.insert(j.name.clone(), Value::Bool(p >= 0.5));
            probabilities.insert(j.name, json!({"true":p, "false":1.0-p}));
            continue;
        }
        let dist = answer
            .get("probabilities")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                malformed(
                    cx,
                    &format!("{path}.probabilities"),
                    "expected a distribution",
                )
            })?;
        if dist.len() != j.keys.len() || j.keys.iter().any(|k| !dist.contains_key(k)) {
            return Err(malformed(
                cx,
                &format!("{path}.probabilities"),
                "expected every declared key and no extra keys",
            ));
        }
        let mut probs = JsonObject::new();
        let mut best = &j.keys[0];
        let mut best_p = -1.0;
        for key in &j.keys {
            let p = probability(dist.get(key), cx, &format!("{path}.probabilities.{key}"))?;
            if p > best_p {
                best = key;
                best_p = p;
            }
            probs.insert(key.clone(), Value::from(p));
        }
        let pick = if j.kind == JudgmentKind::Choice {
            let pick = answer
                .get("choice")
                .and_then(Value::as_str)
                .filter(|k| j.keys.iter().any(|v| v == k))
                .ok_or_else(|| {
                    malformed(
                        cx,
                        &format!("{path}.choice"),
                        "expected a declared choice key",
                    )
                })?;
            Value::from(pick)
        } else {
            j.value_for_key(best)
        };
        value.insert(j.name.clone(), pick);
        probabilities.insert(j.name, Value::Object(probs));
    }
    let usage = match data.get("usage") {
        None | Some(Value::Null) => Usage::default(),
        Some(Value::Object(raw)) => {
            let mut canonical = JsonObject::new();
            for key in ["input_tokens", "output_tokens"] {
                if let Some(v) = raw.get(key) {
                    canonical.insert(key.into(), v.clone());
                }
            }
            Usage::from_json(&Value::Object(canonical))
                .map_err(|e| malformed(cx, "usage", &e.to_string()))?
        }
        Some(_) => return Err(malformed(cx, "usage", "expected an object or null")),
    };
    let model = match data.get("model") {
        None | Some(Value::Null) => cx.model.to_string(),
        Some(Value::String(s)) if !s.is_empty() => s.clone(),
        _ => return Err(malformed(cx, "model", "expected a non-empty string")),
    };
    let measured = !probabilities.is_empty();
    let data = DataPart {
        value: Value::Object(value),
        probabilities: measured.then_some(probabilities),
        method: measured.then_some(JudgmentMethod::ProviderClassification),
        continuation: Vec::new(),
    };
    let response = Response {
        id: None,
        model,
        message: Message::assistant(Part::Data(data))
            .map_err(|e| malformed(cx, "answers", &e.to_string()))?,
        finish_reason: FinishReason::Stop,
        usage,
        logprobs: None,
        logprobs_complete: true,
        adaptations: Vec::new(),
        provider_data: Some(
            json!({"typesafe":{"answers":answers}})
                .as_object()
                .unwrap()
                .clone(),
        ),
    };
    Ok(response)
}

impl Dialect for TypeSafe {
    fn dialect(&self) -> DialectId {
        DialectId::Typesafe
    }
    fn build(
        &self,
        request: &Request,
        stream: bool,
        cx: &BuildContext<'_>,
    ) -> Result<WireRequest, Lm15Error> {
        if stream {
            return Err(refuse(
                cx,
                "stream",
                "systemone answers in one piece; there is no stream",
            ));
        }
        let body = payload(request, cx)?;
        // The caller put an image or a box in the request: that is the opt-in
        // the header exists for (changes/2026-10-14-jev-images.md D3;
        // changes/2026-10-15-bounding-box.md D2).
        let asks_box = body["questions"].as_object().is_some_and(|q| {
            q.values()
                .any(|q| q.get("type").and_then(Value::as_str) == Some("box"))
        });
        let images = body.get("assets").is_some() || asks_box;
        let mut wire = WireRequest::post("/v1/systemone", body);
        if images {
            wire.headers.push((IMAGES_HEADER.into(), "true".into()));
        }
        wire.endpoint = Some("systemone");
        wire.model = Some(cx.model.into());
        apply_static_headers(&mut wire.headers, cx.policy);
        Ok(wire)
    }
    fn parse_response(
        &self,
        request: &Request,
        cx: &BuildContext<'_>,
        body: &[u8],
    ) -> Result<Response, Lm15Error> {
        let value = serde_json::from_slice(body).map_err(|_| malformed(cx, "$", "invalid JSON"))?;
        parse(request, cx, &value)
    }
    fn parse_stream_event(
        &self,
        _: &Request,
        cx: &BuildContext<'_>,
        _: &SseEvent,
        _: &mut Vec<StreamEvent>,
    ) -> Result<(), Lm15Error> {
        Err(refuse(cx, "stream", "systemone has no stream"))
    }
    fn models_request(&self, cx: &BuildContext<'_>) -> Result<WireRequest, Lm15Error> {
        // The reference's `_headers()` sends a JSON content type on the
        // listing too (live capture 2026-09-17, cases/typesafe/models.json).
        let mut wire = WireRequest::get("/v1/models");
        wire.headers
            .push(("Content-Type".into(), "application/json".into()));
        apply_static_headers(&mut wire.headers, cx.policy);
        Ok(wire)
    }
    fn parse_models(
        &self,
        cx: &BuildContext<'_>,
        body: &[u8],
    ) -> Result<Vec<ModelInfo>, Lm15Error> {
        let value: Value =
            serde_json::from_slice(body).map_err(|_| malformed(cx, "$", "invalid JSON catalog"))?;
        Ok(model_infos_from_entries(
            value.get("models"),
            cx.provider,
            "typesafe_systemone",
            |entry| {
                entry
                    .get("name")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            },
        ))
    }
}

/// TypeSafe's native detail envelope, including Pydantic 422 lists.
pub fn normalize_error(status: u16, body: &str, provider: &str) -> Lm15Error {
    let data: Value = serde_json::from_str(body).unwrap_or(Value::Null);
    let detail = data.get("detail");
    let mut message = body.trim().chars().take(500).collect::<String>();
    if message.is_empty() {
        message = format!("HTTP {status}");
    }
    let mut code = None;
    if let Some(Value::Object(detail)) = detail {
        code = detail
            .get("error_type")
            .and_then(Value::as_str)
            .map(str::to_string);
        if let Some(text) = detail.get("message").and_then(Value::as_str) {
            message = text.into();
        }
    } else if let Some(Value::Array(items)) = detail {
        if let Some(first) = items.first() {
            let loc = first
                .get("loc")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter(|v| v.as_str() != Some("body"))
                        .map(|v| {
                            v.as_str()
                                .map(str::to_string)
                                .unwrap_or_else(|| v.to_string())
                        })
                        .collect::<Vec<_>>()
                        .join(".")
                })
                .unwrap_or_default();
            let msg = first
                .get("msg")
                .and_then(Value::as_str)
                .unwrap_or("validation error");
            message = if loc.is_empty() {
                msg.into()
            } else {
                format!("{loc}: {msg}")
            };
        }
    }
    let mut meta = ErrorMeta::new(message);
    meta.provider = Some(provider.into());
    meta.provider_code = code;
    meta.status = Some(status);
    if status == 401 || meta.provider_code.as_deref() == Some("authentication_error") {
        Lm15Error::AuthError(meta)
    } else if status == 429 {
        Lm15Error::RateLimitError(meta)
    } else if status == 400 && meta.message.to_lowercase().contains("unknown model") {
        Lm15Error::UnsupportedModelError(meta)
    } else if matches!(status, 400 | 422) {
        Lm15Error::InvalidRequestError(meta)
    } else if status >= 500 {
        Lm15Error::ServerError(meta)
    } else {
        Lm15Error::ProviderError(meta)
    }
}

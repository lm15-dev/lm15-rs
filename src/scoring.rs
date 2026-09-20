//! Candidate-sequence likelihood using the server's chat template and token trie.
//! MAP-14 §4: tokenize candidates with a terminator, batch all trie nodes in ONE
//! completions request, add raw path log-probabilities, normalize only once.
//! The pure hooks work without a transport; `complete` is an async driver whose
//! callback owns authentication, HTTP errors, and reply diagnostics.
use crate::errors::{ErrorMeta, Lm15Error};
use crate::judgments::{self, Judgment};
use crate::serde::Canonical;
use crate::types::*;
use crate::wire::{apply_static_headers, BuildContext, Dialect, WireRequest};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;

pub type TokenPath = Vec<u64>;
pub type CandidatePaths = BTreeMap<String, TokenPath>;
pub type TrieNodes = BTreeMap<TokenPath, BTreeSet<u64>>;
pub type ScoreTable = BTreeMap<TokenPath, BTreeMap<u64, f64>>;

#[derive(Debug, Clone)]
pub struct CandidateTokenization {
    pub key: String,
    pub open: WireRequest,
    pub closed: WireRequest,
}
#[derive(Debug, Clone)]
pub struct JudgmentPlan {
    pub judgment: Judgment,
    pub prefill: WireRequest,
    pub candidates: Vec<CandidateTokenization>,
}
#[derive(Debug, Clone)]
pub struct ScoringPlan {
    pub judgments: Vec<JudgmentPlan>,
    /// Mixed schemas also need one ordinary structured-output request. It retains
    /// the original schema verbatim; only judgment values are replaced by picks
    /// from measured distributions when folding the two answers.
    pub ordinary: Option<WireRequest>,
}

/// A parsed auxiliary reply with optional HTTP evidence. Pure codec callers may
/// pass `Value`; native drivers should use `from_http` so shape faults retain it.
#[derive(Debug, Clone)]
pub struct ScoringReply {
    pub value: Value,
    pub status: Option<u16>,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}
impl From<Value> for ScoringReply {
    fn from(value: Value) -> Self {
        Self {
            value,
            status: None,
            headers: Vec::new(),
            body: Vec::new(),
        }
    }
}
impl ScoringReply {
    pub fn from_http(
        status: u16,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
        provider: &str,
    ) -> Result<Self, Lm15Error> {
        let value = serde_json::from_slice(&body).map_err(|error| {
            let mut error = provider_fault(provider, format!("invalid scoring JSON: {error}"));
            crate::transport::attach_http_error(&mut error, status, &headers, &body);
            error
        })?;
        Ok(Self {
            value,
            status: Some(status),
            headers,
            body,
        })
    }
    pub(crate) fn parse<T>(
        &self,
        parse: impl FnOnce(&Value) -> Result<T, Lm15Error>,
    ) -> Result<T, Lm15Error> {
        parse(&self.value).map_err(|mut error| {
            if let Some(status) = self.status {
                crate::transport::attach_http_error(&mut error, status, &self.headers, &self.body);
            }
            error
        })
    }
}

#[derive(Debug, Clone)]
pub enum ScoringOutcome {
    Measured(Response),
    /// Caller may execute one ordinary structured-output call, WITHOUT scoring.
    Unavailable(Adaptation, Usage),
}

fn provider_fault(provider: &str, message: impl Into<String>) -> Lm15Error {
    let mut meta = ErrorMeta::new(message);
    meta.provider = Some(provider.to_string());
    Lm15Error::ProviderError(meta)
}

pub fn question(j: &Judgment) -> String {
    let options = j
        .keys
        .iter()
        .map(|k| {
            let tail = match (
                j.titles.get(k).filter(|s| !s.is_empty()),
                j.descriptions.get(k).filter(|s| !s.is_empty()),
            ) {
                (Some(title), Some(desc)) => format!(": {title} - {desc}"),
                (Some(text), None) | (None, Some(text)) => format!(": {text}"),
                _ => String::new(),
            };
            format!("- {k}{tail}")
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "{}\nOptions:\n{}\nAnswer with the option only, spelled exactly as listed.",
        j.instruction.as_deref().unwrap_or(&j.name),
        options
    )
}

pub fn tokenize_request(
    cx: &BuildContext<'_>,
    messages: Vec<Value>,
    continue_final: bool,
) -> WireRequest {
    let mut wire = WireRequest::post(
        "/tokenize",
        json!({"model":cx.model, "messages":messages,
        "add_generation_prompt":false, "continue_final_message":continue_final}),
    );
    let base = cx.base_url.trim_end_matches('/');
    let root = base.strip_suffix("/v1").unwrap_or(base);
    wire.absolute_url = Some(format!("{root}/tokenize"));
    wire.endpoint = Some("tokenize");
    wire.model = Some(cx.model.into());
    apply_static_headers(&mut wire.headers, cx.policy);
    wire
}

/// A network-free plan. Refusals happen here, before any tokenization bill.
pub fn plan(request: &Request, cx: &BuildContext<'_>) -> Result<ScoringPlan, Lm15Error> {
    request
        .validate()
        .map_err(|e| Lm15Error::InvalidRequestError(ErrorMeta::new(e.to_string())))?;
    let found = judgments::request_judgments(request);
    if found.is_empty() {
        return Err(judgments::refusal(
            cx.provider,
            "config.response_format",
            "candidate scoring requires declared judgments",
        ));
    }
    let schema = request
        .config
        .response_format
        .as_ref()
        .and_then(|f| f.get("schema"))
        .expect("judgment schema");
    let mixed = !judgments::non_judgment_properties(schema, &found).is_empty();
    if !request.tools.is_empty() {
        return Err(judgments::refusal(
            cx.provider,
            "tools",
            "candidate scoring cannot execute tools; the program depends on their results",
        ));
    }
    if request.config.tool_choice.is_some() {
        return Err(judgments::refusal(
            cx.provider,
            "config.tool_choice",
            "candidate scoring cannot preserve tool/action semantics",
        ));
    }
    for (field, present) in [
        ("store", request.config.store.is_some()),
        ("user_id", request.config.user_id.is_some()),
        ("service_tier", request.config.service_tier.is_some()),
    ] {
        if present {
            return Err(judgments::refusal(cx.provider, &format!("config.{field}"), "measurement endpoints have no established mapping for this privacy, safety or billing control; use a separate scoring request"));
        }
    }
    if let Some(cache) = &request.config.cache {
        if cache.mode == CacheMode::Off {
            return Err(judgments::refusal(
                cx.provider,
                "config.cache.mode",
                "measurement endpoints cannot guarantee cache writes are disabled",
            ));
        }
        if cache.retention.is_some() {
            return Err(judgments::refusal(
                cx.provider,
                "config.cache.retention",
                "measurement endpoints cannot preserve cache lifetime and billing intent",
            ));
        }
    }
    if request
        .config
        .cache
        .as_ref()
        .is_some_and(|cache| cache.resource.is_some())
    {
        return Err(judgments::refusal(cx.provider, "config.cache.resource", "candidate scoring cannot read the stored cache object; omitting it would lose prompt content"));
    }
    if request
        .config
        .extensions
        .as_ref()
        .and_then(|e| e.get("n"))
        .and_then(Value::as_f64)
        .is_some_and(|n| n > 1.0)
    {
        return Err(judgments::refusal(
            cx.provider,
            "config.extensions.n",
            "n > 1 has no canonical multiple-response representation",
        ));
    }
    if let Some(extensions) = &request.config.extensions {
        for (name, value) in extensions {
            let number = value.as_f64().filter(|n| n.is_finite());
            if (name == "n" && number == Some(1.0))
                || (matches!(
                    name.as_str(),
                    "temperature"
                        | "top_p"
                        | "top_k"
                        | "seed"
                        | "frequency_penalty"
                        | "presence_penalty"
                ) && number.is_some())
            {
                continue;
            }
            return Err(judgments::refusal(cx.provider, &format!("config.extensions.{name}"), "unknown measurement extension semantics; dropping it could lose privacy, money or action controls"));
        }
    }
    // MAP-14 §4: independent measurement (versus joint JSON generation) is
    // documented, not itself an adaptation. Only an extra generation call for
    // ordinary properties introduces additional work that must be visible.
    if mixed {
        crate::adaptation::adapt("config.response_format", AdaptationAction::ClientSide, None, None,
            "an additional structured-output call answers ordinary properties, which are never scored")?;
    }
    // Scoring uses a fixed measurement temperature and no sampling knobs. Record
    // deviations rather than silently billing a different experiment.
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
        "cache",
        "extensions",
    ] {
        if let Some(value) = config.get(name) {
            crate::adaptation::adapt(&format!("config.{name}"), AdaptationAction::Dropped, Some(value.clone()), None,
                "candidate likelihood measures unmodified next-token probabilities with max_tokens=1 per trie node; this setting has no measurement slot (any generated JSON call still uses its usual mapping)")?;
        }
    }
    let ordinary = if mixed {
        let mut generated = request.clone();
        generated.config.probabilities = Some(ProbabilityPolicy::Off);
        Some(crate::dialects::openai_chat::OPENAI_CHAT.build(&generated, false, cx)?)
    } else {
        None
    };
    let mut base = request.clone();
    base.config = Config::default();
    let built = crate::dialects::openai_chat::OPENAI_CHAT.build(&base, false, cx)?;
    let messages = built
        .body
        .as_ref()
        .and_then(|b| b.get("messages"))
        .and_then(Value::as_array)
        .ok_or_else(|| {
            provider_fault(cx.provider, "chat codec did not produce scoring messages")
        })?;
    let mut plans = Vec::new();
    for j in found {
        let mut history = messages.clone();
        history.push(json!({"role":"user", "content":question(&j)}));
        let with_answer = |answer: String| {
            let mut history = history.clone();
            history.push(json!({"role":"assistant", "content":answer}));
            history
        };
        let prefill = tokenize_request(cx, with_answer("Answer:".into()), true);
        let candidates = j
            .keys
            .iter()
            .map(|key| CandidateTokenization {
                key: key.clone(),
                open: tokenize_request(cx, with_answer(format!("Answer: {key}")), true),
                closed: tokenize_request(cx, with_answer(format!("Answer: {key}")), false),
            })
            .collect();
        plans.push(JudgmentPlan {
            judgment: j,
            prefill,
            candidates,
        });
    }
    Ok(ScoringPlan {
        judgments: plans,
        ordinary,
    })
}

pub fn tokens_from_value(value: &Value, provider: &str) -> Result<Vec<u64>, Lm15Error> {
    let tokens = value
        .get("tokens")
        .and_then(Value::as_array)
        .filter(|t| !t.is_empty())
        .ok_or_else(|| {
            provider_fault(provider, "tokenize reply carries no non-empty token list")
        })?;
    tokens
        .iter()
        .map(|t| {
            t.as_u64().ok_or_else(|| {
                provider_fault(
                    provider,
                    "tokenize reply token IDs must be nonnegative integers",
                )
            })
        })
        .collect()
}

/// Use only the first end-of-turn token; the resulting paths must be prefix-free.
pub fn candidate_paths(
    prefix: &[u64],
    candidates: &[(String, Vec<u64>, Vec<u64>)],
    provider: &str,
) -> Result<CandidatePaths, Lm15Error> {
    let mut paths = CandidatePaths::new();
    for (key, open, closed) in candidates {
        if !open.starts_with(prefix)
            || !closed.starts_with(open)
            || open.len() == prefix.len()
            || closed.len() == open.len()
        {
            return Err(judgments::refusal(provider, "config.response_format", format!("key {key:?} is not a non-empty extension of the prefill with an end-of-turn token in this template")));
        }
        let mut path = open[prefix.len()..].to_vec();
        path.push(closed[open.len()]);
        if paths.insert(key.clone(), path).is_some() {
            return Err(judgments::refusal(
                provider,
                "config.response_format",
                "duplicate candidate key",
            ));
        }
    }
    let sequences: Vec<_> = paths.values().collect();
    for (i, left) in sequences.iter().enumerate() {
        for right in sequences.iter().skip(i + 1) {
            if left.starts_with(right) || right.starts_with(left) {
                return Err(judgments::refusal(provider, "config.response_format", "candidate token paths are not prefix-free, so their probability masses overlap"));
            }
        }
    }
    Ok(paths)
}

pub fn trie_nodes(paths: &CandidatePaths) -> TrieNodes {
    let mut nodes = TrieNodes::new();
    for seq in paths.values() {
        for i in 0..seq.len() {
            nodes.entry(seq[..i].to_vec()).or_default().insert(seq[i]);
        }
    }
    nodes
}

pub fn score_request(
    cx: &BuildContext<'_>,
    prompts: Vec<Vec<u64>>,
    ids: &BTreeSet<u64>,
) -> WireRequest {
    let mut wire = WireRequest::post(
        "/completions",
        json!({"model":cx.model, "prompt":prompts,
        "max_tokens":1, "temperature":1.0, "logprobs":0,
        "return_tokens_as_token_ids":true, "logprob_token_ids":ids.iter().copied().collect::<Vec<_>>()}),
    );
    wire.endpoint = Some("completions");
    wire.model = Some(cx.model.into());
    apply_static_headers(&mut wire.headers, cx.policy);
    wire
}

#[derive(Debug, Clone)]
pub struct ScoredNodes {
    pub scores: Vec<BTreeMap<u64, f64>>,
    pub usage: Usage,
    pub model: Option<String>,
}

pub fn scores_from_value(
    value: &Value,
    count: usize,
    provider: &str,
) -> Result<ScoredNodes, Lm15Error> {
    let choices = value
        .get("choices")
        .and_then(Value::as_array)
        .filter(|v| v.len() == count)
        .ok_or_else(|| {
            provider_fault(
                provider,
                format!("completions reply must contain {count} choices"),
            )
        })?;
    let mut indexed = vec![None; count];
    for choice in choices {
        let index = choice
            .get("index")
            .and_then(Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .filter(|n| *n < count)
            .ok_or_else(|| provider_fault(provider, "completions choice has an invalid index"))?;
        if indexed[index].is_some() {
            return Err(provider_fault(
                provider,
                "completions reply has duplicate choice indices",
            ));
        }
        let mut scores = BTreeMap::new();
        // Missing requested IDs is an unsupported capability, not zero likelihood.
        // Malformed fields are provider faults, not permission to spend on fallback.
        let top = match choice.get("logprobs") {
            None | Some(Value::Null) => None,
            Some(Value::Object(logprobs)) => match logprobs.get("top_logprobs") {
                None | Some(Value::Null) => None,
                Some(Value::Array(items))
                    if items.is_empty() || (items.len() == 1 && items[0].is_null()) =>
                {
                    None
                }
                Some(Value::Array(items)) if items.len() == 1 && items[0].is_object() => {
                    items[0].as_object()
                }
                _ => {
                    return Err(provider_fault(
                        provider,
                        "completions must report one top_logprobs object",
                    ))
                }
            },
            _ => {
                return Err(provider_fault(
                    provider,
                    "completions logprobs must be an object or null",
                ))
            }
        };
        if let Some(top) = top {
            for (token, score) in top {
                if let Some(id) = token
                    .strip_prefix("token_id:")
                    .filter(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
                    .and_then(|s| s.parse::<u64>().ok())
                {
                    let score = score.as_f64().filter(|v| !v.is_nan() && *v <= 0.0)
                        .ok_or_else(|| provider_fault(provider, "completions token logprob must be numeric and nonpositive (never NaN or positive infinity)"))?;
                    scores.insert(id, score);
                }
            }
        }
        indexed[index] = Some(scores);
    }
    let usage = match value.get("usage") {
        None | Some(Value::Null) => Usage::default(),
        Some(Value::Object(raw)) => {
            for name in ["prompt_tokens_details", "completion_tokens_details"] {
                if raw
                    .get(name)
                    .is_some_and(|v| !v.is_null() && !v.is_object())
                {
                    return Err(provider_fault(
                        provider,
                        format!("completions usage.{name} must be an object or null"),
                    ));
                }
            }
            crate::dialects::openai_chat::response::usage_from_chat(provider, raw)?
                .normalized()
                .map_err(|e| provider_fault(provider, e.to_string()))?
        }
        _ => {
            return Err(provider_fault(
                provider,
                "completions usage must be an object",
            ))
        }
    };
    let model = match value.get("model") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        _ => {
            return Err(provider_fault(
                provider,
                "completions model must be a non-empty string",
            ))
        }
    };
    Ok(ScoredNodes {
        scores: indexed
            .into_iter()
            .map(|v| v.expect("unique indices cover range"))
            .collect(),
        usage,
        model,
    })
}

pub fn fold(
    request: &Request,
    provider: &str,
    measured: &[(Judgment, CandidatePaths, ScoreTable)],
    usage: Usage,
    model: Option<String>,
    nodes: usize,
    tokenize_calls: usize,
) -> Result<Response, Lm15Error> {
    let mut value = JsonObject::new();
    let mut probabilities = JsonObject::new();
    let mut coverage = JsonObject::new();
    for (j, paths, table) in measured {
        let mut raw = Vec::new();
        for key in &j.keys {
            let path = paths
                .get(key)
                .ok_or_else(|| provider_fault(provider, "missing scored candidate path"))?;
            let mut score = 0.0;
            for i in 0..path.len() {
                score += table
                    .get(&path[..i])
                    .and_then(|t| t.get(&path[i]))
                    .copied()
                    .ok_or_else(|| provider_fault(provider, "missing requested token score"))?;
            }
            raw.push((key.clone(), score));
        }
        let distribution = judgments::normalize_logprobs(&raw)
            .map_err(|e| provider_fault(provider, e.to_string()))?;
        let mass: f64 = raw.iter().map(|(_, score)| score.exp()).sum();
        if !mass.is_finite() {
            return Err(provider_fault(provider, "candidate coverage is not finite"));
        }
        coverage.insert(j.name.clone(), mass.into());
        let mut best = &j.keys[0];
        for key in &j.keys[1..] {
            if distribution[key].as_f64().unwrap() > distribution[best].as_f64().unwrap() {
                best = key;
            }
        }
        value.insert(j.name.clone(), j.value_for_key(best));
        probabilities.insert(j.name.clone(), distribution.into());
    }
    let data = DataPart {
        value: value.into(),
        probabilities: Some(probabilities),
        method: Some(JudgmentMethod::CandidateSequenceLikelihood),
        continuation: Vec::new(),
    };
    let response = Response { id: None, model: model.unwrap_or_else(|| request.model.clone()),
        message: Message::assistant(Part::Data(data)).map_err(|e| provider_fault(provider, e.to_string()))?,
        finish_reason: FinishReason::Stop, usage, logprobs: None, logprobs_complete: true, adaptations: Vec::new(),
        provider_data: Some(json!({"coverage":coverage, "judgments":{"nodes":nodes, "tokenize_calls":tokenize_calls,
            "method":"candidate_sequence_likelihood"}}).as_object().unwrap().clone()) };
    Ok(response)
}

/// Decode generated fields without accepting missing finishes or recovering an
/// incomplete JSON fragment. Measured picks can fill declared judgment fields
/// only on the measured path; fallback needs every required property itself.
pub fn parse_generated(
    request: &Request,
    cx: &BuildContext<'_>,
    raw: &Value,
    measured: bool,
) -> Result<Response, Lm15Error> {
    let mut plain = request.clone();
    plain.config.response_format = None;
    plain.config.probabilities = Some(ProbabilityPolicy::Off);
    let bytes = serde_json::to_vec(raw).map_err(|e| provider_fault(cx.provider, e.to_string()))?;
    if raw.get("error").is_some_and(|v| !v.is_null()) {
        crate::dialects::openai_chat::OPENAI_CHAT.parse_response(&plain, cx, &bytes)?;
    }
    let choices = raw
        .get("choices")
        .and_then(Value::as_array)
        .filter(|v| {
            v.len() == 1 && v[0].get("finish_reason").and_then(Value::as_str) == Some("stop")
        })
        .ok_or_else(|| {
            provider_fault(
                cx.provider,
                "generated judgment reply needs one complete choice with finish_reason='stop'",
            )
        })?;
    let _ = choices;
    let mut response =
        crate::dialects::openai_chat::OPENAI_CHAT.parse_response(&plain, cx, &bytes)?;
    if response.finish_reason != FinishReason::Stop {
        return Err(provider_fault(
            cx.provider,
            "generated judgment answer did not finish completely",
        ));
    }
    let texts: Vec<_> = response
        .message
        .parts
        .iter()
        .enumerate()
        .filter_map(|(i, part)| {
            if let Part::Text(text) = part {
                Some((i, text))
            } else {
                None
            }
        })
        .collect();
    if texts.len() != 1 {
        return Err(provider_fault(
            cx.provider,
            "generated judgment answer needs one JSON text payload",
        ));
    }
    let (index, text) = texts[0];
    let value: Value = serde_json::from_str(&text.text).map_err(|e| {
        provider_fault(
            cx.provider,
            format!("malformed generated judgment JSON: {e}"),
        )
    })?;
    let object = value.as_object().ok_or_else(|| {
        provider_fault(cx.provider, "generated judgment answer needs a JSON object")
    })?;
    let found = judgments::request_judgments(request);
    let schema = request
        .config
        .response_format
        .as_ref()
        .and_then(|f| f.get("schema"));
    if let Some(required) = schema
        .and_then(|s| s.get("required"))
        .and_then(Value::as_array)
    {
        for name in required.iter().filter_map(Value::as_str) {
            if (!measured || !found.iter().any(|j| j.name == name)) && !object.contains_key(name) {
                return Err(provider_fault(
                    cx.provider,
                    format!("generated answer is missing required property {name:?}"),
                ));
            }
        }
    }
    let continuation = text.continuation.clone();
    drop(texts);
    response.message.parts[index] = Part::Data(DataPart {
        value,
        probabilities: None,
        method: None,
        continuation,
    });
    Ok(response)
}

/// Add bills from actual exchanges; a counter unknown on either side remains unknown.
pub fn combined_usage(provider: &str, a: Usage, b: Usage) -> Result<Usage, Lm15Error> {
    let sum = |x: Option<u64>, y: Option<u64>| -> Result<Option<u64>, Lm15Error> {
        match (x, y) {
            (Some(x), Some(y)) => x
                .checked_add(y)
                .map(Some)
                .ok_or_else(|| provider_fault(provider, "combined judgment usage overflow")),
            _ => Ok(None),
        }
    };
    Usage {
        input_tokens: sum(a.input_tokens, b.input_tokens)?,
        output_tokens: sum(a.output_tokens, b.output_tokens)?,
        total_tokens: sum(a.total_tokens, b.total_tokens)?,
        cache_read_tokens: sum(a.cache_read_tokens, b.cache_read_tokens)?,
        cache_write_tokens: sum(a.cache_write_tokens, b.cache_write_tokens)?,
        reasoning_tokens: sum(a.reasoning_tokens, b.reasoning_tokens)?,
        input_audio_tokens: sum(a.input_audio_tokens, b.input_audio_tokens)?,
        output_audio_tokens: sum(a.output_audio_tokens, b.output_audio_tokens)?,
    }
    .normalized()
    .map_err(|e| provider_fault(provider, e.to_string()))
}

pub fn merge_ordinary(
    request: &Request,
    provider: &str,
    measured: &mut Response,
    generated: Response,
) -> Result<(), Lm15Error> {
    if generated.finish_reason != FinishReason::Stop {
        return Err(provider_fault(provider, "mixed judgment schema needs a completed ordinary answer, not a truncated or interrupted one"));
    }
    let Some(Value::Object(mut combined)) = generated.data() else {
        return Err(provider_fault(provider, "mixed judgment schema needs a complete generated JSON object for its ordinary properties"));
    };
    let schema = request
        .config
        .response_format
        .as_ref()
        .and_then(|f| f.get("schema"))
        .expect("judgment schema");
    let found = judgments::request_judgments(request);
    if let Some(required) = schema.get("required").and_then(Value::as_array) {
        for name in required.iter().filter_map(Value::as_str) {
            if !found.iter().any(|j| j.name == name) && !combined.contains_key(name) {
                return Err(provider_fault(
                    provider,
                    format!("generated answer is missing required ordinary property {name:?}"),
                ));
            }
        }
    }
    let data = match measured.message.parts.first_mut() {
        Some(Part::Data(data)) => data,
        _ => return Err(provider_fault(provider, "missing measured judgment data")),
    };
    for (key, value) in data.value.as_object().expect("measured judgment object") {
        combined.insert(key.clone(), value.clone());
    }
    data.value = combined.into();
    if let Some(original) = generated.data_part() {
        data.continuation = original.continuation.clone();
    }
    let data = data.clone();
    let mut parts = generated.message.parts.clone();
    let mut replaced = false;
    for part in &mut parts {
        if !replaced && matches!(part, Part::Data(_) | Part::Text(_)) {
            *part = Part::Data(data.clone());
            replaced = true;
        }
    }
    if !replaced {
        return Err(provider_fault(
            provider,
            "generated answer has no structured data slot",
        ));
    }
    let scoring_usage = measured.usage.to_json();
    measured.usage = combined_usage(provider, measured.usage, generated.usage)?;
    let metadata = measured.provider_data.get_or_insert_with(JsonObject::new);
    metadata.insert("scoring_usage".into(), scoring_usage);
    metadata.insert(
        "generated_response".into(),
        generated.to_json_with_provider_data(),
    );
    measured.id = generated.id;
    measured.finish_reason = generated.finish_reason;
    measured.message.parts = parts;
    measured.message.continuation = generated.message.continuation;
    Ok(())
}

/// Execute all tokenization hooks, then a single batched score call. Mixed
/// schemas additionally generate ordinary properties once under the unchanged
/// schema; these fields never receive probabilities. The caller provides an
/// authenticated JSON transport and attaches adaptations.
pub async fn complete<F, Fut, R>(
    request: &Request,
    cx: &BuildContext<'_>,
    mut send: F,
) -> Result<ScoringOutcome, Lm15Error>
where
    F: FnMut(WireRequest) -> Fut,
    Fut: Future<Output = Result<R, Lm15Error>>,
    R: Into<ScoringReply>,
{
    let plan = plan(request, cx)?;
    let ordinary = plan.ordinary;
    let mut tokenized = Vec::new();
    let mut calls = 0;
    for item in plan.judgments {
        let reply: ScoringReply = send(item.prefill).await?.into();
        let prefix = reply.parse(|value| tokens_from_value(value, cx.provider))?;
        calls += 1;
        let mut candidates = Vec::new();
        for candidate in item.candidates {
            let reply: ScoringReply = send(candidate.open).await?.into();
            let open = reply.parse(|value| tokens_from_value(value, cx.provider))?;
            let reply: ScoringReply = send(candidate.closed).await?.into();
            let closed = reply.parse(|value| tokens_from_value(value, cx.provider))?;
            calls += 2;
            candidates.push((candidate.key, open, closed));
        }
        let paths = candidate_paths(&prefix, &candidates, cx.provider)?;
        tokenized.push((item.judgment, prefix, paths));
    }
    let mut prompts = Vec::new();
    let mut locations = Vec::new();
    let mut ids = BTreeSet::new();
    for (index, (_, prefix, paths)) in tokenized.iter().enumerate() {
        for (node, children) in trie_nodes(paths) {
            let mut prompt = prefix.clone();
            prompt.extend(&node);
            prompts.push(prompt);
            ids.extend(&children);
            locations.push((index, node, children));
        }
    }
    let count = prompts.len();
    let reply: ScoringReply = send(score_request(cx, prompts, &ids)).await?.into();
    let scored = reply.parse(|value| scores_from_value(value, count, cx.provider))?;
    let mut tables = vec![ScoreTable::new(); tokenized.len()];
    for ((index, prefix, children), scores) in locations.into_iter().zip(scored.scores) {
        if children.iter().any(|id| !scores.contains_key(id)) {
            if request.config.probabilities == Some(ProbabilityPolicy::Required) {
                return Err(judgments::refusal(cx.provider, "config.probabilities", "the server ignored requested logprob_token_ids; required probabilities cannot be measured"));
            }
            return Ok(ScoringOutcome::Unavailable(Adaptation { field:"config.probabilities".into(), action:AdaptationAction::Dropped,
                asked:Some(Value::from("if_available")), applied:None,
                reason:"the server omitted requested token IDs; answer by ordinary structured output without a distribution".into() }, scored.usage));
        }
        tables[index].insert(prefix, scores);
    }
    let measured: Vec<_> = tokenized
        .into_iter()
        .zip(tables)
        .map(|((j, _, paths), table)| (j, paths, table))
        .collect();
    let mut native = request.clone();
    native.model = cx.model.to_string();
    let mut response = reply.parse(|_| {
        fold(
            &native,
            cx.provider,
            &measured,
            scored.usage,
            scored.model,
            count,
            calls,
        )
    })?;
    if let Some(wire) = ordinary {
        let generated: ScoringReply = send(wire).await?.into();
        let ordinary = generated.parse(|value| parse_generated(request, cx, value, true))?;
        generated.parse(|_| merge_ordinary(request, cx.provider, &mut response, ordinary))?;
    }
    Ok(ScoringOutcome::Measured(response))
}

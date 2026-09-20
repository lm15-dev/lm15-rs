//! Stateless host-driven candidate scoring protocol. No network is performed.
use super::{configuration, response_json, transport_request_json};
use crate::judgments::Judgment;
use crate::scoring::{self, CandidatePaths, ScoreTable};
use crate::{AdaptationPolicy, Canonical, Lm15Error, ProbabilityPolicy, ProviderLM, Request};
use serde_json::{json, Map, Value};
use std::collections::BTreeSet;

type Tokenized = Vec<(Judgment, Vec<u64>, CandidatePaths)>;

// Hosts may supply the old bare decoded JSON body or an explicit HTTP envelope.
// The latter preserves actual status and duplicate diagnostic headers.
fn generated_reply(
    lm: &ProviderLM,
    request: &Request,
    value: &Value,
    measured: bool,
) -> Result<crate::Response, Lm15Error> {
    if let Some(envelope) = value.as_object().filter(|v| {
        v.get("status").is_some_and(Value::is_number)
            && (v.contains_key("body") || v.contains_key("body_b64"))
    }) {
        super::check_decoded(envelope)?;
        let status = envelope
            .get("status")
            .and_then(Value::as_u64)
            .and_then(|s| u16::try_from(s).ok())
            .ok_or_else(|| configuration("invalid generated-response status"))?;
        return lm.parse_generated_judgment_response(
            request,
            status,
            &super::headers_of(envelope),
            &super::body_of(envelope)?,
            measured,
        );
    }
    let body = serde_json::to_vec(value).map_err(|e| configuration(e.to_string()))?;
    lm.parse_generated_judgment_response(request, 200, &[], &body, measured)
}
fn tokens(value: Option<&Value>, provider: &str) -> Result<Vec<u64>, Lm15Error> {
    scoring::tokens_from_value(
        value.ok_or_else(|| configuration("missing tokenization reply"))?,
        provider,
    )
}
fn compile(
    lm: &ProviderLM,
    request: &Request,
    msg: &Map<String, Value>,
) -> Result<Tokenized, Lm15Error> {
    let expected = crate::judgments::request_judgments(request);
    let replies = msg
        .get("tokenizations")
        .and_then(Value::as_array)
        .filter(|r| r.len() == expected.len())
        .ok_or_else(|| configuration("tokenizations must contain every judgment in plan order"))?;
    let mut out = Vec::new();
    for (j, reply) in expected.into_iter().zip(replies) {
        if reply.get("name").and_then(Value::as_str) != Some(j.name.as_str()) {
            return Err(configuration("tokenization judgment name/order mismatch"));
        }
        let prefix = tokens(reply.get("prefill"), lm.provider())?;
        let entries = reply
            .get("candidates")
            .and_then(Value::as_array)
            .filter(|r| r.len() == j.keys.len())
            .ok_or_else(|| configuration("tokenizations require every candidate"))?;
        let mut candidates = Vec::new();
        for (key, entry) in j.keys.iter().zip(entries) {
            if entry.get("key").and_then(Value::as_str) != Some(key.as_str()) {
                return Err(configuration("tokenization candidate key/order mismatch"));
            }
            candidates.push((
                key.clone(),
                tokens(entry.get("open"), lm.provider())?,
                tokens(entry.get("closed"), lm.provider())?,
            ));
        }
        let paths = scoring::candidate_paths(&prefix, &candidates, lm.provider())?;
        out.push((j, prefix, paths));
    }
    Ok(out)
}
pub(super) fn run(
    op: &str,
    lm: &ProviderLM,
    request: &Request,
    msg: &Map<String, Value>,
) -> Result<Value, Lm15Error> {
    let plan = lm.scoring_plan(request)?;
    if op == "scoring_plan" {
        let mut queries = Vec::new();
        for j in plan.judgments {
            let mut candidates = Vec::new();
            for c in j.candidates {
                candidates.push(json!({"key":c.key,"open":transport_request_json(&lm.surface_request(c.open,0)?),"closed":transport_request_json(&lm.surface_request(c.closed,0)?)}));
            }
            queries.push(json!({"name":j.judgment.name,"prefill":transport_request_json(&lm.surface_request(j.prefill,0)?),"candidates":candidates}));
        }
        let ordinary = plan
            .ordinary
            .map(|wire| {
                lm.surface_request(wire, 0)
                    .map(|r| transport_request_json(&r))
            })
            .transpose()?;
        return Ok(
            json!({"tokenizations":queries,"ordinary_request":ordinary,"adaptations":lm.plan(request)?.iter().map(Canonical::to_json).collect::<Vec<_>>(),"next":"scoring_build"}),
        );
    }
    let tokenized = compile(lm, request, msg)?;
    let mut prompts = Vec::new();
    let mut ids = BTreeSet::new();
    let mut locations = Vec::new();
    for (i, (_, prefix, paths)) in tokenized.iter().enumerate() {
        for (node, children) in scoring::trie_nodes(paths) {
            let mut prompt = prefix.clone();
            prompt.extend(&node);
            prompts.push(prompt);
            ids.extend(&children);
            locations.push((i, node, children));
        }
    }
    let count = prompts.len();
    if op == "scoring_build" {
        let mut out = transport_request_json(&lm.build_score_request(request, prompts, &ids)?);
        out["next"] = Value::from("scoring_parse");
        return Ok(out);
    }
    let body = super::body_of(msg)?;
    let status = msg
        .get("status")
        .and_then(Value::as_u64)
        .and_then(|n| u16::try_from(n).ok())
        .unwrap_or(200);
    let headers = super::headers_of(msg);
    if status >= 400 {
        return Err(lm.http_error(status, &headers, &body));
    }
    let reply = scoring::ScoringReply::from_http(status, headers, body, lm.provider())?;
    let scored =
        scoring::scores_from_value(&reply.value, count, lm.provider()).map_err(|mut e| {
            crate::transport::attach_http_error(&mut e, status, &reply.headers, &reply.body);
            e
        })?;
    let mut tables = vec![ScoreTable::new(); tokenized.len()];
    for ((i, node, children), scores) in locations.into_iter().zip(scored.scores) {
        if children.iter().any(|id| !scores.contains_key(id)) {
            let reason =
                "server omitted requested token IDs; no measured probabilities are available";
            if request.config.probabilities == Some(ProbabilityPolicy::Required)
                || lm.adaptation_policy() == AdaptationPolicy::Refuse
            {
                return Err(crate::adaptation::refusal(
                    lm.provider(),
                    "config.probabilities",
                    reason,
                ));
            }
            let record = crate::Adaptation {
                field: "config.probabilities".into(),
                action: crate::AdaptationAction::Dropped,
                asked: Some("if_available".into()),
                applied: None,
                reason: reason.into(),
            };
            let mut fallback = request.clone();
            fallback.config.probabilities = Some(ProbabilityPolicy::Off);
            let stream = crate::adaptation::has_client_side_stop(&lm.plan(&fallback)?);
            let mut records = lm.plan(request)?;
            for note in lm.plan(&fallback)? {
                if !records.contains(&note) {
                    records.push(note);
                }
            }
            if !records.contains(&record) {
                records.push(record);
            }
            if let Some(answer) = msg
                .get("fallback_response")
                .or_else(|| msg.get("ordinary_response"))
            {
                let mut response = generated_reply(lm, &fallback, answer, false)?;
                response.usage =
                    scoring::combined_usage(lm.provider(), scored.usage, response.usage)?;
                response
                    .provider_data
                    .get_or_insert_with(crate::types::JsonObject::new)
                    .insert("scoring_usage".into(), scored.usage.to_json());
                if lm.adaptation_policy() != AdaptationPolicy::Silent {
                    response.adaptations = records;
                }
                return Ok(response_json(&response));
            }
            return Ok(
                json!({"unavailable":true,"adaptations":if lm.adaptation_policy()==AdaptationPolicy::Silent {vec![]} else {records.iter().map(Canonical::to_json).collect::<Vec<_>>()},
                "scoring_usage":scored.usage.to_json(),"next":"scoring_parse",
                "requires_stream":stream,"fallback_request":transport_request_json(&lm.build_request(&fallback,stream)?),"fallback_canonical_request":fallback.to_json()}),
            );
        }
        tables[i].insert(node, scores);
    }
    let calls = tokenized
        .iter()
        .map(|(_, _, paths)| 1 + 2 * paths.len())
        .sum();
    let measured: Vec<_> = tokenized
        .into_iter()
        .zip(tables)
        .map(|((j, _, paths), table)| (j, paths, table))
        .collect();
    let mut native = request.clone();
    native.model = lm.wire_model(&request.model).into();
    let mut response = scoring::fold(
        &native,
        lm.provider(),
        &measured,
        scored.usage,
        scored.model,
        count,
        calls,
    )?;
    if let Some(ordinary_wire) = plan.ordinary {
        let Some(ordinary) = msg.get("ordinary_response") else {
            return Ok(json!({"needs_ordinary":true,"next":"scoring_parse",
                "ordinary_request":transport_request_json(&lm.surface_request(ordinary_wire,0)?),
                "scoring_usage":scored.usage.to_json()}));
        };
        let mut generated = request.clone();
        generated.config.probabilities = Some(ProbabilityPolicy::Off);
        let generated = generated_reply(lm, &generated, ordinary, true)?;
        scoring::merge_ordinary(request, lm.provider(), &mut response, generated)?;
    }
    if lm.adaptation_policy() != AdaptationPolicy::Silent {
        response.adaptations = lm.plan(request)?;
    }
    Ok(response_json(&response))
}

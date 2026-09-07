//! The response side of the corpus through the library (the calls the
//! shim makes for `parse_response` / `replay_stream`): every wire case
//! with a pinned body and a golden — complete bodies and SSE streams —
//! must yield the golden's canonical Response and event trace, under the
//! harness's rules: volatile paths by presence and type, end-event
//! `provider_data` by presence and type (D9), int/float equality inside
//! `usage`, and a pinned refusal (MAP-9) answered with the same class,
//! ErrorCode, partial Response and trace.
//!
//! The contract checkout is located through `LM15_CONTRACT_DIR`, else the
//! sibling `../lm15-contract`; the corpus is never copied here.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use lm15::registry::adapter_for;
use lm15::stream::materialize_response;
use lm15::types::Request;
use lm15::Canonical;

fn contract_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("LM15_CONTRACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../lm15-contract"));
    if dir.join("cases").is_dir() && dir.join("goldens").is_dir() {
        Some(dir)
    } else {
        eprintln!(
            "contract checkout not found at {}; corpus test not run",
            dir.display()
        );
        None
    }
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(
        &fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
    )
    .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn looks_like_sse(body: &[u8]) -> bool {
    let head: Vec<u8> = body
        .iter()
        .skip_while(|b| b.is_ascii_whitespace())
        .take(512)
        .copied()
        .collect();
    head.starts_with(b"event:")
        || head.starts_with(b"data:")
        || head.windows(6).any(|w| w == b"\ndata:")
}

/// The harness's `first_difference`, strict: absent, null, "", [], {} are
/// five values; `true != 1`; int == zero-fraction float only under a
/// `usage` segment; volatile paths by presence and type.
fn first_difference(
    expected: &Value,
    actual: Option<&Value>,
    path: &str,
    under_usage: bool,
    volatile: &Map<String, Value>,
) -> Option<String> {
    let Some(actual) = actual else {
        return Some(format!("{path}: missing"));
    };
    if is_volatile(path, volatile) {
        return match (expected, actual) {
            (Value::Number(_), Value::Number(_)) => None,
            (e, a) if json_type(e) == json_type(a) => None,
            (e, a) => Some(format!(
                "{path}: volatile type {} != {}",
                json_type(e),
                json_type(a)
            )),
        };
    }
    match (expected, actual) {
        (Value::Object(e), Value::Object(a)) => {
            for (key, ev) in e {
                let sub = format!("{path}.{key}");
                if let Some(d) = first_difference(
                    ev,
                    a.get(key),
                    &sub,
                    under_usage || key == "usage",
                    volatile,
                ) {
                    return Some(d);
                }
            }
            for key in a.keys() {
                if !e.contains_key(key) {
                    return Some(format!("{path}.{key}: unexpected key"));
                }
            }
            None
        }
        (Value::Array(e), Value::Array(a)) => {
            for (i, ev) in e.iter().enumerate() {
                let sub = format!("{path}[{i}]");
                if let Some(d) = first_difference(ev, a.get(i), &sub, under_usage, volatile) {
                    return Some(d);
                }
            }
            if a.len() > e.len() {
                return Some(format!("{path}[{}]: unexpected element", e.len()));
            }
            None
        }
        (Value::Number(e), Value::Number(a)) => {
            let same_kind = e.is_f64() == a.is_f64();
            if same_kind && e == a {
                return None;
            }
            if !same_kind && under_usage {
                if let (Some(ef), Some(af)) = (e.as_f64(), a.as_f64()) {
                    if ef == af && ef.fract() == 0.0 {
                        return None;
                    }
                }
            }
            Some(format!("{path}: {e} != {a}"))
        }
        (e, a) if json_type(e) != json_type(a) => {
            Some(format!("{path}: type {} != {}", json_type(e), json_type(a)))
        }
        (e, a) if e != a => Some(format!("{path}: {e} != {a}")),
        _ => None,
    }
}

fn json_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(n) if n.is_f64() => "float",
        Value::Number(_) => "integer",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn is_volatile(path: &str, volatile: &Map<String, Value>) -> bool {
    let bare = path.strip_prefix("$.").unwrap_or(path);
    let under_response = bare.strip_prefix("canonical_response.");
    volatile.keys().any(|key| {
        let key = key.strip_prefix("$.").unwrap_or(key);
        key == bare || Some(key) == under_response
    })
}

/// D9: `provider_data` on `end` events compares by presence and type only.
fn strip_end_provider_data(events: &Value, golden: &Value) -> Result<(Value, Value), String> {
    let (Value::Array(g), Value::Array(a)) = (golden, events) else {
        return Ok((golden.clone(), events.clone()));
    };
    let mut g_out = Vec::new();
    let mut a_out = a.clone();
    for (i, event) in g.iter().enumerate() {
        let Value::Object(event) = event else {
            g_out.push(event.clone());
            continue;
        };
        if event.get("type") != Some(&Value::String("end".into())) {
            g_out.push(Value::Object(event.clone()));
            continue;
        }
        let mut stripped = event.clone();
        let golden_pd = stripped.remove("provider_data");
        g_out.push(Value::Object(stripped));
        if let Some(Value::Object(actual)) = a.get(i) {
            let mut actual = actual.clone();
            let actual_pd = actual.remove("provider_data");
            a_out[i] = Value::Object(actual);
            if let Some(golden_pd) = golden_pd {
                match actual_pd {
                    None => return Err(format!("events[{i}].provider_data: presence required")),
                    Some(actual_pd) if json_type(&actual_pd) != json_type(&golden_pd) => {
                        return Err(format!("events[{i}].provider_data: type mismatch"))
                    }
                    _ => {}
                }
            }
        }
    }
    Ok((Value::Array(g_out), Value::Array(a_out)))
}

#[test]
fn response_and_stream_goldens_replay_exactly() {
    let Some(dir) = contract_dir() else { return };
    let mut paths: Vec<PathBuf> = fs::read_dir(dir.join("cases"))
        .unwrap()
        .flatten()
        .filter(|e| e.path().is_dir())
        .flat_map(|e| fs::read_dir(e.path()).unwrap().flatten().map(|f| f.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    paths.sort();

    let mut failures = Vec::new();
    let mut checked = (0usize, 0usize);
    for path in paths {
        let case = read_json(&path);
        let id = case["id"].as_str().unwrap_or("?").to_string();
        if matches!(
            case.get("surface").and_then(Value::as_str),
            Some("models" | "live" | "files" | "batch" | "generation" | "video" | "cache")
        ) {
            continue;
        }
        let Some(pinned) = case.get("pinned_body").and_then(Value::as_str) else {
            continue;
        };
        let Some(canonical_request) = case.get("canonical_request") else {
            continue;
        };
        let raises = case
            .get("expect_lm15")
            .and_then(|e| e.get("raises"))
            .and_then(Value::as_object)
            .cloned();
        if raises
            .as_ref()
            .is_some_and(|r| r.get("op").and_then(Value::as_str) == Some("build_request"))
        {
            continue;
        }
        let provider = case["provider"].as_str().unwrap();
        let golden_path = dir
            .join("goldens")
            .join(provider)
            .join(format!("{}.json", case["feature"].as_str().unwrap()));
        if !golden_path.is_file() {
            continue;
        }
        let golden = read_json(&golden_path);
        let body = fs::read(dir.join("bodies").join(&id).join(pinned)).unwrap();
        let is_stream = case.get("stream") == Some(&Value::Bool(true)) || looks_like_sse(&body);
        let volatile = case
            .get("volatile")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();

        let settings = case.get("settings").and_then(Value::as_object).map(|s| {
            s.iter()
                .map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().to_string()))
                .collect()
        });
        let lm = adapter_for(
            provider,
            "test-key",
            case.get("base_url").and_then(Value::as_str),
            settings,
            None,
        )
        .unwrap_or_else(|e| panic!("{id}: {e}"));
        let request = Request::from_json(canonical_request).unwrap_or_else(|e| panic!("{id}: {e}"));

        let want_op = if is_stream {
            "replay_stream"
        } else {
            "parse_response"
        };
        let want_raise = raises.filter(|r| r.get("op").and_then(Value::as_str) == Some(want_op));

        if is_stream {
            checked.1 += 1;
            let events = match lm.replay_stream(&request, &body) {
                Ok(events) => events,
                Err(e) => {
                    failures.push(format!("{id}: replay failed: {e}"));
                    continue;
                }
            };
            let trace = Value::Array(events.iter().map(Canonical::to_json).collect());
            let assembled = materialize_response(events.iter(), &request);
            match (want_raise, assembled) {
                (Some(want), Err(err)) => {
                    if want.get("type").and_then(Value::as_str) != Some(err.class_name())
                        || want.get("code").and_then(Value::as_str) != Some(err.code().as_str())
                    {
                        failures.push(format!(
                            "{id}: raised {} ({})",
                            err.class_name(),
                            err.code()
                        ));
                        continue;
                    }
                    if let Some(partial) = golden.get("partial_response") {
                        let actual = err.partial().map(Canonical::to_json);
                        if let Some(d) = first_difference(
                            partial,
                            actual.as_ref(),
                            "partial_response",
                            false,
                            &volatile,
                        ) {
                            failures.push(format!("{id}: {d}"));
                            continue;
                        }
                    }
                    if let Some(golden_events) = golden.get("events") {
                        match strip_end_provider_data(&trace, golden_events) {
                            Err(d) => failures.push(format!("{id}: {d}")),
                            Ok((g, a)) => {
                                if let Some(d) =
                                    first_difference(&g, Some(&a), "events", false, &volatile)
                                {
                                    failures.push(format!("{id}: {d}"));
                                }
                            }
                        }
                    }
                }
                (Some(want), Ok(_)) => failures.push(format!(
                    "{id}: expected {} but assembled a Response",
                    want.get("type").and_then(Value::as_str).unwrap_or("?")
                )),
                (None, Err(err)) => failures.push(format!("{id}: assembly refused: {err}")),
                (None, Ok(response)) => {
                    if let Some(unmapped) = response
                        .provider_data
                        .as_ref()
                        .and_then(|pd| pd.get("_lm15_unmapped"))
                    {
                        failures.push(format!("{id}: unmapped {unmapped}"));
                        continue;
                    }
                    if let Some(d) = first_difference(
                        &golden["canonical_response"],
                        Some(&response.to_json()),
                        "canonical_response",
                        false,
                        &volatile,
                    ) {
                        failures.push(format!("{id}: {d}"));
                        continue;
                    }
                    if let Some(golden_events) = golden.get("events") {
                        match strip_end_provider_data(&trace, golden_events) {
                            Err(d) => failures.push(format!("{id}: {d}")),
                            Ok((g, a)) => {
                                if let Some(d) =
                                    first_difference(&g, Some(&a), "events", false, &volatile)
                                {
                                    failures.push(format!("{id}: {d}"));
                                }
                            }
                        }
                    }
                }
            }
        } else {
            checked.0 += 1;
            let status = case
                .get("expect")
                .and_then(|e| e.get("status"))
                .and_then(Value::as_u64)
                .unwrap_or(200) as u16;
            match (want_raise, lm.parse_response(&request, status, &body)) {
                (Some(want), Err(err)) => {
                    if want.get("type").and_then(Value::as_str) != Some(err.class_name()) {
                        failures.push(format!("{id}: raised {}", err.class_name()));
                    }
                }
                (Some(_), Ok(_)) => failures.push(format!("{id}: expected a raise")),
                (None, Err(err)) => failures.push(format!("{id}: parse failed: {err}")),
                (None, Ok(response)) => {
                    if let Some(unmapped) = response
                        .provider_data
                        .as_ref()
                        .and_then(|pd| pd.get("_lm15_unmapped"))
                    {
                        failures.push(format!("{id}: unmapped {unmapped}"));
                        continue;
                    }
                    if let Some(d) = first_difference(
                        &golden["canonical_response"],
                        Some(&response.to_json()),
                        "canonical_response",
                        false,
                        &volatile,
                    ) {
                        failures.push(format!("{id}: {d}"));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(checked.0 >= 298, "complete cases checked: {}", checked.0);
    assert!(checked.1 >= 40, "stream cases checked: {}", checked.1);
}

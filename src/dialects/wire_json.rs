//! Reading provider JSON on the response side, with the reference's
//! leniency spelled out once. Provider bodies are not canonical JSON: the
//! reference reads them with Python truthiness (`x or ""`, `int(x or 0)`,
//! `isinstance` guards) and never raises on a stray shape — it records what
//! it could not map (the `_lm15_unmapped` canary, harness/PROTOCOL.md) and
//! moves on. These helpers are those idioms.

use serde_json::{Map, Value};

use crate::errors::{ErrorClass, ErrorMeta, Lm15Error};
use crate::types::{ErrorDetail, JsonObject, TokenLogprob, TopLogprob, Usage};

/// Python truthiness of a JSON value (`bool(x)`).
pub fn truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64() != Some(0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// `str(x)` for a JSON value.
pub fn py_str(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        other => other.to_string(),
    }
}

/// `str(x or "")`: the empty string for a falsy value, else `str(x)`.
pub fn str_or_empty(value: Option<&Value>) -> String {
    match value {
        Some(v) if truthy(Some(v)) => py_str(v),
        _ => String::new(),
    }
}

/// `str(x or "") or None`.
pub fn str_or_none(value: Option<&Value>) -> Option<String> {
    let text = str_or_empty(value);
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

/// `str(x) if x else None` for a scalar id/model field.
pub fn id_or_none(value: Option<&Value>) -> Option<String> {
    str_or_none(value)
}

/// The first truthy value among `keys`, as `str(x)`; `None` when none is.
pub fn first_str(obj: &Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter()
        .map(|k| obj.get(*k))
        .find(|v| truthy(*v))
        .map(|v| py_str(v.expect("truthy is Some")))
}

/// `int(x or 0)` for an index: a JSON number truncated, anything else 0.
pub fn index_of(value: Option<&Value>) -> u64 {
    match value {
        Some(Value::Number(n)) => n
            .as_u64()
            .or_else(|| n.as_f64().map(|f| if f > 0.0 { f as u64 } else { 0 }))
            .unwrap_or(0),
        Some(Value::String(s)) => s.trim().parse::<u64>().unwrap_or(0),
        _ => 0,
    }
}

/// A usage counter (INV-029): absent or `null` is "not reported"; an int
/// or an integral float (proto3 JSON) is the count; anything else — a
/// string, a bool, a fraction, a negative — is a malformed counter and a
/// `ProviderError`, never silently "not reported" (the reference raises a
/// native `TypeError` from the `Usage` constructor).
pub fn count_of(
    provider: &str,
    field: &str,
    value: Option<&Value>,
) -> Result<Option<u64>, Lm15Error> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(n)) => n
            .as_u64()
            .or_else(|| {
                n.as_f64()
                    .filter(|f| f.fract() == 0.0 && *f >= 0.0)
                    .map(|f| f as u64)
            })
            .map(Some)
            .ok_or_else(|| malformed_counter(provider, field, n.to_string())),
        Some(other) => Err(malformed_counter(provider, field, other.to_string())),
    }
}

fn malformed_counter(provider: &str, field: &str, value: String) -> Lm15Error {
    provider_error(
        ErrorClass::ProviderError,
        provider,
        format!("{provider}: usage counter {field} is not a token count: {value}"),
        None,
    )
}

/// `x if isinstance(x, dict) else {}`.
pub fn object_or_empty(value: Option<&Value>) -> &Map<String, Value> {
    static EMPTY: std::sync::OnceLock<Map<String, Value>> = std::sync::OnceLock::new();
    match value {
        Some(Value::Object(o)) => o,
        _ => EMPTY.get_or_init(Map::new),
    }
}

/// `x if isinstance(x, list) else []`.
pub fn array_or_empty(value: Option<&Value>) -> &[Value] {
    match value {
        Some(Value::Array(a)) => a,
        _ => &[],
    }
}

/// The reference's `parse_json_object` (`lm15/providers/common.py`): a
/// dict verbatim; a non-empty string parsed, a non-object wrapped as
/// `{"value": ...}`, an unparseable one kept as `{"partial_json": ...}`;
/// anything else `{}`.
pub fn parse_json_object(value: Option<&Value>) -> JsonObject {
    match value {
        Some(Value::Object(o)) => o.clone(),
        Some(Value::String(s)) if !s.is_empty() => parse_json_text(s),
        _ => JsonObject::new(),
    }
}

/// Best-effort parse of accumulated tool-call JSON text
/// (`lm15/result.py` `_parse_json_best_effort`): empty text is `{}`.
pub fn parse_json_text(text: &str) -> JsonObject {
    if text.is_empty() {
        return JsonObject::new();
    }
    match serde_json::from_str::<Value>(text) {
        Ok(Value::Object(o)) => o,
        Ok(other) => {
            let mut wrapped = JsonObject::new();
            wrapped.insert("value".to_string(), other);
            wrapped
        }
        Err(_) => {
            let mut wrapped = JsonObject::new();
            wrapped.insert("partial_json".to_string(), Value::String(text.to_string()));
            wrapped
        }
    }
}

/// The `_lm15_unmapped` recorder (harness/PROTOCOL.md § Unmapped
/// recorder): response content the adapter could not map, as
/// `{"path", "type"}` entries; falsy types are `"<missing>"`.
#[derive(Debug, Default)]
pub struct Unmapped(pub Vec<(String, String)>);

impl Unmapped {
    pub fn record(&mut self, path: impl Into<String>, type_name: Option<&Value>) {
        let type_text = str_or_empty(type_name);
        self.record_text(path, &type_text);
    }

    pub fn record_text(&mut self, path: impl Into<String>, type_text: &str) {
        let type_text = if type_text.is_empty() {
            "<missing>".to_string()
        } else {
            type_text.to_string()
        };
        self.0.push((path.into(), type_text));
    }

    /// The Python type name of a JSON value that was not the shape the
    /// mapping expected (`type(item).__name__`).
    pub fn record_shape(&mut self, path: impl Into<String>, value: &Value) {
        self.record_text(path, python_type_name(value));
    }

    /// `provider_data` with the canary attached when non-empty.
    pub fn attach(self, mut data: JsonObject) -> JsonObject {
        if self.0.is_empty() {
            return data;
        }
        let entries: Vec<Value> = self
            .0
            .into_iter()
            .map(|(path, type_text)| {
                let mut o = Map::new();
                o.insert("path".to_string(), Value::String(path));
                o.insert("type".to_string(), Value::String(type_text));
                Value::Object(o)
            })
            .collect();
        data.insert("_lm15_unmapped".to_string(), Value::Array(entries));
        data
    }
}

pub fn python_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) => {
            if n.is_f64() {
                "float"
            } else {
                "int"
            }
        }
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// OpenAI-style logprob entries, both wire dialects
/// (`lm15/providers/common.py` `openai_token_logprobs`): malformed entries
/// are skipped, non-list input is empty.
pub fn openai_token_logprobs(entries: Option<&Value>) -> Vec<TokenLogprob> {
    let Some(Value::Array(entries)) = entries else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries {
        let Value::Object(entry) = entry else {
            continue;
        };
        let (Some(token), Some(logprob)) = (entry.get("token"), entry.get("logprob")) else {
            continue;
        };
        let Some(logprob) = logprob.as_f64() else {
            continue;
        };
        let mut top = Vec::new();
        for alt in array_or_empty(entry.get("top_logprobs")) {
            let Value::Object(alt) = alt else { continue };
            let (Some(alt_token), Some(alt_logprob)) = (alt.get("token"), alt.get("logprob"))
            else {
                continue;
            };
            let Some(alt_logprob) = alt_logprob.as_f64() else {
                continue;
            };
            top.push(TopLogprob {
                token: py_str(alt_token),
                logprob: alt_logprob,
                bytes: byte_list(alt.get("bytes")),
                token_id: None,
            });
        }
        out.push(TokenLogprob {
            token: py_str(token),
            logprob,
            bytes: byte_list(entry.get("bytes")),
            token_id: None,
            top,
        });
    }
    out
}

fn byte_list(value: Option<&Value>) -> Option<Vec<u64>> {
    match value {
        Some(Value::Array(items)) => Some(items.iter().filter_map(Value::as_u64).collect()),
        _ => None,
    }
}

/// INV-029 at the wire boundary: the counters as reported, the total
/// summed only when both primaries are present and no total came.
pub fn usage(fields: Usage) -> Usage {
    // Overflow of two provider counts is not a real case; keep the total
    // unknown rather than fail a whole response over it.
    fields.normalized().unwrap_or(fields)
}

/// The `ErrorDetail` of a stream error frame: the class from a dialect's
/// code table (default `ProviderError`), the message and provider code
/// with the reference's fallbacks.
pub fn error_detail(class: ErrorClass, provider_code: &str, message: &str) -> ErrorDetail {
    ErrorDetail {
        code: class.code(),
        message: if !message.is_empty() {
            message.to_string()
        } else if !provider_code.is_empty() {
            provider_code.to_string()
        } else {
            "provider error".to_string()
        },
        provider_code: Some(if provider_code.is_empty() {
            "provider".to_string()
        } else {
            provider_code.to_string()
        }),
    }
}

/// A provider-shaped error raised by a parse (an in-band error envelope,
/// a blocked prompt): the class with the provider and its code attached.
pub fn provider_error(
    class: ErrorClass,
    provider: &str,
    message: impl Into<String>,
    provider_code: Option<String>,
) -> Lm15Error {
    let mut meta = ErrorMeta::new(message);
    meta.provider = Some(provider.to_string());
    meta.provider_code = provider_code.filter(|c| !c.is_empty());
    Lm15Error::of_class(class, meta)
}

/// A body that is not the JSON object the dialect expects.
pub fn malformed(provider: &str, what: &str) -> Lm15Error {
    provider_error(
        ErrorClass::ProviderError,
        provider,
        format!("{provider}: response body is not {what}"),
        None,
    )
}

/// The body as a JSON object, or a `ProviderError`.
pub fn body_object(provider: &str, body: &[u8]) -> Result<JsonObject, Lm15Error> {
    match serde_json::from_slice::<Value>(body) {
        Ok(Value::Object(o)) => Ok(o),
        Ok(_) => Err(malformed(provider, "a JSON object")),
        Err(e) => Err(provider_error(
            ErrorClass::ProviderError,
            provider,
            format!("{provider}: response body is not JSON: {e}"),
            None,
        )),
    }
}

/// A stream frame's payload as a JSON object; `None` for anything else
/// (the reference ignores non-object frames).
pub fn frame_object(data: &str) -> Option<JsonObject> {
    match serde_json::from_str::<Value>(data) {
        Ok(Value::Object(o)) => Some(o),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn truthiness_and_text() {
        assert_eq!(str_or_empty(Some(&json!(""))), "");
        assert_eq!(str_or_empty(Some(&json!(0))), "");
        assert_eq!(str_or_empty(Some(&json!(5))), "5");
        assert_eq!(str_or_empty(Some(&json!(true))), "True");
        assert_eq!(str_or_none(Some(&json!(null))), None);
        assert_eq!(index_of(Some(&json!(2))), 2);
        assert_eq!(index_of(Some(&json!(null))), 0);
        assert_eq!(count_of("p", "f", Some(&json!(3.0))).unwrap(), Some(3));
        assert_eq!(count_of("p", "f", None).unwrap(), None);
        assert_eq!(count_of("p", "f", Some(&json!(null))).unwrap(), None);
        assert!(count_of("p", "f", Some(&json!(3.5))).is_err());
        assert!(count_of("p", "f", Some(&json!("3"))).is_err());
        assert!(count_of("p", "f", Some(&json!(true))).is_err());
    }

    #[test]
    fn json_object_parsing_is_best_effort() {
        assert_eq!(parse_json_text(""), JsonObject::new());
        assert_eq!(parse_json_text("{\"a\": 1}")["a"], json!(1));
        assert_eq!(parse_json_text("[1]")["value"], json!([1]));
        assert_eq!(
            parse_json_text("{\"a\": ")["partial_json"],
            json!("{\"a\": ")
        );
        assert_eq!(parse_json_object(Some(&json!(null))), JsonObject::new());
    }

    #[test]
    fn unmapped_attaches_the_canary() {
        let mut u = Unmapped::default();
        u.record("output[1]", None);
        u.record_shape("output[2]", &json!(3));
        let data = u.attach(JsonObject::new());
        assert_eq!(
            data["_lm15_unmapped"],
            json!([
                {"path": "output[1]", "type": "<missing>"},
                {"path": "output[2]", "type": "int"}
            ])
        );
        assert!(Unmapped::default().attach(JsonObject::new()).is_empty());
    }

    #[test]
    fn logprobs_skip_malformed_entries() {
        let entries = json!([
            {"token": "a", "logprob": -0.5, "bytes": [97], "top_logprobs": [{"token": "b", "logprob": -1}, {"token": "c"}]},
            {"token": "no-logprob"},
            "not an object"
        ]);
        let out = openai_token_logprobs(Some(&entries));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].top.len(), 1);
        assert_eq!(out[0].bytes, Some(vec![97]));
        assert!(openai_token_logprobs(Some(&json!({}))).is_empty());
    }
}

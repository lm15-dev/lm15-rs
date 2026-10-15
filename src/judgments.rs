//! Declared judgments, equivalent provider schemas, and structured answers (MAP-14).
//! This module is pure; probability totals are deliberately not validated (INV-052).
//! A bounding box (changes/2026-10-15-bounding-box.md) is a judgment that
//! declares no keys: it is answered, never scored.
use crate::errors::{ErrorMeta, Lm15Error};
use crate::types::*;
use serde_json::{json, Value};
use std::collections::BTreeSet;

pub const MAX_ORDERED_LEVELS: usize = 10;
pub const MAX_CHOICE_KEYS: usize = 255;

/// A box's edges, in Jev's order (changes/2026-10-15-bounding-box.md D1):
/// fractions of the image from its top-left corner, `top`/`bottom` of its
/// height, `left`/`right` of its width (docs.typesafe.ai/primitives/box).
pub const BOX_EDGES: [&str; 4] = ["top", "bottom", "left", "right"];
const BOX_EDGE_DESCRIPTIONS: [&str; 4] = [
    "Top edge, as a fraction of the image's height from the top, from 0 to 1",
    "Bottom edge, as a fraction of the image's height from the top, from 0 to 1",
    "Left edge, as a fraction of the image's width from the left, from 0 to 1",
    "Right edge, as a fraction of the image's width from the left, from 0 to 1",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JudgmentKind {
    Boolean,
    Choice,
    Ordered,
    /// A bounding box: four edges, no declared keys (2026-10-15).
    Box,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Judgment {
    pub name: String,
    pub kind: JudgmentKind,
    pub keys: Vec<String>,
    pub instruction: Option<String>,
    pub descriptions: std::collections::BTreeMap<String, String>,
    pub titles: std::collections::BTreeMap<String, String>,
}
impl Judgment {
    pub fn ordered(&self) -> bool {
        self.kind == JudgmentKind::Ordered
    }
    /// Whether a distribution over declared keys exists to measure: a box
    /// declares no keys, so it is answered, never scored (MAP-14 §1).
    pub fn measurable(&self) -> bool {
        self.kind != JudgmentKind::Box
    }
    pub fn value_for_key(&self, key: &str) -> Value {
        match self.kind {
            JudgmentKind::Boolean => Value::Bool(key == "true"),
            JudgmentKind::Ordered => Value::from(key.parse::<u64>().expect("declared ordered key")),
            JudgmentKind::Choice | JudgmentKind::Box => Value::String(key.to_string()),
        }
    }
}

/// An object whose properties are exactly the four edges, each a `number`
/// (changes/2026-10-15-bounding-box.md D1). Other keywords are the caller's
/// and travel verbatim.
fn is_box(prop: &JsonObject) -> bool {
    if prop
        .get("type")
        .is_some_and(|t| t.as_str() != Some("object"))
    {
        return false;
    }
    let Some(props) = prop.get("properties").and_then(Value::as_object) else {
        return false;
    };
    props.len() == BOX_EDGES.len()
        && BOX_EDGES.iter().all(|edge| {
            props
                .get(*edge)
                .and_then(Value::as_object)
                .is_some_and(|s| s.get("type").and_then(Value::as_str) == Some("number"))
        })
}

fn judgment_of(name: &str, property: &Value) -> Option<Judgment> {
    let prop = property.as_object()?;
    let mut j = Judgment {
        name: name.to_string(),
        kind: JudgmentKind::Boolean,
        keys: Vec::new(),
        instruction: prop
            .get("description")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string),
        descriptions: Default::default(),
        titles: Default::default(),
    };
    if is_box(prop) {
        j.kind = JudgmentKind::Box;
        return Some(j);
    }
    if prop.get("type").and_then(Value::as_str) == Some("boolean") {
        j.keys = vec!["true".into(), "false".into()];
        return Some(j);
    }
    let branches = prop.get("anyOf").and_then(Value::as_array).filter(|b| {
        !b.is_empty()
            && b.iter()
                .all(|v| v.as_object().is_some_and(|o| o.contains_key("const")))
    });
    let values: Vec<&Value> = match (prop.get("enum"), branches) {
        (Some(Value::Array(values)), None) if !values.is_empty() => values.iter().collect(),
        (None, Some(branches)) => branches.iter().map(|b| &b["const"]).collect(),
        _ => return None,
    };
    if values
        .iter()
        .all(|v| v.as_str().is_some_and(|s| !s.is_empty()))
    {
        if prop
            .get("type")
            .is_some_and(|v| v.as_str() != Some("string"))
        {
            return None;
        }
        j.kind = JudgmentKind::Choice;
        j.keys = values
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        if j.keys.iter().collect::<BTreeSet<_>>().len() != j.keys.len() {
            return None;
        }
    } else if values.len() >= 2
        && values
            .iter()
            .enumerate()
            .all(|(i, v)| v.as_u64() == Some(i as u64))
    {
        if prop
            .get("type")
            .is_some_and(|v| v.as_str() != Some("integer"))
        {
            return None;
        }
        j.kind = JudgmentKind::Ordered;
        j.keys = (0..values.len()).map(|i| i.to_string()).collect();
    } else {
        return None;
    }
    if let Some(branches) = branches {
        for (key, branch) in j.keys.iter().zip(branches) {
            if let Some(text) = branch.get("description").and_then(Value::as_str) {
                j.descriptions.insert(key.clone(), text.to_string());
            }
            if let Some(text) = branch.get("title").and_then(Value::as_str) {
                j.titles.insert(key.clone(), text.to_string());
            }
        }
    }
    Some(j)
}

pub fn judgments_in_schema(schema: &Value) -> Vec<Judgment> {
    let Some(schema) = schema.as_object() else {
        return Vec::new();
    };
    if schema
        .get("type")
        .is_some_and(|t| t.as_str() != Some("object"))
    {
        return Vec::new();
    }
    schema
        .get("properties")
        .and_then(Value::as_object)
        .map(|props| {
            props
                .iter()
                .filter_map(|(name, prop)| judgment_of(name, prop))
                .collect()
        })
        .unwrap_or_default()
}

pub fn request_judgments(request: &Request) -> Vec<Judgment> {
    request
        .config
        .response_format
        .as_ref()
        .filter(|f| f.get("type").and_then(Value::as_str) == Some("json_schema"))
        .and_then(|f| f.get("schema"))
        .map(judgments_in_schema)
        .unwrap_or_default()
}

/// The judgments with declared keys: the ones a distribution can be measured
/// for (boxes are left out).
pub fn measurable_judgments(found: Vec<Judgment>) -> Vec<Judgment> {
    found.into_iter().filter(Judgment::measurable).collect()
}

pub fn request_measurable_judgments(request: &Request) -> Vec<Judgment> {
    measurable_judgments(request_judgments(request))
}

pub fn non_judgment_properties(schema: &Value, found: &[Judgment]) -> Vec<String> {
    schema
        .get("properties")
        .and_then(Value::as_object)
        .map(|props| {
            props
                .keys()
                .filter(|name| !found.iter().any(|j| &j.name == *name))
                .cloned()
                .collect()
        })
        .unwrap_or_default()
}

pub(crate) fn refusal(provider: &str, feature: &str, reason: impl Into<String>) -> Lm15Error {
    let mut meta = ErrorMeta::new(reason);
    meta.provider = Some(provider.to_string());
    meta.feature = Some(feature.to_string());
    Lm15Error::UnsupportedFeatureError(meta)
}

/// MAP-14 §3 on a wire with no distribution. A request whose judgments are
/// all boxes asked for nothing measurable: nothing is recorded or refused.
pub fn note_unmeasurable_probabilities(request: &Request, provider: &str) -> Result<(), Lm15Error> {
    if request_measurable_judgments(request).is_empty() {
        return Ok(());
    }
    match request.config.probabilities {
        Some(ProbabilityPolicy::Required) => Err(refusal(provider, "config.probabilities", "this wire cannot measure a distribution over declared keys; the program depends on required probabilities")),
        Some(ProbabilityPolicy::IfAvailable) => crate::adaptation::adapt("config.probabilities", AdaptationAction::Dropped,
            Some(Value::String("if_available".into())), None, "this wire returns a pick only, not measured probabilities"),
        _ => Ok(()),
    }
}

pub fn anthropic_schema(schema: &JsonObject, found: &[Judgment]) -> JsonObject {
    let mut out = schema.clone();
    if let Some(props) = out.get_mut("properties").and_then(Value::as_object_mut) {
        for j in found {
            if let Some(prop) = props.get_mut(&j.name).and_then(Value::as_object_mut) {
                if prop.get("anyOf").is_some_and(Value::is_array) {
                    if let Some(kind) = prop.remove("type") {
                        for branch in prop.get_mut("anyOf").unwrap().as_array_mut().unwrap() {
                            if let Some(branch) = branch.as_object_mut() {
                                branch.entry("type").or_insert_with(|| kind.clone());
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

pub fn gemini_schema(schema: &JsonObject, found: &[Judgment]) -> JsonObject {
    let mut out = schema.clone();
    if let Some(props) = out.get_mut("properties").and_then(Value::as_object_mut) {
        for j in found {
            let Some(prop) = props.get_mut(&j.name).and_then(Value::as_object_mut) else {
                continue;
            };
            if matches!(j.kind, JudgmentKind::Boolean | JudgmentKind::Box)
                || !prop.contains_key("anyOf")
            {
                continue;
            }
            prop.remove("anyOf");
            prop.insert(
                "type".into(),
                Value::from(if j.ordered() { "integer" } else { "string" }),
            );
            prop.insert(
                "enum".into(),
                Value::Array(j.keys.iter().map(|k| j.value_for_key(k)).collect()),
            );
            if j.keys.iter().any(|k| {
                j.titles.get(k).is_some_and(|s| !s.is_empty())
                    || j.descriptions.get(k).is_some_and(|s| !s.is_empty())
            }) {
                let options = j
                    .keys
                    .iter()
                    .map(|k| {
                        match (
                            j.titles.get(k).filter(|s| !s.is_empty()),
                            j.descriptions.get(k).filter(|s| !s.is_empty()),
                        ) {
                            (Some(title), Some(desc)) => format!("{k} = {title}: {desc}"),
                            (Some(text), None) | (None, Some(text)) => format!("{k} = {text}"),
                            _ => k.clone(),
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("; ");
                let head = prop
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let description = format!(
                    "{}{}{}: {}",
                    head,
                    if head.is_empty() { "" } else { " " },
                    if j.ordered() { "Levels" } else { "Options" },
                    options
                );
                prop.insert("description".into(), description.trim().into());
            }
        }
    }
    out
}

pub fn data_part_from_text(text: &str, found: &[Judgment]) -> Option<DataPart> {
    if found.is_empty() {
        return None;
    }
    let value: Value = serde_json::from_str(text.trim()).ok()?;
    value.is_object().then(|| DataPart::new(value))
}

pub fn replace_text_with_data(parts: &mut [Part], found: &[Judgment]) {
    let indices: Vec<usize> = parts
        .iter()
        .enumerate()
        .filter_map(|(i, p)| matches!(p, Part::Text(_)).then_some(i))
        .collect();
    if indices.len() != 1 {
        return;
    }
    let i = indices[0];
    if let Part::Text(text) = &parts[i] {
        if let Some(mut data) = data_part_from_text(&text.text, found) {
            data.continuation = text.continuation.clone();
            parts[i] = Part::Data(data);
        }
    }
}

pub fn normalize_logprobs(scores: &[(String, f64)]) -> Result<JsonObject, ValidationError> {
    if scores.is_empty()
        || scores
            .iter()
            .any(|(_, v)| v.is_nan() || *v == f64::INFINITY)
    {
        return Err(ValidationError::value(
            "candidate log-scores must be non-empty and contain neither NaN nor positive infinity",
        ));
    }
    let top = scores
        .iter()
        .map(|(_, v)| *v)
        .fold(f64::NEG_INFINITY, f64::max);
    if !top.is_finite() {
        return Err(ValidationError::value(
            "every declared candidate has zero likelihood; no distribution can be normalized",
        ));
    }
    let total: f64 = scores.iter().map(|(_, v)| (v - top).exp()).sum();
    Ok(scores
        .iter()
        .map(|(k, v)| (k.clone(), Value::from((v - top).exp() / total)))
        .collect())
}

pub fn expected_level(distribution: &JsonObject) -> Result<f64, ValidationError> {
    distribution
        .iter()
        .map(|(k, p)| {
            let index = k
                .parse::<u64>()
                .map_err(|_| ValidationError::value("expected level requires integer keys"))?;
            let probability = p
                .as_f64()
                .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
                .ok_or_else(|| ValidationError::value("probability must be finite in [0, 1]"))?;
            Ok(index as f64 * probability)
        })
        .sum()
}

/// Bare choice keys; use `choice_described` to attach descriptions.
pub fn choice(
    instruction: impl Into<String>,
    options: impl IntoIterator<Item = impl Into<String>>,
) -> Result<JsonObject, ValidationError> {
    choice_described(
        instruction,
        options.into_iter().map(|key| (key.into(), None)),
    )
}

pub fn choice_described(
    instruction: impl Into<String>,
    options: impl IntoIterator<Item = (String, Option<String>)>,
) -> Result<JsonObject, ValidationError> {
    let options: Vec<_> = options.into_iter().collect();
    if options.is_empty()
        || options.iter().any(|(k, _)| k.is_empty())
        || options
            .iter()
            .map(|(k, _)| k)
            .collect::<BTreeSet<_>>()
            .len()
            != options.len()
    {
        return Err(ValidationError::value(
            "choice needs non-empty, unique option keys",
        ));
    }
    let mut out = json!({"type":"string", "description":instruction.into()})
        .as_object()
        .unwrap()
        .clone();
    if options.iter().all(|(_, d)| d.is_none()) {
        out.insert(
            "enum".into(),
            Value::Array(options.into_iter().map(|(k, _)| k.into()).collect()),
        );
    } else {
        out.insert(
            "anyOf".into(),
            Value::Array(
                options
                    .into_iter()
                    .map(|(k, desc)| {
                        let mut branch = json!({"const":k});
                        if let Some(desc) = desc.filter(|s| !s.is_empty()) {
                            branch["description"] = desc.into();
                        }
                        branch
                    })
                    .collect(),
            ),
        );
    }
    Ok(out)
}

/// A bounding-box judgment: where in the request's image is what
/// `instruction` describes (changes/2026-10-15-bounding-box.md).
///
/// The answer is `{top, bottom, left, right}`, each a fraction from 0 to 1 of
/// the image's height (`top`, `bottom`) or width (`left`, `right`), measured
/// from the top-left corner; multiply by the image's size in pixels to draw
/// or crop. There is no "not found" answer: ask a [`yes_no`] beside it. On
/// `typesafe` this is Jev's Box question and needs Jev 1.14 (`jev-preview` or
/// `jev-1.14.0`); elsewhere it is ordinary structured output.
///
/// The edges carry no `minimum`/`maximum`: Anthropic refuses both on a
/// number (HTTP 400, receipted 2026-10-15); the range is in each edge's
/// description, which every wire passes to the model. (Not `box`: a
/// reserved word in Rust.)
pub fn bounding_box(instruction: impl Into<String>) -> JsonObject {
    let edges: JsonObject = BOX_EDGES
        .iter()
        .zip(BOX_EDGE_DESCRIPTIONS)
        .map(|(edge, description)| {
            (
                (*edge).to_string(),
                json!({"type":"number", "description":description}),
            )
        })
        .collect();
    json!({"type":"object", "description":instruction.into(), "properties":edges,
           "required":BOX_EDGES, "additionalProperties":false})
    .as_object()
    .unwrap()
    .clone()
}

pub fn yes_no(instruction: impl Into<String>) -> JsonObject {
    json!({"type":"boolean", "description":instruction.into()})
        .as_object()
        .unwrap()
        .clone()
}

/// Ordered level descriptions, lowest first; use `score_named` for titles.
pub fn score(
    instruction: impl Into<String>,
    levels: impl IntoIterator<Item = impl Into<String>>,
) -> Result<JsonObject, ValidationError> {
    score_named(
        instruction,
        levels
            .into_iter()
            .map(|description| (None, description.into())),
    )
}
pub fn score_named(
    instruction: impl Into<String>,
    levels: impl IntoIterator<Item = (Option<String>, String)>,
) -> Result<JsonObject, ValidationError> {
    let levels: Vec<_> = levels.into_iter().collect();
    if !(2..=MAX_ORDERED_LEVELS).contains(&levels.len()) {
        return Err(ValidationError::value("score needs 2 to 10 ordered levels"));
    }
    let branches: Vec<Value> = levels
        .into_iter()
        .enumerate()
        .map(|(i, (title, description))| {
            let mut branch = json!({"const":i});
            if let Some(title) = title.filter(|s| !s.is_empty()) {
                branch["title"] = title.into();
            }
            if !description.is_empty() {
                branch["description"] = description.into();
            }
            branch
        })
        .collect();
    Ok(
        json!({"type":"integer", "description":instruction.into(), "anyOf":branches})
            .as_object()
            .unwrap()
            .clone(),
    )
}

pub fn judgments(properties: JsonObject) -> Result<JsonObject, ValidationError> {
    judgments_named(properties, "judgments", true)
}
pub fn judgments_named(
    properties: JsonObject,
    name: &str,
    strict: bool,
) -> Result<JsonObject, ValidationError> {
    if properties.is_empty() || name.is_empty() {
        return Err(ValidationError::value(
            "judgments requires properties and a non-empty name",
        ));
    }
    let required: Vec<String> = properties.keys().cloned().collect();
    Ok(json!({"type":"json_schema", "name":name, "strict":strict, "schema":{
        "type":"object", "properties":properties, "required":required, "additionalProperties":false
    }}).as_object().unwrap().clone())
}

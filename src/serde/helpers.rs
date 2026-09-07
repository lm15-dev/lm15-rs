//! Reading and writing helpers for the canonical JSON form.
//!
//! `Reader` implements the lenient read side (INV-040..048 and the Number
//! rule coercions INV-007/INV-008); `Obj` implements the write side
//! (the omission rule: a typed serializer omits its own `null`/`""`/`[]`/`{}`
//! optional fields, at its own level only — docs/serde-rules.md).

use serde_json::{Number, Value};

use crate::types::{JsonObject, ValidationError};

pub type VResult<T> = Result<T, ValidationError>;

/// Python `str(x)` for a JSON scalar that stands in for text (INV-041,
/// INV-047).
pub fn scalar_to_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        other => other.to_string(),
    }
}

/// True for the values the omission rule drops.
pub fn is_empty(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::String(s) => s.is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
        _ => false,
    }
}

/// Coerce a JSON number to an int under the Number rule (INV-007):
/// same-valued floats coerce, other floats and bools are rejected.
pub fn int_from_value(value: &Value, field: &str) -> VResult<i64> {
    match value {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                return Ok(i);
            }
            if let Some(f) = n.as_f64() {
                if f.fract() == 0.0 && f.abs() < 9.0e18 {
                    return Ok(f as i64);
                }
            }
            Err(ValidationError::type_error(format!(
                "{field} must be an int"
            )))
        }
        _ => Err(ValidationError::type_error(format!(
            "{field} must be an int"
        ))),
    }
}

/// Coerce a JSON number to a float under the Number rule (INV-008); bools
/// never coerce.
pub fn float_from_value(value: &Value, field: &str) -> VResult<f64> {
    match value {
        Value::Number(n) => n
            .as_f64()
            .ok_or_else(|| ValidationError::type_error(format!("{field} must be numeric"))),
        _ => Err(ValidationError::type_error(format!(
            "{field} must be numeric"
        ))),
    }
}

/// A typed view of one JSON object being read.
pub struct Reader<'a> {
    obj: &'a JsonObject,
    type_name: &'static str,
}

impl<'a> Reader<'a> {
    pub fn new(value: &'a Value, type_name: &'static str) -> VResult<Self> {
        match value {
            Value::Object(obj) => Ok(Reader { obj, type_name }),
            _ => Err(ValidationError::type_error(format!(
                "{type_name} must be a JSON object"
            ))),
        }
    }

    fn label(&self, key: &str) -> String {
        format!("{}.{}", self.type_name, key)
    }

    /// The value at `key`; absent and `null` both read as `None`.
    pub fn get(&self, key: &str) -> Option<&'a Value> {
        match self.obj.get(key) {
            None | Some(Value::Null) => None,
            Some(v) => Some(v),
        }
    }

    pub fn has(&self, key: &str) -> bool {
        self.obj.contains_key(key)
    }

    pub fn req(&self, key: &str) -> VResult<&'a Value> {
        self.get(key)
            .ok_or_else(|| ValidationError::type_error(format!("{} is required", self.label(key))))
    }

    pub fn req_str(&self, key: &str) -> VResult<String> {
        self.as_str(self.req(key)?, key)
    }

    pub fn opt_str(&self, key: &str) -> VResult<Option<String>> {
        self.get(key).map(|v| self.as_str(v, key)).transpose()
    }

    /// INV-040: a missing text field reads as `""`.
    pub fn str_or_empty(&self, key: &str) -> VResult<String> {
        Ok(self.opt_str(key)?.unwrap_or_default())
    }

    pub fn str_or(&self, key: &str, default: &str) -> VResult<String> {
        Ok(self.opt_str(key)?.unwrap_or_else(|| default.to_string()))
    }

    fn as_str(&self, value: &Value, key: &str) -> VResult<String> {
        match value {
            Value::String(s) => Ok(s.clone()),
            _ => Err(ValidationError::type_error(format!(
                "{} must be a string",
                self.label(key)
            ))),
        }
    }

    /// A non-negative int field (Number rule coercion, then `>= 0`).
    pub fn opt_u64(&self, key: &str) -> VResult<Option<u64>> {
        match self.get(key) {
            None => Ok(None),
            Some(v) => {
                let i = int_from_value(v, &self.label(key))?;
                u64::try_from(i).map(Some).map_err(|_| {
                    ValidationError::value(format!("{} must be >= 0", self.label(key)))
                })
            }
        }
    }

    pub fn req_u64(&self, key: &str) -> VResult<u64> {
        self.opt_u64(key)?
            .ok_or_else(|| ValidationError::type_error(format!("{} is required", self.label(key))))
    }

    /// A non-negative int field WITHOUT float coercion (fields outside the
    /// INV-007 list: `index`, `seconds`, `progress`).
    pub fn opt_strict_u64(&self, key: &str) -> VResult<Option<u64>> {
        match self.get(key) {
            None => Ok(None),
            Some(Value::Number(n)) if n.is_u64() => Ok(n.as_u64()),
            Some(_) => Err(ValidationError::value(format!(
                "{} must be a non-negative int",
                self.label(key)
            ))),
        }
    }

    pub fn opt_i64(&self, key: &str) -> VResult<Option<i64>> {
        self.get(key)
            .map(|v| int_from_value(v, &self.label(key)))
            .transpose()
    }

    pub fn opt_f64(&self, key: &str) -> VResult<Option<f64>> {
        self.get(key)
            .map(|v| float_from_value(v, &self.label(key)))
            .transpose()
    }

    pub fn req_f64(&self, key: &str) -> VResult<f64> {
        float_from_value(self.req(key)?, &self.label(key))
    }

    pub fn opt_bool(&self, key: &str) -> VResult<Option<bool>> {
        match self.get(key) {
            None => Ok(None),
            Some(Value::Bool(b)) => Ok(Some(*b)),
            Some(_) => Err(ValidationError::type_error(format!(
                "{} must be a bool",
                self.label(key)
            ))),
        }
    }

    pub fn bool_or(&self, key: &str, default: bool) -> VResult<bool> {
        Ok(self.opt_bool(key)?.unwrap_or(default))
    }

    /// An opaque object field; a present non-object is a `TypeError`.
    pub fn opt_object(&self, key: &str) -> VResult<Option<JsonObject>> {
        match self.get(key) {
            None => Ok(None),
            Some(Value::Object(o)) => Ok(Some(o.clone())),
            Some(_) => Err(ValidationError::type_error(format!(
                "{} must be a JSON object",
                self.label(key)
            ))),
        }
    }

    /// An opaque object that defaults to `{}` when absent (INV-045).
    pub fn object_or_empty(&self, key: &str) -> VResult<JsonObject> {
        Ok(self.opt_object(key)?.unwrap_or_default())
    }

    /// A telemetry nest read leniently (INV-042 second half): a non-object
    /// value reads as absent.
    pub fn lenient_object(&self, key: &str) -> Option<&'a JsonObject> {
        match self.get(key) {
            Some(Value::Object(o)) => Some(o),
            _ => None,
        }
    }

    pub fn opt_array(&self, key: &str) -> VResult<Option<&'a Vec<Value>>> {
        match self.get(key) {
            None => Ok(None),
            Some(Value::Array(a)) => Ok(Some(a)),
            Some(_) => Err(ValidationError::type_error(format!(
                "{} must be a list",
                self.label(key)
            ))),
        }
    }

    pub fn array_or_empty(&self, key: &str) -> VResult<&'a [Value]> {
        Ok(self.opt_array(key)?.map_or(&[][..], Vec::as_slice))
    }

    /// A list of strings; a bare string coerces to a one-element list
    /// (INV-020). Absent reads as empty (INV-048).
    pub fn str_list(&self, key: &str) -> VResult<Vec<String>> {
        match self.get(key) {
            None => Ok(Vec::new()),
            Some(Value::String(s)) => Ok(vec![s.clone()]),
            Some(Value::Array(items)) => items.iter().map(|item| self.as_str(item, key)).collect(),
            Some(_) => Err(ValidationError::type_error(format!(
                "{} must be a list of strings",
                self.label(key)
            ))),
        }
    }
}

/// An output object under construction.
#[derive(Default)]
pub struct Obj(JsonObject);

impl Obj {
    pub fn new() -> Self {
        Obj(JsonObject::new())
    }

    pub fn typed(type_name: &str) -> Self {
        let mut obj = Obj::new();
        obj.set("type", Value::String(type_name.to_string()));
        obj
    }

    /// Always emitted (required-with-shape).
    pub fn set(&mut self, key: &str, value: impl Into<Value>) -> &mut Self {
        self.0.insert(key.to_string(), value.into());
        self
    }

    /// Emitted when present (omit-null; empty strings survive).
    pub fn opt<T: Into<Value>>(&mut self, key: &str, value: Option<T>) -> &mut Self {
        if let Some(v) = value {
            self.set(key, v);
        }
        self
    }

    /// The omission rule: dropped when `null`/`""`/`[]`/`{}`.
    pub fn omit_empty(&mut self, key: &str, value: impl Into<Value>) -> &mut Self {
        let value = value.into();
        if !is_empty(&value) {
            self.set(key, value);
        }
        self
    }

    pub fn omit_empty_opt<T: Into<Value>>(&mut self, key: &str, value: Option<T>) -> &mut Self {
        if let Some(v) = value {
            self.omit_empty(key, v);
        }
        self
    }

    pub fn finish(self) -> Value {
        Value::Object(self.0)
    }
}

pub fn float(value: f64) -> Value {
    Number::from_f64(value).map_or(Value::Null, Value::Number)
}

pub fn strings(values: &[String]) -> Value {
    Value::Array(values.iter().cloned().map(Value::String).collect())
}

pub fn opt_strings(values: Option<&[u64]>) -> Option<Value> {
    values.map(|v| Value::Array(v.iter().map(|n| Value::from(*n)).collect()))
}

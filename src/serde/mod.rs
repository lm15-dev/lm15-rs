//! Canonical JSON serde (docs/serde-rules.md, spec/invariants.md
//! § Serde leniency).
//!
//! Every canonical type implements [`Canonical`]: `from_json` reads the
//! lenient canonical form and runs the constructor invariants (INV-046);
//! `to_json` writes the one canonical wire form under the omission rule.
//! The impls are written by hand over `serde_json::Value` because the
//! leniency rules (INV-040..048), the Number rule coercions and the
//! omission rule are not expressible with derive attributes. `Serialize`
//! and `Deserialize` delegate to them, so `serde_json::to_string(&x)` and
//! `serde_json::from_str::<Request>(s)` work as the api-family expects.

mod config;
mod credential;
mod endpoints;
mod helpers;
mod kinds;
mod live;
mod message;
mod model_info;
mod parts;
mod request;
mod stream;

use serde_json::Value;

use crate::types::ValidationError;

pub use kinds::{roundtrip, validate, KINDS};

/// The canonical JSON form of a type.
pub trait Canonical: Sized {
    /// Read the canonical form leniently, then validate (INV-046).
    fn from_json(value: &Value) -> Result<Self, ValidationError>;

    /// Write the one canonical wire form (omission rule).
    fn to_json(&self) -> Value;

    /// `from_json` on a JSON text.
    fn from_json_str(text: &str) -> Result<Self, ValidationError> {
        let value: Value = serde_json::from_str(text)
            .map_err(|e| ValidationError::value(format!("invalid JSON: {e}")))?;
        Self::from_json(&value)
    }

    /// `to_json` as compact JSON text.
    fn to_json_string(&self) -> String {
        self.to_json().to_string()
    }
}

/// `Serialize`/`Deserialize` through the canonical form.
macro_rules! impl_serde_via_canonical {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl ::serde::Serialize for $ty {
                fn serialize<S: ::serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                    $crate::serde::Canonical::to_json(self).serialize(serializer)
                }
            }

            impl<'de> ::serde::Deserialize<'de> for $ty {
                fn deserialize<D: ::serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                    let value = ::serde_json::Value::deserialize(deserializer)?;
                    <$ty as $crate::serde::Canonical>::from_json(&value).map_err(::serde::de::Error::custom)
                }
            }
        )+
    };
}

pub(crate) use impl_serde_via_canonical;

//! The AUTH-2 credential value on the wire (the `credential` serde kind;
//! `lm15/credentials.py` `credential_from_dict` / `credential_to_dict`).
//!
//! | Kind | Canonical JSON |
//! |---|---|
//! | `ApiKey` | `{"kind":"api_key","value"}` |
//! | `BearerToken` | `{"kind":"bearer_token","value","expires_at"?}` |
//! | `AwsCredentials` | `{"kind":"aws","access_key_id","secret_access_key","session_token"?,"expires_at"?}` |
//!
//! `expires_at` is RFC 3339 on the wire, read with the reference's
//! leniency and written as whole seconds in UTC with a `Z` suffix; in
//! memory it is Unix seconds (stated in the README). Absent fields are
//! omitted, never null.

use serde_json::Value;

use super::helpers::{Obj, Reader, VResult};
use super::{impl_serde_via_canonical, Canonical};
use crate::auth::{format_rfc3339, Credential};
use crate::types::{parse_rfc3339_lenient, ValidationError};

impl Canonical for Credential {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "Credential")?;
        let expires_at = r
            .opt_str("expires_at")?
            .as_deref()
            .map(parse_rfc3339_lenient)
            .transpose()?;
        let result = match r.req_str("kind")?.as_str() {
            "api_key" => Credential::api_key(r.req_str("value")?),
            "bearer_token" => Credential::bearer_token(r.req_str("value")?, expires_at),
            "aws" => Credential::aws(
                r.req_str("access_key_id")?,
                r.req_str("secret_access_key")?,
                r.opt_str("session_token")?,
                expires_at,
            ),
            _ => return Err(ValidationError::value("unknown credential kind")),
        };
        // The constructors refuse empty strings (INV-046); the refusal is
        // the native ValueError on the protocol.
        result.map_err(|e| ValidationError::value(e.to_string()))
    }

    fn to_json(&self) -> Value {
        // AUTH-2: absent fields are omitted, never null.
        let mut o = Obj::new();
        o.set("kind", self.kind().as_str());
        match self {
            Credential::ApiKey { value } => {
                o.set("value", value.as_str());
            }
            Credential::BearerToken { value, expires_at } => {
                o.set("value", value.as_str());
                o.opt("expires_at", expires_at.map(format_rfc3339));
            }
            Credential::AwsCredentials {
                access_key_id,
                secret_access_key,
                session_token,
                expires_at,
            } => {
                o.set("access_key_id", access_key_id.as_str())
                    .set("secret_access_key", secret_access_key.as_str());
                o.opt("session_token", session_token.clone());
                o.opt("expires_at", expires_at.map(format_rfc3339));
            }
        }
        o.finish()
    }
}

impl_serde_via_canonical!(Credential);

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn expires_at_offsets_normalize_to_z() {
        let token = Credential::from_json(
            &json!({"kind": "bearer_token", "value": "t", "expires_at": "2026-09-03T15:00:00+02:00"}),
        )
        .unwrap();
        assert_eq!(token.expires_at(), Some(1_788_440_400));
        assert_eq!(
            token.to_json(),
            json!({"kind": "bearer_token", "value": "t", "expires_at": "2026-09-03T13:00:00Z"})
        );
    }

    #[test]
    fn null_expires_at_reads_as_absent_and_non_string_is_a_type_error() {
        let token = Credential::from_json(
            &json!({"kind": "bearer_token", "value": "t", "expires_at": null}),
        )
        .unwrap();
        assert_eq!(token.expires_at(), None);
        assert!(token.to_json().get("expires_at").is_none());
        let err =
            Credential::from_json(&json!({"kind": "bearer_token", "value": "t", "expires_at": 5}))
                .unwrap_err();
        assert_eq!(err.type_name(), "TypeError");
    }

    #[test]
    fn refusals_are_value_errors() {
        for value in [
            json!({"kind": "password", "value": "x"}),
            json!({"kind": "api_key", "value": ""}),
            json!({"kind": "aws", "access_key_id": "a", "secret_access_key": "s", "session_token": ""}),
            json!({"kind": "bearer_token", "value": "t", "expires_at": "yesterday"}),
            // The F1 reproduction: a multibyte character at byte 10.
            json!({"kind": "bearer_token", "value": "t", "expires_at": "2026-09-0\u{e9}X"}),
        ] {
            let err = Credential::from_json(&value).unwrap_err();
            assert_eq!(err.type_name(), "ValueError", "{value}");
        }
        assert_eq!(
            Credential::from_json(&json!({"kind": 5}))
                .unwrap_err()
                .type_name(),
            "TypeError"
        );
    }

    #[test]
    fn serde_json_goes_through_the_canonical_form() {
        let text = r#"{"kind":"aws","access_key_id":"a","secret_access_key":"s"}"#;
        let aws: Credential = serde_json::from_str(text).unwrap();
        let back: Value = serde_json::from_str(&serde_json::to_string(&aws).unwrap()).unwrap();
        assert_eq!(back, serde_json::from_str::<Value>(text).unwrap());
    }
}

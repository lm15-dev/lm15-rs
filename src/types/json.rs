//! Shared validation primitives: the validation error, opaque JSON objects,
//! base64 shape checks (INV-012) and RFC 3339 normalization.

use std::fmt;

use serde_json::Value;

/// An opaque JSON object payload (INV-001, INV-002). `serde_json::Value`
/// can only hold finite numbers and string keys, so strict-JSON validity
/// holds by construction; payloads are stored as given and never rewritten.
pub type JsonObject = serde_json::Map<String, Value>;

/// The two kinds of construction failure the spec names: a well-typed but
/// forbidden value (`ValueError`, e.g. INV-016, INV-037, INV-044) and a
/// value of the wrong shape (`TypeError`, e.g. INV-003, INV-042).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationKind {
    Value,
    Type,
}

/// A constructor or `from_json` refusal. Not an [`crate::errors::Lm15Error`]:
/// the reference raises Python's native `ValueError`/`TypeError` here, and
/// the vet protocol reports those native names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationError {
    pub kind: ValidationKind,
    pub message: String,
}

impl ValidationError {
    pub fn value(message: impl Into<String>) -> Self {
        ValidationError {
            kind: ValidationKind::Value,
            message: message.into(),
        }
    }

    pub fn type_error(message: impl Into<String>) -> Self {
        ValidationError {
            kind: ValidationKind::Type,
            message: message.into(),
        }
    }

    /// The exception name the vet protocol reports for a native refusal.
    pub fn type_name(&self) -> &'static str {
        match self.kind {
            ValidationKind::Value => "ValueError",
            ValidationKind::Type => "TypeError",
        }
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.type_name(), self.message)
    }
}

impl std::error::Error for ValidationError {}

pub type VResult<T> = Result<T, ValidationError>;

// ─── Text helpers ────────────────────────────────────────────────────

pub(crate) fn non_empty(value: &str, field: &str) -> VResult<()> {
    if value.is_empty() {
        return Err(ValidationError::value(format!("{field} cannot be empty")));
    }
    Ok(())
}

pub(crate) fn opt_non_empty(value: Option<&String>, field: &str) -> VResult<()> {
    match value {
        Some(v) => non_empty(v, field),
        None => Ok(()),
    }
}

pub(crate) fn positive(value: Option<u64>, field: &str) -> VResult<()> {
    if value == Some(0) {
        return Err(ValidationError::value(format!("{field} must be > 0")));
    }
    Ok(())
}

// ─── Base64 (INV-012) ────────────────────────────────────────────────

/// Return the base64 payload of `data`: a data-URI prefix is stripped and
/// embedded whitespace is collapsed. Never decodes.
pub fn base64_payload(data: &str) -> String {
    let payload = match data.strip_prefix("data:") {
        Some(_) => match data.find(";base64,") {
            Some(i) => &data[i + ";base64,".len()..],
            None => data,
        },
        None => data,
    };
    if payload.chars().any(char::is_whitespace) {
        payload.split_whitespace().collect()
    } else {
        payload.to_string()
    }
}

/// `^[A-Za-z0-9+/]*={0,2}$` with `len % 4 == 0` on the payload (INV-012).
pub fn is_base64_shaped(data: &str) -> bool {
    let payload = base64_payload(data);
    if !payload.len().is_multiple_of(4) {
        return false;
    }
    let bytes = payload.as_bytes();
    let trimmed = bytes.iter().rev().take_while(|b| **b == b'=').count();
    if trimmed > 2 {
        return false;
    }
    bytes[..bytes.len() - trimmed]
        .iter()
        .all(|b| b.is_ascii_alphanumeric() || *b == b'+' || *b == b'/')
}

pub(crate) fn validate_base64(part_type: &str, data: &str) -> VResult<()> {
    if data.is_empty() {
        return Err(ValidationError::value(format!(
            "{part_type}.data cannot be empty"
        )));
    }
    if !is_base64_shaped(data) {
        return Err(ValidationError::value(format!(
            "{part_type}.data must be a valid base64 string"
        )));
    }
    Ok(())
}

// ─── Base64 codec (FileUploadRequest bytes on the wire) ──────────────

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub(crate) fn base64_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(B64[(n >> 18) as usize & 63] as char);
        out.push(B64[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            B64[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            B64[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

pub(crate) fn base64_decode(text: &str) -> VResult<Vec<u8>> {
    let payload = base64_payload(text);
    let bad = || ValidationError::value("bytes_data must be a valid base64 string");
    let clean: Vec<u8> = payload.bytes().filter(|b| *b != b'=').collect();
    let mut out = Vec::with_capacity(clean.len() * 3 / 4);
    let mut acc: u32 = 0;
    let mut bits = 0u32;
    for b in clean {
        let v = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return Err(bad()),
        };
        acc = (acc << 6) | u32::from(v);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

// ─── RFC 3339 (credential expiry) ────────────────────────────────────

/// Normalize an RFC 3339 timestamp to `YYYY-MM-DDTHH:MM:SSZ` (whole
/// seconds, UTC). Accepts `Z`/`z`, a numeric offset, or no offset (UTC).
pub fn normalize_rfc3339(text: &str) -> VResult<String> {
    let bad = || ValidationError::value(format!("invalid RFC 3339 timestamp: {text:?}"));
    let s = text.trim();
    if s.len() < 10 {
        return Err(bad());
    }
    let (date, rest) = s.split_at(10);
    let mut parts = date.split('-');
    let year: i64 = parts.next().and_then(|p| p.parse().ok()).ok_or_else(bad)?;
    let month: i64 = parts.next().and_then(|p| p.parse().ok()).ok_or_else(bad)?;
    let day: i64 = parts.next().and_then(|p| p.parse().ok()).ok_or_else(bad)?;
    if parts.next().is_some() || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return Err(bad());
    }
    let mut hour = 0i64;
    let mut minute = 0i64;
    let mut second = 0i64;
    let mut offset_minutes = 0i64;
    if !rest.is_empty() {
        let time = rest
            .strip_prefix('T')
            .or_else(|| rest.strip_prefix('t'))
            .or_else(|| rest.strip_prefix(' '))
            .ok_or_else(bad)?;
        let (clock, offset) = match time.find(['Z', 'z', '+', '-']) {
            Some(i) => (&time[..i], &time[i..]),
            None => (time, ""),
        };
        let clock = clock.split('.').next().unwrap_or(clock);
        let mut fields = clock.split(':');
        hour = fields.next().and_then(|p| p.parse().ok()).ok_or_else(bad)?;
        minute = fields.next().and_then(|p| p.parse().ok()).ok_or_else(bad)?;
        second = fields
            .next()
            .map(|p| p.parse::<i64>().map_err(|_| bad()))
            .transpose()?
            .unwrap_or(0);
        if fields.next().is_some() || hour > 23 || minute > 59 || second > 60 {
            return Err(bad());
        }
        match offset {
            "" | "Z" | "z" => {}
            _ => {
                let sign = if offset.starts_with('-') { -1 } else { 1 };
                let body = &offset[1..];
                let (oh, om) = match body.split_once(':') {
                    Some((h, m)) => (h, m),
                    None if body.len() == 4 => body.split_at(2),
                    None => (body, "0"),
                };
                let oh: i64 = oh.parse().map_err(|_| bad())?;
                let om: i64 = om.parse().map_err(|_| bad())?;
                offset_minutes = sign * (oh * 60 + om);
            }
        }
    }
    let days = days_from_civil(year, month, day);
    let total = days * 86_400 + hour * 3600 + minute * 60 + second - offset_minutes * 60;
    let (days, secs) = (total.div_euclid(86_400), total.rem_euclid(86_400));
    let (y, m, d) = civil_from_days(days);
    Ok(format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        secs / 3600,
        (secs % 3600) / 60,
        secs % 60
    ))
}

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_shape() {
        assert!(is_base64_shaped("aGk="));
        assert!(is_base64_shaped("data:image/png;base64,aGk="));
        assert!(is_base64_shaped("aG\nk="));
        assert!(!is_base64_shaped("aGk"));
        assert!(!is_base64_shaped("aG!="));
    }

    #[test]
    fn base64_codec_roundtrip() {
        assert_eq!(base64_encode(b"hello"), "aGVsbG8=");
        assert_eq!(base64_decode("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(base64_encode(b"hi"), "aGk=");
    }

    #[test]
    fn rfc3339_normalizes_to_utc_z() {
        assert_eq!(
            normalize_rfc3339("2026-09-03T13:00:00Z").unwrap(),
            "2026-09-03T13:00:00Z"
        );
        assert_eq!(
            normalize_rfc3339("2026-09-03T15:30:00.250+02:30").unwrap(),
            "2026-09-03T13:00:00Z"
        );
        assert_eq!(
            normalize_rfc3339("2026-01-01T00:30:00-01:00").unwrap(),
            "2026-01-01T01:30:00Z"
        );
        assert!(normalize_rfc3339("yesterday").is_err());
    }
}

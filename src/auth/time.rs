//! RFC 3339 timestamps as Unix seconds, without a date crate.
//!
//! AUTH-2: `expires_at` is RFC 3339; the reference formats whole seconds in
//! UTC with a `Z` suffix (`lm15/credentials.py` `format_rfc3339`). This
//! module parses any RFC 3339 offset and always formats UTC.

use std::time::{SystemTime, UNIX_EPOCH};

/// Now, in Unix seconds.
pub(crate) fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0)
}

/// Now, in Unix milliseconds (the borrowed CLI files store `expiresAt` /
/// `expires` in milliseconds; AUTH-8).
pub(crate) fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

/// Parses `YYYY-MM-DDTHH:MM:SS[.frac](Z|±HH:MM)` into Unix seconds (UTC).
/// Fractional seconds are truncated. Returns `None` on any malformed input.
pub fn parse_rfc3339(text: &str) -> Option<i64> {
    let text = text.trim();
    let bytes = text.as_bytes();
    if bytes.len() < 20 {
        return None;
    }
    let digits = |from: usize, to: usize| -> Option<i64> {
        let slice = text.get(from..to)?;
        if !slice.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        slice.parse().ok()
    };
    let year = digits(0, 4)?;
    let month = digits(5, 7)?;
    let day = digits(8, 10)?;
    let hour = digits(11, 13)?;
    let minute = digits(14, 16)?;
    let second = digits(17, 19)?;
    if bytes[4] != b'-' || bytes[7] != b'-' || bytes[10] != b'T' && bytes[10] != b't' {
        return None;
    }
    if bytes[13] != b':' || bytes[16] != b':' {
        return None;
    }
    let mut index = 19;
    if bytes.get(index) == Some(&b'.') {
        index += 1;
        let start = index;
        while bytes.get(index).is_some_and(u8::is_ascii_digit) {
            index += 1;
        }
        if index == start {
            return None;
        }
    }
    let offset_seconds = match bytes.get(index) {
        Some(b'Z') | Some(b'z') if index + 1 == bytes.len() => 0,
        Some(sign @ (b'+' | b'-')) if index + 6 == bytes.len() && bytes[index + 3] == b':' => {
            let oh = digits(index + 1, index + 3)?;
            let om = digits(index + 4, index + 6)?;
            if oh > 23 || om > 59 {
                return None;
            }
            let total = oh * 3600 + om * 60;
            if *sign == b'+' {
                total
            } else {
                -total
            }
        }
        _ => return None,
    };
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    if hour > 23 || minute > 59 || second > 60 {
        return None;
    }
    let days = days_from_civil(year, month, day);
    Some(days * 86_400 + hour * 3600 + minute * 60 + second - offset_seconds)
}

/// Formats Unix seconds as `YYYY-MM-DDTHH:MM:SSZ`.
pub fn format_rfc3339(unix: i64) -> String {
    let days = unix.div_euclid(86_400);
    let rest = unix.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rest / 3600,
        (rest % 3600) / 60,
        rest % 60
    )
}

// Howard Hinnant's proleptic-Gregorian day arithmetic.
pub(crate) fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_formats_the_vector_timestamps() {
        for text in [
            "2026-09-03T13:00:00Z",
            "2026-09-03T18:00:00Z",
            "1970-01-01T00:00:00Z",
        ] {
            let unix = parse_rfc3339(text).unwrap();
            assert_eq!(format_rfc3339(unix), text);
        }
        assert_eq!(parse_rfc3339("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(parse_rfc3339("2000-03-01T00:00:00Z"), Some(951_868_800));
    }

    #[test]
    fn offsets_and_fractions_normalize_to_utc() {
        assert_eq!(
            parse_rfc3339("2026-09-03T15:00:00+02:00"),
            parse_rfc3339("2026-09-03T13:00:00Z")
        );
        assert_eq!(
            parse_rfc3339("2026-09-03T13:00:00.250Z"),
            parse_rfc3339("2026-09-03T13:00:00Z")
        );
        assert_eq!(parse_rfc3339("2026-09-03T13:00:00"), None);
        assert_eq!(parse_rfc3339("2026-13-03T13:00:00Z"), None);
        assert_eq!(parse_rfc3339("not a date"), None);
    }
}

//! RFC 3986 percent-encoding with the same contract as Python's
//! `urllib.parse.quote` / `unquote`: the unreserved bytes (ASCII letters,
//! digits, `-_.~`) are never encoded, the caller names the other safe
//! bytes, everything else is `%XX` over its UTF-8 bytes. Decoding keeps a
//! malformed `%` literally.

/// `quote(text, safe)`.
pub fn encode(text: &str, safe: &[u8]) -> String {
    encode_bytes(text.as_bytes(), safe)
}

pub(crate) fn encode_bytes(bytes: &[u8], safe: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len());
    for &byte in bytes {
        if byte.is_ascii_alphanumeric() || b"-_.~".contains(&byte) || safe.contains(&byte) {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

/// `unquote(text)` to bytes: `%XX` becomes the byte; a lone or malformed
/// `%` stays as it is.
pub(crate) fn decode_bytes(text: &str) -> Vec<u8> {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let (hi, lo) = (bytes[i + 1], bytes[i + 2]);
            if hi.is_ascii_hexdigit() && lo.is_ascii_hexdigit() {
                let digit = |b: u8| (b as char).to_digit(16).unwrap_or(0) as u8;
                out.push(digit(hi) * 16 + digit(lo));
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    out
}

/// Percent-decoding of a query component (`%XX` sequences and `+` as a
/// space); malformed escapes are kept verbatim.
pub(crate) fn decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hex = &text[i + 1..i + 3];
                match u8::from_str_radix(hex, 16) {
                    Ok(byte) => {
                        out.push(byte);
                        i += 3;
                    }
                    Err(_) => {
                        out.push(b'%');
                        i += 1;
                    }
                }
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            byte => {
                out.push(byte);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_like_quote() {
        assert_eq!(encode("a b/c", b"/"), "a%20b/c");
        assert_eq!(encode("ሴ", b""), "%E1%88%B4");
        assert_eq!(encode("-._~", b""), "-._~");
        assert_eq!(encode("$delete", b"-_.~"), "%24delete");
    }

    #[test]
    fn decodes_like_unquote() {
        let decode = |s: &str| String::from_utf8(decode_bytes(s)).unwrap();
        assert_eq!(decode("%E1%88%B4"), "ሴ");
        assert_eq!(decode("a%2"), "a%2");
        assert_eq!(decode("100%"), "100%");
        assert_eq!(decode("%zz"), "%zz");
        assert_eq!(decode("%+1"), "%+1");
        assert_eq!(decode_bytes("%41%42"), b"AB");
    }
}

//! RS256 (RSASSA-PKCS1-v1_5 / SHA-256) for the two JWT assertions of
//! spec/auth.md AUTH-11 `jwt-rs256`: Google service accounts and Entra
//! certificate credentials. The reference (`lm15/cloud/rs256.py`) signs
//! in pure Python and states it is not hardened; this port signs with
//! `aws-lc-rs`, the provider rustls already compiles — constant-time,
//! blinded, audited — so that trade-off does not carry over.
//!
//! PEM handled, and only that: `PRIVATE KEY` (PKCS#8), `RSA PRIVATE KEY`
//! (PKCS#1), `CERTIFICATE`. Encrypted PEM and PKCS#12 are not parsed; the
//! error names `openssl pkey` / `openssl pkcs12 -nodes`.
//!
//! The compact JWS serialization lm15 pins: compact JSON, keys in the
//! caller's order, base64url without padding — the vectors compare the
//! JWT byte for byte.

use aws_lc_rs::rand::SystemRandom;
use aws_lc_rs::signature::{RsaKeyPair, RSA_PKCS1_SHA256};
use serde_json::Value;

use crate::auth::AuthError;

/// base64url without padding.
pub fn b64url(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let mut buf = [0u8; 3];
        buf[..chunk.len()].copy_from_slice(chunk);
        let n = u32::from_be_bytes([0, buf[0], buf[1], buf[2]]);
        for i in 0..=chunk.len() {
            out.push(ALPHABET[((n >> (18 - 6 * i)) & 63) as usize] as char);
        }
    }
    out
}

fn not_configured(message: String, hint: Option<&str>) -> AuthError {
    AuthError::NotConfigured {
        provider: None,
        message,
        hint: hint.map(str::to_string),
    }
}

/// The DER bytes of the first `-----BEGIN {label}-----` block.
pub fn pem_block(text: &str, label: &str) -> Result<Option<Vec<u8>>, AuthError> {
    let head = format!("-----BEGIN {label}-----");
    let tail = format!("-----END {label}-----");
    let Some(start) = text.find(&head) else {
        return Ok(None);
    };
    let after = start + head.len();
    let Some(end) = text[after..].find(&tail) else {
        return Err(not_configured(
            format!("PEM block {label:?} has no END line"),
            None,
        ));
    };
    let body: String = text[after..after + end].split_whitespace().collect();
    crate::types::base64_decode(&body)
        .map(Some)
        .map_err(|err| not_configured(format!("PEM block {label:?}: {}", err.message), None))
}

/// The private key of a PEM text: PKCS#8, else PKCS#1.
pub fn load_private_key(pem: &str) -> Result<RsaKeyPair, AuthError> {
    if pem.contains("ENCRYPTED") {
        return Err(not_configured(
            "encrypted private keys are not supported; decrypt with `openssl pkey`".into(),
            Some("openssl pkey -in key.pem -out key-plain.pem"),
        ));
    }
    if let Some(der) = pem_block(pem, "PRIVATE KEY")? {
        return RsaKeyPair::from_pkcs8(&der)
            .map_err(|err| not_configured(format!("PKCS#8 private key rejected: {err}"), None));
    }
    if let Some(der) = pem_block(pem, "RSA PRIVATE KEY")? {
        return RsaKeyPair::from_der(&der)
            .map_err(|err| not_configured(format!("PKCS#1 private key rejected: {err}"), None));
    }
    Err(not_configured(
        "no PRIVATE KEY / RSA PRIVATE KEY block in the PEM (PKCS#12? use `openssl pkcs12 -nodes`)"
            .into(),
        None,
    ))
}

/// The DER of the PEM's `CERTIFICATE` block.
pub fn certificate_der(pem: &str) -> Result<Vec<u8>, AuthError> {
    pem_block(pem, "CERTIFICATE")?
        .ok_or_else(|| not_configured("no CERTIFICATE block in the PEM".into(), None))
}

/// SHA-1 of the DER certificate, base64url: the Entra `x5t` header.
pub fn x5t(der: &[u8]) -> String {
    let digest = aws_lc_rs::digest::digest(&aws_lc_rs::digest::SHA1_FOR_LEGACY_USE_ONLY, der);
    b64url(digest.as_ref())
}

/// RSASSA-PKCS1-v1_5 / SHA-256 over `message`.
pub fn sign(key: &RsaKeyPair, message: &[u8]) -> Result<Vec<u8>, AuthError> {
    let mut signature = vec![0u8; key.public_modulus_len()];
    key.sign(
        &RSA_PKCS1_SHA256,
        &SystemRandom::new(),
        message,
        &mut signature,
    )
    .map_err(|_| not_configured("RS256 signing failed".into(), None))?;
    Ok(signature)
}

/// The compact JWS: `b64url(header).b64url(payload).b64url(signature)`,
/// each JSON compact with keys in the given order.
pub fn jwt_encode(header: &Value, payload: &Value, key: &RsaKeyPair) -> Result<String, AuthError> {
    let signing_input = format!(
        "{}.{}",
        b64url(
            serde_json::to_string(header)
                .expect("serializes")
                .as_bytes()
        ),
        b64url(
            serde_json::to_string(payload)
                .expect("serializes")
                .as_bytes()
        )
    );
    let signature = sign(key, signing_input.as_bytes())?;
    Ok(format!("{signing_input}.{}", b64url(&signature)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn b64url_is_unpadded() {
        assert_eq!(b64url(b""), "");
        assert_eq!(b64url(b"f"), "Zg");
        assert_eq!(b64url(b"fo"), "Zm8");
        assert_eq!(b64url(b"foo"), "Zm9v");
        assert_eq!(b64url(&[0xfb, 0xff]), "-_8");
    }
}

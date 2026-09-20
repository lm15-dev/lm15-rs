//! Cloud doors (spec/auth.md AUTH-10 `host`, AUTH-11 signing).
//!
//! - [`hosts`]: settings resolution, base-URL rendering and the host's
//!   closed set of request rewrites (`lm15/cloud/hosts.py`).
//! - [`sigv4`]: AWS Signature Version 4 (`lm15/cloud/sigv4.py`), pinned
//!   byte for byte by `auth/sigv4-vectors.json`.
//!
//! `chains` (native builds) resolves the vendor chains or deterministic named
//! subsets and records their provenance. The pure signer accepts an explicit
//! `AwsCredentials` value, including in codec-only builds.

#[cfg(feature = "native")]
pub mod chains;
pub mod hosts;
pub mod ini;
pub mod percent;
#[cfg(feature = "native")]
pub mod rs256;
pub mod sigv4;

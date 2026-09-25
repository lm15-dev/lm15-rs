# Releasing lm15 (Rust)

## The first version: once, by hand

crates.io's trusted publishing is configured per crate, so the crate must
exist first. The maintainer publishes `1.0.0-rc.1` from a clean checkout:

```bash
git status --short            # nothing
cargo login                   # a token from https://crates.io/settings/tokens (scope: publish-new)
cargo publish --locked        # packages, builds the package on its own, uploads
```

Then revoke that token, and on crates.io: crate **lm15** → Settings → Trusted
Publishing → GitHub, repository `lm15-dev/lm15-rs`, workflow `release.yml`,
environment `crates-io`. From then on no token is needed.

Done 2026-09-25: `1.0.0-rc.1` published (tag `v1.0.0-rc.1`), the token
revoked, and trusted publishing configured (`lm15-dev/lm15-rs`, `release.yml`,
environment `crates-io`). Once one release has gone through GitHub, turn on
"Require trusted publishing for all new versions" in the crate's settings, so
no token can ever publish it again.

## Every later version: from GitHub

1. Set `version` in `Cargo.toml`, commit, push; `ci` must be green (tests on
   Linux, macOS and Windows, the wasm build, the contract harness, the
   packaged crate).
2. Create a GitHub release with tag `v<version>` (mark release candidates as
   pre-releases). The `release` workflow checks the tag, runs everything
   again, and waits for the maintainer's approval (environment `crates-io`).

## History

`v1.0.0` was once a git tag here (2026-06-11, an early prototype). It was
never published to crates.io; the tag is now `prototype-2026-06-11`.

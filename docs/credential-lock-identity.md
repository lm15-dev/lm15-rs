# Credential lock identity and upgrade safety

Rust now resolves the same credential-path identity as Python and TypeScript,
including missing targets behind directory aliases and dangling symlinks.
Symlinks are resolved before following `..`; errors, non-Unicode paths and loops
fail closed rather than hashing a different spelling. Forty total symlink
expansions are allowed. Ordinary existing POSIX hashes remain unchanged.

`auth::lock_path_for` now returns `Result<PathBuf, AuthError>`: handle the error
with `?` rather than relying on an empty or guessed path. The explicit
`try_lock_path_for` spelling is also exported. This is an intentional source
API change in the pre-1.0 SDK; acquisition already uses the fallible path.

Windows identities strip ordinary verbatim prefixes, normalize separators and
use whole-string Unicode lowercase. Case-sensitive Windows directories may be
overlocked. Missing non-ASCII components and non-ASCII UNC roots are refused
rather than risking inconsistent locks. Provision such local files first or use
an ASCII missing suffix. Device paths/names, alternate streams, drive-relative
paths and trailing dots/spaces are refused.

**Upgrade all participants together.** Stop old credential refresh/store writers
before restarting corrected versions on Windows or affected POSIX aliases.
Mixed versions can choose different lock filenames. Use the same lock directory;
never delete a live lock file to resolve contention.

Path identity does not unify hardlinks, bind mounts, mapped-drive/UNC aliases or
changing filesystem namespaces, and foreign CLIs do not cooperate. A valid
current directory is required. Explicit absolute paths avoid home discovery
changes; shared expansion supports the current user's `~` only. Runtime Unicode
table differences and actual cross-platform exclusion remain unverified.

Regression test sources were added. No builds or tests were run in this pass.

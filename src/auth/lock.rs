//! AUTH-4 storage semantics: the cross-process credential-file lock and
//! the atomic private write (the reference's `lm15/_authlock.py`).
//!
//! - [`FileLock`]: an advisory, exclusive, cross-process lock scoped to a
//!   credential file's canonical path. Lock files live in an lm15-owned
//!   directory ([`lock_dir`]: `$LM15_LOCK_DIR`, else
//!   `$XDG_CACHE_HOME/lm15/locks`, else `~/.cache/lm15/locks`; AUTH-8),
//!   never next to the guarded file — `~/.claude` and `~/.codex` are
//!   foreign territory.
//! - [`write_private_json_atomic`]: temp file created 0600 in the target's
//!   directory → write → fsync → rename over the target → fsync the
//!   directory. A reader sees the complete old file or the complete new
//!   file, never a partial one.
//!
//! Stated limitations (the spec's, not oversights): the lock is
//! cooperative among lm15 processes — the Claude Code and Codex CLIs do
//! not take it, and AUTH-3's double-checked re-read is the mitigation;
//! `flock` is unreliable on some network filesystems.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde_json::Value;
use sha2::{Digest, Sha256};

use super::error::AuthError;

/// The reference's default: wait this long for a sibling's refresh.
pub const DEFAULT_LOCK_TIMEOUT: Duration = Duration::from_secs(60);
const LOCK_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// The lm15-owned lock directory (AUTH-8), from `env`.
pub fn lock_dir(env: &dyn Fn(&str) -> Option<String>) -> Option<PathBuf> {
    let value = |key: &str| env(key).filter(|v| !v.is_empty());
    if let Some(dir) = value("LM15_LOCK_DIR") {
        return Some(PathBuf::from(dir));
    }
    let base = match value("XDG_CACHE_HOME") {
        Some(cache_home) => PathBuf::from(cache_home),
        None => PathBuf::from(value("HOME")?).join(".cache"),
    };
    Some(base.join("lm15").join("locks"))
}

/// Resolve the shared lock filename, or explain why identity cannot be established.
///
/// This is now fallible: callers must handle inaccessible/ambiguous paths rather
/// than treating an empty or guessed filename as a valid lock identity.
pub fn lock_path_for(lock_dir: &Path, path: &Path) -> Result<PathBuf, AuthError> {
    try_lock_path_for(lock_dir, path)
}

/// AUTH-4: `<lock_dir>/<sha256(canonical path)[:32]>.lock`, even for missing
/// leaves behind directory aliases or dangling links. Ordinary POSIX hashes
/// are unchanged. Only NotFound is recoverable during component resolution.
///
/// WINDOWS MIGRATION: old and new SDK processes MUST NOT overlap. Verbatim
/// prefix removal and Unicode lowercase change old lock filenames. Like
/// Python normcase, case-sensitive Windows directories intentionally overlock.
pub fn try_lock_path_for(lock_dir: &Path, path: &Path) -> Result<PathBuf, AuthError> {
    let result = (|| -> std::io::Result<PathBuf> {
        let canonical = real_path_allow_missing(path)?;
        let text = lock_path_text(&canonical)?;
        let key = if cfg!(windows) {
            windows_identity_key(text)?
        } else {
            text.to_owned()
        };
        let digest = Sha256::digest(key.as_bytes());
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        Ok(lock_dir.join(format!("{}.lock", &hex[..32])))
    })();
    result.map_err(|e| io_error("resolve credential lock identity", path, &e))
}

fn invalid_lock_path(message: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message)
}

fn lock_path_text(path: &Path) -> std::io::Result<&str> {
    path.to_str()
        .ok_or_else(|| invalid_lock_path("credential lock path must be Unicode"))
}

fn strip_windows_lock_verbatim(value: &str) -> std::io::Result<String> {
    let value = value.replace('/', "\\");
    if value
        .get(..8)
        .is_some_and(|s| s.eq_ignore_ascii_case("\\\\?\\UNC\\"))
    {
        return Ok(format!("\\\\{}", &value[8..]));
    }
    if let Some(rest) = value.strip_prefix("\\\\?\\") {
        let bytes = rest.as_bytes();
        if bytes.len() < 3
            || !bytes[0].is_ascii_alphabetic()
            || bytes[1] != b':'
            || bytes[2] != b'\\'
        {
            return Err(invalid_lock_path(
                "unsupported Windows credential path namespace",
            ));
        }
        return Ok(rest.to_owned());
    }
    Ok(value)
}

// Pure post-resolution spelling, testable on POSIX. str::to_lowercase matches
// Python str.lower / JS toLowerCase (including context-sensitive Greek sigma),
// unlike per-character lowercase, ASCII-only lowercase, or Unicode casefold.
fn windows_identity_key(canonical: &str) -> std::io::Result<String> {
    Ok(strip_windows_lock_verbatim(canonical)?.to_lowercase())
}

fn lock_path_parts(
    value: &str,
    base: &Path,
) -> std::io::Result<(PathBuf, std::collections::VecDeque<String>)> {
    if value.contains('\0') {
        return Err(invalid_lock_path("NUL in credential path"));
    }
    let mut resolved = base.to_path_buf();
    let value = if cfg!(windows) {
        strip_windows_lock_verbatim(value)?
    } else {
        value.to_owned()
    };
    let mut tail = value.as_str();
    let separator = if cfg!(windows) { '\\' } else { '/' };
    if cfg!(windows) {
        if value.starts_with("\\\\.\\") {
            return Err(invalid_lock_path(
                "unsupported Windows credential path namespace",
            ));
        }
        if let Some(unc) = value.strip_prefix("\\\\") {
            let mut names = unc.splitn(3, '\\');
            let server = names.next().unwrap_or("");
            let share = names.next().unwrap_or("");
            if [server, share].iter().any(|s| {
                s.is_empty()
                    || !s.is_ascii()
                    || *s == "."
                    || *s == ".."
                    || s.contains(['?', ':'])
                    || s.ends_with(['.', ' '])
            }) {
                return Err(invalid_lock_path(
                    "unsupported Windows credential path root",
                ));
            }
            resolved = PathBuf::from(format!("\\\\{server}\\{share}\\"));
            tail = &value[2 + server.len() + 1 + share.len()..];
        } else if value.as_bytes().get(1) == Some(&b':') {
            let bytes = value.as_bytes();
            if !bytes[0].is_ascii_alphabetic() || bytes.get(2) != Some(&b'\\') {
                return Err(invalid_lock_path(
                    "drive-relative credential path is unsupported",
                ));
            }
            resolved = PathBuf::from(&value[..3]);
            tail = &value[3..];
        } else if value.starts_with('\\') {
            resolved = base
                .ancestors()
                .last()
                .ok_or_else(|| invalid_lock_path("missing Windows drive"))?
                .to_path_buf();
        }
    } else if value.starts_with('/') {
        resolved = PathBuf::from("/");
    }
    let names: std::collections::VecDeque<String> = tail
        .split(separator)
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .collect();
    if cfg!(windows) {
        resolved = PathBuf::from(strip_windows_lock_verbatim(lock_path_text(
            &std::fs::canonicalize(&resolved)?,
        )?)?);
        for name in &names {
            if name == "." || name == ".." {
                continue;
            }
            let stem = name.split('.').next().unwrap_or("").to_uppercase();
            let numbered_device = (stem.starts_with("COM") || stem.starts_with("LPT"))
                && stem.get(3..).is_some_and(|s| {
                    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "¹", "²", "³"].contains(&s)
                });
            if name.ends_with(['.', ' '])
                || name.contains(':')
                || numbered_device
                || ["CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"].contains(&stem.as_str())
            {
                return Err(invalid_lock_path(
                    "unsupported Windows credential path component",
                ));
            }
        }
    }
    Ok((resolved, names))
}

fn real_path_allow_missing(path: &Path) -> std::io::Result<PathBuf> {
    let text = lock_path_text(path)?;
    // Concatenate raw text: PathBuf::push on Windows verbatim paths can
    // normalize symlink/.. before the filesystem has seen the link.
    let expanded;
    let text =
        if text == "~" || text.starts_with("~/") || (cfg!(windows) && text.starts_with("~\\")) {
            let home = if cfg!(windows) {
                std::env::var_os("USERPROFILE").or_else(|| {
                    let mut drive = std::env::var_os("HOMEDRIVE")?;
                    drive.push(std::env::var_os("HOMEPATH")?);
                    Some(drive)
                })
            } else {
                std::env::var_os("HOME")
            };
            let home = home
                .ok_or_else(|| invalid_lock_path("no home directory for credential lock path"))?;
            expanded = format!("{}{}", lock_path_text(Path::new(&home))?, &text[1..]);
            expanded.as_str()
        } else {
            if text.starts_with('~') {
                return Err(invalid_lock_path(
                    "named-user home expansion is unsupported for credential locks",
                ));
            }
            text
        };
    let cwd = std::fs::canonicalize(std::env::current_dir()?)?;
    let cwd = if cfg!(windows) {
        PathBuf::from(strip_windows_lock_verbatim(lock_path_text(&cwd)?)?)
    } else {
        cwd
    };
    let (mut resolved, mut pending) = lock_path_parts(text, &cwd)?;
    let mut links = 0;
    while let Some(name) = pending.pop_front() {
        if name == "." {
            continue;
        }
        if name == ".." {
            resolved.pop();
            continue;
        }
        let candidate = resolved.join(&name);
        let metadata = match std::fs::symlink_metadata(&candidate) {
            Ok(metadata) => metadata,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Filesystem upcase tables are not Unicode lowercase (e.g.
                // sigma/final sigma). No on-disk spelling: fail closed.
                if cfg!(windows) && !name.is_ascii() {
                    return Err(invalid_lock_path(
                        "missing non-ASCII Windows credential path component is unsupported",
                    ));
                }
                resolved = candidate;
                continue;
            }
            Err(e) => return Err(e),
        };
        if metadata.file_type().is_symlink() {
            links += 1;
            if links > 40 {
                return Err(invalid_lock_path("too many credential path symlinks"));
            }
            let linked = std::fs::read_link(&candidate)?;
            let (base, mut names) = lock_path_parts(lock_path_text(&linked)?, &resolved)?;
            names.append(&mut pending);
            pending = names;
            resolved = base;
        } else {
            if !pending.is_empty() && !metadata.is_dir() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotADirectory,
                    "credential path ancestor is not a directory",
                ));
            }
            resolved = if cfg!(windows) {
                PathBuf::from(strip_windows_lock_verbatim(lock_path_text(
                    &std::fs::canonicalize(&candidate)?,
                )?)?)
            } else {
                candidate
            };
        }
    }
    Ok(resolved)
}

/// A held lock; released on drop.
pub struct FileLock {
    file: File,
    lock_path: PathBuf,
}

impl FileLock {
    /// Try once, without waiting.
    pub fn try_acquire(lock_dir: &Path, guarded: &Path) -> Result<Option<FileLock>, AuthError> {
        let lock_path = try_lock_path_for(lock_dir, guarded)?;
        std::fs::create_dir_all(lock_dir)
            .map_err(|e| io_error("create lock directory", lock_dir, &e))?;
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true).truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = options
            .open(&lock_path)
            .map_err(|e| io_error("open lock file", &lock_path, &e))?;
        match file.try_lock() {
            Ok(()) => Ok(Some(FileLock { file, lock_path })),
            Err(std::fs::TryLockError::WouldBlock) => Ok(None),
            Err(std::fs::TryLockError::Error(e)) => Err(io_error("lock", &lock_path, &e)),
        }
    }

    /// Wait up to `timeout` (polling, so one code path serves every
    /// platform), then [`AuthError::LockTimeout`] — a local, transient
    /// condition, deliberately not the credential's fault (AUTH-6).
    pub async fn acquire(
        lock_dir: &Path,
        guarded: &Path,
        timeout: Duration,
    ) -> Result<FileLock, AuthError> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(lock) = FileLock::try_acquire(lock_dir, guarded)? {
                return Ok(lock);
            }
            if Instant::now() >= deadline {
                return Err(AuthError::LockTimeout {
                    path: guarded.display().to_string(),
                    lock_path: try_lock_path_for(lock_dir, guarded)?.display().to_string(),
                    timeout_secs: timeout.as_secs(),
                });
            }
            tokio::time::sleep(LOCK_POLL_INTERVAL).await;
        }
    }

    /// The blocking form, for callers outside a runtime.
    pub fn acquire_blocking(
        lock_dir: &Path,
        guarded: &Path,
        timeout: Duration,
    ) -> Result<FileLock, AuthError> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(lock) = FileLock::try_acquire(lock_dir, guarded)? {
                return Ok(lock);
            }
            if Instant::now() >= deadline {
                return Err(AuthError::LockTimeout {
                    path: guarded.display().to_string(),
                    lock_path: try_lock_path_for(lock_dir, guarded)?.display().to_string(),
                    timeout_secs: timeout.as_secs(),
                });
            }
            std::thread::sleep(LOCK_POLL_INTERVAL);
        }
    }

    pub fn lock_path(&self) -> &Path {
        &self.lock_path
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

impl std::fmt::Debug for FileLock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileLock")
            .field("lock_path", &self.lock_path)
            .finish()
    }
}

/// Atomically replace `path` with `data` as private (0600), pretty JSON
/// with a trailing newline (the reference's `json.dumps(indent=2)`).
pub fn write_private_json_atomic(path: &Path, data: &Value) -> Result<(), AuthError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent).map_err(|e| io_error("create directory", parent, &e))?;
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "credentials".into());
    let temp = parent.join(format!(
        ".{name}.{}.{}.tmp",
        std::process::id(),
        unique_suffix()
    ));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let result = (|| -> std::io::Result<()> {
        let mut file = options.open(&temp)?;
        let mut text = serde_json::to_string_pretty(data)?;
        text.push('\n');
        file.write_all(text.as_bytes())?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&temp, path)?;
        Ok(())
    })();
    if let Err(e) = result {
        let _ = std::fs::remove_file(&temp);
        return Err(io_error("write", path, &e));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        // Best effort, like the reference: the temp file was born 0600.
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
        if let Ok(dir) = File::open(parent) {
            let _ = dir.sync_all();
        }
    }
    Ok(())
}

fn unique_suffix() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

fn io_error(what: &str, path: &Path, error: &std::io::Error) -> AuthError {
    AuthError::NotConfigured {
        provider: None,
        message: format!("could not {what} {}: {}", path.display(), error.kind()),
        hint: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tempdir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "lm15-lock-{}-{}",
            std::process::id(),
            unique_suffix()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn lock_dir_follows_auth_8() {
        let env = |pairs: &'static [(&str, &str)]| {
            move |k: &str| {
                pairs
                    .iter()
                    .find(|(key, _)| *key == k)
                    .map(|(_, v)| v.to_string())
            }
        };
        assert_eq!(
            lock_dir(&env(&[("HOME", "/h")])),
            Some(PathBuf::from("/h/.cache/lm15/locks"))
        );
        assert_eq!(
            lock_dir(&env(&[("HOME", "/h"), ("XDG_CACHE_HOME", "/x")])),
            Some(PathBuf::from("/x/lm15/locks"))
        );
        assert_eq!(
            lock_dir(&env(&[("HOME", "/h"), ("LM15_LOCK_DIR", "/l")])),
            Some(PathBuf::from("/l"))
        );
        assert_eq!(lock_dir(&env(&[])), None);
    }

    #[test]
    fn lock_path_is_stable_and_outside_the_guarded_directory() {
        let dir = tempdir();
        let guarded = dir.join("foreign/.credentials.json");
        let a = lock_path_for(&dir.join("locks"), &guarded).unwrap();
        let b = lock_path_for(&dir.join("locks"), &guarded).unwrap();
        assert_eq!(a, b);
        assert!(a.starts_with(dir.join("locks")));
        assert!(!a.starts_with(dir.join("foreign")));
        assert!(a.extension().is_some_and(|e| e == "lock"));
    }

    struct IdentityDir(PathBuf);

    impl Drop for IdentityDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[cfg(any(unix, windows))]
    fn identity_symlink(target: &Path, link: &Path, directory: bool) {
        #[cfg(unix)]
        {
            let _ = directory;
            std::os::unix::fs::symlink(target, link).unwrap();
        }
        // Windows CI needs Developer Mode or symlink privilege. Do not turn
        // inability to create a link into a passing test without assertions.
        #[cfg(windows)]
        if directory {
            std::os::windows::fs::symlink_dir(target, link).unwrap();
        } else {
            std::os::windows::fs::symlink_file(target, link).unwrap();
        }
    }

    #[test]
    fn lock_identity_keeps_existing_realpath_hash() {
        let fixture = IdentityDir(tempdir());
        let target = fixture.0.join("credentials.json");
        std::fs::write(&target, "{}").unwrap();
        let canonical = std::fs::canonicalize(&target).unwrap();
        let text = canonical.to_str().unwrap();
        let key = if cfg!(windows) {
            windows_identity_key(text).unwrap()
        } else {
            text.to_owned()
        };
        let hex: String = Sha256::digest(key.as_bytes())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(
            try_lock_path_for(&fixture.0.join("locks"), &target)
                .unwrap()
                .file_name()
                .unwrap(),
            std::ffi::OsStr::new(&format!("{}.lock", &hex[..32]))
        );
    }

    #[test]
    #[cfg(any(unix, windows))]
    fn lock_identity_missing_leaf_nested_directories_and_dangling_aliases() {
        let fixture = IdentityDir(tempdir());
        let dir = &fixture.0;
        let real = dir.join("real");
        let alias = dir.join("alias");
        let locks = dir.join("locks");
        std::fs::create_dir(&real).unwrap();
        identity_symlink(&real, &alias, true);
        for suffix in [
            "credentials.json",
            "one/two/credentials.json",
            "one/./two/../credentials.json",
        ] {
            assert_eq!(
                try_lock_path_for(&locks, Path::new(&format!("{}/{suffix}", alias.display())))
                    .unwrap(),
                try_lock_path_for(&locks, Path::new(&format!("{}/{suffix}", real.display())))
                    .unwrap()
            );
        }
        assert_eq!(
            try_lock_path_for(
                &locks,
                Path::new(&format!(
                    "{}/one/./two/../credentials.json",
                    alias.display()
                ))
            )
            .unwrap(),
            try_lock_path_for(&locks, &real.join("one/credentials.json")).unwrap()
        );
        assert!(!real.join("one").exists());
        let before = try_lock_path_for(&locks, &alias.join("credentials.json")).unwrap();
        std::fs::write(real.join("credentials.json"), "{}").unwrap();
        assert_eq!(
            try_lock_path_for(&locks, &alias.join("credentials.json")).unwrap(),
            before
        );
        identity_symlink(
            Path::new("real/missing/credentials.json"),
            &dir.join("dangling"),
            false,
        );
        identity_symlink(Path::new("dangling"), &dir.join("leaf-alias"), false);
        identity_symlink(
            &real.join("missing/credentials.json"),
            &dir.join("absolute"),
            false,
        );
        let expected = try_lock_path_for(&locks, &real.join("missing/credentials.json")).unwrap();
        for name in ["dangling", "leaf-alias", "absolute"] {
            assert_eq!(
                try_lock_path_for(&locks, &dir.join(name)).unwrap(),
                expected
            );
        }
    }

    #[test]
    #[cfg(any(unix, windows))]
    fn lock_identity_dotdot_follows_links_before_normalizing() {
        let fixture = IdentityDir(tempdir());
        let dir = &fixture.0;
        let locks = dir.join("locks");
        std::fs::create_dir_all(dir.join("real/child")).unwrap();
        identity_symlink(Path::new("real/child"), &dir.join("alias"), true);
        let expected = try_lock_path_for(&locks, &dir.join("real/credentials.json")).unwrap();
        for suffix in [
            "alias/../credentials.json",
            "absent/../alias/./../credentials.json",
        ] {
            // Raw text preserves .. even if the temp root is Windows verbatim.
            let target = PathBuf::from(format!("{}/{suffix}", dir.display()));
            assert_eq!(try_lock_path_for(&locks, &target).unwrap(), expected);
        }
        assert_ne!(
            expected,
            try_lock_path_for(&locks, &dir.join("credentials.json")).unwrap()
        );
        identity_symlink(
            Path::new("real/not-created/child"),
            &dir.join("dangling-dir"),
            true,
        );
        assert_eq!(
            try_lock_path_for(
                &locks,
                Path::new(&format!(
                    "{}/dangling-dir/../credentials.json",
                    dir.display()
                ))
            )
            .unwrap(),
            try_lock_path_for(&locks, &dir.join("real/not-created/credentials.json")).unwrap()
        );
    }

    #[test]
    #[cfg(any(unix, windows))]
    fn lock_identity_resolution_failures_never_acquire_a_guessed_lock() {
        let fixture = IdentityDir(tempdir());
        let dir = &fixture.0;
        let locks = dir.join("locks");
        identity_symlink(Path::new("loop-b"), &dir.join("loop-a"), false);
        identity_symlink(Path::new("loop-a"), &dir.join("loop-b"), false);
        let guarded = dir.join("loop-a");
        assert!(try_lock_path_for(&locks, &guarded).is_err());
        assert!(lock_path_for(&locks, &guarded).is_err());
        assert!(FileLock::try_acquire(&locks, &guarded).is_err());
        assert!(!locks.exists());
        std::fs::write(dir.join("file"), "{}").unwrap();
        for suffix in ["child", "../credentials.json", "./credentials.json"] {
            assert!(try_lock_path_for(
                &locks,
                Path::new(&format!("{}/file/{suffix}", dir.display()))
            )
            .is_err());
        }
    }

    #[test]
    #[cfg(unix)]
    fn lock_identity_rejects_non_unicode_symlink_targets() {
        use std::os::unix::ffi::OsStrExt;
        let fixture = IdentityDir(tempdir());
        let target = Path::new(std::ffi::OsStr::from_bytes(b"invalid-\xff"));
        identity_symlink(target, &fixture.0.join("invalid"), false);
        assert!(try_lock_path_for(&fixture.0.join("locks"), &fixture.0.join("invalid")).is_err());
    }

    #[test]
    fn windows_lock_key_vectors_run_on_every_platform() {
        for (raw, expected) in [
            (r"C:\Users\MAX\Auth.JSON", r"c:\users\max\auth.json"),
            (r"\\?\C:\Users\MAX\Auth.JSON", r"c:\users\max\auth.json"),
            ("C:/Users/MAX/Auth.JSON", r"c:\users\max\auth.json"),
            (
                r"\\?\UNC\Server\Share\Auth.JSON",
                r"\\server\share\auth.json",
            ),
            (r"\\SERVER\SHARE\Auth.JSON", r"\\server\share\auth.json"),
            (r"\\?\UNC\Server\Share\", r"\\server\share\"),
            (
                r"C:\ÉCOLE\İ\ΟΣ\Auth.JSON",
                "c:\\école\\i\u{307}\\ος\\auth.json",
            ),
        ] {
            assert_eq!(windows_identity_key(raw).unwrap(), expected);
        }
    }

    #[test]
    #[cfg(windows)]
    fn windows_lock_identity_prefix_case_and_missing_leaf() {
        let fixture = IdentityDir(tempdir());
        let real = fixture.0.join("MixedCase");
        std::fs::create_dir(&real).unwrap();
        let target = strip_windows_lock_verbatim(real.join("Auth.JSON").to_str().unwrap()).unwrap();
        let locks = fixture.0.join("locks");
        let expected = try_lock_path_for(&locks, Path::new(&target)).unwrap();
        for alias in [
            target.to_uppercase(),
            format!("\\\\?\\{target}"),
            target.replace('\\', "/"),
        ] {
            assert_eq!(
                try_lock_path_for(&locks, Path::new(&alias)).unwrap(),
                expected
            );
        }
        std::fs::write(&target, "{}").unwrap();
        assert_eq!(
            try_lock_path_for(&locks, Path::new(&target.to_uppercase())).unwrap(),
            expected
        );
        let unicode_file = real.join("ΟΣ.json");
        std::fs::write(&unicode_file, "{}").unwrap();
        assert_eq!(
            try_lock_path_for(&locks, &real.join("οσ.json")).unwrap(),
            try_lock_path_for(&locks, &unicode_file).unwrap()
        );
        // Final sigma: whether the volume's upcase table folds "ς" to "Σ"
        // varies (the GitHub Windows runner's NTFS does not: "ος.json" is
        // another, missing file there). The safety property is never a
        // DIFFERENT lock for what may be the same file: the same lock, or a
        // refusal (a missing non-ASCII name).
        if let Ok(other) = try_lock_path_for(&locks, &real.join("ος.json")) {
            assert_eq!(other, try_lock_path_for(&locks, &unicode_file).unwrap());
        }
    }

    #[test]
    #[cfg(windows)]
    fn windows_lock_identity_ambiguous_missing_names_fail_closed() {
        let fixture = IdentityDir(tempdir());
        let locks = fixture.0.join("locks");
        for name in ["ΟΣ.json", "οσ.json", "trailing. ", "file:stream", "NUL"] {
            assert!(try_lock_path_for(&locks, &fixture.0.join(name)).is_err());
        }
        for target in [r"C:relative.json", r"\\.\NUL", r"\\?\GLOBALROOT\Device\X"] {
            assert!(try_lock_path_for(&locks, Path::new(target)).is_err());
        }
    }

    #[test]
    fn lock_is_exclusive_across_open_descriptions_and_times_out() {
        let dir = tempdir();
        let locks = dir.join("locks");
        let guarded = dir.join("c.json");
        let held = FileLock::acquire_blocking(&locks, &guarded, Duration::from_secs(1)).unwrap();
        assert!(FileLock::try_acquire(&locks, &guarded).unwrap().is_none());
        let err =
            FileLock::acquire_blocking(&locks, &guarded, Duration::from_millis(120)).unwrap_err();
        assert!(matches!(err, AuthError::LockTimeout { .. }));
        assert_eq!(err.class_name(), "LockTimeoutError");
        assert_eq!(err.code(), "lock_timeout");
        assert!(err.to_string().contains("c.json"), "{err}");
        drop(held);
        assert!(FileLock::try_acquire(&locks, &guarded).unwrap().is_some());
    }

    #[test]
    fn atomic_write_is_private_and_complete() {
        let dir = tempdir();
        let path = dir.join("nested/store.json");
        write_private_json_atomic(&path, &json!({"xai": {"access": "a"}})).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.ends_with('\n'));
        assert_eq!(
            serde_json::from_str::<Value>(&text).unwrap(),
            json!({"xai": {"access": "a"}})
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }
        // No temp file left behind.
        let leftovers: Vec<_> = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty());
        // Overwrite is a replacement, not a merge.
        write_private_json_atomic(&path, &json!({"other": 1})).unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&std::fs::read_to_string(&path).unwrap()).unwrap(),
            json!({"other": 1})
        );
    }
}

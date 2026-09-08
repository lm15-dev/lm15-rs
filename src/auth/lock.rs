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

/// The lock file guarding `path`: `<lock_dir>/<sha256(canonical path)[:32]>.lock`.
/// Keyed by the absolute path, resolved through symlinks when the file
/// exists, so every lm15 process agrees on one lock even before the
/// credential file is created.
pub fn lock_path_for(lock_dir: &Path, path: &Path) -> PathBuf {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    };
    let canonical = std::fs::canonicalize(&absolute).unwrap_or(absolute);
    let digest = Sha256::digest(canonical.to_string_lossy().as_bytes());
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    lock_dir.join(format!("{}.lock", &hex[..32]))
}

/// A held lock; released on drop.
pub struct FileLock {
    file: File,
    lock_path: PathBuf,
}

impl FileLock {
    /// Try once, without waiting.
    pub fn try_acquire(lock_dir: &Path, guarded: &Path) -> Result<Option<FileLock>, AuthError> {
        let lock_path = lock_path_for(lock_dir, guarded);
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
                    lock_path: lock_path_for(lock_dir, guarded).display().to_string(),
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
                    lock_path: lock_path_for(lock_dir, guarded).display().to_string(),
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
        let a = lock_path_for(&dir.join("locks"), &guarded);
        let b = lock_path_for(&dir.join("locks"), &guarded);
        assert_eq!(a, b);
        assert!(a.starts_with(dir.join("locks")));
        assert!(!a.starts_with(dir.join("foreign")));
        assert!(a.extension().is_some_and(|e| e == "lock"));
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

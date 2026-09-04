//! Claude Code local OAuth access token (file or macOS keychain).
//!
//! Env `ANTHROPIC_*` still wins at the caller. Never log the token.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use serde_json::Value;

const DEFAULT_AUTH_FILE: &str = ".claude/.credentials.json";
const KEYCHAIN_SERVICE: &str = "Claude Code-credentials";
const KEYCHAIN_ACCOUNTS: &[&str] = &["Claude Code", "credentials"];

static KEYCHAIN_DISABLES: AtomicU32 = AtomicU32::new(0);

/// Skip live keychain reads until this guard is dropped.
#[must_use]
pub struct ClaudeCodeKeychainIsolation {
    _private: (),
}

impl ClaudeCodeKeychainIsolation {
    /// Disable keychain lookups for the lifetime of the returned guard.
    pub fn hold() -> Self {
        KEYCHAIN_DISABLES.fetch_add(1, Ordering::SeqCst);
        Self { _private: () }
    }
}

impl Drop for ClaudeCodeKeychainIsolation {
    fn drop(&mut self) {
        KEYCHAIN_DISABLES.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Access token from `CLAUDE_CODE_OAUTH_TOKEN`, `~/.claude/.credentials.json`,
/// or the macOS `Claude Code-credentials` keychain item.
pub fn claude_code_access_token() -> Option<String> {
    env_oauth_token()
        .or_else(load_from_default_file)
        .or_else(load_from_keychain)
}

/// `accessToken` from a Claude Code credentials JSON blob.
pub fn access_token_from_json(raw: &str) -> Option<String> {
    let value: Value = serde_json::from_str(raw).ok()?;
    access_token_from_value(&value)
}

fn access_token_from_value(value: &Value) -> Option<String> {
    let obj = value
        .get("claudeAiOauth")
        .and_then(Value::as_object)
        .or_else(|| value.as_object())?;
    obj.get("accessToken")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
}

fn env_oauth_token() -> Option<String> {
    std::env::var("CLAUDE_CODE_OAUTH_TOKEN")
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
}

fn load_from_default_file() -> Option<String> {
    load_from_file(&default_auth_path()?)
}

fn default_auth_path() -> Option<PathBuf> {
    Some(dirs::home_dir()?.join(DEFAULT_AUTH_FILE))
}

fn load_from_file(path: &Path) -> Option<String> {
    let raw = std::fs::read_to_string(path).ok()?;
    access_token_from_json(&raw)
}

fn load_from_keychain() -> Option<String> {
    if KEYCHAIN_DISABLES.load(Ordering::SeqCst) > 0 {
        return None;
    }
    #[cfg(not(target_os = "macos"))]
    {
        return None;
    }
    #[cfg(target_os = "macos")]
    {
        for account in keychain_accounts() {
            if let Some(token) = keychain_token(&account) {
                return Some(token);
            }
        }
        None
    }
}

fn keychain_accounts() -> Vec<String> {
    let mut accounts = Vec::new();
    for key in ["USER", "LOGNAME"] {
        if let Ok(name) = std::env::var(key) {
            let name = name.trim();
            if !name.is_empty() && !accounts.iter().any(|a| a == name) {
                accounts.push(name.to_owned());
            }
        }
    }
    for name in KEYCHAIN_ACCOUNTS {
        if !accounts.iter().any(|a| a == name) {
            accounts.push((*name).to_owned());
        }
    }
    accounts
}

#[cfg(target_os = "macos")]
fn keychain_token(account: &str) -> Option<String> {
    if account.is_empty() {
        return None;
    }
    let output = std::process::Command::new("security")
        .args([
            "find-generic-password",
            "-s",
            KEYCHAIN_SERVICE,
            "-a",
            account,
            "-w",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let raw = String::from_utf8(output.stdout).ok()?;
    access_token_from_json(raw.trim())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn access_token_from_nested_oauth_object() {
        let raw = r#"{"claudeAiOauth":{"accessToken":"sk-ant-oat01-nested","refreshToken":"rt"}}"#;
        assert_eq!(
            access_token_from_json(raw).as_deref(),
            Some("sk-ant-oat01-nested")
        );
    }

    #[test]
    fn access_token_from_flat_object() {
        let raw = r#"{"accessToken":"sk-ant-oat01-flat"}"#;
        assert_eq!(
            access_token_from_json(raw).as_deref(),
            Some("sk-ant-oat01-flat")
        );
    }

    #[test]
    fn access_token_ignores_empty_and_non_object() {
        assert_eq!(access_token_from_json(r#"{"accessToken":"  "}"#), None);
        assert_eq!(access_token_from_json(r#"{"claudeAiOauth":"x"}"#), None);
        assert_eq!(access_token_from_json("not-json"), None);
    }

    #[test]
    fn access_token_from_file_reads_nested() {
        let dir = tempfile::tempdir().expect("temp");
        let path = dir.path().join("creds.json");
        std::fs::write(
            &path,
            r#"{"claudeAiOauth":{"accessToken":"sk-ant-oat01-file"}}"#,
        )
        .expect("write");
        assert_eq!(load_from_file(&path).as_deref(), Some("sk-ant-oat01-file"));
    }

    #[test]
    fn keychain_isolation_skips_live_lookup() {
        let _guard = ClaudeCodeKeychainIsolation::hold();
        assert_eq!(load_from_keychain(), None);
    }

    #[test]
    fn keychain_accounts_prefer_user_then_known_names() {
        let accounts = keychain_accounts();
        assert!(accounts.iter().any(|a| a == "Claude Code"), "{accounts:?}");
        assert!(accounts.iter().any(|a| a == "credentials"), "{accounts:?}");
    }
}

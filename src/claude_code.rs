//! Claude Code local OAuth access token (file or macOS keychain).
//!
//! Env `ANTHROPIC_*` still wins at the caller. Expired file/keychain oats
//! refresh when a refresh token is present. Env oat is access-only.
//! Never log the token.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

const DEFAULT_AUTH_FILE: &str = ".claude/.credentials.json";
const CLAUDE_CODE_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const CLAUDE_CODE_TOKEN_URL: &str = "https://platform.claude.com/v1/oauth/token";
const CLAUDE_CODE_TOKEN_URL_FALLBACK: &str = "https://console.anthropic.com/v1/oauth/token";
const DEFAULT_LIFETIME_SECS: u64 = 3600;
const EXPIRY_SKEW_MS: u64 = 60_000;
#[cfg(target_os = "macos")]
const KEYCHAIN_SERVICE: &str = "Claude Code-credentials";
#[cfg(any(target_os = "macos", test))]
const KEYCHAIN_ACCOUNTS: &[&str] = &["Claude Code", "credentials"];

static KEYCHAIN_DISABLES: AtomicU32 = AtomicU32::new(0);

#[cfg(test)]
thread_local! {
    static TEST_TOKEN_URLS: std::cell::RefCell<Option<(String, Option<String>)>> =
        const { std::cell::RefCell::new(None) };
}

#[derive(Debug)]
struct ParsedCreds {
    access_token: String,
    refresh_token: Option<String>,
    expires_at_ms: Option<u64>,
}

struct RefreshResponse {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: Option<u64>,
}

enum CredStore {
    File(PathBuf),
    Keychain { account: String },
}

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
#[cfg(test)]
fn access_token_from_json(raw: &str) -> Option<String> {
    parse_creds_from_json(raw).map(|c| c.access_token)
}

fn parse_creds_from_json(raw: &str) -> Option<ParsedCreds> {
    let value: Value = serde_json::from_str(raw).ok()?;
    parse_creds_value(&value)
}

fn parse_creds_value(value: &Value) -> Option<ParsedCreds> {
    let obj = value
        .get("claudeAiOauth")
        .and_then(Value::as_object)
        .or_else(|| value.as_object())?;
    let access_token = obj
        .get("accessToken")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())?
        .to_owned();
    let refresh_token = obj
        .get("refreshToken")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned);
    let expires_at_ms = obj.get("expiresAt").and_then(json_u64);
    Some(ParsedCreds {
        access_token,
        refresh_token,
        expires_at_ms,
    })
}

fn json_u64(value: &Value) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_i64().and_then(|n| u64::try_from(n).ok()))
        .or_else(|| {
            value
                .as_f64()
                .filter(|n| n.is_finite() && *n >= 0.0)
                .map(|n| n as u64)
        })
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn now_plus_secs_ms(secs: u64) -> u64 {
    now_ms().saturating_add(secs.saturating_mul(1000))
}

fn access_needs_refresh(expires_at_ms: Option<u64>, now_ms: u64) -> bool {
    match expires_at_ms {
        None => false,
        Some(ms) => ms <= now_ms.saturating_add(EXPIRY_SKEW_MS),
    }
}

/// Write refresh tokens onto the same object [`parse_creds_value`] reads.
///
/// Nested `claudeAiOauth` is used only when it is an object. A string or
/// array at that key falls through to the top-level object.
fn apply_refresh_to_json(
    doc: &mut Value,
    access_token: &str,
    refresh_token: Option<&str>,
    expires_at_ms: u64,
) -> Result<(), &'static str> {
    let use_nested = matches!(doc.get("claudeAiOauth"), Some(Value::Object(_)));
    let target = if use_nested {
        doc.get_mut("claudeAiOauth")
    } else {
        Some(doc)
    };
    let Some(Value::Object(map)) = target else {
        return Err("Claude Code credentials JSON is not an object");
    };
    map.insert(
        "accessToken".to_owned(),
        Value::String(access_token.to_owned()),
    );
    if let Some(rt) = refresh_token {
        map.insert("refreshToken".to_owned(), Value::String(rt.to_owned()));
    }
    map.insert("expiresAt".to_owned(), json!(expires_at_ms));
    Ok(())
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
    token_from_store(&raw, &CredStore::File(path.to_owned()))
}

fn token_from_store(raw: &str, store: &CredStore) -> Option<String> {
    let parsed = parse_creds_from_json(raw)?;
    if !access_needs_refresh(parsed.expires_at_ms, now_ms()) {
        return Some(parsed.access_token);
    }
    let Some(rt) = parsed
        .refresh_token
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    else {
        return Some(parsed.access_token);
    };
    let Some(resp) = refresh_access_token(rt) else {
        return Some(parsed.access_token);
    };
    let new_rt = resp
        .refresh_token
        .as_deref()
        .or(parsed.refresh_token.as_deref());
    let expires_in = resp.expires_in.unwrap_or(DEFAULT_LIFETIME_SECS);
    let _ = persist_refresh(store, raw, &resp.access_token, new_rt, expires_in);
    Some(resp.access_token)
}

fn persist_refresh(
    store: &CredStore,
    current_raw: &str,
    access_token: &str,
    refresh_token: Option<&str>,
    expires_in: u64,
) -> Result<(), ()> {
    let expires_at_ms = now_plus_secs_ms(expires_in);
    match store {
        CredStore::File(path) => persist_file(
            path,
            current_raw,
            access_token,
            refresh_token,
            expires_at_ms,
        ),
        CredStore::Keychain { account } => persist_keychain(
            account,
            current_raw,
            access_token,
            refresh_token,
            expires_at_ms,
        ),
    }
}

fn persist_file(
    path: &Path,
    current_raw: &str,
    access_token: &str,
    refresh_token: Option<&str>,
    expires_at_ms: u64,
) -> Result<(), ()> {
    let raw = std::fs::read_to_string(path).unwrap_or_else(|_| current_raw.to_owned());
    let mut doc: Value = serde_json::from_str(&raw).map_err(|_| ())?;
    apply_refresh_to_json(&mut doc, access_token, refresh_token, expires_at_ms).map_err(|_| ())?;
    let updated = serde_json::to_string_pretty(&doc).map_err(|_| ())?;
    std::fs::write(path, updated).map_err(|_| ())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

fn persist_keychain(
    account: &str,
    current_raw: &str,
    access_token: &str,
    refresh_token: Option<&str>,
    expires_at_ms: u64,
) -> Result<(), ()> {
    if KEYCHAIN_DISABLES.load(Ordering::SeqCst) > 0 {
        return Ok(());
    }
    persist_keychain_os(
        account,
        current_raw,
        access_token,
        refresh_token,
        expires_at_ms,
    )
}

#[cfg(target_os = "macos")]
fn persist_keychain_os(
    account: &str,
    current_raw: &str,
    access_token: &str,
    refresh_token: Option<&str>,
    expires_at_ms: u64,
) -> Result<(), ()> {
    let mut doc: Value = serde_json::from_str(current_raw).map_err(|_| ())?;
    apply_refresh_to_json(&mut doc, access_token, refresh_token, expires_at_ms).map_err(|_| ())?;
    let updated = serde_json::to_string(&doc).map_err(|_| ())?;
    write_keychain_secret(account, &updated)
}

#[cfg(not(target_os = "macos"))]
fn persist_keychain_os(
    _account: &str,
    _current_raw: &str,
    _access_token: &str,
    _refresh_token: Option<&str>,
    _expires_at_ms: u64,
) -> Result<(), ()> {
    Ok(())
}

fn token_urls() -> (String, Option<String>) {
    #[cfg(test)]
    {
        if let Some(urls) = TEST_TOKEN_URLS.with(|c| c.borrow().clone()) {
            return urls;
        }
    }
    (
        CLAUDE_CODE_TOKEN_URL.to_owned(),
        Some(CLAUDE_CODE_TOKEN_URL_FALLBACK.to_owned()),
    )
}

fn refresh_access_token(refresh_token: &str) -> Option<RefreshResponse> {
    let refresh_token = refresh_token.to_owned();
    let (primary, fallback) = token_urls();
    std::thread::Builder::new()
        .name("canact-claude-refresh".into())
        .spawn(move || {
            refresh_access_token_on_thread(&refresh_token, &primary, fallback.as_deref())
        })
        .ok()?
        .join()
        .ok()?
}

fn refresh_access_token_on_thread(
    refresh_token: &str,
    primary: &str,
    fallback: Option<&str>,
) -> Option<RefreshResponse> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .ok()?;
    rt.block_on(refresh_access_token_async(refresh_token, primary, fallback))
}

async fn refresh_access_token_async(
    refresh_token: &str,
    primary: &str,
    fallback: Option<&str>,
) -> Option<RefreshResponse> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .ok()?;
    let body = json!({
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": CLAUDE_CODE_CLIENT_ID,
    });
    let resp = client.post(primary).json(&body).send().await.ok()?;
    let resp = if resp.status() == reqwest::StatusCode::NOT_FOUND {
        if let Some(fb) = fallback {
            client.post(fb).json(&body).send().await.ok()?
        } else {
            resp
        }
    } else {
        resp
    };
    if !resp.status().is_success() {
        return None;
    }
    let value: Value = resp.json().await.ok()?;
    let access_token = value
        .get("access_token")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())?
        .to_owned();
    let refresh_token = value
        .get("refresh_token")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned);
    let expires_in = value.get("expires_in").and_then(json_u64);
    Some(RefreshResponse {
        access_token,
        refresh_token,
        expires_in,
    })
}

fn load_from_keychain() -> Option<String> {
    if KEYCHAIN_DISABLES.load(Ordering::SeqCst) > 0 {
        return None;
    }
    load_from_keychain_os()
}

#[cfg(target_os = "macos")]
fn load_from_keychain_os() -> Option<String> {
    for account in keychain_accounts() {
        if let Some(raw) = keychain_secret(&account) {
            if let Some(token) = token_from_store(&raw, &CredStore::Keychain { account }) {
                return Some(token);
            }
        }
    }
    None
}

#[cfg(not(target_os = "macos"))]
fn load_from_keychain_os() -> Option<String> {
    None
}

#[cfg(any(target_os = "macos", test))]
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
fn keychain_secret(account: &str) -> Option<String> {
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
    let raw = raw.trim();
    if raw.is_empty() {
        None
    } else {
        Some(raw.to_owned())
    }
}

#[cfg(target_os = "macos")]
fn write_keychain_secret(account: &str, secret: &str) -> Result<(), ()> {
    if account.is_empty() {
        return Err(());
    }
    let status = std::process::Command::new("security")
        .args([
            "add-generic-password",
            "-s",
            KEYCHAIN_SERVICE,
            "-a",
            account,
            "-w",
            secret,
            "-U",
        ])
        .status()
        .map_err(|_| ())?;
    if status.success() { Ok(()) } else { Err(()) }
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

    #[test]
    fn parse_creds_reads_nested_refresh_and_expires() {
        let raw = r#"{"claudeAiOauth":{"accessToken":"sk-ant-oat01-nested","refreshToken":"rt-nested","expiresAt":1770000000000}}"#;
        let parsed = parse_creds_from_json(raw).expect("parse");
        assert_eq!(parsed.access_token, "sk-ant-oat01-nested");
        assert_eq!(parsed.refresh_token.as_deref(), Some("rt-nested"));
        assert_eq!(parsed.expires_at_ms, Some(1_770_000_000_000));
    }

    #[test]
    fn parse_creds_reads_flat_refresh_and_expires() {
        let raw = r#"{"accessToken":"sk-ant-oat01-flat","refreshToken":"rt-flat","expiresAt":1770000000000}"#;
        let parsed = parse_creds_from_json(raw).expect("parse");
        assert_eq!(parsed.access_token, "sk-ant-oat01-flat");
        assert_eq!(parsed.refresh_token.as_deref(), Some("rt-flat"));
        assert_eq!(parsed.expires_at_ms, Some(1_770_000_000_000));
    }

    #[test]
    fn access_needs_refresh_uses_sixty_second_skew() {
        let now = 1_800_000_000_000;
        assert!(access_needs_refresh(Some(1), now));
        assert!(access_needs_refresh(Some(now), now));
        assert!(access_needs_refresh(Some(now + EXPIRY_SKEW_MS), now));
        assert!(!access_needs_refresh(Some(now + EXPIRY_SKEW_MS + 1), now));
        assert!(!access_needs_refresh(None, now));
    }

    #[test]
    fn apply_refresh_to_json_updates_nested_object() {
        let mut doc = serde_json::json!({
            "claudeAiOauth": {
                "accessToken": "sk-ant-oat01-old",
                "refreshToken": "rt-old",
                "expiresAt": 1
            }
        });
        apply_refresh_to_json(&mut doc, "sk-ant-oat01-new", Some("rt-new"), 99)
            .expect("nested object is writable");
        assert_eq!(
            doc["claudeAiOauth"]["accessToken"].as_str(),
            Some("sk-ant-oat01-new")
        );
        assert_eq!(
            doc["claudeAiOauth"]["refreshToken"].as_str(),
            Some("rt-new")
        );
        assert_eq!(doc["claudeAiOauth"]["expiresAt"].as_u64(), Some(99));
    }

    #[test]
    fn apply_refresh_to_json_uses_top_level_when_nested_is_not_object() {
        let mut doc = serde_json::json!({
            "claudeAiOauth": "not-an-object",
            "accessToken": "sk-ant-oat01-old",
            "refreshToken": "rt-old",
            "expiresAt": 1
        });
        apply_refresh_to_json(&mut doc, "sk-ant-oat01-new", Some("rt-new"), 99)
            .expect("top-level object is writable");
        assert_eq!(doc["accessToken"].as_str(), Some("sk-ant-oat01-new"));
        assert_eq!(doc["refreshToken"].as_str(), Some("rt-new"));
        assert_eq!(doc["expiresAt"].as_u64(), Some(99));
        assert_eq!(doc["claudeAiOauth"].as_str(), Some("not-an-object"));
        assert!(parse_creds_value(&doc).is_some());
    }

    #[test]
    fn apply_refresh_to_json_errors_on_non_object_document() {
        let mut doc = serde_json::json!(["not", "an", "object"]);
        let err = apply_refresh_to_json(&mut doc, "sk-ant-oat01-new", Some("rt-new"), 99)
            .expect_err("non-object must not no-op");
        assert!(
            err.to_ascii_lowercase().contains("object"),
            "expected not-an-object error, got: {err}"
        );
        assert_eq!(doc, serde_json::json!(["not", "an", "object"]));
    }

    #[test]
    fn expired_without_refresh_token_returns_stale_access() {
        let dir = tempfile::tempdir().expect("temp");
        let path = dir.path().join("creds.json");
        std::fs::write(
            &path,
            r#"{"claudeAiOauth":{"accessToken":"sk-ant-oat01-stale","expiresAt":1}}"#,
        )
        .expect("write");
        assert_eq!(load_from_file(&path).as_deref(), Some("sk-ant-oat01-stale"));
    }

    #[test]
    fn fresh_file_returns_access_without_http() {
        let dir = tempfile::tempdir().expect("temp");
        let path = dir.path().join("creds.json");
        let future = now_ms().saturating_add(3_600_000);
        std::fs::write(
            &path,
            format!(
                r#"{{"claudeAiOauth":{{"accessToken":"sk-ant-oat01-fresh","refreshToken":"rt","expiresAt":{future}}}}}"#
            ),
        )
        .expect("write");
        assert_eq!(load_from_file(&path).as_deref(), Some("sk-ant-oat01-fresh"));
    }

    #[test]
    fn expired_file_refresh_writes_back() {
        let _guard = ClaudeCodeKeychainIsolation::hold();
        let dir = tempfile::tempdir().expect("temp");
        let path = dir.path().join("creds.json");
        std::fs::write(
            &path,
            r#"{"claudeAiOauth":{"accessToken":"sk-ant-oat01-old","refreshToken":"rt-old","expiresAt":1}}"#,
        )
        .expect("write");
        let (url, seen) = spawn_refresh_seq(vec![(
            200,
            r#"{"access_token":"sk-ant-oat01-refreshed","refresh_token":"rt-new","expires_in":3600}"#,
        )]);
        let _urls = TokenUrlOverride::set(url, None);
        assert_eq!(
            load_from_file(&path).as_deref(),
            Some("sk-ant-oat01-refreshed")
        );
        let req = seen.lock().expect("seen")[0].clone();
        assert!(
            req.contains("\"grant_type\":\"refresh_token\""),
            "refresh body must be JSON grant_type=refresh_token, got: {req}"
        );
        assert!(
            req.contains("\"refresh_token\":\"rt-old\""),
            "refresh body must send stored refresh token, got: {req}"
        );
        assert!(
            req.contains(&format!("\"client_id\":\"{CLAUDE_CODE_CLIENT_ID}\"")),
            "refresh body must send client id, got: {req}"
        );
        let written = std::fs::read_to_string(&path).expect("reread");
        let parsed = parse_creds_from_json(&written).expect("parse written");
        assert_eq!(parsed.access_token, "sk-ant-oat01-refreshed");
        assert_eq!(parsed.refresh_token.as_deref(), Some("rt-new"));
        assert!(
            parsed.expires_at_ms.unwrap_or(0) > now_ms(),
            "write-back expiresAt must be in the future: {parsed:?}"
        );
    }

    #[test]
    fn expired_file_refresh_falls_back_on_http_404() {
        let _guard = ClaudeCodeKeychainIsolation::hold();
        let dir = tempfile::tempdir().expect("temp");
        let path = dir.path().join("creds.json");
        std::fs::write(
            &path,
            r#"{"accessToken":"sk-ant-oat01-old","refreshToken":"rt-old","expiresAt":1}"#,
        )
        .expect("write");
        let (url, seen) = spawn_refresh_seq(vec![
            (404, r#"{"error":"not_found"}"#),
            (
                200,
                r#"{"access_token":"sk-ant-oat01-fallback","expires_in":3600}"#,
            ),
        ]);
        let _urls = TokenUrlOverride::set(url.clone(), Some(url));
        assert_eq!(
            load_from_file(&path).as_deref(),
            Some("sk-ant-oat01-fallback")
        );
        let seen = seen.lock().expect("seen");
        assert_eq!(seen.len(), 2, "primary 404 then fallback, got: {seen:?}");
        assert!(
            seen[1].contains("\"grant_type\":\"refresh_token\""),
            "fallback must receive grant_type=refresh_token, got: {}",
            seen[1]
        );
        let written = std::fs::read_to_string(&path).expect("reread");
        let parsed = parse_creds_from_json(&written).expect("parse written");
        assert_eq!(parsed.access_token, "sk-ant-oat01-fallback");
        assert_eq!(
            parsed.refresh_token.as_deref(),
            Some("rt-old"),
            "missing refresh_token in response keeps stored refresh"
        );
    }

    #[test]
    fn write_back_failure_still_returns_new_access() {
        let _guard = ClaudeCodeKeychainIsolation::hold();
        let dir = tempfile::tempdir().expect("temp");
        let path = dir.path().join("creds.json");
        std::fs::write(
            &path,
            r#"{"claudeAiOauth":{"accessToken":"sk-ant-oat01-old","refreshToken":"rt-old","expiresAt":1}}"#,
        )
        .expect("write");
        let mut perms = std::fs::metadata(&path).expect("meta").permissions();
        perms.set_readonly(true);
        std::fs::set_permissions(&path, perms).expect("readonly");
        let (url, _seen) = spawn_refresh_seq(vec![(
            200,
            r#"{"access_token":"sk-ant-oat01-refreshed","refresh_token":"rt-new","expires_in":3600}"#,
        )]);
        let _urls = TokenUrlOverride::set(url, None);
        let token = load_from_file(&path);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
        }
        #[cfg(not(unix))]
        {
            let mut perms = std::fs::metadata(&path).expect("meta").permissions();
            #[allow(clippy::permissions_set_readonly_false)]
            perms.set_readonly(false);
            let _ = std::fs::set_permissions(&path, perms);
        }
        assert_eq!(token.as_deref(), Some("sk-ant-oat01-refreshed"));
        let written = std::fs::read_to_string(&path).expect("reread");
        assert!(
            written.contains("sk-ant-oat01-old"),
            "readonly write-back must leave the file unchanged"
        );
    }

    struct TokenUrlOverride {
        _private: (),
    }

    impl TokenUrlOverride {
        fn set(primary: String, fallback: Option<String>) -> Self {
            TEST_TOKEN_URLS.with(|c| *c.borrow_mut() = Some((primary, fallback)));
            Self { _private: () }
        }
    }

    impl Drop for TokenUrlOverride {
        fn drop(&mut self) {
            TEST_TOKEN_URLS.with(|c| *c.borrow_mut() = None);
        }
    }

    fn spawn_refresh_seq(
        replies: Vec<(u16, &'static str)>,
    ) -> (String, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
        use std::io::Write;
        use std::net::TcpListener;
        use std::sync::{Arc, Mutex};
        use std::thread;

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen_thread = Arc::clone(&seen);
        thread::spawn(move || {
            for (status, body) in replies {
                if let Ok((mut stream, _)) = listener.accept() {
                    let req = read_http_request(&mut stream);
                    if let Ok(mut log) = seen_thread.lock() {
                        log.push(req);
                    }
                    let resp = format!(
                        "HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = stream.write_all(resp.as_bytes());
                    let _ = stream.flush();
                }
            }
        });
        (format!("http://{addr}/v1/oauth/token"), seen)
    }

    fn read_http_request(stream: &mut std::net::TcpStream) -> String {
        use std::io::Read;
        use std::time::Duration;

        let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
        let mut buf = Vec::new();
        let mut tmp = [0u8; 4096];
        loop {
            match stream.read(&mut tmp) {
                Ok(0) => break,
                Ok(n) => buf.extend_from_slice(&tmp[..n]),
                Err(_) => break,
            }
            if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                let content_len = std::str::from_utf8(&buf[..pos])
                    .unwrap_or("")
                    .lines()
                    .find_map(|line| {
                        line.split_once(':').and_then(|(k, v)| {
                            k.eq_ignore_ascii_case("content-length")
                                .then_some(v.trim().parse::<usize>().unwrap_or(0))
                        })
                    })
                    .unwrap_or(0);
                let header_end = pos + 4;
                while buf.len() < header_end + content_len {
                    match stream.read(&mut tmp) {
                        Ok(0) => break,
                        Ok(n) => buf.extend_from_slice(&tmp[..n]),
                        Err(_) => break,
                    }
                }
                break;
            }
            if buf.len() > 32_768 {
                break;
            }
        }
        String::from_utf8_lossy(&buf).into_owned()
    }
}

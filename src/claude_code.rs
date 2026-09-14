//! Claude Code local OAuth access token via `wiremux-auth`.
//!
//! Refresh, keychain, and file parse live in the shipped `anthropic-oauth`
//! profile. Env `ANTHROPIC_*` still wins at the caller.

/// Test hook kept so existing CLI/MCP tests compile. Isolation is owned by
/// `wiremux-auth` `IsolatedHome` (feature `test-util`).
#[derive(Debug)]
pub struct ClaudeCodeKeychainIsolation {
    _private: (),
}

impl ClaudeCodeKeychainIsolation {
    /// No-op hold. Wiremux owns keychain tests.
    #[must_use]
    pub fn hold() -> Self {
        Self { _private: () }
    }
}

/// Access token from catalog id `anthropic-oauth` (file, keychain, or env).
///
/// Runs on a helper thread so a current-thread Tokio runtime can call this
/// from `async` without nesting `block_on`.
pub fn claude_code_access_token() -> Option<String> {
    std::thread::spawn(|| {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .ok()?;
        rt.block_on(async {
            let opts = wiremux_auth::LoadOptions::default();
            let mut profile = wiremux_auth::load_profile("anthropic-oauth", &opts).ok()?;
            prepend_login_user_keychain_account(&mut profile);
            let provider = wiremux_auth::provider_from_profile(&profile).ok()?;
            wiremux_auth::TokenProvider::get_token(&provider).await.ok()
        })
    })
    .join()
    .ok()
    .flatten()
}

/// Claude Code stores the oat under the login `USER` account. The shipped
/// preset only lists `Claude Code` and `credentials`.
fn login_user_account() -> Option<String> {
    std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .ok()
        .filter(|s| !s.is_empty())
}

fn prepend_login_user_keychain_account(profile: &mut wiremux_auth::ResolvedProfile) {
    let Some(oauth) = profile.oauth.as_mut() else {
        return;
    };
    let Some(user) = login_user_account() else {
        return;
    };
    if oauth.keychain_accounts.iter().any(|a| a == &user) {
        return;
    }
    oauth.keychain_accounts.insert(0, user);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepends_login_user_before_shipped_keychain_accounts() {
        let opts = wiremux_auth::LoadOptions::default();
        let mut profile =
            wiremux_auth::load_profile("anthropic-oauth", &opts).expect("shipped profile");
        let user = login_user_account().expect("USER or USERNAME");
        prepend_login_user_keychain_account(&mut profile);
        let accounts = profile.oauth.expect("oauth").keychain_accounts;
        assert_eq!(accounts.first().map(String::as_str), Some(user.as_str()));
        assert!(accounts.iter().any(|a| a == "Claude Code"));
        assert!(accounts.iter().any(|a| a == "credentials"));
    }
}

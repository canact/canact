//! Claude Code local OAuth access token via `wiremux-auth`.
//!
//! Refresh, keychain, and file parse live in the shipped `anthropic-oauth`
//! profile. Env `ANTHROPIC_*` still wins at the caller. wiremux-auth 0.3.0
//! tries the process login name before the shipped `Claude Code` /
//! `credentials` keychain accounts.

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
        rt.block_on(wiremux_auth::token_for_profile("anthropic-oauth"))
            .ok()
    })
    .join()
    .ok()
    .flatten()
}

/// Access token from catalog id `xai-oauth` (`~/.grok/auth.json`).
///
/// Env `XAI_API_KEY` / `GROK_API_KEY` still win at the caller.
pub fn xai_oauth_access_token() -> Option<String> {
    std::thread::spawn(|| {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .ok()?;
        rt.block_on(wiremux_auth::token_for_profile("xai-oauth"))
            .ok()
    })
    .join()
    .ok()
    .flatten()
}

#[cfg(test)]
mod tests {
    #[test]
    fn shipped_anthropic_oauth_profile_loads() {
        let opts = wiremux_auth::LoadOptions::default();
        let profile =
            wiremux_auth::load_profile("anthropic-oauth", &opts).expect("shipped anthropic-oauth");
        assert_eq!(profile.id, "anthropic-oauth");
        let accounts = profile.oauth.expect("oauth").keychain_accounts;
        assert!(accounts.iter().any(|a| a == "Claude Code"));
        assert!(accounts.iter().any(|a| a == "credentials"));
    }

    #[test]
    fn shipped_xai_oauth_profile_loads() {
        let opts = wiremux_auth::LoadOptions::default();
        let profile = wiremux_auth::load_profile("xai-oauth", &opts).expect("shipped xai-oauth");
        assert_eq!(profile.id, "xai-oauth");
        assert_eq!(profile.http.base_url.as_deref(), Some("https://api.x.ai"));
    }
}

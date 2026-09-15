//! Claude Code local OAuth access token via `wiremux-auth`.
//!
//! Refresh, keychain, and file parse live in the shipped `anthropic-oauth`
//! profile. Env `ANTHROPIC_*` still wins at the caller. wiremux-auth 0.4.0
//! tries the process login name before the shipped `Claude Code` /
//! `credentials` keychain accounts. Hosts that only need a stored
//! Bearer use `token_for_profile_cached` so an expired oat does not
//! POST `token_url`.

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
        rt.block_on(wiremux_auth::token_for_profile_cached("anthropic-oauth"))
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
        rt.block_on(wiremux_auth::token_for_profile_cached("xai-oauth"))
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
        assert!(
            !profile.http.headers.contains_key("x-grok-client-version"),
            "api.x.ai must not send a Grok CLI version header"
        );
    }

    #[test]
    fn shipped_xai_grok_build_profile_loads() {
        let opts = wiremux_auth::LoadOptions::default();
        let profile =
            wiremux_auth::load_profile("xai-grok-build", &opts).expect("shipped xai-grok-build");
        assert_eq!(profile.id, "xai-grok-build");
        assert_eq!(
            profile.http.base_url.as_deref(),
            Some("https://cli-chat-proxy.grok.com")
        );
        assert_eq!(
            profile
                .http
                .headers
                .get("x-grok-client-version")
                .map(String::as_str),
            Some("0.1.202"),
            "cli-chat-proxy returns HTTP 426 without a Grok CLI version"
        );
        assert_eq!(
            profile
                .http
                .headers
                .get("x-grok-client-identifier")
                .map(String::as_str),
            Some("wiremux")
        );
        let oauth = profile.oauth.expect("oauth");
        let client = oauth.client_id.as_deref().map(str::trim).unwrap_or("");
        assert!(client.is_empty(), "must not ship a product client id");
    }
}

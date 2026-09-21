//! Claude Code local OAuth access token via `wiremux-auth`.
//!
//! Refresh, keychain, and file parse live in the shipped `anthropic-oauth`
//! profile. Env `ANTHROPIC_*` still wins at the caller. wiremux-auth 0.8.0
//! tries the process login name before the shipped `Claude Code` /
//! `credentials` keychain accounts. Login uses `token_for_profile` so an
//! expired oat POSTs `token_url` (bounded refresh). Do not call that
//! helper from a unit test against the developer keychain; isolate HOME.

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
/// from `async` without nesting `block_on`. Refreshes an expired oat.
/// `Ok(None)` means no login. `Err` is a failed refresh (Auth), not a
/// missing-key.
pub fn claude_code_access_token() -> Result<Option<String>, String> {
    oauth_access_token("anthropic-oauth")
}

/// Access token from catalog id `xai-oauth` (`~/.grok/auth.json`).
///
/// Env `XAI_API_KEY` / `GROK_API_KEY` still win at the caller. Refreshes
/// an expired oat. Same `Ok(None)` vs `Err` split as
/// [`claude_code_access_token`].
pub fn xai_oauth_access_token() -> Result<Option<String>, String> {
    oauth_access_token("xai-oauth")
}

fn classify_oauth_result(
    result: Result<String, wiremux_auth::AuthError>,
) -> Result<Option<String>, String> {
    match result {
        Ok(token) => Ok(Some(token)),
        Err(wiremux_auth::AuthError::MissingField(_)) => Ok(None),
        Err(err) => {
            let msg = err.to_string();
            if msg.contains("no credentials") {
                Ok(None)
            } else {
                Err(msg)
            }
        }
    }
}

fn oauth_access_token(profile_id: &'static str) -> Result<Option<String>, String> {
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|err| err.to_string())?;
        classify_oauth_result(rt.block_on(wiremux_auth::token_for_profile(profile_id)))
    })
    .join()
    .unwrap_or_else(|_| Err("oauth helper thread panicked".to_owned()))
}

#[cfg(test)]
mod tests {
    #[test]
    fn login_uses_refreshing_token_for_profile() {
        let src = include_str!("claude_code.rs");
        let prod = src.split("mod tests").next().expect("production");
        assert!(
            prod.contains("token_for_profile(profile_id)"),
            "Claude/xAI login must refresh via token_for_profile"
        );
        assert!(
            !prod.contains("token_for_profile_cached"),
            "cached Bearer skips token_url and leaves an expired oat"
        );
    }

    #[test]
    fn oauth_missing_field_is_absent() {
        let err = wiremux_auth::AuthError::MissingField("refresh_token".into());
        assert_eq!(super::classify_oauth_result(Err(err)).unwrap(), None);
    }

    #[test]
    fn oauth_no_credentials_is_absent() {
        let err = wiremux_auth::AuthError::TokenProvider(
            "no credentials (file); re-authenticate using the profile login flow".into(),
        );
        assert_eq!(super::classify_oauth_result(Err(err)).unwrap(), None);
    }

    #[test]
    fn oauth_vendor_rejected_is_auth_not_missing() {
        let err = wiremux_auth::AuthError::VendorRejected {
            status: 400,
            summary: "invalid_grant".into(),
        };
        let msg = super::classify_oauth_result(Err(err)).unwrap_err();
        assert!(msg.contains("vendor rejected"), "{msg}");
        assert!(msg.contains("invalid_grant"), "{msg}");
    }

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

    #[test]
    fn shipped_xai_grok_build_messages_profile_loads() {
        let opts = wiremux_auth::LoadOptions::default();
        let profile = wiremux_auth::load_profile("xai-grok-build-messages", &opts)
            .expect("shipped xai-grok-build-messages");
        assert_eq!(profile.id, "xai-grok-build-messages");
        assert_eq!(profile.dialect.wire, Some(wiremux_auth::Wire::Messages));
        assert_eq!(
            profile.http.base_url.as_deref(),
            Some("https://cli-chat-proxy.grok.com")
        );
        assert_eq!(
            profile.http.chat_path.as_deref(),
            Some("/v1/messages"),
            "Messages family must not reuse chat-completions"
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
        let oauth = profile.oauth.expect("oauth");
        let client = oauth.client_id.as_deref().map(str::trim).unwrap_or("");
        assert!(client.is_empty(), "must not ship a product client id");
    }

    #[test]
    fn shipped_openai_codex_profile_loads() {
        let opts = wiremux_auth::LoadOptions::default();
        let profile =
            wiremux_auth::load_profile("openai-codex", &opts).expect("shipped openai-codex");
        assert_eq!(profile.id, "openai-codex");
        assert_eq!(profile.dialect.wire, Some(wiremux_auth::Wire::Responses));
        assert_eq!(
            profile.http.base_url.as_deref(),
            Some("https://api.openai.com")
        );
        assert_eq!(
            profile.http.chat_path.as_deref(),
            Some("/v1/responses"),
            "Codex must not reuse chat-completions"
        );
        assert_eq!(profile.access_env, ["OPENAI_API_KEY"]);
    }

    #[test]
    fn shipped_groq_profile_loads() {
        let opts = wiremux_auth::LoadOptions::default();
        let profile = wiremux_auth::load_profile("groq", &opts).expect("shipped groq");
        assert_eq!(profile.id, "groq");
        assert_eq!(
            profile.http.base_url.as_deref(),
            Some("https://api.groq.com")
        );
        assert_eq!(
            profile.http.chat_path.as_deref(),
            Some("/openai/v1/chat/completions")
        );
        assert_eq!(profile.access_env, ["GROQ_API_KEY"]);
    }

    #[test]
    fn shipped_amazon_bedrock_profile_loads() {
        let opts = wiremux_auth::LoadOptions::default();
        let profile =
            wiremux_auth::load_profile("amazon-bedrock", &opts).expect("shipped amazon-bedrock");
        assert_eq!(profile.id, "amazon-bedrock");
        assert_eq!(profile.dialect.wire, Some(wiremux_auth::Wire::Converse));
        assert_eq!(profile.access_env, ["AWS_BEARER_TOKEN_BEDROCK"]);
    }

    #[test]
    fn shipped_profile_ids_load_offline() {
        let opts = wiremux_auth::LoadOptions::default();
        let ids = wiremux_auth::shipped_profile_ids();
        assert!(
            ids.contains(&"xai-grok-build-messages"),
            "0.8.0 must ship the Messages Grok Build catalog: {ids:?}"
        );
        assert!(
            ids.contains(&"amazon-bedrock"),
            "0.8.0 must ship the Bedrock catalog: {ids:?}"
        );
        assert!(
            ids.contains(&"groq"),
            "0.8.0 must ship the Groq catalog: {ids:?}"
        );
        assert!(
            ids.contains(&"openai-codex"),
            "0.8.0 must ship the Responses Codex catalog: {ids:?}"
        );
        for id in ids {
            let profile = wiremux_auth::load_profile(id, &opts)
                .unwrap_or_else(|err| panic!("shipped {id} must load: {err}"));
            assert_eq!(profile.id, *id);
        }
    }
}

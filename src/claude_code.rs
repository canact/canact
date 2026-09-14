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
        rt.block_on(wiremux_auth::token_for_profile("anthropic-oauth"))
            .ok()
    })
    .join()
    .ok()
    .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_catalog_token_is_none() {
        let _guard = ClaudeCodeKeychainIsolation::hold();
        // No planted Claude creds in this process. Absence is None, not panic.
        let _ = claude_code_access_token();
    }
}

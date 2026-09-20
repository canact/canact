//! Default OpenAI-compatible base URLs for known local providers.

/// Ollama's OpenAI-compatible listener.
pub const OLLAMA_BASE_URL: &str = "http://127.0.0.1:11434/v1";
/// LM Studio's OpenAI-compatible listener.
pub const LMSTUDIO_BASE_URL: &str = "http://127.0.0.1:1234/v1";
/// Common vLLM OpenAI-compatible listener.
pub const VLLM_BASE_URL: &str = "http://127.0.0.1:8000/v1";
/// xAI OpenAI-compatible listener (`--provider xai` / `grok`).
pub const XAI_BASE_URL: &str = "https://api.x.ai/v1";
/// Grok Build CLI proxy (`--provider grok-build` / `xai-grok-build`).
pub const GROK_BUILD_BASE_URL: &str = "https://cli-chat-proxy.grok.com/v1";
/// Anthropic OpenAI-compatible listener (`--provider claude` / `anthropic`).
pub const ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com/v1";
/// Groq OpenAI-compatible listener (`--provider groq`).
pub const GROQ_BASE_URL: &str = "https://api.groq.com/openai/v1";
/// Amazon Bedrock Converse refuse-gate URL (`--provider amazon-bedrock` / `bedrock`).
///
/// The shipped profile substitutes `{env:AWS_REGION}`. This default is
/// only the cloud-key gate and the omitted `--base-url` host.
pub const BEDROCK_BASE_URL: &str = "https://bedrock-runtime.us-east-1.amazonaws.com";

/// Local-provider default when the user omitted `--base-url`.
pub fn local_provider_base_url(provider: &str) -> Option<String> {
    let provider = provider.to_ascii_lowercase();
    match provider.as_str() {
        "ollama" | "localhost" | "127.0.0.1" | "::1" | "[::1]" | "0.0.0.0" => {
            Some(OLLAMA_BASE_URL.to_owned())
        }
        "lmstudio" => Some(LMSTUDIO_BASE_URL.to_owned()),
        "vllm" => Some(VLLM_BASE_URL.to_owned()),
        other => loopback_host_port_base_url(other),
    }
}

/// Base URL when `--base-url` is omitted.
pub fn default_compat_base_url(provider: &str, from_openrouter: bool) -> String {
    if let Some(local) = local_provider_base_url(provider) {
        return local;
    }
    let provider = provider.to_ascii_lowercase();
    if is_grok_build_provider_label(&provider) || is_grok_build_messages_provider_label(&provider) {
        GROK_BUILD_BASE_URL.to_owned()
    } else if is_xai_provider_label(&provider) {
        XAI_BASE_URL.to_owned()
    } else if is_anthropic_provider_label(&provider) {
        ANTHROPIC_BASE_URL.to_owned()
    } else if is_groq_provider_label(&provider) {
        GROQ_BASE_URL.to_owned()
    } else if is_bedrock_provider_label(&provider) {
        BEDROCK_BASE_URL.to_owned()
    } else if from_openrouter || provider == "openrouter" || provider == "openrouter.ai" {
        "https://openrouter.ai/api/v1".to_owned()
    } else {
        "https://api.openai.com/v1".to_owned()
    }
}

/// `--provider xai` / `grok` / `api.x.ai` (not a loopback host).
pub fn is_xai_provider_label(provider: &str) -> bool {
    matches!(
        provider.to_ascii_lowercase().as_str(),
        "xai" | "grok" | "api.x.ai" | "x.ai"
    )
}

/// `--provider grok-build` / `xai-grok-build` / `cli-chat-proxy.grok.com`.
///
/// Distinct from [`is_xai_provider_label`]: that family chats at
/// `api.x.ai`. This family needs the shipped `xai-grok-build` header
/// pack (`x-grok-client-version`) or the proxy returns HTTP 426.
pub fn is_grok_build_provider_label(provider: &str) -> bool {
    matches!(
        provider.to_ascii_lowercase().as_str(),
        "grok-build" | "xai-grok-build" | "cli-chat-proxy.grok.com"
    )
}

/// `--provider grok-build-messages` / `xai-grok-build-messages`.
///
/// Same host and xAI keys as [`is_grok_build_provider_label`], but
/// shipped `wire = "messages"` (`/v1/messages`).
pub fn is_grok_build_messages_provider_label(provider: &str) -> bool {
    matches!(
        provider.to_ascii_lowercase().as_str(),
        "grok-build-messages" | "xai-grok-build-messages"
    )
}

/// True when the route uses xAI env keys or `~/.grok/auth.json`.
pub fn uses_xai_credentials(provider: &str) -> bool {
    is_xai_provider_label(provider)
        || is_grok_build_provider_label(provider)
        || is_grok_build_messages_provider_label(provider)
}

/// `--provider claude` / `anthropic` / `api.anthropic.com`.
pub fn is_anthropic_provider_label(provider: &str) -> bool {
    matches!(
        provider.to_ascii_lowercase().as_str(),
        "claude" | "anthropic" | "api.anthropic.com"
    )
}

/// `--provider groq` / `api.groq.com`.
pub fn is_groq_provider_label(provider: &str) -> bool {
    matches!(
        provider.to_ascii_lowercase().as_str(),
        "groq" | "api.groq.com"
    )
}

/// `--provider amazon-bedrock` / `bedrock` / a Bedrock runtime host.
pub fn is_bedrock_provider_label(provider: &str) -> bool {
    let provider = provider.to_ascii_lowercase();
    matches!(provider.as_str(), "amazon-bedrock" | "bedrock") || is_bedrock_runtime_host(&provider)
}

fn is_bedrock_runtime_host(host: &str) -> bool {
    let host = host.trim_end_matches('.');
    host == "bedrock-runtime.amazonaws.com"
        || (host.starts_with("bedrock-runtime.") && host.ends_with(".amazonaws.com"))
}

/// True when extra Anthropic headers are required (OAuth + version).
pub fn is_anthropic_cloud_host(base_url: &str) -> bool {
    let host = url_host_hint(base_url);
    let host = host.trim_end_matches('.');
    host == "api.anthropic.com" || host.ends_with(".anthropic.com")
}

/// True when `{base}` is an Ollama OpenAI-compat listener (`*:11434`).
///
/// Native `/api/show` is only safe on this family. Do not POST that
/// path to cloud OpenAI-compat hosts.
pub fn is_ollama_compat_base(base_url: &str) -> bool {
    let hostport = url_host_port_hint(base_url);
    let host = host_without_port(&hostport);
    let host = host.trim_matches(|c| c == '[' || c == ']');
    let loopback = matches!(host, "localhost" | "127.0.0.1" | "0.0.0.0" | "::1");
    loopback && hostport.ends_with(":11434")
}

/// Ollama's listen URL is `:11434` with no path. canact talks to `/v1`.
pub fn normalize_ollama_compat_base(base_url: &str) -> String {
    if !is_ollama_compat_base(base_url) {
        return base_url.to_owned();
    }
    let trimmed = base_url.trim().trim_end_matches('/');
    if trimmed.ends_with("/v1") {
        return trimmed.to_owned();
    }
    if url_path_after_authority(trimmed).is_empty() {
        return format!("{trimmed}/v1");
    }
    base_url.to_owned()
}

fn url_path_after_authority(url: &str) -> &str {
    let rest = url.split_once("://").map(|(_, r)| r).unwrap_or(url);
    rest.find('/').map(|i| &rest[i..]).unwrap_or("")
}

/// Which probe key to send and which default host flags it implies.
///
/// `key` is never logged. Do not `#[derive(Debug)]`.
#[derive(Clone, PartialEq, Eq)]
pub struct KeyRoute {
    /// Bearer token after CLI flag / env resolution. Never log this.
    pub key: Option<String>,
    /// True when the OpenRouter env key selected the default host.
    pub from_openrouter: bool,
    /// True when `XAI_API_KEY` selected the default host.
    pub from_xai: bool,
    /// True when an Anthropic env key selected the default host.
    pub from_anthropic: bool,
}

impl KeyRoute {
    /// Default OpenAI-compat base URL for this route when `--base-url` is omitted.
    pub fn default_base_url(&self, provider: &str) -> String {
        if self.from_xai && provider.is_empty() {
            XAI_BASE_URL.to_owned()
        } else if self.from_anthropic && provider.is_empty() {
            ANTHROPIC_BASE_URL.to_owned()
        } else {
            let from_openrouter = self.from_openrouter && openrouter_default_ok(provider);
            default_compat_base_url(provider, from_openrouter)
        }
    }
}

fn openrouter_default_ok(provider: &str) -> bool {
    let p = provider.to_ascii_lowercase();
    p.is_empty() || p == "openrouter" || p == "openrouter.ai"
}

fn xai_default_ok(provider: &str) -> bool {
    provider.is_empty() || uses_xai_credentials(provider)
}

fn anthropic_default_ok(provider: &str) -> bool {
    provider.is_empty() || is_anthropic_provider_label(provider)
}

/// Whether to call `token_for_profile_cached("anthropic-oauth")`.
///
/// Skip on named non-Anthropic routes so an expired Claude Code keychain
/// item cannot stall an Ollama or xAI probe while wiremux refreshes.
/// The first resolve (empty `--provider`) also skips when `--base-url` is
/// set; `finalize_key_route` re-resolves with the URL-derived provider.
pub fn should_load_claude_code_login(
    provider: &str,
    other_cloud_keys: bool,
    explicit_base_url: bool,
) -> bool {
    if provider.is_empty() && explicit_base_url {
        return false;
    }
    is_anthropic_provider_label(provider) || (provider.is_empty() && !other_cloud_keys)
}

/// Whether to call `token_for_profile_cached("xai-oauth")` (`~/.grok/auth.json`).
///
/// Same skip rules as [`should_load_claude_code_login`]: named non-xAI
/// routes and a first resolve with `--base-url` must not read Grok login.
/// `--provider grok-build` uses the same pack (different host and headers).
pub fn should_load_xai_oauth(
    provider: &str,
    other_cloud_keys: bool,
    explicit_base_url: bool,
) -> bool {
    if provider.is_empty() && explicit_base_url {
        return false;
    }
    uses_xai_credentials(provider) || (provider.is_empty() && !other_cloud_keys)
}

/// Pick a key and host flags from injected values. Parse `provider` first.
///
/// Callers read env vars (or MCP `api_key_env`) and pass the values in.
/// Tests inject keys so they do not race on process-global env.
pub fn resolve_api_key_from(
    cli: Option<String>,
    openai: Option<String>,
    openrouter: Option<String>,
    xai: Option<String>,
    anthropic: Option<String>,
    provider: &str,
) -> KeyRoute {
    if is_groq_provider_label(provider) || is_bedrock_provider_label(provider) {
        return KeyRoute {
            key: cli.filter(|s| !s.is_empty()),
            from_openrouter: false,
            from_xai: false,
            from_anthropic: false,
        };
    }
    let from_openrouter =
        openrouter.is_some() && openai.is_none() && openrouter_default_ok(provider);
    let from_xai = xai.is_some() && openai.is_none() && xai_default_ok(provider);
    let from_anthropic = anthropic.is_some() && openai.is_none() && anthropic_default_ok(provider);
    if let Some(key) = cli
        && !key.is_empty()
    {
        return KeyRoute {
            key: Some(key),
            from_openrouter,
            from_xai: from_xai && !from_openrouter,
            from_anthropic: from_anthropic && !from_openrouter && !from_xai,
        };
    }
    if uses_xai_credentials(provider) {
        let key = xai.filter(|s| !s.is_empty());
        return KeyRoute {
            from_xai: key.is_some(),
            key,
            from_openrouter: false,
            from_anthropic: false,
        };
    }
    if is_anthropic_provider_label(provider) {
        let key = anthropic.filter(|s| !s.is_empty());
        return KeyRoute {
            from_anthropic: key.is_some(),
            key,
            from_openrouter: false,
            from_xai: false,
        };
    }
    if let Some(key) = openai {
        return KeyRoute {
            key: Some(key),
            from_openrouter: false,
            from_xai: false,
            from_anthropic: false,
        };
    }
    if from_xai {
        return KeyRoute {
            key: xai,
            from_openrouter: false,
            from_xai: true,
            from_anthropic: false,
        };
    }
    if from_anthropic {
        return KeyRoute {
            key: anthropic,
            from_openrouter: false,
            from_xai: false,
            from_anthropic: true,
        };
    }
    if let Some(key) = openrouter {
        return KeyRoute {
            key: Some(key),
            from_openrouter,
            from_xai: false,
            from_anthropic: false,
        };
    }
    KeyRoute {
        key: None,
        from_openrouter: false,
        from_xai: false,
        from_anthropic: false,
    }
}

/// Cloud hosts that must not be called without an API key.
pub fn cloud_endpoint_requires_key(base_url: &str) -> bool {
    let host = url_host_hint(base_url);
    let host = host.trim_end_matches('.');
    host == "api.openai.com"
        || host.ends_with(".openai.com")
        || host == "openrouter.ai"
        || host.ends_with(".openrouter.ai")
        || host == "openai.azure.com"
        || host.ends_with(".openai.azure.com")
        || host == "api.x.ai"
        || host == "x.ai"
        || host.ends_with(".x.ai")
        || host == "api.anthropic.com"
        || host.ends_with(".anthropic.com")
        || host == "cli-chat-proxy.grok.com"
        || host.ends_with(".cli-chat-proxy.grok.com")
        || is_groq_cloud_host(base_url)
        || is_bedrock_cloud_host(base_url)
}

/// True when a cloud host must not be called without an API key.
pub fn refuse_cloud_without_key(api_key: Option<&str>, base_url: &str) -> bool {
    api_key.is_none() && cloud_endpoint_requires_key(base_url)
}

/// True when `{base}` is api.x.ai (or another xAI host).
fn is_xai_cloud_host(base_url: &str) -> bool {
    let host = url_host_hint(base_url);
    let host = host.trim_end_matches('.');
    host == "api.x.ai" || host == "x.ai" || host.ends_with(".x.ai")
}

/// True when `{base}` is the Grok Build CLI proxy.
pub fn is_grok_build_cloud_host(base_url: &str) -> bool {
    let host = url_host_hint(base_url);
    let host = host.trim_end_matches('.');
    host == "cli-chat-proxy.grok.com" || host.ends_with(".cli-chat-proxy.grok.com")
}

/// True when `{base}` is api.groq.com.
pub fn is_groq_cloud_host(base_url: &str) -> bool {
    let host = url_host_hint(base_url);
    let host = host.trim_end_matches('.');
    host == "api.groq.com" || host.ends_with(".groq.com")
}

/// True when `{base}` is a Bedrock runtime host.
pub fn is_bedrock_cloud_host(base_url: &str) -> bool {
    let host = url_host_hint(base_url);
    is_bedrock_runtime_host(&host)
}

/// Named-provider missing-key text. Do not list env vars the route will ignore.
pub fn missing_cloud_key_message(provider: &str, base_url: &str) -> &'static str {
    if uses_xai_credentials(provider)
        || is_xai_cloud_host(base_url)
        || is_grok_build_cloud_host(base_url)
    {
        "error: set --api-key or XAI_API_KEY for xAI (OPENAI_API_KEY is not sent)"
    } else if is_anthropic_provider_label(provider) || is_anthropic_cloud_host(base_url) {
        "error: set --api-key, ANTHROPIC_AUTH_TOKEN, or ANTHROPIC_API_KEY for Anthropic (OPENAI_API_KEY is not sent)"
    } else if is_groq_provider_label(provider) || is_groq_cloud_host(base_url) {
        "error: set --api-key or GROQ_API_KEY for Groq (OPENAI_API_KEY is not sent)"
    } else if is_bedrock_provider_label(provider) || is_bedrock_cloud_host(base_url) {
        "error: set --api-key or AWS_BEARER_TOKEN_BEDROCK for Amazon Bedrock (OPENAI_API_KEY is not sent)"
    } else if openrouter_default_ok(provider) && !provider.is_empty() {
        "error: set --api-key, OPENROUTER_API_KEY, or OPENAI_API_KEY for OpenRouter"
    } else if is_openai_provider_label(provider) || is_openai_cloud_host(base_url) {
        "error: set --api-key or OPENAI_API_KEY"
    } else {
        "error: set --api-key, OPENAI_API_KEY, OPENROUTER_API_KEY, XAI_API_KEY, ANTHROPIC_AUTH_TOKEN, or ANTHROPIC_API_KEY (or pass --base-url for a local host)"
    }
}

fn is_openai_provider_label(provider: &str) -> bool {
    matches!(
        provider.to_ascii_lowercase().as_str(),
        "openai" | "api.openai.com"
    )
}

fn is_openai_cloud_host(base_url: &str) -> bool {
    let host = url_host_hint(base_url);
    let host = host.trim_end_matches('.');
    host == "api.openai.com" || host.ends_with(".openai.com")
}

/// Present `--base-url` / MCP `base_url` after trim. Whitespace-only is absent.
pub fn present_base_url(raw: Option<&str>) -> Option<&str> {
    raw.map(str::trim).filter(|s| !s.is_empty())
}

/// After an explicit base URL is known, re-resolve the key when
/// the user omitted `--provider` / MCP `provider`.
pub fn finalize_key_route(
    provider_given: &str,
    explicit_base_url: Option<String>,
    first: KeyRoute,
    re_resolve: impl FnOnce(&str) -> KeyRoute,
) -> (KeyRoute, String, String) {
    let explicit_base_url = explicit_base_url.and_then(|s| {
        let t = s.trim();
        if t.is_empty() {
            None
        } else {
            Some(t.to_owned())
        }
    });
    let has_explicit = explicit_base_url.is_some();
    let base_url = normalize_ollama_compat_base(
        &explicit_base_url.unwrap_or_else(|| first.default_base_url(provider_given)),
    );
    let provider = if provider_given.is_empty() {
        provider_from_base_url(&base_url)
    } else {
        provider_given.to_owned()
    };
    let route = if provider_given.is_empty() && has_explicit {
        re_resolve(&provider)
    } else {
        first
    };
    (route, base_url, provider)
}

/// True when the host or model looks local/free so the cheap suite is enough.
/// Host label used as `provider` when the user omitted `--provider`.
/// Loopback URLs keep `host:port` so different listeners do not share a cache row.
pub fn provider_from_base_url(base_url: &str) -> String {
    let hostport = url_host_port_hint(base_url);
    if hostport.is_empty() {
        "openai-compat".to_owned()
    } else {
        hostport
    }
}

pub fn looks_cheap(provider: &str, model: &str, base_url: &str) -> bool {
    let provider = provider.to_ascii_lowercase();
    let host = url_host_hint(base_url);
    model.contains(":free")
        || is_local_provider_label(&provider)
        || matches!(host.as_str(), "localhost" | "127.0.0.1" | "0.0.0.0" | "::1")
}

fn is_local_provider_label(provider: &str) -> bool {
    matches!(
        provider,
        "ollama" | "lmstudio" | "vllm" | "localhost" | "127.0.0.1" | "::1" | "[::1]" | "0.0.0.0"
    ) || loopback_host_port_base_url(provider).is_some()
}

fn loopback_host_port_base_url(provider: &str) -> Option<String> {
    let host = host_without_port(provider);
    if host == provider {
        return None;
    }
    let port = provider.rsplit_once(':').map(|(_, p)| p)?;
    if port.is_empty() || !port.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let bare = host
        .strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(host);
    if !matches!(bare, "localhost" | "127.0.0.1" | "0.0.0.0" | "::1") {
        return None;
    }
    if bare == "::1" {
        Some(format!("http://[::1]:{port}/v1"))
    } else {
        Some(format!("http://{host}:{port}/v1"))
    }
}

fn url_host_hint(url: &str) -> String {
    let hostport = url_host_port_hint(url);
    host_without_port(&hostport).to_owned()
}

fn url_host_port_hint(url: &str) -> String {
    let url = url.trim().to_ascii_lowercase();
    let after_scheme = url.split("://").nth(1).unwrap_or(&url);
    let authority = after_scheme
        .split(['/', '?', '#'])
        .next()
        .unwrap_or(after_scheme);
    let hostport = authority
        .rsplit_once('@')
        .map(|(_, h)| h)
        .unwrap_or(authority);
    if let Some(rest) = hostport.strip_prefix('[') {
        let (host, after) = rest.split_once(']').unwrap_or((rest, ""));
        if let Some(port) = after.strip_prefix(':').filter(|p| !p.is_empty()) {
            return format!("{host}:{port}");
        }
        return host.to_owned();
    }
    hostport.to_owned()
}

fn host_without_port(hostport: &str) -> &str {
    if let Some((host, port)) = hostport.rsplit_once(':')
        && !host.is_empty()
        && !port.is_empty()
        && port.bytes().all(|b| b.is_ascii_digit())
    {
        return host;
    }
    hostport
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openrouter_provider_uses_openrouter_key_when_xai_also_set() {
        let route = resolve_api_key_from(
            None,
            None,
            Some("sk-or-env".to_owned()),
            Some("xai-env".to_owned()),
            None,
            "openrouter",
        );
        assert_eq!(route.key.as_deref(), Some("sk-or-env"));
        assert!(route.from_openrouter);
        assert!(!route.from_xai);
        assert_eq!(
            route.default_base_url("openrouter"),
            "https://openrouter.ai/api/v1"
        );
    }

    #[test]
    fn anthropic_provider_uses_anthropic_key_when_xai_also_set() {
        let route = resolve_api_key_from(
            None,
            None,
            None,
            Some("xai-env".to_owned()),
            Some("sk-ant-env".to_owned()),
            "anthropic",
        );
        assert_eq!(route.key.as_deref(), Some("sk-ant-env"));
        assert!(route.from_anthropic);
        assert!(!route.from_xai);
        assert_eq!(route.default_base_url("anthropic"), ANTHROPIC_BASE_URL);
        let xai_only = resolve_api_key_from(
            None,
            None,
            None,
            Some("xai-env".to_owned()),
            None,
            "anthropic",
        );
        assert!(
            xai_only.key.is_none(),
            "provider=anthropic must not reuse XAI_API_KEY"
        );
    }

    #[test]
    fn refuse_cloud_without_key_gates_cloud_hosts() {
        assert!(refuse_cloud_without_key(None, "https://api.openai.com/v1"));
        assert!(refuse_cloud_without_key(None, "https://api.x.ai/v1"));
        assert!(refuse_cloud_without_key(
            None,
            "https://api.anthropic.com/v1"
        ));
        assert!(refuse_cloud_without_key(
            None,
            "https://cli-chat-proxy.grok.com/v1"
        ));
        assert!(!refuse_cloud_without_key(None, "http://127.0.0.1:11434/v1"));
        assert!(!refuse_cloud_without_key(
            Some("sk"),
            "https://api.openai.com/v1"
        ));
    }

    #[test]
    fn should_load_claude_code_login_only_for_anthropic_or_empty_no_other_keys() {
        assert!(should_load_claude_code_login("claude", false, false));
        assert!(should_load_claude_code_login("anthropic", true, false));
        assert!(should_load_claude_code_login(
            "api.anthropic.com",
            true,
            true
        ));
        assert!(should_load_claude_code_login("", false, false));
        assert!(!should_load_claude_code_login("", true, false));
        assert!(!should_load_claude_code_login("", false, true));
        assert!(!should_load_claude_code_login("xai", false, false));
        assert!(!should_load_claude_code_login("ollama", false, false));
        assert!(!should_load_claude_code_login("openai", false, false));
    }

    #[test]
    fn should_load_xai_oauth_only_for_xai_or_empty_no_other_keys() {
        assert!(should_load_xai_oauth("xai", false, false));
        assert!(should_load_xai_oauth("grok", true, false));
        assert!(should_load_xai_oauth("api.x.ai", true, true));
        assert!(should_load_xai_oauth("", false, false));
        assert!(!should_load_xai_oauth("", true, false));
        assert!(!should_load_xai_oauth("", false, true));
        assert!(!should_load_xai_oauth("claude", false, false));
        assert!(!should_load_xai_oauth("ollama", false, false));
        assert!(!should_load_xai_oauth("openai", false, false));
    }

    #[test]
    fn missing_cloud_key_message_names_the_route() {
        let xai = missing_cloud_key_message("xai", "https://api.x.ai/v1");
        assert!(xai.contains("XAI_API_KEY"), "{xai}");
        assert!(
            !xai.contains("set --api-key, OPENAI_API_KEY"),
            "xAI must not list OPENAI_API_KEY as the fix: {xai}"
        );
        let claude = missing_cloud_key_message("claude", "https://api.anthropic.com/v1");
        assert!(claude.contains("ANTHROPIC_AUTH_TOKEN"), "{claude}");
        assert!(
            !claude.contains("set --api-key, OPENAI_API_KEY"),
            "Claude must not list OPENAI_API_KEY as the fix: {claude}"
        );
        let openai = missing_cloud_key_message("openai", "https://api.openai.com/v1");
        assert!(openai.contains("OPENAI_API_KEY"), "{openai}");
        let groq = missing_cloud_key_message("groq", GROQ_BASE_URL);
        assert!(groq.contains("GROQ_API_KEY"), "{groq}");
        assert!(
            !groq.contains("set --api-key, OPENAI_API_KEY"),
            "Groq must not list OPENAI_API_KEY as the fix: {groq}"
        );
        let bedrock = missing_cloud_key_message("amazon-bedrock", BEDROCK_BASE_URL);
        assert!(bedrock.contains("AWS_BEARER_TOKEN_BEDROCK"), "{bedrock}");
        assert!(
            !bedrock.contains("set --api-key, OPENAI_API_KEY"),
            "Bedrock must not list OPENAI_API_KEY as the fix: {bedrock}"
        );
    }

    #[test]
    fn ollama_defaults_to_loopback_not_openai() {
        assert_eq!(default_compat_base_url("ollama", false), OLLAMA_BASE_URL);
        assert_eq!(default_compat_base_url("Ollama", true), OLLAMA_BASE_URL);
        assert!(!cloud_endpoint_requires_key(OLLAMA_BASE_URL));
    }

    #[test]
    fn all_zeros_provider_defaults_to_ollama_url() {
        assert_eq!(
            default_compat_base_url("0.0.0.0", false),
            OLLAMA_BASE_URL,
            "provider 0.0.0.0 must default like ollama"
        );
        assert!(looks_cheap("0.0.0.0", "qwen", "http://example.invalid/v1"));
    }

    #[test]
    fn looks_cheap_uses_url_host_not_raw_substring() {
        assert!(
            !looks_cheap("openai", "gpt-4o", "https://10.0.0.0/v1"),
            "0.0.0.0 substring in 10.0.0.0 must not look cheap"
        );
        assert!(
            !looks_cheap("openai", "gpt-4o", "https://127.0.0.1@api.openai.com/v1"),
            "loopback userinfo must not make a cloud host look cheap"
        );
        assert!(looks_cheap(
            "openai-compat",
            "llama3",
            "http://0.0.0.0:11434/v1"
        ));
        assert!(looks_cheap(
            "openai-compat",
            "llama3",
            "http://127.0.0.1:11434/v1"
        ));
    }

    #[test]
    fn ollama_compat_base_is_loopback_11434_only() {
        assert!(is_ollama_compat_base(OLLAMA_BASE_URL));
        assert!(is_ollama_compat_base("http://localhost:11434/v1"));
        assert!(is_ollama_compat_base("http://[::1]:11434/v1"));
        assert!(is_ollama_compat_base("http://0.0.0.0:11434/v1"));
        assert!(!is_ollama_compat_base(LMSTUDIO_BASE_URL));
        assert!(!is_ollama_compat_base(VLLM_BASE_URL));
        assert!(!is_ollama_compat_base("https://api.openai.com/v1"));
        assert!(!is_ollama_compat_base("https://openrouter.ai/api/v1"));
        assert!(!is_ollama_compat_base(XAI_BASE_URL));
        assert!(!is_ollama_compat_base(ANTHROPIC_BASE_URL));
    }

    #[test]
    fn is_ollama_compat_base_trims_whitespace() {
        assert!(is_ollama_compat_base("http://127.0.0.1:11434 "));
        assert!(is_ollama_compat_base(" http://127.0.0.1:11434"));
    }

    #[test]
    fn normalize_ollama_listen_url_appends_v1() {
        assert_eq!(
            normalize_ollama_compat_base("http://127.0.0.1:11434"),
            "http://127.0.0.1:11434/v1"
        );
        assert_eq!(
            normalize_ollama_compat_base("http://127.0.0.1:11434/"),
            "http://127.0.0.1:11434/v1"
        );
        assert_eq!(
            normalize_ollama_compat_base("http://localhost:11434/v1"),
            "http://localhost:11434/v1"
        );
        assert_eq!(
            normalize_ollama_compat_base("http://127.0.0.1:11434/api"),
            "http://127.0.0.1:11434/api"
        );
        assert_eq!(
            normalize_ollama_compat_base("http://127.0.0.1:1234"),
            "http://127.0.0.1:1234"
        );
        assert!(is_ollama_compat_base("http://127.0.0.1:11434 "));
        assert_eq!(
            normalize_ollama_compat_base("http://127.0.0.1:11434 "),
            "http://127.0.0.1:11434/v1"
        );
        assert_eq!(
            normalize_ollama_compat_base(" http://127.0.0.1:11434"),
            "http://127.0.0.1:11434/v1"
        );
    }

    #[test]
    fn loopback_provider_label_defaults_to_ollama_url() {
        for provider in ["localhost", "127.0.0.1", "::1", "[::1]"] {
            assert_eq!(
                default_compat_base_url(provider, false),
                OLLAMA_BASE_URL,
                "provider {provider} must not default to OpenAI"
            );
            assert_eq!(
                default_compat_base_url(provider, true),
                OLLAMA_BASE_URL,
                "provider {provider} must stay local even when OpenRouter env is set"
            );
        }
    }

    #[test]
    fn unknown_provider_stays_on_openai_or_openrouter() {
        assert_eq!(
            default_compat_base_url("openai", false),
            "https://api.openai.com/v1"
        );
        assert_eq!(
            default_compat_base_url("openai", true),
            "https://openrouter.ai/api/v1"
        );
        assert!(cloud_endpoint_requires_key("https://api.openai.com/v1"));
        assert!(cloud_endpoint_requires_key("https://openrouter.ai/api/v1"));
        assert!(cloud_endpoint_requires_key(
            "https://eastus.openai.azure.com/openai/v1"
        ));
        assert_eq!(
            default_compat_base_url("openrouter", false),
            "https://openrouter.ai/api/v1"
        );
        assert!(!cloud_endpoint_requires_key(
            "https://myopenrouter.ai.internal/v1"
        ));
        assert_eq!(
            provider_from_base_url("https://api.openai.com/v1"),
            "api.openai.com"
        );
    }

    #[test]
    fn ipv6_loopback_host_is_not_open_bracket() {
        assert_eq!(
            provider_from_base_url("http://[::1]:11434/v1"),
            "::1:11434",
            "IPv6 authority must keep host:port and not split on the first colon"
        );
        assert!(looks_cheap("::1", "qwen", "http://[::1]:11434/v1"));
    }

    #[test]
    fn url_derived_provider_keeps_loopback_port() {
        assert_eq!(
            provider_from_base_url("http://127.0.0.1:1234/v1"),
            "127.0.0.1:1234"
        );
        assert_eq!(
            provider_from_base_url("http://localhost:1234/v1"),
            "localhost:1234"
        );
        assert_eq!(
            provider_from_base_url("https://api.openai.com/v1"),
            "api.openai.com",
            "cloud hosts without an explicit port stay host-only"
        );
    }

    #[test]
    fn host_port_loopback_provider_defaults_to_loopback_url() {
        assert_eq!(
            default_compat_base_url("127.0.0.1:1234", false),
            "http://127.0.0.1:1234/v1",
            "provider 127.0.0.1:1234 must stay on loopback, not api.openai.com"
        );
        assert_eq!(
            default_compat_base_url("localhost:11434", false),
            "http://localhost:11434/v1"
        );
        assert_eq!(
            default_compat_base_url("[::1]:11434", false),
            "http://[::1]:11434/v1"
        );
        assert_eq!(
            default_compat_base_url("127.0.0.1:1234", true),
            "http://127.0.0.1:1234/v1",
            "loopback host:port must stay local even when OpenRouter env is set"
        );
        assert_ne!(
            default_compat_base_url("127.0.0.1:1234", false),
            "https://api.openai.com/v1"
        );
        assert!(looks_cheap(
            "127.0.0.1:1234",
            "qwen",
            "http://example.invalid/v1"
        ));
        assert!(looks_cheap(
            "localhost:11434",
            "qwen",
            "http://example.invalid/v1"
        ));
        assert!(looks_cheap(
            "[::1]:11434",
            "qwen",
            "http://example.invalid/v1"
        ));
        assert_ne!(
            local_provider_base_url("127.0.0.1:1234"),
            local_provider_base_url("ollama"),
            "cache isolation of :1234 vs the ollama label must stay"
        );
    }

    #[test]
    fn userinfo_does_not_hide_openai_cloud_host() {
        assert!(
            cloud_endpoint_requires_key("https://user:pass@api.openai.com/v1"),
            "userinfo must not skip the cloud-key gate"
        );
        assert_eq!(
            provider_from_base_url("https://user:pass@api.openai.com/v1"),
            "api.openai.com"
        );
    }

    #[test]
    fn trailing_dot_fqdn_still_requires_cloud_key() {
        assert!(
            cloud_endpoint_requires_key("https://api.openai.com./v1"),
            "trailing-dot api.openai.com. must still require a key"
        );
        assert!(
            cloud_endpoint_requires_key("https://openrouter.ai./api/v1"),
            "trailing-dot openrouter.ai. must still require a key"
        );
    }

    #[test]
    fn xai_provider_defaults_to_api_x_ai_and_requires_key() {
        for provider in ["xai", "grok", "api.x.ai", "x.ai", "Xai"] {
            assert_eq!(
                default_compat_base_url(provider, false),
                XAI_BASE_URL,
                "provider {provider} must not default to OpenAI"
            );
            assert_eq!(
                default_compat_base_url(provider, true),
                XAI_BASE_URL,
                "provider {provider} must stay on xAI even when OpenRouter env is set"
            );
        }
        assert!(cloud_endpoint_requires_key(XAI_BASE_URL));
        assert!(
            cloud_endpoint_requires_key("https://api.x.ai./v1"),
            "trailing-dot api.x.ai. must still require a key"
        );
        assert!(!cloud_endpoint_requires_key("https://notx.ai.internal/v1"));
        assert_eq!(provider_from_base_url(XAI_BASE_URL), "api.x.ai");
    }

    #[test]
    fn grok_build_provider_defaults_to_cli_proxy_and_requires_key() {
        for provider in ["grok-build", "xai-grok-build", "cli-chat-proxy.grok.com"] {
            assert_eq!(
                default_compat_base_url(provider, false),
                GROK_BUILD_BASE_URL,
                "provider {provider} must not default to api.x.ai"
            );
            assert_eq!(
                default_compat_base_url(provider, true),
                GROK_BUILD_BASE_URL,
                "provider {provider} must stay on grok-build even when OpenRouter env is set"
            );
            assert!(
                is_grok_build_provider_label(provider),
                "{provider} is the Grok Build family"
            );
            assert!(
                !is_xai_provider_label(provider),
                "{provider} must not share the api.x.ai label"
            );
            assert!(uses_xai_credentials(provider));
        }
        assert_eq!(
            default_compat_base_url("grok", false),
            XAI_BASE_URL,
            "`grok` stays on api.x.ai; grok-build is the CLI proxy"
        );
        assert!(cloud_endpoint_requires_key(GROK_BUILD_BASE_URL));
        assert!(is_grok_build_cloud_host(GROK_BUILD_BASE_URL));
        assert!(
            cloud_endpoint_requires_key("https://cli-chat-proxy.grok.com./v1"),
            "trailing-dot cli-chat-proxy.grok.com. must still require a key"
        );
        assert!(!is_grok_build_cloud_host(XAI_BASE_URL));
        let msg = missing_cloud_key_message("grok-build", GROK_BUILD_BASE_URL);
        assert!(msg.contains("XAI_API_KEY"), "{msg}");
        assert!(
            !msg.contains("set --api-key, OPENAI_API_KEY"),
            "grok-build must not list OPENAI_API_KEY as the fix: {msg}"
        );
        assert!(should_load_xai_oauth("grok-build", false, false));
        assert!(should_load_xai_oauth("xai-grok-build", true, false));
        assert!(!should_load_claude_code_login("grok-build", false, false));
    }

    #[test]
    fn grok_build_messages_provider_stays_on_cli_proxy_and_xai_keys() {
        for provider in ["grok-build-messages", "xai-grok-build-messages"] {
            assert_eq!(
                default_compat_base_url(provider, false),
                GROK_BUILD_BASE_URL,
                "provider {provider} must not default to api.x.ai"
            );
            assert!(is_grok_build_messages_provider_label(provider));
            assert!(
                !is_grok_build_provider_label(provider),
                "{provider} is Messages, not chat-completions"
            );
            assert!(!is_xai_provider_label(provider));
            assert!(uses_xai_credentials(provider));
            assert!(should_load_xai_oauth(provider, false, false));
            assert!(!should_load_claude_code_login(provider, false, false));
        }
    }

    #[test]
    fn claude_provider_defaults_to_anthropic_and_requires_key() {
        for provider in ["claude", "anthropic", "api.anthropic.com", "Claude"] {
            assert_eq!(
                default_compat_base_url(provider, false),
                ANTHROPIC_BASE_URL,
                "provider {provider} must not default to OpenAI"
            );
            assert_eq!(
                default_compat_base_url(provider, true),
                ANTHROPIC_BASE_URL,
                "provider {provider} must stay on Anthropic even when OpenRouter env is set"
            );
        }
        assert!(cloud_endpoint_requires_key(ANTHROPIC_BASE_URL));
        assert!(
            cloud_endpoint_requires_key("https://api.anthropic.com./v1"),
            "trailing-dot api.anthropic.com. must still require a key"
        );
        assert!(is_anthropic_cloud_host(ANTHROPIC_BASE_URL));
        assert!(
            is_anthropic_cloud_host("https://api.anthropic.com./v1"),
            "trailing-dot api.anthropic.com. still needs Anthropic OAuth headers"
        );
        assert!(!is_anthropic_cloud_host(XAI_BASE_URL));
        assert_eq!(
            provider_from_base_url(ANTHROPIC_BASE_URL),
            "api.anthropic.com"
        );
    }

    #[test]
    fn groq_provider_defaults_to_groq_and_requires_key() {
        for provider in ["groq", "api.groq.com", "Groq"] {
            assert_eq!(
                default_compat_base_url(provider, false),
                GROQ_BASE_URL,
                "provider {provider} must not default to OpenAI"
            );
            assert_eq!(
                default_compat_base_url(provider, true),
                GROQ_BASE_URL,
                "provider {provider} must stay on Groq even when OpenRouter env is set"
            );
            assert!(is_groq_provider_label(provider));
        }
        assert!(cloud_endpoint_requires_key(GROQ_BASE_URL));
        assert!(
            cloud_endpoint_requires_key("https://api.groq.com./openai/v1"),
            "trailing-dot api.groq.com. must still require a key"
        );
        assert!(is_groq_cloud_host(GROQ_BASE_URL));
        assert!(!is_groq_cloud_host(XAI_BASE_URL));
        assert_eq!(provider_from_base_url(GROQ_BASE_URL), "api.groq.com");
        let openai =
            resolve_api_key_from(None, Some("sk-openai".to_owned()), None, None, None, "groq");
        assert!(
            openai.key.is_none(),
            "provider=groq must not send OPENAI_API_KEY"
        );
        let named = resolve_api_key_from(
            Some("gsk-test".to_owned()),
            Some("sk-openai".to_owned()),
            None,
            None,
            None,
            "groq",
        );
        assert_eq!(named.key.as_deref(), Some("gsk-test"));
        assert!(!named.from_openrouter);
        assert!(!named.from_xai);
        assert!(!named.from_anthropic);
        assert!(!should_load_xai_oauth("groq", false, false));
        assert!(!should_load_claude_code_login("groq", false, false));
    }

    #[test]
    fn bedrock_provider_defaults_to_bedrock_and_requires_key() {
        for provider in ["amazon-bedrock", "bedrock", "Bedrock"] {
            assert_eq!(
                default_compat_base_url(provider, false),
                BEDROCK_BASE_URL,
                "provider {provider} must not default to OpenAI"
            );
            assert_eq!(
                default_compat_base_url(provider, true),
                BEDROCK_BASE_URL,
                "provider {provider} must stay on Bedrock even when OpenRouter env is set"
            );
            assert!(is_bedrock_provider_label(provider));
        }
        assert!(is_bedrock_provider_label(
            "bedrock-runtime.us-west-2.amazonaws.com"
        ));
        assert!(cloud_endpoint_requires_key(BEDROCK_BASE_URL));
        assert!(cloud_endpoint_requires_key(
            "https://bedrock-runtime.eu-west-1.amazonaws.com"
        ));
        assert!(
            cloud_endpoint_requires_key("https://bedrock-runtime.us-east-1.amazonaws.com./"),
            "trailing-dot Bedrock host must still require a key"
        );
        assert!(!cloud_endpoint_requires_key(
            "https://notbedrock-runtime.amazonaws.com"
        ));
        assert!(is_bedrock_cloud_host(BEDROCK_BASE_URL));
        assert!(!is_bedrock_cloud_host(XAI_BASE_URL));
        let openai = resolve_api_key_from(
            None,
            Some("sk-openai".to_owned()),
            None,
            None,
            None,
            "amazon-bedrock",
        );
        assert!(
            openai.key.is_none(),
            "provider=amazon-bedrock must not send OPENAI_API_KEY"
        );
        let named = resolve_api_key_from(
            Some("bedrock-token".to_owned()),
            Some("sk-openai".to_owned()),
            None,
            None,
            None,
            "bedrock",
        );
        assert_eq!(named.key.as_deref(), Some("bedrock-token"));
        assert!(!should_load_xai_oauth("amazon-bedrock", false, false));
        assert!(!should_load_claude_code_login("bedrock", false, false));
    }

    #[test]
    fn finalize_key_route_empty_provider_explicit_anthropic_url_uses_anthropic_key() {
        let first = resolve_api_key_from(
            None,
            None,
            None,
            Some("xai-env".to_owned()),
            Some("sk-ant-env".to_owned()),
            "",
        );
        assert!(first.from_xai, "precondition: empty provider prefers xAI");
        let (route, base_url, provider) =
            finalize_key_route("", Some(ANTHROPIC_BASE_URL.to_owned()), first, |p| {
                resolve_api_key_from(
                    None,
                    None,
                    None,
                    Some("xai-env".to_owned()),
                    Some("sk-ant-env".to_owned()),
                    p,
                )
            });
        assert_eq!(route.key.as_deref(), Some("sk-ant-env"));
        assert!(route.from_anthropic);
        assert!(!route.from_xai);
        assert_eq!(base_url, ANTHROPIC_BASE_URL);
        assert_eq!(provider, "api.anthropic.com");
    }

    #[test]
    fn finalize_key_route_empty_provider_explicit_xai_url_does_not_use_anthropic_key() {
        let first = resolve_api_key_from(None, None, None, None, Some("sk-ant".to_owned()), "");
        assert!(
            first.from_anthropic,
            "precondition: empty provider with Anthropic-only key selects Anthropic"
        );
        let (route, base_url, provider) =
            finalize_key_route("", Some(XAI_BASE_URL.to_owned()), first, |p| {
                resolve_api_key_from(None, None, None, None, Some("sk-ant".to_owned()), p)
            });
        assert!(
            route.key.is_none(),
            "must not send an Anthropic key to api.x.ai"
        );
        assert!(!route.from_anthropic);
        assert_eq!(base_url, XAI_BASE_URL);
        assert_eq!(provider, "api.x.ai");
    }

    #[test]
    fn finalize_key_route_empty_provider_no_url_still_prefers_xai() {
        let first = resolve_api_key_from(
            None,
            None,
            None,
            Some("xai-env".to_owned()),
            Some("sk-ant".to_owned()),
            "",
        );
        let (route, base_url, provider) = finalize_key_route("", None, first, |_| {
            panic!("must not re-resolve when URL was omitted")
        });
        assert_eq!(route.key.as_deref(), Some("xai-env"));
        assert!(route.from_xai);
        assert!(!route.from_anthropic);
        assert_eq!(base_url, XAI_BASE_URL);
        assert_eq!(provider, "api.x.ai");
    }

    #[test]
    fn finalize_key_route_explicit_provider_wins_over_url() {
        let first = resolve_api_key_from(
            None,
            None,
            None,
            Some("xai-env".to_owned()),
            Some("sk-ant-env".to_owned()),
            "claude",
        );
        assert!(first.from_anthropic);
        let (route, base_url, provider) =
            finalize_key_route("claude", Some(XAI_BASE_URL.to_owned()), first, |_| {
                panic!("must not re-resolve when --provider was given")
            });
        assert_eq!(route.key.as_deref(), Some("sk-ant-env"));
        assert!(route.from_anthropic);
        assert!(!route.from_xai);
        assert_eq!(base_url, XAI_BASE_URL);
        assert_eq!(provider, "claude");
    }

    #[test]
    fn named_xai_provider_does_not_use_openai_key() {
        let route = resolve_api_key_from(
            None,
            Some("sk-openai".to_owned()),
            None,
            Some("xai-env".to_owned()),
            None,
            "api.x.ai",
        );
        assert_eq!(route.key.as_deref(), Some("xai-env"));
        assert!(route.from_xai);
        assert!(!route.from_anthropic);
    }

    #[test]
    fn finalize_key_route_explicit_xai_url_does_not_use_openai_key() {
        let first = resolve_api_key_from(
            None,
            Some("sk-openai".to_owned()),
            None,
            Some("xai-env".to_owned()),
            None,
            "",
        );
        assert_eq!(
            first.key.as_deref(),
            Some("sk-openai"),
            "precondition: empty provider still prefers OPENAI_API_KEY"
        );
        let (route, base_url, provider) =
            finalize_key_route("", Some(XAI_BASE_URL.to_owned()), first, |p| {
                resolve_api_key_from(
                    None,
                    Some("sk-openai".to_owned()),
                    None,
                    Some("xai-env".to_owned()),
                    None,
                    p,
                )
            });
        assert_eq!(route.key.as_deref(), Some("xai-env"));
        assert!(route.from_xai);
        assert_eq!(base_url, XAI_BASE_URL);
        assert_eq!(provider, "api.x.ai");
    }

    #[test]
    fn finalize_key_route_explicit_anthropic_url_does_not_use_openai_key() {
        let first = resolve_api_key_from(
            None,
            Some("sk-openai".to_owned()),
            None,
            None,
            Some("sk-ant-env".to_owned()),
            "",
        );
        let (route, _, provider) =
            finalize_key_route("", Some(ANTHROPIC_BASE_URL.to_owned()), first, |p| {
                resolve_api_key_from(
                    None,
                    Some("sk-openai".to_owned()),
                    None,
                    None,
                    Some("sk-ant-env".to_owned()),
                    p,
                )
            });
        assert_eq!(route.key.as_deref(), Some("sk-ant-env"));
        assert!(route.from_anthropic);
        assert_eq!(provider, "api.anthropic.com");
    }

    #[test]
    fn finalize_key_route_explicit_ollama_url_trims_and_appends_v1() {
        let first = resolve_api_key_from(None, None, None, None, None, "");
        let (route, base_url, provider) =
            finalize_key_route("", Some("http://127.0.0.1:11434 ".to_owned()), first, |p| {
                resolve_api_key_from(None, None, None, None, None, p)
            });
        assert!(route.key.is_none());
        assert_eq!(base_url, "http://127.0.0.1:11434/v1");
        assert_eq!(provider, "127.0.0.1:11434");
    }

    #[test]
    fn finalize_key_route_whitespace_only_url_is_absent() {
        let first = resolve_api_key_from(
            None,
            None,
            None,
            Some("xai-env".to_owned()),
            Some("sk-ant".to_owned()),
            "",
        );
        let (route, base_url, provider) =
            finalize_key_route("", Some("  ".to_owned()), first, |_| {
                panic!("must not re-resolve when URL was whitespace-only")
            });
        assert_eq!(route.key.as_deref(), Some("xai-env"));
        assert!(route.from_xai);
        assert!(!route.from_anthropic);
        assert_eq!(base_url, XAI_BASE_URL);
        assert_eq!(provider, "api.x.ai");
    }

    #[test]
    fn present_base_url_whitespace_only_is_absent() {
        assert_eq!(present_base_url(Some("  ")), None);
        assert_eq!(present_base_url(Some("")), None);
        assert_eq!(present_base_url(None), None);
        assert_eq!(
            present_base_url(Some(" http://127.0.0.1:11434 ")),
            Some("http://127.0.0.1:11434")
        );
        assert!(should_load_xai_oauth(
            "",
            false,
            present_base_url(Some("  ")).is_some()
        ));
        assert!(!should_load_xai_oauth(
            "",
            false,
            present_base_url(Some("http://127.0.0.1:11434")).is_some()
        ));
    }
}

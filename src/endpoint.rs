//! Default OpenAI-compatible base URLs for known local providers.

/// Ollama's OpenAI-compatible listener.
pub const OLLAMA_BASE_URL: &str = "http://127.0.0.1:11434/v1";
/// LM Studio's OpenAI-compatible listener.
pub const LMSTUDIO_BASE_URL: &str = "http://127.0.0.1:1234/v1";
/// Common vLLM OpenAI-compatible listener.
pub const VLLM_BASE_URL: &str = "http://127.0.0.1:8000/v1";

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
    if from_openrouter || provider == "openrouter" || provider == "openrouter.ai" {
        "https://openrouter.ai/api/v1".to_owned()
    } else {
        "https://api.openai.com/v1".to_owned()
    }
}

/// Cloud hosts that must not be called without an API key.
pub fn cloud_endpoint_requires_key(base_url: &str) -> bool {
    let host = url_host_hint(base_url);
    host == "api.openai.com"
        || host.ends_with(".openai.com")
        || host == "openrouter.ai"
        || host.ends_with(".openrouter.ai")
        || host == "openai.azure.com"
        || host.ends_with(".openai.azure.com")
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
    let url = url.to_ascii_lowercase();
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
    if let Some((host, port)) = hostport.rsplit_once(':') {
        if !host.is_empty() && !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit()) {
            return host;
        }
    }
    hostport
}

#[cfg(test)]
mod tests {
    use super::*;

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
}

//! Default OpenAI-compatible base URLs for known local providers.

/// Ollama's OpenAI-compatible listener.
pub const OLLAMA_BASE_URL: &str = "http://127.0.0.1:11434/v1";
/// LM Studio's OpenAI-compatible listener.
pub const LMSTUDIO_BASE_URL: &str = "http://127.0.0.1:1234/v1";
/// Common vLLM OpenAI-compatible listener.
pub const VLLM_BASE_URL: &str = "http://127.0.0.1:8000/v1";

/// Local-provider default when the user omitted `--base-url`.
pub fn local_provider_base_url(provider: &str) -> Option<&'static str> {
    match provider.to_ascii_lowercase().as_str() {
        "ollama" => Some(OLLAMA_BASE_URL),
        "lmstudio" => Some(LMSTUDIO_BASE_URL),
        "vllm" => Some(VLLM_BASE_URL),
        _ => None,
    }
}

/// Base URL when `--base-url` is omitted.
pub fn default_compat_base_url(provider: &str, from_openrouter: bool) -> String {
    if let Some(local) = local_provider_base_url(provider) {
        return local.to_owned();
    }
    if from_openrouter {
        "https://openrouter.ai/api/v1".to_owned()
    } else {
        "https://api.openai.com/v1".to_owned()
    }
}

/// Cloud hosts that must not be called without an API key.
pub fn cloud_endpoint_requires_key(base_url: &str) -> bool {
    let url = base_url.to_ascii_lowercase();
    url.contains("api.openai.com") || url.contains("openrouter.ai")
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
    }
}

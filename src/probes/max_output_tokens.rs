//! Cheap measured output-token cap (`maxOutputTokens`).
//!
//! Ask for a large `max_tokens` on a one-word prompt. Parse a provider
//! reject (`32768 > 8192`). Do not generate a 4k-32k completion.

use crate::ProbeError;
use crate::client::{ProbeClient, ProbeRequest};

use super::user_text;

/// `max_tokens` large enough to trip 4k/8k output caps. Not a generation climb.
pub const OVERSIZE_MAX_TOKENS: u32 = 32_768;

/// Probe the provider output cap. `None` when unmeasured.
pub async fn probe_max_output_tokens<C: ProbeClient>(llm: &C) -> Result<Option<u32>, ProbeError> {
    let request = ProbeRequest {
        messages: vec![user_text("Reply with the single word ok.")],
        tools: vec![],
        model: llm.model_id().to_string(),
        temperature: Some(0.0),
        max_tokens: Some(OVERSIZE_MAX_TOKENS),
    };
    match llm.chat(request).await {
        Ok(_) => Ok(None),
        Err(err @ ProbeError::Auth(_)) | Err(err @ ProbeError::NotFound(_)) => Err(err),
        Err(err @ ProbeError::Llm(_)) => Ok(parse_max_output_cap(&err.to_string())),
        Err(err) => Err(err),
    }
}

/// Parse a provider reject for the allowed `max_tokens` ceiling.
pub fn parse_max_output_cap(err: &str) -> Option<u32> {
    let lower = err.to_ascii_lowercase();
    if !mentions_output_budget(&lower) {
        return None;
    }
    let n = parse_greater_than(err).or_else(|| parse_named_maximum(err))?;
    if n == 0 || n == OVERSIZE_MAX_TOKENS {
        return None;
    }
    Some(n)
}

fn mentions_output_budget(lower: &str) -> bool {
    lower.contains("max_tokens")
        || lower.contains("max_completion_tokens")
        || lower.contains("maxoutputtokens")
        || lower.contains("output token limit")
        || lower.contains("output_token_limit")
}

fn parse_greater_than(err: &str) -> Option<u32> {
    for (i, _) in err.char_indices() {
        if let Some((left, after_left)) = take_u32(&err[i..]) {
            let rest = err[i + after_left..].trim_start();
            if let Some(rest) = rest.strip_prefix('>') {
                let rest = rest.trim_start();
                if let Some((right, _)) = take_u32(rest) {
                    // Status-code pairs (HTTP 502 > 400) are not the output cap.
                    if left != OVERSIZE_MAX_TOKENS && right != OVERSIZE_MAX_TOKENS {
                        continue;
                    }
                    let cap = left.min(right);
                    if cap > 0 {
                        return Some(cap);
                    }
                }
            }
        }
    }
    None
}

fn parse_named_maximum(err: &str) -> Option<u32> {
    let lower = err.to_ascii_lowercase();
    for needle in [
        "maximum is ",
        "max is ",
        "at most ",
        "less than or equal to ",
        "must be <= ",
        "must be <=",
        "output token limit is ",
        "output_token_limit is ",
    ] {
        if let Some(idx) = lower.find(needle) {
            let rest = &err[idx + needle.len()..];
            if let Some((n, _)) = take_u32(rest.trim_start()) {
                if n > 0 {
                    return Some(n);
                }
            }
        }
    }
    None
}

fn take_u32(s: &str) -> Option<(u32, usize)> {
    let trimmed = s.trim_start();
    let skip = s.len() - trimmed.len();
    let digits: String = trimmed.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    let n = digits.parse().ok()?;
    Some((n, skip + digits.len()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::{MockLlm, ProbeClient, ProbeResponse};
    use crate::error::ProbeError;
    use std::future::Future;

    #[test]
    fn parse_anthropic_greater_than() {
        assert_eq!(parse_max_output_cap("max_tokens: 32768 > 8192"), Some(8192));
    }

    #[test]
    fn parse_openai_maximum_is() {
        assert_eq!(
            parse_max_output_cap("max_tokens is too large: 32768. This model's maximum is 4096"),
            Some(4096)
        );
    }

    #[test]
    fn parse_less_than_or_equal() {
        assert_eq!(
            parse_max_output_cap("'max_tokens' must be less than or equal to 4096"),
            Some(4096)
        );
    }

    #[test]
    fn parse_ignores_unrelated_errors() {
        assert_eq!(parse_max_output_cap("rate limited"), None);
        assert_eq!(parse_max_output_cap("context_length 200000"), None);
        assert_eq!(
            parse_max_output_cap("max_tokens request failed; rate limit is 10000 TPM"),
            None
        );
    }

    #[test]
    fn parse_does_not_use_input_window_words_alone() {
        assert_eq!(
            parse_max_output_cap("advertised context window is 200000 tokens"),
            None
        );
        assert_eq!(
            parse_max_output_cap("prompt exceeds maximum context 32768 > 8192"),
            None
        );
        assert_eq!(
            parse_max_output_cap("prompt + completion_tokens exceed maximum context 32768 > 8192"),
            None
        );
        assert_eq!(
            parse_max_output_cap(
                r#"{"error":{"message":"context token limit is 128000","param":"max_tokens"}}"#
            ),
            None
        );
    }

    #[test]
    fn parse_ignores_the_oversize_ask_as_the_cap() {
        assert_eq!(parse_max_output_cap("max_tokens: 32768 > 32768"), None);
    }

    #[test]
    fn parse_greater_than_skips_non_ascii_without_panic() {
        assert_eq!(
            parse_max_output_cap("max_tokens \u{2014} 32768 > 8192"),
            Some(8192)
        );
    }

    #[test]
    fn parse_skips_http_status_greater_than_for_named_maximum() {
        assert_eq!(
            parse_max_output_cap(
                "LLM error: HTTP 502 > 400; max_tokens is too large. This model's maximum is 8192"
            ),
            Some(8192)
        );
    }

    #[test]
    fn parse_json_400_body_that_names_an_output_budget() {
        assert_eq!(
            parse_max_output_cap(
                r#"{"error":{"message":"max_tokens is too large: 32768. This model's maximum is 4096","code":"invalid_request_error"}}"#
            ),
            Some(4096)
        );
    }

    #[test]
    fn parse_output_token_limit_is() {
        assert_eq!(
            parse_max_output_cap("output token limit is 8192"),
            Some(8192)
        );
        assert_eq!(
            parse_max_output_cap("output_token_limit is 2048"),
            Some(2048)
        );
    }

    #[test]
    fn parse_wrapped_reject_one_layer_down() {
        assert_eq!(
            parse_max_output_cap("error decoding response body: max_tokens: 32768 > 8192"),
            Some(8192)
        );
    }

    struct RejectLlm(&'static str);

    impl ProbeClient for RejectLlm {
        fn chat(
            &self,
            _req: crate::client::ProbeRequest,
        ) -> impl Future<Output = Result<ProbeResponse, ProbeError>> + Send {
            let msg = self.0;
            async move { Err(ProbeError::Llm(msg.to_owned())) }
        }

        fn stream_chat(
            &self,
            _req: crate::client::ProbeRequest,
        ) -> impl futures::Stream<Item = Result<crate::client::ProbeStreamChunk, ProbeError>> + Send
        {
            futures::stream::empty()
        }

        fn model_id(&self) -> &str {
            "m"
        }

        fn provider(&self) -> &str {
            "p"
        }
    }

    #[tokio::test]
    async fn reject_yields_measured_cap() {
        let llm = RejectLlm("max_tokens: 32768 > 4096");
        assert_eq!(probe_max_output_tokens(&llm).await.unwrap(), Some(4096));
    }

    #[tokio::test]
    async fn success_is_unmeasured() {
        let llm = MockLlm::new("m", "p");
        assert_eq!(probe_max_output_tokens(&llm).await.unwrap(), None);
    }

    #[tokio::test]
    async fn auth_aborts() {
        let llm = MockLlm::new("m", "p").with_error(ProbeError::Auth("no".into()));
        let err = probe_max_output_tokens(&llm).await.unwrap_err();
        assert!(matches!(err, ProbeError::Auth(_)));
    }

    #[tokio::test]
    async fn rate_limit_is_err_not_unmeasured() {
        let llm = MockLlm::new("m", "p").with_error(ProbeError::RateLimit { retry_after: None });
        let err = probe_max_output_tokens(&llm).await.unwrap_err();
        assert!(
            matches!(err, ProbeError::RateLimit { .. }),
            "429 on the oversize ask must not become Ok(None): {err:?}"
        );
    }

    #[tokio::test]
    async fn transient_is_err_not_unmeasured() {
        let llm = MockLlm::new("m", "p").with_error(ProbeError::Transient("overload".into()));
        let err = probe_max_output_tokens(&llm).await.unwrap_err();
        assert!(
            matches!(err, ProbeError::Transient(_)),
            "transient on the oversize ask must not become Ok(None): {err:?}"
        );
    }
}

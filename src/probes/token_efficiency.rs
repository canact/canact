//! Token efficiency probe.
//!
//! Measures how concise the model is for a simple factual question.
//! Verbose models burn through context windows faster and cost more.

use crate::ProbeError;
use crate::client::{ProbeClient, ProbeRequest};
use crate::types::{ProbeResult, classify};

use super::user_text;

/// Probe how many output tokens the model uses for a trivial question.
///
/// Asks "What is 2+2?" with no formatting constraints (the instruction-
/// following probe already tests constraint adherence). This measures
/// the model's *natural* verbosity.
///
/// Scoring uses provider completion tokens when the host reported them.
/// Fall back to a character estimate (`len.div_ceil(4)`) when usage is
/// missing. Reasoning tokens are reported separately, else 0.
///
/// - `1.0` - completion <= 10 (concise)
/// - `0.5` - 11-50 (moderate)
/// - `0.0` - > 50 (verbose)
pub async fn probe_token_efficiency<C: ProbeClient>(llm: &C) -> Result<ProbeResult, ProbeError> {
    let request = ProbeRequest {
        messages: vec![user_text("What is 2+2?")],
        tools: vec![],
        model: llm.model_id().to_string(),
        temperature: Some(0.0),
        max_tokens: Some(256),
    };

    let response = llm.chat(request).await?;
    let empty_text = response.text.trim().is_empty();

    let (completion_tokens, reasoning_tokens) = match response.usage.as_ref() {
        Some(usage) => {
            let completion = match usage.completion_tokens {
                Some(n) if n > 0 => n,
                _ if empty_text => u32::MAX,
                Some(n) => n,
                None => (response.text.len() as u32).div_ceil(4),
            };
            (completion, usage.reasoning_tokens.unwrap_or(0))
        }
        None if empty_text => (u32::MAX, 0),
        None => ((response.text.len() as u32).div_ceil(4), 0),
    };

    let band = if completion_tokens <= 10 {
        "concise"
    } else if completion_tokens <= 50 {
        "moderate"
    } else {
        "verbose"
    };
    let score = if completion_tokens <= 10 {
        1.0
    } else if completion_tokens <= 50 {
        0.5
    } else {
        0.0
    };
    let details = format!(
        "completion={completion_tokens} reasoning={reasoning_tokens} \
         (effort=unset, score on completion tokens only, {band})"
    );

    Ok(ProbeResult {
        name: "token_efficiency".to_string(),
        score,
        max_score: 1.0,
        level: classify(score),
        details,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::{ProbeFinish, ProbeResponse, ProbeUsage};
    use crate::probes::test_support::*;
    use crate::types::CapabilityLevel;

    #[tokio::test]
    async fn concise_response_is_strong() {
        let llm = MockLlm {
            response: text_response("4"),
        };
        let result = probe_token_efficiency(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Strong);
        assert!(
            result.details.contains("completion=")
                && result.details.contains("reasoning=")
                && result.details.contains("effort=unset"),
            "details must expose completion/reasoning/effort (#1340): {}",
            result.details
        );
    }

    #[tokio::test]
    async fn verbose_response_is_weak() {
        let verbose = "Well, that's a great question! Let me think about this carefully. \
            The sum of two plus two is four. This is a basic arithmetic operation \
            that forms the foundation of mathematics. In fact, addition is one of \
            the four basic operations, along with subtraction, multiplication, and division.";
        let llm = MockLlm {
            response: text_response(verbose),
        };
        let result = probe_token_efficiency(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Weak);
    }

    #[tokio::test]
    async fn live_completion_tokens_override_short_text() {
        let llm = MockLlm {
            response: ProbeResponse {
                text: "4".into(),
                tool_calls: Vec::new(),
                finish: ProbeFinish::Stop,
                usage: Some(ProbeUsage {
                    prompt_tokens: Some(12),
                    completion_tokens: Some(80),
                    reasoning_tokens: Some(40),
                }),
            },
        };
        let result = probe_token_efficiency(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
        assert!(
            result.details.contains("completion=80")
                && result.details.contains("reasoning=40")
                && result.details.contains("verbose"),
            "{}",
            result.details
        );
    }

    #[tokio::test]
    async fn empty_text_without_usage_is_weak() {
        let llm = MockLlm {
            response: text_response(""),
        };
        let result = probe_token_efficiency(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn whitespace_only_without_usage_is_weak() {
        let llm = MockLlm {
            response: text_response("  \n\t"),
        };
        let result = probe_token_efficiency(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn empty_text_zero_completion_tokens_is_weak() {
        let llm = MockLlm {
            response: ProbeResponse {
                text: String::new(),
                tool_calls: Vec::new(),
                finish: ProbeFinish::Stop,
                usage: Some(ProbeUsage {
                    prompt_tokens: Some(8),
                    completion_tokens: Some(0),
                    reasoning_tokens: Some(0),
                }),
            },
        };
        let result = probe_token_efficiency(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Weak);
    }

    #[tokio::test]
    async fn missing_usage_keeps_char_estimate() {
        let llm = MockLlm {
            response: text_response("4"),
        };
        let result = probe_token_efficiency(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Strong);
        assert!(
            result.details.contains("completion=1") && result.details.contains("reasoning=0"),
            "{}",
            result.details
        );
    }
}

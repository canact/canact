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
/// Scoring uses **visible completion tokens only**. The host client does
/// not yet report usage, so this probe estimates tokens from character
/// length (`len.div_ceil(4)`). Reasoning tokens are reported as 0.
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

    let completion_tokens = (response.text.len() as u32).div_ceil(4);
    let reasoning_tokens = 0u32;

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
         (effort=unset, score on visible completion only, {band})"
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
}
